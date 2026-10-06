#!/usr/bin/env python3
"""Convert a qwen3.c Q8 checkpoint into matched Core ML decoder blocks."""
import argparse
import gc
import importlib.metadata
import os
import platform
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
import coremltools as ct

from .common import Checkpoint, ROOT, sha256, source_hashes, write_json
from .model import Block, inputs


def share_identical_weights(directory, source):
    """Share immutable coefficients only after checking their complete bytes."""
    shared = []
    for path in directory.rglob('*'):
        if not path.is_file() or (path.suffix != '.npy' and path.name != 'weight.bin'):
            continue
        relative = path.relative_to(directory)
        original = source/relative
        if not original.is_file() or path.samefile(original):
            continue
        if path.stat().st_size != original.stat().st_size or sha256(path) != sha256(original):
            continue
        replacement = path.with_name(path.name+'.shared')
        os.link(original, replacement)
        replacement.replace(path)
        shared.append(str(relative))
    return shared


def export(checkpoint_path, output, context, batch, block_layers, reuse_weights_from=None, single_function=False):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('output already exists; use a new model directory')
    if min(context, batch, block_layers) < 1:
        raise ValueError('context, batch and block size must be positive')
    checkpoint = Checkpoint(checkpoint_path)
    if context > checkpoint.config['context']:
        raise ValueError('context exceeds checkpoint capacity')
    c = checkpoint.config
    capacity = context + batch - 1  # Padding never overwrites a logical cache slot.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='qwen3-coreml-', dir=output.parent))
    torch.set_num_threads(4)
    shared_weights = []
    try:
        for name in ('embedding', 'head'):
            if name == 'head' and c['tied']:
                continue
            np.save(temporary / (name + '.npy'), checkpoint.matrix(name))
        np.save(temporary / 'final_norm.npy', checkpoint.norms['final'])
        if reuse_weights_from is not None:
            shared_weights += share_identical_weights(temporary, Path(reuse_weights_from))
        artifacts = []
        for start in range(0, c['layers'], block_layers):
            stop = min(start + block_layers, c['layers'])
            block = Block(checkpoint, start, stop).eval()
            functions = []
            for count in ([batch] if single_function else sorted(set([1, batch]))):
                print(f'Converting layers {start}:{stop}, input width {count}', flush=True)
                values = dict(hidden=np.zeros((1, count, c['dim']), 'f4'),
                              keys=np.zeros((stop-start, 1, c['kv_heads'], capacity, c['head_dim']), 'f4'),
                              values=np.zeros((stop-start, 1, c['kv_heads'], capacity, c['head_dim']), 'f4'),
                              **inputs(c, count, 0, capacity))
                args = tuple(torch.from_numpy(values[name]) for name in values)
                traced = torch.jit.trace(block, args, check_trace=False)
                converted = ct.convert(traced, inputs=[ct.TensorType(name=k, shape=v.shape,
                                       dtype=np.int32 if k == 'indices' else np.float32) for k,v in values.items()],
                                       outputs=[ct.TensorType(name='hidden_out'), ct.TensorType(name='keys_out'),
                                                ct.TensorType(name='values_out')],
                                       minimum_deployment_target=ct.target.macOS15, convert_to='mlprogram',
                                       compute_precision=ct.transform.FP16ComputePrecision(
                                           op_selector=lambda op: op.op_type in ('linear', 'conv')),
                                       skip_model_load=True)
                name = f'block-{start:02d}-{stop:02d}-s{count}.mlpackage'
                converted.save(str(temporary / name))
                functions.append((count, name))
                del converted, traced, args, values
                gc.collect()
            shared_name = f'block-{start:02d}-{stop:02d}.mlpackage'
            if single_function:
                (temporary/functions[0][1]).rename(temporary/shared_name)
                artifacts.append(dict(start=start, stop=stop, width=batch, package=shared_name, function='main'))
            else:
                descriptor = ct.utils.MultiFunctionDescriptor()
                for count, name in functions:
                    function = f'width_{count}'
                    descriptor.add_function(str(temporary/name), 'main', function)
                    artifacts.append(dict(start=start, stop=stop, width=count, package=shared_name,
                                          function=function))
                descriptor.default_function_name = 'width_1'
                ct.utils.save_multifunction(descriptor, str(temporary/shared_name))
                for _, name in functions:
                    shutil.rmtree(temporary/name)
                del descriptor
            if reuse_weights_from is not None:
                shared_weights += share_identical_weights(temporary, Path(reuse_weights_from))
            del block
            gc.collect()
        file_hashes = {str(p.relative_to(temporary)): sha256(p) for p in temporary.rglob('*') if p.is_file()}
        manifest = dict(protocol='qwen3-coreml-v1', checkpoint_sha256=sha256(checkpoint.path),
                        checkpoint_name=checkpoint.path.name, config=c, context=context, batch=batch,
                        cache_capacity=capacity, block_layers=block_layers, artifacts=artifacts,
                        arithmetic='Q8 weights dequantized and rounded to FP16; exact per-row coefficient gains '
                                   '1..32 with FP32 restoration; FP16 static projections; FP32 attention and pointwise math',
                        host_operations=['embedding', 'final RMSNorm', 'FP32 vocabulary projection', 'argmax'],
                        file_sha256=file_hashes,
                        shared_weight_files=shared_weights,
                        source_sha256=source_hashes(),
                        packages={n: importlib.metadata.version(n) for n in ('numpy','torch','coremltools')},
                        platform=platform.platform())
        write_json(temporary/'manifest.json', manifest)
        temporary.rename(output)
        print(f'Saved {output}', flush=True)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('checkpoint', type=Path)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--context', type=int, default=512)
    cli.add_argument('--batch', type=int, default=32)
    cli.add_argument('--block-layers', type=int, default=4)
    cli.add_argument('--reuse-weights-from', type=Path,
                     help='share identical immutable coefficient files from an existing conversion')
    cli.add_argument('--single-function', action='store_true',
                     help='export only the fixed batch width used by padded prefill and decode')
    args = cli.parse_args()
    export(args.checkpoint, args.output, args.context, args.batch, args.block_layers,
           args.reuse_weights_from, args.single_function)


if __name__ == '__main__':
    main()
