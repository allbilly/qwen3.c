# Milestone protocol

1. Primitive/backend WIP: commits 84b02da and 1c94b6e. The exported route target builds byte-identically to external benchmark binary 11d2f9b7. Six CPU and six GPU projection checks passed.
2. Phase placement WIP: all nine dense prefill/decode pairs and one explicit three-device placement, 128/256 inputs and 32 output tokens. Reject rows with any non-target clock; retain unaffected complete jobs transparently. Two warmups and two measured requests. The final CPU256 row needs a lower-temperature retry.
3. Full phase costs WIP: complete 14 root stages plus CPU expansion/BLAS/layout, OpenCL upload/kernel/download, and NPU native packing/copy/sync/driver/unpack timers. Diagnostic profile before final complete-request decisions.
4. Independent optimizations WIP: parallel GPU softmax, fused softmax/PV, GPU 16-row register reuse, shared eight-head NPU KV packing. Primitive gates are unchanged. GPU attention candidates passed eight cases each; GPU row-reuse passed six actual-weight projection cases. Full-model quality and phase/performance tests still required.
5. Complete-request WIP: one direct monotonic clock from prefill start through final greedy output. Covers transfers, waits, packing and device handoffs, not just the sum of separately rounded phases. Weights resident; initialization is separate. All placements use the same successful thermal protocol and fixed clocks.

Rejected data remain immutable. The selected runner and existing paired-stock claims stay tied to their original exact source/binary hashes. No push requested.

INTENT: the selected runner has NPU projections and a tested GPU-attention route but no full CPU/GPU projection comparison; the user requests device combinations and stepwise profiling/optimization; fp16/README.md specifies the shared FP16 checkpoint, native NPU projection reference and numerical verification.

AUTH: user said "wip commit per milestone".
