"""Check the recorded attention batch size against the exact frozen source."""
def checked_rows(metadata,source_base):
    rows=metadata.get('attention_prefill_max_rows',64)
    assert rows in [64,128]
    if rows==128:
        backend=(source_base/'source/fp16/fp16_backend.c').read_text()
        attention=(source_base/'source/fp16/attention_backend.h').read_text()
        assert 'p->max_rows = K <= 512 ? 128 : (K > 2048 ? 16 : (K > 1024 ? 32 : 64));' in backend
        assert 'if (!banks || banks > 4)' in backend
        assert 'attention_prob[i]=malloc(128*512*4);' in attention
        assert 'for(int start=0;start<rows;start+=128) {' in attention
        assert 'int m=rows-start<128?rows-start:128;' in attention
    return rows
