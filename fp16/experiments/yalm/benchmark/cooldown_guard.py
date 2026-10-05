"""Bound only the idle thermal wait; normal device cleanup follows on timeout."""
def bounded(source):
    def change(old,new):
        nonlocal source
        assert source.count(old)==1,old
        source=source.replace(old,new)
    change('    for (int run = 1-warmups; run <= runs; run++) {',
           '    int request_status=0;\n    for (int run = 1-warmups; run <= runs; run++) {')
    change('                if(!f||fscanf(f,"%d",&temp)!=1)exit(1);fclose(f);',
           '                if(!f||fscanf(f,"%d",&temp)!=1){if(f)fclose(f);request_status=1;goto request_cleanup;}fclose(f);')
    change('                if(!initial)initial=temp;if(temp>target)usleep(500000);',
        '''                if(!initial)initial=temp;
                if(temp>target){
                    if(now_ms()-cool_start>=180000){
                        fprintf(stderr,"{\\"event\\":\\"cooldown_abort\\",\\"run\\":%d,\\"target_millidegrees\\":%d,\\"temperature_millidegrees\\":%d,\\"wait_ms\\":%.3f}\\n",run,target,temp,now_ms()-cool_start);
                        request_status=1;goto request_cleanup;
                    }
                    usleep(500000);
                }''')
    change('    gpu_attention_shutdown();','request_cleanup:\n    gpu_attention_shutdown();')
    change('    free_transformer(&t);\n    return 0;','    free_transformer(&t);\n    return request_status;')
    return source
