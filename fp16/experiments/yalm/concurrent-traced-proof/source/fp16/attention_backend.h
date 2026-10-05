// Prompt attention uses FP16 operands and FP32 NPU accumulation, as in the
// captured stock attention matmuls. CPU keeps causal masking and softmax.
static Plan *attention_score[3],*attention_value[3];
static float *attention_prob[3];
static int attention_enabled;
extern void softmax(float *,int);

int npu_attention_enabled(void) { return attention_enabled; }

static void attention_shape(Plan *p,int K,int N) {
    if(p->K==K && p->N==N)return;
    p->K=K;p->N=N;p->current_rows=-1;
    uint64_t *commands=p->mem.weights;
    struct rknpu_task *task=p->mem.tasks;
    for(uint32_t i=0;i<task[0].regcfg_amount;i++) {
        uint32_t address=commands[i]&65535,value=(commands[i]>>16)&0xffffffff;
        switch(address) {
        case REG_CNA_DATA_SIZE1:value=CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(K-1)|CNA_DATA_SIZE1_DATAIN_CHANNEL(K);break;
        case REG_CNA_WEIGHT_SIZE0:value=K*N*2;break;
        case REG_CNA_WEIGHT_SIZE1:value=CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL(K*2);break;
        case REG_CNA_WEIGHT_SIZE2:value=CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(1)|CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(1)|CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(N);break;
        case REG_CNA_FC_DATA_SIZE1:value=CNA_FC_DATA_SIZE1_DMA_CHANNEL(K);break;
        case REG_CORE_DATAOUT_SIZE_1:value=CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(N-1);break;
        case REG_DPU_DATA_CUBE_CHANNEL:value=DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(N-1)|DPU_DATA_CUBE_CHANNEL_CHANNEL(N-1);break;
        case REG_DPU_WDMA_SIZE_0:value=DPU_WDMA_SIZE_0_CHANNEL_WDMA(N-1);break;
        }
        commands[i]=(commands[i]&0xffff00000000ffffull)|((uint64_t)value<<16);
    }
}

static int attention_prepare(void) {
    __fp16 *zero=calloc(512*128,2);
    if(!zero)return 0;
    for(int i=0;i<3;i++) {
        attention_score[i]=prepare(zero,128,512,0);
        attention_value[i]=prepare(zero,512,128,0);
        attention_prob[i]=malloc(64*512*4);
        if(!attention_score[i]||!attention_value[i]||!attention_prob[i]){free(zero);return 0;}
        if(i){attention_score[i-1]->next=attention_score[i];attention_value[i-1]->next=attention_value[i];}
    }
    free(zero);
    prepare_bundles(attention_score[0]);prepare_bundles(attention_value[0]);
    return 1;
}

static void attention_release(void) {
    for(int i=0;i<3;i++) {
        Plan *p[2]={attention_score[i],attention_value[i]};
        for(int j=0;j<2;j++)if(p[j]) {
            release_memhandles(device_fd,&p[j]->mem);free(p[j]->packed_input);free(p[j]->native_output);free(p[j]);
        }
        free(attention_prob[i]);attention_prob[i]=NULL;
        attention_score[i]=attention_value[i]=NULL;
    }
    attention_enabled=0;
}

static void attention_weights(Plan *score,Plan *value,const float *keys,const float *values,
                              int rows,int kv_stride,int kv_head) {
    __fp16 *kw=(__fp16*)((char*)score->mem.weights+REGCMD_RESERVED);
    __fp16 *vw=(__fp16*)((char*)value->mem.weights+REGCMD_RESERVED);
    memset(kw,0,score->N*128*2);memset(vw,0,value->K*128*2);
    for(int time=0;time<rows;time++) {
        const float *k=keys+(size_t)time*kv_stride+kv_head*128;
        const float *v=values+(size_t)time*kv_stride+kv_head*128;
        for(int feature=0;feature<128;feature+=8) {
            float16x8_t h=vcombine_f16(vcvt_f16_f32(vld1q_f32(k+feature)),vcvt_f16_f32(vld1q_f32(k+feature+4)));
            vst1q_f16(kw+weight_fp16(128,time+1,feature+1),h);
        }
        for(int feature=0;feature<128;feature++)vw[weight_fp16(value->K,feature+1,time+1)]=(__fp16)v[feature];
    }
}

static void attention_pack(Plan *p,const float *source,int stride,int rows) {
    for(int k=0;k<p->K;k+=8)for(int row=0;row<rows;row++) {
        const float *s=source+(size_t)row*stride+k;
        float16x8_t h=vcombine_f16(vcvt_f16_f32(vld1q_f32(s)),vcvt_f16_f32(vld1q_f32(s+4)));
        vst1q_f16(p->packed_input+(size_t)(k/8)*rows*8+row*8,h);
    }
    memcpy(p->mem.input,p->packed_input,(size_t)rows*p->K*2);
    mem_sync(device_fd,p->mem.input_obj,0,(size_t)rows*p->K*2,RKNPU_MEM_SYNC_TO_DEVICE);
}

static void attention_read(Plan *p,float *destination,int stride,int rows) {
    mem_sync(device_fd,p->mem.output_obj,0,(size_t)rows*p->N*4,RKNPU_MEM_SYNC_FROM_DEVICE);
    memcpy(p->native_output,p->mem.output,(size_t)rows*p->N*4);
    for(int row_block=0;row_block<rows;row_block+=4)for(int n=0;n<p->N;n+=16)
        for(int row=row_block;row<rows && row<row_block+4;row++)for(int channel=0;channel<16;channel+=4)
            vst1q_f32(destination+(size_t)row*stride+n+channel,
                       vld1q_f32(p->native_output+(size_t)((n+channel)/4)*rows*4+row*4));
}

int npu_attention_prefill(NpuMatmulContext *ctx,const float *queries,const float *keys,
                          const float *values,float *output,int rows) {
    if(!attention_enabled||rows<1||rows>512)exit(1);
    int qdim=ctx->all_heads_dim,kdim=ctx->kv_dim;
    int padded_rows=(rows+31)&~31;
    for(int i=0;i<3;i++) {
        attention_shape(attention_score[i],128,padded_rows);
        attention_shape(attention_value[i],padded_rows,128);
    }
    for(int head=0;head<16;head+=3) {
        int count=16-head<3?16-head:3;
        attention_score[0]->bundle_count=count;attention_value[0]->bundle_count=count;
        // CPU writes independent weight maps in parallel; all ioctls stay serial.
        #pragma omp parallel for
        for(int i=0;i<count;i++)attention_weights(attention_score[i],attention_value[i],keys,values,rows,kdim,(head+i)/2);
        for(int i=0;i<count;i++) {
            mem_sync(device_fd,attention_score[i]->mem.weights_obj,REGCMD_RESERVED,attention_score[i]->N*128*2,RKNPU_MEM_SYNC_TO_DEVICE);
            mem_sync(device_fd,attention_value[i]->mem.weights_obj,REGCMD_RESERVED,attention_value[i]->K*128*2,RKNPU_MEM_SYNC_TO_DEVICE);
        }
        for(int start=0;start<rows;start+=64) {
            int m=rows-start<64?rows-start:64;
            for(int i=0;i<count;i++) {
                configure_rows(attention_score[i],m,1);configure_rows(attention_value[i],m,1);
                attention_pack(attention_score[i],queries+(size_t)start*qdim+(head+i)*128,qdim,m);
            }
            if(submit_bundle(attention_score[0])<0){perror("NPU attention scores");exit(1);}
            for(int i=0;i<count;i++)attention_read(attention_score[i],attention_prob[i],512,m);
            #pragma omp parallel for collapse(2)
            for(int i=0;i<count;i++)for(int row=0;row<m;row++) {
                float *prob=attention_prob[i]+row*512;
                int length=start+row+1;
                for(int t=0;t<length;t++)prob[t]/=sqrtf(128);
                softmax(prob,length);
                memset(prob+length,0,(512-length)*4);
            }
            for(int i=0;i<count;i++)attention_pack(attention_value[i],attention_prob[i],512,m);
            if(submit_bundle(attention_value[0])<0){perror("NPU attention values");exit(1);}
            for(int i=0;i<count;i++)attention_read(attention_value[i],output+(size_t)start*qdim+(head+i)*128,qdim,m);
            ctx->npu_ops+=2;
        }
    }
    return 1;
}
