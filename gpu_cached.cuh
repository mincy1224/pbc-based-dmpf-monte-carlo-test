#pragma once
#include "cuckoo_core.hpp"

// Each table row contains 32 copies. Lane L always accesses bank L, even when
// every lane requests a different S-box index. Only T0 is stored; others rotate.
CUCKOO_CORE uint32_t lane_te(const uint32_t* table,unsigned lane,unsigned index) {
    return table[index*32+lane];
}
CUCKOO_CORE uint32_t rotate8(uint32_t x) {return (x>>8)|(x<<24);}
CUCKOO_CORE uint32_t rotate16(uint32_t x) {return (x>>16)|(x<<16);}
CUCKOO_CORE uint32_t rotate24(uint32_t x) {return (x>>24)|(x<<8);}
CUCKOO_CORE uint32_t lane_sb(const uint32_t* table,unsigned lane,unsigned index) {
    return (lane_te(table,lane,index)>>16)&255;
}
CUCKOO_CORE uint32_t lane_subword(uint32_t x,const uint32_t* table,unsigned lane) {
    return (lane_sb(table,lane,x>>24)<<24)|(lane_sb(table,lane,(x>>16)&255)<<16)|
           (lane_sb(table,lane,(x>>8)&255)<<8)|lane_sb(table,lane,x&255);
}
CUCKOO_CORE void lane_expand(uint32_t a,uint32_t b,uint32_t c,uint32_t d,
                             const uint32_t* table,unsigned lane,AesKey& key) {
    key.w[0]=a;key.w[1]=b;key.w[2]=c;key.w[3]=d;uint32_t rcon=1;
#ifdef __CUDACC__
    #pragma unroll
#endif
    for(int i=4;i<44;++i) {
        uint32_t v=key.w[i-1];
        if((i&3)==0) {
            v=lane_subword((v<<8)|(v>>24),table,lane)^(rcon<<24);
            rcon=(rcon<<1)^((rcon&0x80)?0x1b:0);
        }
        key.w[i]=key.w[i-4]^v;
    }
}
CUCKOO_CORE void lane_aes(const AesKey& key,const uint32_t* table,unsigned lane,
                          uint32_t a,uint32_t b,uint32_t c,uint32_t d,uint32_t out[4]) {
    uint32_t s0=a^key.w[0],s1=b^key.w[1],s2=c^key.w[2],s3=d^key.w[3];
#ifdef __CUDACC__
    #pragma unroll
#endif
    for(unsigned r=1;r<10;++r) {
        const uint32_t t0=lane_te(table,lane,s0>>24)^rotate8(lane_te(table,lane,(s1>>16)&255))^
            rotate16(lane_te(table,lane,(s2>>8)&255))^rotate24(lane_te(table,lane,s3&255))^key.w[4*r];
        const uint32_t t1=lane_te(table,lane,s1>>24)^rotate8(lane_te(table,lane,(s2>>16)&255))^
            rotate16(lane_te(table,lane,(s3>>8)&255))^rotate24(lane_te(table,lane,s0&255))^key.w[4*r+1];
        const uint32_t t2=lane_te(table,lane,s2>>24)^rotate8(lane_te(table,lane,(s3>>16)&255))^
            rotate16(lane_te(table,lane,(s0>>8)&255))^rotate24(lane_te(table,lane,s1&255))^key.w[4*r+2];
        const uint32_t t3=lane_te(table,lane,s3>>24)^rotate8(lane_te(table,lane,(s0>>16)&255))^
            rotate16(lane_te(table,lane,(s1>>8)&255))^rotate24(lane_te(table,lane,s2&255))^key.w[4*r+3];
        s0=t0;s1=t1;s2=t2;s3=t3;
    }
    out[0]=((lane_sb(table,lane,s0>>24)<<24)|(lane_sb(table,lane,(s1>>16)&255)<<16)|
            (lane_sb(table,lane,(s2>>8)&255)<<8)|lane_sb(table,lane,s3&255))^key.w[40];
    out[1]=((lane_sb(table,lane,s1>>24)<<24)|(lane_sb(table,lane,(s2>>16)&255)<<16)|
            (lane_sb(table,lane,(s3>>8)&255)<<8)|lane_sb(table,lane,s0&255))^key.w[41];
    out[2]=((lane_sb(table,lane,s2>>24)<<24)|(lane_sb(table,lane,(s3>>16)&255)<<16)|
            (lane_sb(table,lane,(s0>>8)&255)<<8)|lane_sb(table,lane,s1&255))^key.w[42];
    out[3]=((lane_sb(table,lane,s3>>24)<<24)|(lane_sb(table,lane,(s0>>16)&255)<<16)|
            (lane_sb(table,lane,(s1>>8)&255)<<8)|lane_sb(table,lane,s2&255))^key.w[43];
}

#ifndef CUCKOO_HOST_TEST
__global__ void cached_aes_kat(AesKey input,uint32_t* out) {
    extern __shared__ uint32_t replicated[];
    for(unsigned i=threadIdx.x;i<8192;i+=blockDim.x) replicated[i]=c_te[i/32];
    __syncthreads();
    const unsigned lane=threadIdx.x&31;
    if(threadIdx.x<32) {
        AesKey expanded;lane_expand(input.w[0],input.w[1],input.w[2],input.w[3],replicated,lane,expanded);
        lane_aes(expanded,replicated,lane,0x00112233u,0x44556677u,0x8899aabbu,0xccddeeffu,out+lane*4);
    }
}
template<bool BGH>
__global__ void cuckoo_cached(uint32_t count,uint32_t first,uint32_t pnum,uint32_t buckets,
        uint32_t kick,uint64_t width,uint64_t domain,uint32_t half,uint64_t mask,
        uint64_t range_mask,uint32_t input_mode,AesKey batch_key,size_t pitch,
        uint32_t* tables,uint16_t* rounds,uint32_t* failed_count,uint32_t* failed_offsets) {
    extern __shared__ uint32_t replicated[];
    for(unsigned i=threadIdx.x;i<8192;i+=blockDim.x) replicated[i]=c_te[i/32];
    __syncthreads();
    const unsigned tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=count) return;
    const unsigned lane=threadIdx.x&31,offset=first+tid;
    const StridedBins bins{tables+tid,pitch};
    for(unsigned k=0;k<buckets;++k) bins[k]=0xffffffffu;
    uint32_t seed[4];lane_aes(batch_key,replicated,lane,0x4b455931u,0,offset,0x50525031u,seed);
    AesKey key;lane_expand(seed[0],seed[1],seed[2],seed[3],replicated,lane,key);
    lane_aes(batch_key,replicated,lane,0x57414c4bu,0,offset,0x524e4731u,seed);
    uint64_t walk=(uint64_t(seed[0])<<32)|seed[1];
    const uint64_t shift=((uint64_t(seed[2])<<32)|seed[3])&range_mask;
    uint64_t input_state=shift^0xd1b54a32d192ed03ULL;
    const uint64_t step=(core_splitmix(input_state)|1ULL)&range_mask;
    const uint64_t randomness=core_splitmix(walk);
    uint16_t* values=rounds+tid;
    #pragma unroll 1
    for(unsigned r=0;r<8;++r) {
        #pragma unroll 1
        for(unsigned right=0;right<(1u<<half);++right) {
            uint32_t output[4];lane_aes(key,replicated,lane,0x43455350u,r,half,right,output);
            values[((size_t(r)<<half)+right)*pitch]=static_cast<uint16_t>(output[3]&mask);
        }
    }
    const CachedBuckets bucket{values,pitch,domain,width,mask,range_mask,shift,step,half,input_mode==0};
    if(insert_core<BGH>(pnum,kick,bins,bucket,walk,randomness))
        failed_offsets[atomicAdd(failed_count,1u)]=offset;
}
#endif
