#pragma once
#include "cuckoo_core.hpp"

// Warp-private trial state.  One warp processes one Monte Carlo experiment:
// its bucket table and its PRP round cache both live in shared memory.  The
// previous global layout stored 24 KiB of bins plus 8 KiB of rounds *per
// thread*; this one stores 20 KiB *per warp*, removes random global loads from
// the insertion loop, and lets a single block host several independent trials.

// Launch mapping shared by the kernel and its host test.  The micro-launch
// bounds cover [first, first+count); the block-local trial index must be
// compared with count, not with the absolute offset.
CUCKOO_HD inline bool shared_trial_index(uint32_t block,uint32_t warp,uint32_t warps,
                                         uint32_t count,uint32_t first,uint32_t& trial) {
    const uint32_t local=block*warps+warp;
    if(local>=count) return false;
    trial=first+local;
    return true;
}

#ifndef CUCKOO_HOST_TEST
// Dynamic shared layout: [1024 T-table words][256 S-box bytes][warp state...].
// Each warp state is [uint16 bins][uint16 round cache].
template<bool BGH>
__global__ void cuckoo_shared(uint32_t count,uint32_t first,uint32_t pnum,uint32_t buckets,
        uint32_t kick,uint64_t width,uint64_t domain,uint32_t half,uint64_t mask,
        uint64_t range_mask,uint32_t input_mode,AesKey batch_key,uint32_t warps,
        uint32_t* __restrict__ failed_count,uint32_t* __restrict__ failed_offsets) {
    extern __shared__ uint32_t sm[];
    uint8_t* smem=reinterpret_cast<uint8_t*>(sm);
    uint32_t* te=sm;
    uint8_t* sb=smem+4096;
    for(unsigned i=threadIdx.x;i<1024;i+=blockDim.x) te[i]=c_te[i];
    for(unsigned i=threadIdx.x;i<256;i+=blockDim.x) sb[i]=c_sbox[i];
    __syncthreads();

    const uint32_t warp=threadIdx.x>>5,lane=threadIdx.x&31;
    uint32_t trial=0;
    if(warp>=warps || !shared_trial_index(blockIdx.x,warp,warps,count,first,trial)) return;

    uint8_t* state=smem+4352+uint64_t(warp)*shared_warp_bytes(buckets,half);
    uint16_t* bins=reinterpret_cast<uint16_t*>(state);
    uint16_t* rounds=reinterpret_cast<uint16_t*>(state+shared_bins_bytes(buckets));
    const uint16_t empty=SharedBins::empty();
    for(uint32_t k=lane;k<buckets;k+=32) bins[k]=empty;
    __syncwarp();

    // Exactly the host/CPU derivation: one PRP key and one walk per trial.
    uint32_t a,b,c,d;
    aes128(batch_key,te,sb,0x4b455931u,0,trial,0x50525031u,a,b,c,d);
    AesKey key;
    expand_gpu_key(a,b,c,d,sb,key);
    aes128(batch_key,te,sb,0x57414c4bu,0,trial,0x524e4731u,a,b,c,d);
    uint64_t walk=(uint64_t(a)<<32)|b;
    const uint64_t shift=((uint64_t(c)<<32)|d)&range_mask;
    uint64_t input_state=shift^0xd1b54a32d192ed03ULL;
    const uint64_t step=(splitmix_next(input_state)|1ULL)&range_mask;
    const uint64_t randomness=splitmix_next(walk);

    // All 32 lanes fill the cache in parallel.  Only lane 0 then runs the
    // inherently sequential insertion, but every access is now on-chip.
    const uint32_t entries=8u<<half;
    const uint32_t right_mask=(1u<<half)-1u;
    for(uint32_t e=lane;e<entries;e+=32) {
        uint32_t h0,h1,h2,h3;
        aes128(key,te,sb,0x43455350u,e>>half,half,e&right_mask,h0,h1,h2,h3);
        rounds[e]=static_cast<uint16_t>(h3&mask);
    }
    __syncwarp();

    if(lane==0) {
        const CachedBucketsT<uint16_t> bucket{rounds,1,domain,width,mask,range_mask,
                                              shift,step,half,input_mode==0};
        const SharedBins table{bins,1};
        if(insert_core<BGH>(pnum,kick,table,bucket,walk,randomness))
            failed_offsets[atomicAdd(failed_count,1u)]=trial;
    }
}
#endif
