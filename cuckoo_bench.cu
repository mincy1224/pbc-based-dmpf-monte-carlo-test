// CUDA cuckoo Monte Carlo runner: direct port of prp_cuckoo's PRP and eviction.
// The A[i] distribution is configurable (sequential or affine-distinct).
// AES key/walk PRNG derivation is deterministic but replaces OS randfill().
// All GPU hot-path logic uses result flags, never throw. See README.md.
#ifndef CUCKOO_HOST_TEST
#include <cuda_runtime.h>
#endif
#include "checkpoint.hpp"
#include "cuckoo_core.hpp"
#include <memory>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
#define CK(expr) do { cudaError_t e_ = (expr); if (e_ != cudaSuccess) { \
    throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(e_)); } } while (0)

constexpr unsigned THREADS = 128;
constexpr uint32_t JOURNAL_MAGIC = 0x43554b33u; // CUK3
constexpr uint32_t RECORD_MAGIC = 0x454e4433u; // END3

// AES-128 T-tables are uploaded once per GPU. Every CTA stages them into shared RAM.
#ifndef CUCKOO_HOST_TEST
__constant__ uint32_t c_te[1024];
__constant__ uint8_t c_sbox[256];
#endif

static const uint8_t HOST_SBOX[256] = {
0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};
static uint8_t xtime(uint8_t x) { return (uint8_t)((x << 1) ^ ((x & 0x80) ? 0x1b : 0)); }
#ifndef CUCKOO_HOST_TEST
static void upload_aes_tables() {
    uint32_t t[1024];
    for (unsigned i=0;i<256;++i) {
        const uint32_t s=HOST_SBOX[i], a=xtime((uint8_t)s), b=a^s;
        t[i]       = (a<<24)|(s<<16)|(s<<8)|b;
        t[256+i]   = (b<<24)|(a<<16)|(s<<8)|s;
        t[512+i]   = (s<<24)|(b<<16)|(a<<8)|s;
        t[768+i]   = (s<<24)|(s<<16)|(b<<8)|a;
    }
    CK(cudaMemcpyToSymbol(c_te,t,sizeof(t)));
    CK(cudaMemcpyToSymbol(c_sbox,HOST_SBOX,sizeof(HOST_SBOX)));
}

#endif

struct AesKey { uint32_t w[44]; };
static uint32_t subword(uint32_t x) {
    return (uint32_t(HOST_SBOX[x>>24])<<24) |
           (uint32_t(HOST_SBOX[(x>>16)&255])<<16) |
           (uint32_t(HOST_SBOX[(x>>8)&255])<<8) | HOST_SBOX[x&255];
}
static uint32_t rotword(uint32_t x) { return (x<<8)|(x>>24); }
static AesKey expand_key(uint64_t lo, uint64_t hi) {
    uint8_t bytes[16];
    for (int i=0;i<8;++i) { bytes[i]=(uint8_t)(lo>>(8*i)); bytes[8+i]=(uint8_t)(hi>>(8*i)); }
    AesKey k{};
    for (int i=0;i<4;++i) k.w[i] = (uint32_t(bytes[4*i])<<24) |
        (uint32_t(bytes[4*i+1])<<16) | (uint32_t(bytes[4*i+2])<<8) | bytes[4*i+3];
    uint8_t rcon=1;
    for (int i=4;i<44;++i) {
        uint32_t v=k.w[i-1];
        if (i%4==0) { v=subword(rotword(v))^(uint32_t(rcon)<<24); rcon=xtime(rcon); }
        k.w[i]=k.w[i-4]^v;
    }
    return k;
}

#ifndef CUCKOO_HOST_TEST
__device__ __forceinline__ void aes128(const AesKey& rk, const uint32_t* te,
                        const uint8_t* sb, uint32_t a, uint32_t b, uint32_t c,
                        uint32_t d, uint32_t& h0, uint32_t& h1,
                        uint32_t& h2, uint32_t& h3) {
    uint32_t s0=a^rk.w[0], s1=b^rk.w[1], s2=c^rk.w[2], s3=d^rk.w[3];
    #pragma unroll
    for (int r=1;r<10;++r) {
        uint32_t t0=te[         (s0>>24)      ]^te[256+((s1>>16)&255)]^
                    te[512+((s2>>8)&255)]^te[768+(s3&255)]^rk.w[4*r];
        uint32_t t1=te[         (s1>>24)      ]^te[256+((s2>>16)&255)]^
                    te[512+((s3>>8)&255)]^te[768+(s0&255)]^rk.w[4*r+1];
        uint32_t t2=te[         (s2>>24)      ]^te[256+((s3>>16)&255)]^
                    te[512+((s0>>8)&255)]^te[768+(s1&255)]^rk.w[4*r+2];
        uint32_t t3=te[         (s3>>24)      ]^te[256+((s0>>16)&255)]^
                    te[512+((s1>>8)&255)]^te[768+(s2&255)]^rk.w[4*r+3];
        s0=t0; s1=t1; s2=t2; s3=t3;
    }
    h0=((uint32_t)sb[s0>>24]<<24)|((uint32_t)sb[(s1>>16)&255]<<16)|
       ((uint32_t)sb[(s2>>8)&255]<<8)|sb[s3&255];
    h1=((uint32_t)sb[s1>>24]<<24)|((uint32_t)sb[(s2>>16)&255]<<16)|
       ((uint32_t)sb[(s3>>8)&255]<<8)|sb[s0&255];
    h2=((uint32_t)sb[s2>>24]<<24)|((uint32_t)sb[(s3>>16)&255]<<16)|
       ((uint32_t)sb[(s0>>8)&255]<<8)|sb[s1&255];
    h3=((uint32_t)sb[s3>>24]<<24)|((uint32_t)sb[(s0>>16)&255]<<16)|
       ((uint32_t)sb[(s1>>8)&255]<<8)|sb[s2&255];
    h0^=rk.w[40]; h1^=rk.w[41]; h2^=rk.w[42]; h3^=rk.w[43];
}

// S-box based GPU key schedule. Only once per independent experiment; all
// candidate PRPs use the same per-experiment AES-128 key, as in prp_cuckoo.
__device__ __forceinline__ uint32_t gpu_subword(uint32_t x,const uint8_t* sb) {
    return (uint32_t(sb[x>>24])<<24) | (uint32_t(sb[(x>>16)&255])<<16) |
           (uint32_t(sb[(x>>8)&255])<<8) | sb[x&255];
}
__device__ __forceinline__ void expand_gpu_key(uint32_t a,uint32_t b,uint32_t c,uint32_t d,
                                                const uint8_t* sb,AesKey& rk) {
    rk.w[0]=a; rk.w[1]=b; rk.w[2]=c; rk.w[3]=d;
    uint32_t rcon=1;
    #pragma unroll 1
    for(int i=4;i<44;++i) {
        uint32_t v=rk.w[i-1];
        if((i&3)==0) {
            v=gpu_subword((v<<8)|(v>>24),sb)^(rcon<<24);
            rcon=(rcon<<1)^((rcon&0x80)?0x1b:0);
        }
        rk.w[i]=rk.w[i-4]^v;
    }
}
__device__ __forceinline__ uint64_t splitmix_next(uint64_t& state) {
    state+=0x9E3779B97F4A7C15ULL;
    uint64_t z=state;
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;
    z=(z^(z>>27))*0x94D049BB133111EBULL;
    return z^(z>>31);
}

// Exactly mirrors prp_helper_::round_function, permute_power2 and permute:
// 8 Feistel rounds, AES input "CESP" || BE32(round) || BE32(half_bits)
// || BE32(right), final 32 bits of AES ciphertext, cycle walk on [0,domain).
__device__ __forceinline__ uint32_t prp_bucket(
      uint64_t elem,uint32_t which, const AesKey& rk,
      const uint32_t* te,const uint8_t* sb,
      uint32_t half_bits,uint64_t mask,uint64_t domain,uint64_t bucket_size) {
    uint64_t y=elem*3ULL+which;
    do {
        uint64_t left=y>>half_bits;
        uint64_t right=y&mask;
        #pragma unroll
        for(uint32_t round=0;round<8;++round) {
            uint32_t h0,h1,h2,h3;
            aes128(rk,te,sb,0x43455350u,round,half_bits,(uint32_t)right,
                   h0,h1,h2,h3);
            const uint64_t next_right=(left^(uint64_t(h3)&mask))&mask;
            left=right;
            right=next_right;
        }
        y=(left<<half_bits)|right;
    } while(y>=domain);
    // 32-bit division is substantially cheaper on the GPU for the common
    // ell_i<=30 domain; the quotient is < buckets either way.
    return domain<=0xffffffffULL?uint32_t(uint32_t(y)/uint32_t(bucket_size))
                               :uint32_t(y/bucket_size);
}

// Both supplied input modes yield PNUM distinct A[i] in [0,2^ELL_I).
// 'affine' randomizes which set of elements is inserted without duplicate
// handling: A[i]=(shift+i*odd_step) mod 2^ELL_I.
__device__ __forceinline__ uint64_t trial_element(uint32_t ordinal,
         uint32_t input_mode,uint64_t shift,uint64_t step,uint64_t range_mask) {
    if(input_mode==0) return ordinal;
    return (shift+(uint64_t)ordinal*step)&range_mask;
}

// Packed table entry: 16-bit item ordinal, 2-bit current choice.
// 0xffffffff indicates an empty bucket. Distinct inputs guarantee no repeat.
constexpr uint32_t EMPTY_ENTRY = 0xffffffffu;
__device__ __forceinline__ uint32_t pack_entry(uint32_t ord,uint32_t choice) {
    return ord | (choice<<16);
}

template<bool BGH>
__global__ void cuckoo_trials(uint32_t n,uint32_t first_offset,uint32_t pnum,
        uint32_t buckets,uint32_t maxkick,uint64_t bucket_size,
        uint64_t domain,uint32_t half_bits,uint64_t half_mask,
        uint64_t range_mask,uint32_t input_mode,AesKey batch_rk,
        bool in_shared,uint32_t* __restrict__ scratch,
        uint32_t* __restrict__ failed_count,
        uint32_t* __restrict__ failed_offsets) {
    extern __shared__ uint32_t sm[];
    uint32_t* te=sm;
    uint8_t* sb=reinterpret_cast<uint8_t*>(sm+1024);
    uint32_t* shared_bins=reinterpret_cast<uint32_t*>(sb+256);
    for(unsigned i=threadIdx.x;i<1024;i+=blockDim.x) te[i]=c_te[i];
    for(unsigned i=threadIdx.x;i<256;i+=blockDim.x) sb[i]=c_sbox[i];
    __syncthreads();
    const uint32_t tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n) return;
    uint32_t* bins=in_shared ? shared_bins+(size_t)threadIdx.x*buckets
                             : scratch+(size_t)tid*buckets;
    for(uint32_t k=0;k<buckets;++k) bins[k]=EMPTY_ENTRY;
    const uint32_t trial=first_offset+tid;

    // One PRP key per *trial*. A single batch key for all PRPs would not
    // reproduce the original independent initialization of each prp_cuckoo.
    uint32_t a,b,c,d;
    aes128(batch_rk,te,sb,0x4b455931u,0,trial,0x50525031u,a,b,c,d);
    AesKey rk;
    expand_gpu_key(a,b,c,d,sb,rk);
    aes128(batch_rk,te,sb,0x57414c4bu,0,trial,0x524e4731u,a,b,c,d);
    uint64_t walk_state=(uint64_t(a)<<32)|b;
    // walk_state is advanced ONCE before the first BGH selection, exactly as
    // the original constructor initializes bucket_randomness_. Derive affine
    // inputs from separate bytes so they do not consume the random-walk stream.
    const uint64_t shift=((uint64_t(c)<<32)|d)&range_mask;
    uint64_t input_state=shift^0xd1b54a32d192ed03ULL;
    const uint64_t step=(splitmix_next(input_state)|1ULL)&range_mask;
    uint64_t bucket_randomness=splitmix_next(walk_state);

    bool failed=false;
    for(uint32_t item=0;item<pnum && !failed;++item) {
        if constexpr(BGH) {
            // Exact BGH25 behavior: choose a random candidate on EACH walk
            // step, including the initial placement. Limit is on collisions:
            // if (++collisions > max_times) fail BEFORE evicting.
            uint32_t cur=item;
            uint32_t collisions=0;
            while(true) {
                if(bucket_randomness<3) bucket_randomness=splitmix_next(walk_state);
                const uint32_t which=(uint32_t)(bucket_randomness%3);
                bucket_randomness/=3;
                const uint64_t elem=trial_element(cur,input_mode,shift,step,range_mask);
                const uint32_t bucket=prp_bucket(elem,which,rk,te,sb,
                        half_bits,half_mask,domain,bucket_size);
                const uint32_t old=bins[bucket];
                if(old==EMPTY_ENTRY) {
                    bins[bucket]=pack_entry(cur,which);
                    break;
                }
                if(++collisions>maxkick) { failed=true; break; }
                bins[bucket]=pack_entry(cur,which);
                cur=old&65535u;
            }
        } else {
            // Exact DRRT18/pirpsi behavior: try ALL candidates in order for
            // the new item. On collision begin with choice 0, then the
            // displaced item's *stored* choice cycles to the first candidate
            // with bucket != previous_bucket (no random choice here).
            const uint64_t elem=trial_element(item,input_mode,shift,step,range_mask);
            uint32_t initial0=prp_bucket(elem,0,rk,te,sb,half_bits,half_mask,domain,bucket_size);
            if(bins[initial0]==EMPTY_ENTRY) {
                bins[initial0]=pack_entry(item,0);
                continue;
            }
            bool inserted=false;
            for(uint32_t k=1;k<3;++k) {
                const uint32_t bucket=prp_bucket(elem,k,rk,te,sb,
                        half_bits,half_mask,domain,bucket_size);
                if(bins[bucket]==EMPTY_ENTRY) {
                    bins[bucket]=pack_entry(item,k);
                    inserted=true; break;
                }
            }
            if(inserted) continue;
            uint32_t cur=item, choice=0, bucket=initial0, kicks=0;
            while(true) {
                const uint32_t displaced=bins[bucket];
                if(displaced==EMPTY_ENTRY) {
                    bins[bucket]=pack_entry(cur,choice);
                    break;
                }
                if(kicks>=maxkick) {failed=true;break;}
                bins[bucket]=pack_entry(cur,choice);
                cur=displaced&65535u;
                choice=(displaced>>16)&3u;
                const uint32_t previous_bucket=bucket;
                const uint64_t cur_elem=trial_element(cur,input_mode,shift,step,range_mask);
                bool found=false;
                #pragma unroll
                for(int step_idx=0;step_idx<3;++step_idx) {
                    choice=(choice+1==3)?0:choice+1;
                    const uint32_t next_bucket=prp_bucket(cur_elem,choice,rk,te,sb,
                            half_bits,half_mask,domain,bucket_size);
                    if(next_bucket!=previous_bucket) {
                        bucket=next_bucket;found=true;break;
                    }
                }
                if(!found) {failed=true;break;}
                ++kicks;
            }
        }
    }
    if(failed) {
        const uint32_t at=atomicAdd(failed_count,1u);
        failed_offsets[at]=trial;
    }
}

#include "gpu_cached.cuh"
#include "gpu_shared.cuh"

__global__ void aes_kat(AesKey key, uint32_t* out) {
    extern __shared__ uint32_t sm[];
    uint32_t* te=sm;
    uint8_t* sb=reinterpret_cast<uint8_t*>(sm+1024);
    for (unsigned i=threadIdx.x;i<1024;i+=blockDim.x) te[i]=c_te[i];
    for (unsigned i=threadIdx.x;i<256;i+=blockDim.x) sb[i]=c_sbox[i];
    __syncthreads();
    if (threadIdx.x==0) aes128(key,te,sb,0x00112233,0x44556677,0x8899aabb,0xccddeeff,
                              out[0],out[1],out[2],out[3]);
}
static void aes_self_test() {
    // NIST FIPS-197 AES-128 known answer test.
    uint8_t b[16]; for(unsigned i=0;i<16;++i) b[i]=(uint8_t)i;
    uint64_t lo=0,hi=0;
    for(int i=0;i<8;++i) { lo|=(uint64_t)b[i]<<(8*i); hi|=(uint64_t)b[i+8]<<(8*i); }
    AesKey k=expand_key(lo,hi);
    uint32_t *o=nullptr, h[4]; CK(cudaMalloc((void**)&o,sizeof(h)));
    aes_kat<<<1,32,4352>>>(k,o); CK(cudaGetLastError());
    CK(cudaMemcpy(h,o,sizeof(h),cudaMemcpyDeviceToHost)); CK(cudaFree(o));
    const uint32_t correct[]={0x69c4e0d8,0x6a7b0430,0xd8cdb780,0x70b4c55a};
    if (std::memcmp(h,correct,sizeof(correct))!=0)
        throw std::runtime_error("AES-128 FIPS known-answer test FAILED");
}
static void cached_aes_self_test() {
    const auto key=expand_key(0x0706050403020100ULL,0x0f0e0d0c0b0a0908ULL);
    uint32_t host[32*4],*device=nullptr;CK(cudaMalloc((void**)&device,sizeof(host)));
    cached_aes_kat<<<1,32,32768>>>(key,device);CK(cudaGetLastError());
    CK(cudaMemcpy(host,device,sizeof(host),cudaMemcpyDeviceToHost));CK(cudaFree(device));
    const uint32_t expected[]={0x69c4e0d8u,0x6a7b0430u,0xd8cdb780u,0x70b4c55au};
    for(unsigned lane=0;lane<32;++lane)
        if(std::memcmp(host+lane*4,expected,sizeof(expected)))
            throw std::runtime_error("Replicated-table GPU AES/key expansion NIST test failed");
}

#endif // CUCKOO_HOST_TEST

// Deterministic AES-128 PRG for batch seeds. This is host-side: one block
// encryption per batch, so it does not affect the per-experiment GPU hot path.
static void host_aes128(const AesKey& key, uint32_t a, uint32_t b,
                        uint32_t c, uint32_t d, uint32_t out[4]) {
    uint8_t st[16];
    const uint32_t input[4]={a,b,c,d};
    for(int j=0;j<4;++j)
        for(int q=0;q<4;++q) st[4*j+q]=(uint8_t)(input[j]>>(24-8*q));
    auto add_round=[&](int round) {
        for(int j=0;j<4;++j)
            for(int q=0;q<4;++q) st[4*j+q]^=(uint8_t)(key.w[4*round+j]>>(24-8*q));
    };
    add_round(0);
    for(int round=1;round<=10;++round) {
        for(int i=0;i<16;++i) st[i]=HOST_SBOX[st[i]];
        uint8_t copy[16]; std::memcpy(copy,st,16);
        for(int col=0;col<4;++col)
            for(int row=0;row<4;++row)
                st[4*col+row]=copy[4*((col+row)%4)+row];
        if(round!=10) {
            for(int col=0;col<4;++col) {
                uint8_t* x=&st[4*col];
                const uint8_t t=x[0]^x[1]^x[2]^x[3], v=x[0];
                x[0]^=t^xtime(x[0]^x[1]);
                x[1]^=t^xtime(x[1]^x[2]);
                x[2]^=t^xtime(x[2]^x[3]);
                x[3]^=t^xtime(x[3]^v);
            }
        }
        add_round(round);
    }
    for(int j=0;j<4;++j)
        out[j]=(uint32_t(st[4*j])<<24)|(uint32_t(st[4*j+1])<<16)|
               (uint32_t(st[4*j+2])<<8)|st[4*j+3];
}
static AesKey batch_key(uint64_t seed, uint64_t batch_id) {
    // A 64-bit master seed is expanded into the 128-bit AES-PRG master key.
    // The experiment's requested formula is literal: batch_seed = seed XOR PRG(b).
    const AesKey master=expand_key(seed,seed^0x9e3779b97f4a7c15ULL);
    uint32_t x[4];
    host_aes128(master,0x42415443u,(uint32_t)(batch_id>>32),
                (uint32_t)batch_id,0x50524731u,x);
    const uint64_t prg_lo=(uint64_t(x[0])<<32)|x[1];
    const uint64_t prg_hi=(uint64_t(x[2])<<32)|x[3];
    const uint64_t batch_seed=seed^prg_lo;
    return expand_key(batch_seed,prg_hi^(seed^0x243f6a8885a308d3ULL));
}

struct Options {
    uint32_t et=30, bt=20, pnum=16, ell_i=32, micro=1u<<17;
    uint64_t seed=1, stop_after=1ULL<<20;
    std::string mode="drrt18", out="cuckoo_results", input_mode="sequential";
    int workers=0;
    uint32_t scratch_mib=256;
    uint32_t block_threads=256;
    uint32_t warps_per_block=0; // 0 = choose the largest warp-private layout that fits
    std::string prp_cache="auto";
    std::string table_layout="auto"; // auto|global|shared
    std::string shared_variant="auto"; // auto|round|map
    std::string backend="cuda";
    bool force_global=false, status_only=false;
};
static uint64_t parse_u64(const std::string& s) {
    if(s.empty() || s[0]=='-') throw std::runtime_error("Invalid unsigned number: "+s);
    size_t at=0; auto v=std::stoull(s,&at,0);
    if(at!=s.size()) throw std::runtime_error("Invalid number: "+s);
    return v;
}
static uint32_t parse_u32(const std::string& s) {
    const auto v=parse_u64(s);
    if(v>std::numeric_limits<uint32_t>::max()) throw std::runtime_error("Number exceeds uint32: "+s);
    return static_cast<uint32_t>(v);
}
static Options parse(int argc,char** argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        const std::string a=argv[i];
        if(a=="--help" || a=="-h") {
            std::cout<<"Usage: ./cuckoo_bench --mode bgh25|drrt18 --et 40 --bt 20 "
                "--pnum 16 --seed 123 --workers 1 --micro 131072 "
                "--out results [--input-mode sequential|affine] "
                "[--scratch-mib 256] [--prp-cache auto|on|off] [--block-threads 256] "
                "[--table-layout auto|global|shared] [--shared-variant auto|round|map] "
                "[--warps-per-block N] [--stop-after 1048576] [--global-table] [--status]\n";
            std::exit(0);
        }
        if(a=="--global-table") { o.force_global=true; continue; }
        if(a=="--status") { o.status_only=true; continue; }
        if(i+1>=argc) throw std::runtime_error("Missing value after "+a);
        std::string v=argv[++i];
        if(a=="--et") o.et=parse_u32(v);
        else if(a=="--bt") o.bt=parse_u32(v);
        else if(a=="--pnum") o.pnum=parse_u32(v);
        else if(a=="--ell-i") o.ell_i=parse_u32(v);
        else if(a=="--micro") o.micro=parse_u32(v);
        else if(a=="--scratch-mib") o.scratch_mib=parse_u32(v);
        else if(a=="--prp-cache") o.prp_cache=v;
        else if(a=="--block-threads") o.block_threads=parse_u32(v);
        else if(a=="--table-layout") o.table_layout=v;
        else if(a=="--shared-variant") o.shared_variant=v;
        else if(a=="--warps-per-block") o.warps_per_block=parse_u32(v);
        else if(a=="--seed") o.seed=parse_u64(v);
        else if(a=="--workers") { auto n=parse_u64(v); if(n>256) throw std::runtime_error("workers exceeds 256"); o.workers=static_cast<int>(n); }
        else if(a=="--stop-after") o.stop_after=parse_u64(v);
        else if(a=="--mode") o.mode=v;
        else if(a=="--input-mode") o.input_mode=v;
        else if(a=="--out") o.out=v;
        else throw std::runtime_error("Unknown option "+a);
    }
    if (o.et>40 || o.bt>30 || o.bt>o.et || o.et==0 || o.bt==0 ||
        o.pnum<3 || o.pnum>65534 || o.micro==0 || o.scratch_mib==0 || o.ell_i>61 ||
        (o.block_threads!=64 && o.block_threads!=128 && o.block_threads!=256) ||
        (o.prp_cache!="auto" && o.prp_cache!="on" && o.prp_cache!="off") ||
        (o.table_layout!="auto" && o.table_layout!="global" && o.table_layout!="shared") ||
        (o.shared_variant!="auto" && o.shared_variant!="round" && o.shared_variant!="map") ||
        o.warps_per_block>32 ||
        (o.ell_i<16 && (1ULL<<o.ell_i)<o.pnum) ||
        (o.mode!="bgh25" && o.mode!="drrt18") ||
        (o.input_mode!="sequential" && o.input_mode!="affine"))
        throw std::runtime_error("Invalid config (ET 1..40, BT 1..min(ET,30), "
                                 "PNUM 3..65534, ell-i <=61, 2^ell-i >=PNUM, input mode sequential|affine)");
    return o;
}
// C++17 equivalent of std::bit_width, including bit_width_u64(0) == 0.
// Host-side parameter calculation only; GPU AES/PRP/insertion code is unchanged.
static unsigned bit_width_u64(uint64_t value) noexcept {
    unsigned bits=0;
    for(;value!=0;value>>=1) ++bits;
    return bits;
}
static float original_log2_u64(uint64_t n) {
    if(n==0) return -std::numeric_limits<float>::infinity();
    const unsigned e=bit_width_u64(n)-1;
    const double x=(double)n/(double)(1ULL<<e);
    if(x>=2.0) return (float)(e+1);
    const double z=(x-1.0)/(x+1.0), z2=z*z;
    double term=z, sum=0.0;
    for(int k=1;k<=23;k+=2) {sum+=term/k;term*=z2;}
    constexpr double inv_ln2=1.4426950408889634074;
    return (float)((double)e+2.0*sum*inv_ln2);
}
struct Params { uint32_t buckets=0, kick=0; uint64_t bucket_size=0, domain=0, mask=0; uint32_t ell_d=0, half_bits=0; };
static Params parameters(const Options& o) {
    Params x;
    if(o.mode=="bgh25") {
        const uint64_t overhead=o.pnum<30?100:50;
        x.buckets=(uint32_t)((uint64_t)o.pnum*(100+overhead)/100);
        x.kick=3*o.pnum;
    } else {
        // Matches the original code's float-valued PIRPSI_BUCKETS calculation.
        const float lf=original_log2_u64(o.pnum);
        const float f=(float)((double)o.pnum*((double)(170.0f+lf)/123.5));
        x.buckets=(uint32_t)std::ceil((double)f);
        x.kick=100;
    }
    const uint64_t range=uint64_t(1)<<o.ell_i;
    // ell_i <= 61 makes range*3 fit uint64_t, also on Windows/MSVC.
    const uint64_t source=range*3;
    x.bucket_size=source/x.buckets+(source%x.buckets!=0);
    x.ell_d=(uint32_t)std::ceil(original_log2_u64(x.bucket_size));
    if(x.buckets>std::numeric_limits<uint64_t>::max()/x.bucket_size)
        throw std::runtime_error("PRP domain overflow");
    x.domain=x.buckets*x.bucket_size;
    uint32_t bits=std::max<uint32_t>(bit_width_u64(x.domain-1),2u);
    if(bits&1u) ++bits;
    x.half_bits=bits/2;
    x.mask=(1ULL<<x.half_bits)-1ULL;
    return x;
}
// One experiment per warp, table and round cache in shared memory.  Auto uses
// it only where the old per-thread table would be large enough that random
// global bucket loads dominate; small tables keep the much wider per-thread
// mapping (32 trials per warp).
static bool shared_trials_enabled(const Options& o,const Params& p) {
    if(o.table_layout=="shared" && o.prp_cache=="off")
        throw std::runtime_error("Shared table layout uses the PRP round cache; drop --prp-cache off");
    if(o.table_layout=="shared" && o.force_global)
        throw std::runtime_error("--table-layout shared conflicts with --global-table");
    if(o.prp_cache=="off" || o.table_layout=="global" || o.force_global) return false;
    if(!shared_layout_supported(o.pnum,p.half_bits)) {
        if(o.table_layout=="shared")
            throw std::runtime_error("Shared table layout needs PNUM < 16384 and half_bits <= 12");
        return false;
    }
    if(o.table_layout=="shared") return true;
    return o.pnum>=1024;
}
static bool round_cache_enabled(const Options& o,const Params& p) {
    if(o.prp_cache=="off") return false;
    if(shared_trials_enabled(o,p)) return true; // the shared kernel always caches
    if(p.half_bits>10) {
        if(o.prp_cache=="on") throw std::runtime_error("PRP round cache requires half_bits <=10; use auto/off");
        return false;
    }
    return o.prp_cache=="on" || o.pnum>=512;
}
static bool shared_map_planned(const Options& o,const Params& p) {
    if(!shared_map_supported(o.pnum,p.half_bits,p.buckets)) return false;
    return o.shared_variant=="map" || (o.shared_variant=="auto" && o.pnum>=1024);
}
static const char* table_layout_name(const Options& o,const Params& p) {
    if(o.backend=="cpu-aesni") return round_cache_enabled(o,p)?"cpu-cached":"cpu-direct";
    if(shared_trials_enabled(o,p)) return shared_map_planned(o,p)?"shared-map":"shared-trial";
    if(round_cache_enabled(o,p)) return "cached-global";
    return "direct-global";
}

// Journal is append-only; every complete batch consists of a header, offsets and
// a checksum footer. Torn tail records are dropped on resume.
struct FileHeader {
    uint32_t magic=JOURNAL_MAGIC, version=3;
    uint64_t seed=0, planned=0;
    uint32_t et=0, bt=0, pnum=0, ell_i=0, mode=0, buckets=0, kick=0, input_mode=0;
};
struct RecordHeader { uint64_t batch; uint32_t tried, failed; };
struct RecordFooter { uint32_t check, magic; };
static uint32_t checksum(uint64_t batch,uint32_t tried,const std::vector<uint32_t>& offsets) {
    uint32_t h=0x811c9dc5u;
    auto mix=[&](uint32_t x){h=(h^x)*0x01000193u;};
    mix((uint32_t)batch); mix((uint32_t)(batch>>32)); mix(tried);
    mix((uint32_t)offsets.size());
    for(uint32_t x:offsets) mix(x);
    return h;
}
static uint64_t batch_length(const Options& o,uint64_t id) {
    const uint64_t total=1ULL<<o.et, start=id*(1ULL<<o.bt);
    return std::min<uint64_t>(1ULL<<o.bt,total-start);
}
static std::string journal_name(const Options& o,int worker) {
    return o.out+"/worker_"+std::to_string(worker)+".bin";
}
static std::vector<std::string> journal_paths(const Options& o) {
    std::vector<std::string> paths;
    for (const auto& e: fs::directory_iterator(o.out)) {
        if (!e.is_regular_file()) continue;
        const std::string name=e.path().filename().string();
        if(name.rfind("worker_",0)!=0 || name.size()<12 ||
           name.substr(name.size()-4)!=".bin") continue;
        const std::string middle=name.substr(7,name.size()-11);
        if(middle.empty() || !std::all_of(middle.begin(),middle.end(),
                                          [](char c){return c>='0' && c<='9';})) continue;
        paths.push_back(e.path().string());
    }
    std::sort(paths.begin(),paths.end());
    return paths;
}
static FileHeader create_header(const Options& o,const Params& p) {
    FileHeader h; h.seed=o.seed; h.planned=1ULL<<o.et;
    h.et=o.et; h.bt=o.bt; h.pnum=o.pnum; h.ell_i=o.ell_i;
    h.mode=o.mode=="bgh25"?1:2; h.buckets=p.buckets; h.kick=p.kick;
    h.input_mode=(o.input_mode=="sequential")?0:1;
    return h;
}
static void append_record(FILE* f,uint64_t id,uint32_t attempted,
                          const std::vector<uint32_t>& offsets) {
    if (offsets.size()>attempted) throw std::runtime_error("Invalid failure count");
    if(!std::is_sorted(offsets.begin(),offsets.end()) ||
       std::adjacent_find(offsets.begin(),offsets.end())!=offsets.end() ||
       (!offsets.empty() && offsets.back()>=attempted))
        throw std::runtime_error("Invalid failure offsets");
    const RecordHeader h{id,attempted,(uint32_t)offsets.size()};
    const RecordFooter tail{checksum(id,attempted,offsets),RECORD_MAGIC};
    if(fwrite(&h,sizeof(h),1,f)!=1 ||
       (!offsets.empty() && fwrite(offsets.data(),sizeof(uint32_t),offsets.size(),f)!=offsets.size()) ||
       fwrite(&tail,sizeof(tail),1,f)!=1)
        throw std::runtime_error("Journal write failed");
    durable_flush(f);
}
static void validate_header(FileHeader a,FileHeader b) {
    // No padding comparison: compare fields explicitly.
    if(a.magic!=b.magic||a.version!=b.version||a.seed!=b.seed||a.planned!=b.planned||
       a.et!=b.et||a.bt!=b.bt||a.pnum!=b.pnum||a.ell_i!=b.ell_i||
       a.mode!=b.mode||a.buckets!=b.buckets||a.kick!=b.kick||
       a.input_mode!=b.input_mode)
        throw std::runtime_error("Existing journal has different experiment parameters");
}
struct Counters {
    std::atomic<uint64_t> completed{0}, failed{0}, batches{0};
    std::atomic<bool> stop{false}, abort_inflight{false};
    std::atomic<int> active{0};
    std::atomic<uint64_t> next_batch{0};
    std::atomic<int64_t> finished_ns{0};
};
static int64_t steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void mark_finished(Counters& s) {
    const auto now=steady_ns();
    auto previous=s.finished_ns.load();
    while(previous<now && !s.finished_ns.compare_exchange_weak(previous,now)) {}
}
static std::vector<uint8_t> resume(const Options& o,const Params& p, Counters& s) {
    const uint64_t total_batches=1ULL<<(o.et-o.bt);
    if(total_batches>(1ULL<<28))
        throw std::runtime_error("Too many batches for the completion bitmap; increase --bt");
    std::vector<uint8_t> done(static_cast<size_t>(total_batches),0);
    const FileHeader want=create_header(o,p);
    const auto paths=journal_paths(o);
    // Validate every complete file header before repairing any tail.
    for(const auto& path:paths) {
        if(fs::file_size(path)<sizeof(FileHeader)) continue;
        std::ifstream in(path,std::ios::binary);
        FileHeader head{}; in.read((char*)&head,sizeof(head));
        if(!in) throw std::runtime_error("Cannot read journal header: "+path);
        validate_header(head,want);
    }
    for(const auto& path:paths) {
        const uint64_t file_size=fs::file_size(path);
        if(file_size<sizeof(FileHeader)) { fs::resize_file(path,0); continue; }
        std::ifstream in(path,std::ios::binary);
        in.seekg(sizeof(FileHeader));
        uint64_t safe=sizeof(FileHeader);
        while(safe<file_size) {
            if(file_size-safe<sizeof(RecordHeader)) break; // torn header
            RecordHeader r{}; in.read((char*)&r,sizeof(r));
            if(!in) throw std::runtime_error("Journal read failed: "+path);
            if(r.batch>=total_batches || r.tried!=batch_length(o,r.batch) || r.failed>r.tried)
                throw std::runtime_error("Invalid journal record header: "+path);
            const uint64_t record_bytes=sizeof(r)+uint64_t(r.failed)*sizeof(uint32_t)+sizeof(RecordFooter);
            if(file_size-safe<record_bytes) break; // uncommitted partial tail
            std::vector<uint32_t> offsets(r.failed);
            if(r.failed) in.read((char*)offsets.data(),r.failed*sizeof(uint32_t));
            RecordFooter tail{}; in.read((char*)&tail,sizeof(tail));
            if(!in) throw std::runtime_error("Journal read failed: "+path);
            if(tail.magic!=RECORD_MAGIC || tail.check!=checksum(r.batch,r.tried,offsets))
                throw std::runtime_error("Corrupt complete journal record (file preserved): "+path);
            if(!std::is_sorted(offsets.begin(),offsets.end()) ||
               std::adjacent_find(offsets.begin(),offsets.end())!=offsets.end() ||
               (!offsets.empty() && offsets.back()>=r.tried))
                throw std::runtime_error("Invalid journal offsets: "+path);
            if(done[r.batch]) throw std::runtime_error("Duplicate committed batch: "+std::to_string(r.batch));
            done[r.batch]=1;
            s.batches.fetch_add(1); s.completed.fetch_add(r.tried); s.failed.fetch_add(r.failed);
            safe+=record_bytes;
        }
        in.close();
        if(file_size!=safe) fs::resize_file(path,safe);
    }
    return done;
}
static FILE* open_journal(const Options& o,const Params& p,int worker) {
    const auto path=journal_name(o,worker);
    const bool initialized=fs::exists(path) && fs::file_size(path)!=0;
    FILE* opened=nullptr;
#ifdef _WIN32
    if(fopen_s(&opened,path.c_str(),initialized?"ab":"wb")!=0) opened=nullptr;
#else
    opened=fopen(path.c_str(),initialized?"ab":"wb");
#endif
    std::unique_ptr<FILE,decltype(&fclose)> f(opened,fclose);
    if(!f) throw std::runtime_error("Unable to open: "+path);
    if(!initialized) {
        const auto h=create_header(o,p);
        if(fwrite(&h,sizeof(h),1,f.get())!=1) throw std::runtime_error("Journal header write failed");
        durable_flush(f.get());
    }
    return f.release();
}
static bool pause_requested(const Options& o) {
    return pause_signal.load(std::memory_order_relaxed) || fs::exists(fs::path(o.out)/"PAUSE");
}

// Shared host controller, also exercised without a CUDA toolkit by host_tests.cpp.
// A CPU worker owns the key for one claimed batch. Each device trial still derives
// its own PRP key from this batch key and its offset; scheduling never changes it.
template<class Launch,class Commit>
static bool process_batch(const Options& o,uint64_t id,uint32_t micro,Counters& s,
                          Launch&& launch,Commit&& commit) {
    if(micro==0) throw std::runtime_error("micro must be positive");
    const auto key=batch_key(o.seed,id);
    const auto actual=static_cast<uint32_t>(batch_length(o,id));
    std::vector<uint32_t> failures;
    for(uint32_t offset=0;offset<actual;) {
        if(s.abort_inflight.load() || pause_requested(o)) { s.stop.store(true); return false; }
        const auto n=static_cast<uint32_t>(std::min<uint64_t>(micro,actual-offset));
        launch(n,offset,key,failures);
        offset+=n;
    }
    // Once all micro-launches have finished, commit even if pause just arrived.
    std::sort(failures.begin(),failures.end());
    commit(id,actual,failures); // includes disk sync; never publish partial batches
    s.completed.fetch_add(actual,std::memory_order_relaxed);
    const auto total=s.failed.fetch_add(failures.size())+failures.size();
    s.batches.fetch_add(1);
    if(total>o.stop_after) s.stop.store(true);
    return true;
}
static void final_outputs(const Options& o,const Params& p,
                          const Counters& s,double seconds,const std::string& status="complete",
                          double work_seconds=0) {
    auto completed=s.completed.load(), fails=s.failed.load(), batches=s.batches.load();
    {
        std::ofstream out; out.exceptions(std::ios::failbit|std::ios::badbit);
        out.open(o.out+"/summary.json",std::ios::trunc);
        out<<"{\n  \"status\": \""<<status<<"\",\n  \"seed_scheme\": \"aes-batch-v1\",\n"
           <<"  \"mode\": \""<<o.mode<<"\",\n  \"algorithm\": \"prp-cuckoo-aes-feistel-port\",\n"
           <<"  \"backend\": \""<<o.backend<<"\",\n  \"workers\": "<<o.workers<<",\n"
           <<"  \"micro_requested\": "<<o.micro<<",\n  \"scratch_mib\": "<<o.scratch_mib<<",\n"
           <<"  \"prp_cache_requested\": \""<<o.prp_cache<<"\",\n"
           <<"  \"prp_cache_enabled\": "<<(round_cache_enabled(o,p)?"true":"false")<<",\n"
           <<"  \"table_layout_requested\": \""<<o.table_layout<<"\",\n"
           <<"  \"table_layout_planned\": \""<<table_layout_name(o,p)<<"\",\n"
           <<"  \"shared_variant_requested\": \""<<o.shared_variant<<"\",\n"
           <<"  \"block_threads_requested\": "<<o.block_threads<<",\n"
           <<"  \"warps_per_block_requested\": "<<o.warps_per_block<<",\n"
           <<"  \"input_mode\": \""<<o.input_mode<<"\",\n"
           <<"  \"seed\": "<<o.seed<<",\n  \"ET\": "<<o.et<<",\n  \"BT\": "<<o.bt<<",\n"
           <<"  \"seed_decimal\": \""<<o.seed<<"\",\n  \"batch_size\": "<<(1ULL<<o.bt)<<",\n"
           <<"  \"PNUM_T\": "<<o.pnum<<",\n  \"ELL_I\": "<<o.ell_i<<",\n"
           <<"  \"BUCKETS_NUM\": "<<p.buckets<<",\n  \"MAX_KICK_TIMES\": "<<p.kick<<",\n"
           <<"  \"BUCKET_SIZE\": "<<p.bucket_size<<",\n  \"PRP_DOMAIN\": "<<p.domain<<",\n"
           <<"  \"planned\": "<<(1ULL<<o.et)<<",\n  \"completed\": "<<completed
           <<",\n  \"failed\": "<<fails<<",\n  \"completed_batches\": "<<batches
           <<",\n  \"pending_batches\": "<<((1ULL<<(o.et-o.bt))-batches)
           <<",\n  \"failed_ratio\": "<<std::setprecision(17)<<(completed?(double)fails/completed:0.0)
           <<",\n  \"elapsed_seconds\": "<<seconds
           <<",\n  \"work_elapsed_seconds\": "<<work_seconds<<",\n  \"threshold_exceeded\": "
           <<(fails>o.stop_after?"true":"false")<<"\n}\n";
        out.flush(); out.close();
    }
    std::ofstream csv; csv.exceptions(std::ios::failbit|std::ios::badbit);
    csv.open(o.out+"/failures.csv",std::ios::trunc);
    csv<<"batch_id,offset\n";
    uint64_t count=0;
    for(const auto& path:journal_paths(o)) {
        std::ifstream in(path,std::ios::binary);
        if(!in) continue;
        in.seekg(sizeof(FileHeader),std::ios::beg);
        while(true) {
            RecordHeader r{};
            in.read((char*)&r,sizeof(r)); if(!in) break;
            for(uint32_t i=0;i<r.failed;++i) {
                uint32_t off=0; in.read((char*)&off,sizeof(off));
                csv<<r.batch<<','<<off<<'\n'; ++count;
            }
            in.seekg(sizeof(RecordFooter),std::ios::cur);
            if(!in) throw std::runtime_error("Corrupt journal in final export");
        }
    }
    if(count!=fails) throw std::runtime_error("Exported failure count mismatch");
    csv.flush(); csv.close();
}
#ifndef CUCKOO_HOST_TEST
class DeviceAllocation {
    uint32_t* ptr_=nullptr;
public:
    explicit DeviceAllocation(size_t bytes) {
        if(bytes) CK(cudaMalloc((void**)&ptr_,bytes));
    }
    ~DeviceAllocation() { if(ptr_) cudaFree(ptr_); }
    uint32_t* get() const { return ptr_; }
    void free_checked() { if(ptr_) { CK(cudaFree(ptr_)); ptr_=nullptr; } }
    DeviceAllocation(const DeviceAllocation&)=delete;
};
static void gpu_worker(int worker,int gpu,const Options& o,const Params& p,
                       const std::vector<uint8_t>& done, Counters& s,std::mutex& print_mutex) {
    CK(cudaSetDevice(gpu));
    const uint64_t batch_size=1ULL<<o.bt;
    const uint64_t total_batches=1ULL<<(o.et-o.bt);
    cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop,gpu));
    // Shared-trial layout: one experiment per warp, so the per-trial table and
    // the round cache never touch global memory.  Capacity is per SM, not per
    // thread; micro is no longer capped by the scratch budget.
    enum class SharedKind { None, Round, Map };
    SharedKind shared_kind=SharedKind::None;
    uint32_t shared_warps=0;
    uint64_t shared_bytes=0;
    if(shared_trials_enabled(o,p)) {
        const bool map_possible=shared_map_supported(o.pnum,p.half_bits,p.buckets);
        const uint64_t per_round=shared_warp_bytes(p.buckets,p.half_bits);
        const uint64_t per_map=map_possible?shared_map_warp_bytes(p.buckets,p.half_bits,o.pnum):0;
        auto max_warps=[&](uint64_t per)->uint32_t {
            if(per==0) return 0;
            const uint64_t fit=(uint64_t(prop.sharedMemPerBlockOptin)-4352)/per;
            uint32_t w=o.warps_per_block?o.warps_per_block:8;
            if(w>fit) {
                if(o.warps_per_block)
                    throw std::runtime_error("Requested warps per block do not fit in shared memory");
                w=static_cast<uint32_t>(fit);
            }
            if(w>32) w=32;
            return w;
        };
        const uint32_t round_warps=max_warps(per_round);
        const uint32_t map_warps=max_warps(per_map);
        bool choose_map=false;
        if(o.shared_variant=="map") {
            if(map_warps==0)
                throw std::runtime_error("Shared bucket map is not supported or does not fit on this GPU");
            choose_map=true;
        } else if(o.shared_variant=="round") {
            choose_map=false;
        } else {
            // The map removes eight dependent Feistel lookups per probe at the
            // cost of 3*pnum extra shared bytes.  Prefer it whenever at least
            // two experiments still fit per SM.
            choose_map=map_possible && map_warps>=2;
        }
        uint32_t want=choose_map?map_warps:round_warps;
        // A single warp per SM leaves too little latency hiding; auto keeps the
        // global layout in that case, while an explicit request is honoured.
        if(o.table_layout=="auto" && want<2) want=0;
        if(want>0) {
            shared_kind=choose_map?SharedKind::Map:SharedKind::Round;
            shared_warps=want;
            shared_bytes=4352+uint64_t(want)*(choose_map?per_map:per_round);
        } else if(o.table_layout=="shared") {
            throw std::runtime_error("Shared table layout does not fit on this GPU");
        }
    }
    const bool use_shared_trials=shared_kind!=SharedKind::None;
    bool cached=!use_shared_trials && round_cache_enabled(o,p);
    // If the shared layout did not fit, do not silently inflate the global
    // per-thread cache beyond the previously supported half_bits range.
    if(!use_shared_trials && cached && p.half_bits>10 && o.prp_cache!="on") cached=false;
    const uint32_t block_threads=cached?o.block_threads:THREADS;
    const uint64_t cache_entries=cached?(uint64_t{8}<<p.half_bits):0;
    const uint64_t shared_need=4352ULL+uint64_t(THREADS)*p.buckets*sizeof(uint32_t);
    const bool use_shared=!cached && !use_shared_trials && !o.force_global && shared_need<=
        (uint64_t)prop.sharedMemPerBlockOptin;
    const size_t sh_bytes=use_shared_trials?shared_bytes:
                          cached?32768:use_shared?shared_need:4352;
    if(sh_bytes>(size_t)prop.sharedMemPerBlock) {
        if(use_shared_trials && o.mode=="bgh25") {
            if(shared_kind==SharedKind::Map)
                CK(cudaFuncSetAttribute(cuckoo_shared<true,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh_bytes));
            else
                CK(cudaFuncSetAttribute(cuckoo_shared<true,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh_bytes));
        } else if(use_shared_trials) {
            if(shared_kind==SharedKind::Map)
                CK(cudaFuncSetAttribute(cuckoo_shared<false,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh_bytes));
            else
                CK(cudaFuncSetAttribute(cuckoo_shared<false,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh_bytes));
        } else if(cached && o.mode=="bgh25") {
            CK(cudaFuncSetAttribute(cuckoo_cached<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh_bytes));
        } else if(cached) {
            CK(cudaFuncSetAttribute(cuckoo_cached<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh_bytes));
        } else if(o.mode=="bgh25") {
            CK(cudaFuncSetAttribute(cuckoo_trials<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh_bytes));
        } else {
            CK(cudaFuncSetAttribute(cuckoo_trials<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh_bytes));
        }
    }
    // Table-space budget per worker, independent of experiment seeds/journals.
    const uint64_t limit_bytes=uint64_t(o.scratch_mib)<<20;
    uint64_t micro=o.micro;
    const uint64_t bytes_per_trial=uint64_t(p.buckets)*sizeof(uint32_t)+cache_entries*sizeof(uint16_t);
    if(use_shared_trials) { /* on-chip tables: no scratch cap */ }
    else if(cached) micro=std::min(micro,(limit_bytes/bytes_per_trial/32)*32);
    else if(!use_shared) micro=std::min(micro,limit_bytes/bytes_per_trial);
    micro=std::min(micro,batch_size);
    if(micro==0) throw std::runtime_error("Insufficient scratch memory or too many bins");
    const size_t pitch=cached?((micro+31)/32)*32:micro;
    DeviceAllocation bins_memory(use_shared||use_shared_trials?0:pitch*p.buckets*sizeof(uint32_t));
    DeviceAllocation rounds_memory(cached?pitch*cache_entries*sizeof(uint16_t):0);
    DeviceAllocation count_memory(sizeof(uint32_t)), offsets_memory(micro*sizeof(uint32_t));
    auto* d_bins=bins_memory.get();
    auto* d_count=count_memory.get();
    auto* d_offsets=offsets_memory.get();
    auto* d_rounds=reinterpret_cast<uint16_t*>(rounds_memory.get());
    std::vector<uint32_t> chunk_failures(micro);
    std::unique_ptr<FILE,decltype(&fclose)> journal(open_journal(o,p,worker),fclose);
    cudaFuncAttributes attributes{}; int active_blocks=0;
    if(use_shared_trials && o.mode=="bgh25") {
        if(shared_kind==SharedKind::Map) {
            CK(cudaFuncGetAttributes(&attributes,cuckoo_shared<true,true>));
            CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,cuckoo_shared<true,true>,
                                                             shared_warps*32,sh_bytes));
        } else {
            CK(cudaFuncGetAttributes(&attributes,cuckoo_shared<true,false>));
            CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,cuckoo_shared<true,false>,
                                                             shared_warps*32,sh_bytes));
        }
    } else if(use_shared_trials) {
        if(shared_kind==SharedKind::Map) {
            CK(cudaFuncGetAttributes(&attributes,cuckoo_shared<false,true>));
            CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,cuckoo_shared<false,true>,
                                                             shared_warps*32,sh_bytes));
        } else {
            CK(cudaFuncGetAttributes(&attributes,cuckoo_shared<false,false>));
            CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,cuckoo_shared<false,false>,
                                                             shared_warps*32,sh_bytes));
        }
    } else if(cached && o.mode=="bgh25") {
        CK(cudaFuncGetAttributes(&attributes,cuckoo_cached<true>));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,cuckoo_cached<true>,block_threads,sh_bytes));
    } else if(cached) {
        CK(cudaFuncGetAttributes(&attributes,cuckoo_cached<false>));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,cuckoo_cached<false>,block_threads,sh_bytes));
    } else if(o.mode=="bgh25") {
        CK(cudaFuncGetAttributes(&attributes,cuckoo_trials<true>));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,cuckoo_trials<true>,block_threads,sh_bytes));
    } else {
        CK(cudaFuncGetAttributes(&attributes,cuckoo_trials<false>));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,cuckoo_trials<false>,block_threads,sh_bytes));
    }
    const uint64_t scratch_bytes=
        (use_shared||use_shared_trials?0:uint64_t(pitch)*p.buckets*sizeof(uint32_t))+
        (cached?uint64_t(pitch)*cache_entries*sizeof(uint16_t):0);
    {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cerr<<"Worker "<<worker<<" GPU "<<gpu<<" ("<<prop.name<<") bins="<<p.buckets
                 <<" table="<<(use_shared_trials?
                               (shared_kind==SharedKind::Map?"shared-map":"shared-trial"):
                               cached?"cached-strided":use_shared?"shared":"global")
                 <<" micro="<<micro
                 <<" block="<<(use_shared_trials?shared_warps*32:block_threads)
                 <<" warps="<<(use_shared_trials?shared_warps:block_threads/32)
                 <<" shared_bytes="<<sh_bytes<<" scratch_bytes="<<scratch_bytes
                 <<" shared_per_sm="<<prop.sharedMemPerMultiprocessor
                 <<" shared_max_block="<<prop.sharedMemPerBlockOptin
                 <<" SMs="<<prop.multiProcessorCount
                 <<" regs/thread="<<attributes.numRegs<<" local_bytes/thread="<<attributes.localSizeBytes
                 <<" active_blocks/SM="<<active_blocks<<"\n";
    }
    while(!s.stop.load(std::memory_order_relaxed)) {
        if(pause_requested(o)) { s.stop.store(true); break; }
        const uint64_t id=s.next_batch.fetch_add(1);
        if(id>=total_batches) break;
        if(done[id]) continue;
        const bool committed=process_batch(o,id,static_cast<uint32_t>(micro),s,
          [&](uint32_t n,uint32_t offset,const AesKey& key,std::vector<uint32_t>& failures) {
            CK(cudaMemset(d_count,0,sizeof(uint32_t)));
            const unsigned blocks=(n+block_threads-1)/block_threads;
            const uint64_t range_mask=(1ULL<<o.ell_i)-1ULL;
            const uint32_t input_mode=(o.input_mode=="sequential")?0u:1u;
            if(use_shared_trials && o.mode=="bgh25") {
                const unsigned trial_blocks=(n+shared_warps-1)/shared_warps;
                if(shared_kind==SharedKind::Map) {
                    cuckoo_shared<true,true><<<trial_blocks,shared_warps*32,sh_bytes>>>(n,offset,o.pnum,
                        p.buckets,p.kick,p.bucket_size,p.domain,p.half_bits,p.mask,range_mask,
                        input_mode,key,shared_warps,d_count,d_offsets);
                } else {
                    cuckoo_shared<true,false><<<trial_blocks,shared_warps*32,sh_bytes>>>(n,offset,o.pnum,
                        p.buckets,p.kick,p.bucket_size,p.domain,p.half_bits,p.mask,range_mask,
                        input_mode,key,shared_warps,d_count,d_offsets);
                }
            } else if(use_shared_trials) {
                const unsigned trial_blocks=(n+shared_warps-1)/shared_warps;
                if(shared_kind==SharedKind::Map) {
                    cuckoo_shared<false,true><<<trial_blocks,shared_warps*32,sh_bytes>>>(n,offset,o.pnum,
                        p.buckets,p.kick,p.bucket_size,p.domain,p.half_bits,p.mask,range_mask,
                        input_mode,key,shared_warps,d_count,d_offsets);
                } else {
                    cuckoo_shared<false,false><<<trial_blocks,shared_warps*32,sh_bytes>>>(n,offset,o.pnum,
                        p.buckets,p.kick,p.bucket_size,p.domain,p.half_bits,p.mask,range_mask,
                        input_mode,key,shared_warps,d_count,d_offsets);
                }
            } else if(cached && o.mode=="bgh25") {
                cuckoo_cached<true><<<blocks,block_threads,sh_bytes>>>(n,offset,o.pnum,p.buckets,p.kick,
                    p.bucket_size,p.domain,p.half_bits,p.mask,range_mask,input_mode,key,pitch,
                    d_bins,d_rounds,d_count,d_offsets);
            } else if(cached) {
                cuckoo_cached<false><<<blocks,block_threads,sh_bytes>>>(n,offset,o.pnum,p.buckets,p.kick,
                    p.bucket_size,p.domain,p.half_bits,p.mask,range_mask,input_mode,key,pitch,
                    d_bins,d_rounds,d_count,d_offsets);
            } else if(o.mode=="bgh25") {
                cuckoo_trials<true><<<blocks,THREADS,sh_bytes>>>(n,offset,o.pnum,p.buckets,p.kick,
                     p.bucket_size,p.domain,p.half_bits,p.mask,range_mask,input_mode,
                     key,use_shared,d_bins,d_count,d_offsets);
            } else {
                cuckoo_trials<false><<<blocks,THREADS,sh_bytes>>>(n,offset,o.pnum,p.buckets,p.kick,
                     p.bucket_size,p.domain,p.half_bits,p.mask,range_mask,input_mode,
                     key,use_shared,d_bins,d_count,d_offsets);
            }
            CK(cudaGetLastError());
            uint32_t nf=0;
            CK(cudaMemcpy(&nf,d_count,sizeof(uint32_t),cudaMemcpyDeviceToHost));
            if(nf>n) throw std::runtime_error("GPU failure count out of range");
            if(nf) {
                CK(cudaMemcpy(chunk_failures.data(),d_offsets,nf*sizeof(uint32_t),cudaMemcpyDeviceToHost));
                failures.insert(failures.end(),chunk_failures.begin(),chunk_failures.begin()+nf);
            }
          },[&](uint64_t batch,uint32_t actual,const std::vector<uint32_t>& failures) {
              append_record(journal.get(),batch,actual,failures);
          });
        if(!committed) break;
    }
    bins_memory.free_checked(); rounds_memory.free_checked(); count_memory.free_checked(); offsets_memory.free_checked();
}
#endif // CUCKOO_HOST_TEST
static void print_progress(const Options& o,const Counters& s,
                           std::chrono::steady_clock::time_point start) {
    const uint64_t initial=s.completed.load();
    const uint64_t total=1ULL<<o.et;
    while(s.active.load()>0) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        const auto now=std::chrono::steady_clock::now();
        const auto done=s.completed.load(),fail=s.failed.load();
        double elapsed=std::chrono::duration<double>(now-start).count();
        const double average=(done-initial)/std::max(elapsed,1e-6);
        double eta=average>0?(double)(total-done)/average:-1.0;
        const unsigned filled=(unsigned)std::min<uint64_t>(30,(done*30)/total);
        std::cerr<<'\r'<<'['<<std::string(filled,'#')<<std::string(30-filled,'-')<<"] "
                 <<std::fixed<<std::setprecision(2)
                 <<100.0*done/total<<"% | "<<done<<"/"<<total
                 <<" | Fail "<<fail<<" | "<<std::setprecision(0)<<average<<" avg exp/s"
                 <<" | elapsed "<<std::setprecision(1)<<elapsed<<"s"
                 <<" | ETA "<<(eta>=0?std::to_string((uint64_t)eta)+"s":"--")<<"     "<<std::flush;
    }
    std::cerr<<'\n';
}
#ifndef CUCKOO_HOST_TEST
int main(int argc,char** argv) {
    try {
        auto o=parse(argc,argv);
        const auto p=parameters(o);
        fs::create_directories(o.out);
        RunLock run_lock(o.out);
        PauseSignals signals;
        Counters s;
        auto done=resume(o,p,s);
        const auto planned=1ULL<<o.et;
        const auto start=std::chrono::steady_clock::now();
        std::cout<<"mode="<<o.mode<<" ET="<<o.et<<" BT="<<o.bt
                 <<" PNUM="<<o.pnum<<" buckets="<<p.buckets<<" kick="<<p.kick
                 <<" bucket_size="<<p.bucket_size<<" ell_d="<<p.ell_d
                 <<" input_mode="<<o.input_mode<<"\n"
                 <<"Resume: "<<s.completed<<" completed / "<<s.failed<<" failed\n";
        if(o.status_only || s.completed.load()==planned || pause_requested(o) || s.failed.load()>o.stop_after) {
            const std::string status=s.completed.load()==planned?"complete":pause_requested(o)?"paused":
                                      s.failed.load()>o.stop_after?"threshold":"ready";
            final_outputs(o,p,s,0,status);
            std::cout<<"Status: "<<status<<"; pending batches="<<(done.size()-s.batches.load())<<"\n";
            return 0;
        }
        int gpus=0; CK(cudaGetDeviceCount(&gpus));
        if(gpus==0) throw std::runtime_error("No NVIDIA CUDA GPUs");
        if(o.workers==0) o.workers=gpus;
        if(o.workers<1 || o.workers>256) throw std::runtime_error("workers must be in 1..256");
        if(o.workers>gpus) std::cerr<<"WARNING: Multiple workers share GPUs; usually use one per GPU.\n";
        // Initialize constant tables before any worker can launch a kernel.
        // Re-uploading them concurrently on a shared GPU could race with readers.
        for(int gpu=0;gpu<std::min(gpus,o.workers);++gpu) {
            CK(cudaSetDevice(gpu)); upload_aes_tables(); aes_self_test();
            if(round_cache_enabled(o,p)) cached_aes_self_test();
        }
        std::mutex print_mutex;
        std::vector<std::thread> threads;
        std::vector<std::exception_ptr> errors(o.workers);
        std::exception_ptr launch_error;
        const auto work_start=steady_ns();
        if(s.failed.load()<=o.stop_after) {
            for(int g=0;g<o.workers;++g) {
                s.active.fetch_add(1);
                try {
                    threads.emplace_back([&,g]() {
                        try { gpu_worker(g,g%gpus,o,p,done,s,print_mutex); }
                        catch (...) {
                            errors[g]=std::current_exception();
                            s.abort_inflight.store(true); s.stop.store(true);
                        }
                        mark_finished(s); s.active.fetch_sub(1);
                    });
                } catch (...) {
                    s.active.fetch_sub(1);
                    launch_error=std::current_exception();
                    s.abort_inflight.store(true); s.stop.store(true); break;
                }
            }
            print_progress(o,s,start);
            for(auto& t:threads) t.join();
        }
        // Journals, rather than live counters, are the authoritative checkpoint.
        Counters recovered;
        resume(o,p,recovered);
        const bool error=bool(launch_error)||std::any_of(errors.begin(),errors.end(),[](const auto& e){return bool(e);});
        const std::string status=error?"error":recovered.completed.load()==planned?"complete":
                                  pause_requested(o)?"paused":"threshold";
        const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        const auto work_seconds=std::max<int64_t>(0,s.finished_ns.load()-work_start)*1e-9;
        final_outputs(o,p,recovered,elapsed,status,work_seconds);
        std::cout<<"RESULT status="<<status<<" completed="<<recovered.completed<<" failed="<<recovered.failed
                 <<" elapsed_s="<<elapsed<<"\n";
        if(pause_requested(o)) std::cout<<"Paused: unfinished batches were not counted. Restart the same command to resume. "
                                        "If using the PAUSE file, remove it first.\n";
        if(launch_error) std::rethrow_exception(launch_error);
        for(auto& e:errors) if(e) std::rethrow_exception(e);
        return 0;
    } catch (const std::exception& e) {
        std::cerr<<"FATAL: "<<e.what()<<"\n";
        return 1;
    }
}
#endif // CUCKOO_HOST_TEST
