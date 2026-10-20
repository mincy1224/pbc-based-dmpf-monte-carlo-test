#pragma once
#include <cstdint>
#include <cstddef>
#if (defined(__CUDACC__) || defined(__CUDA__)) && !defined(CUCKOO_HOST_TEST)
#define CUCKOO_CORE __device__ __forceinline__
#define CUCKOO_HD __host__ __device__
#else
#define CUCKOO_CORE inline
#define CUCKOO_HD
#endif

// Packed cuckoo-table entry.  Two layouts are used:
//   32-bit: 16-bit ordinal | 2-bit choice << 16 (original GPU/CPU table)
//   16-bit: 14-bit ordinal | 2-bit choice << 14 (shared-memory tables)
// 0xff..ff is empty.  A valid 16-bit entry never produces 0xffff because the
// choice is at most 2, so the top two bits of a packed entry are at most 0b10.
template<typename Value,unsigned ChoiceShift>
struct PackedBins {
    using value_type=Value;
    Value* base;
    size_t stride;
    CUCKOO_CORE Value& operator[](uint32_t index) const {
        return base[size_t(index)*stride];
    }
    CUCKOO_CORE static Value empty() {return static_cast<Value>(-1);}
    CUCKOO_CORE static Value pack(uint32_t ordinal,uint32_t choice) {
        return static_cast<Value>(ordinal|(choice<<ChoiceShift));
    }
    CUCKOO_CORE static uint32_t ordinal(Value entry) {
        return uint32_t(entry)&((uint32_t(1)<<ChoiceShift)-1u);
    }
    CUCKOO_CORE static uint32_t choice(Value entry) {
        return (uint32_t(entry)>>ChoiceShift)&3u;
    }
};
using StridedBins = PackedBins<uint32_t,16>;
using SharedBins = PackedBins<uint16_t,14>;

// Round cache for one experiment: entry (round, right) stores the low
// half_bits bits of the Feistel round function.  This is exactly the value
// used by the direct AES path, only evaluated once per distinct input.
template<typename Value=uint16_t>
struct CachedBucketsT {
    const Value* rounds;
    size_t stride;
    uint64_t domain,width,mask,range_mask,shift,step;
    uint32_t half_bits;
    bool sequential;
    CUCKOO_CORE uint32_t bucket_of(uint64_t y) const {
        // 64-bit division is expensive on the GPU.  The PRP domain is below
        // 2^32 for the common ell_i<=30 cases, where 32-bit division is much
        // cheaper; the quotient is still < buckets either way.
        if(domain<=0xffffffffULL) return uint32_t(uint32_t(y)/uint32_t(width));
        return uint32_t(y/width);
    }
    CUCKOO_CORE uint32_t operator()(uint32_t ordinal,uint32_t which) const {
        const uint64_t element=sequential?uint64_t(ordinal):(shift+uint64_t(ordinal)*step)&range_mask;
        uint64_t y=element*3+which;
        do {
            auto left=y>>half_bits,right=y&mask;
#ifdef __CUDACC__
            #pragma unroll
#endif
            for(unsigned r=0;r<8;++r) {
                const uint64_t next=left^uint64_t(rounds[((size_t(r)<<half_bits)+right)*stride]);
                left=right;right=next&mask;
            }
            y=(left<<half_bits)|right;
        } while(y>=domain);
        return bucket_of(y);
    }
};
using CachedBuckets = CachedBucketsT<uint16_t>;

// Precomputed bucket positions for every (ordinal, candidate) pair.  The map
// build is parallel across all lanes; the sequential insertion then needs one
// shared lookup per probe instead of the eight dependent Feistel lookups.
struct PrecomputedBuckets {
    const uint16_t* values;
    CUCKOO_CORE uint32_t operator()(uint32_t ordinal,uint32_t which) const {
        return uint32_t(values[size_t(ordinal)*3+which]);
    }
};

// Warp-private shared-memory layout: one experiment per warp.  The 16-bit table
// needs 14 ordinal bits, so pnum<16384; the round cache has 8*2^half entries.
CUCKOO_HD inline bool shared_layout_supported(uint32_t pnum,uint32_t half_bits) {
    return pnum<(1u<<14) && half_bits<=12;
}
CUCKOO_HD inline uint64_t shared_bins_bytes(uint32_t buckets) {
    return uint64_t(buckets)*sizeof(uint16_t);
}
CUCKOO_HD inline uint64_t shared_rounds_bytes(uint32_t half_bits) {
    return (uint64_t(8)<<half_bits)*sizeof(uint16_t);
}
CUCKOO_HD inline uint64_t shared_warp_bytes(uint32_t buckets,uint32_t half_bits) {
    return shared_bins_bytes(buckets)+shared_rounds_bytes(half_bits);
}
CUCKOO_HD inline bool shared_map_supported(uint32_t pnum,uint32_t half_bits,uint32_t buckets) {
    return shared_layout_supported(pnum,half_bits) && buckets<=0xffffu;
}
CUCKOO_HD inline uint64_t shared_map_bytes(uint32_t pnum) {
    return uint64_t(pnum)*3*sizeof(uint16_t);
}
// Map variant: bins + 3*pnum bucket values + the round cache used to build it.
CUCKOO_HD inline uint64_t shared_map_warp_bytes(uint32_t buckets,uint32_t half_bits,uint32_t pnum) {
    return shared_bins_bytes(buckets)+shared_map_bytes(pnum)+shared_rounds_bytes(half_bits);
}

CUCKOO_CORE uint64_t core_splitmix(uint64_t& state) {
    state+=0x9E3779B97F4A7C15ULL;uint64_t z=state;
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;z=(z^(z>>27))*0x94D049BB133111EBULL;
    return z^(z>>31);
}
// Shared by every GPU/CPU path.  Choices and termination conditions exactly
// match the original direct kernel; only the table representation changes.
template<bool BGH,class Bins,class Buckets>
CUCKOO_CORE bool insert_core(uint32_t pnum,uint32_t maxkick,Bins bins,Buckets bucket,
                            uint64_t walk,uint64_t randomness) {
    using Value=typename Bins::value_type;
    const Value empty=Bins::empty();
    for(uint32_t item=0;item<pnum;++item) {
        if constexpr(BGH) {
            uint32_t cur=item,collisions=0;
            for(;;) {
                if(randomness<3) randomness=core_splitmix(walk);
                const uint32_t which=static_cast<uint32_t>(randomness%3);randomness/=3;
                const uint32_t target=bucket(cur,which);
                const Value old=bins[target];
                if(old==empty) {bins[target]=Bins::pack(cur,which);break;}
                if(++collisions>maxkick) return true;
                bins[target]=Bins::pack(cur,which);cur=Bins::ordinal(old);
            }
        } else {
            const auto initial=bucket(item,0);
            if(bins[initial]==empty) {bins[initial]=Bins::pack(item,0);continue;}
            bool inserted=false;
            for(uint32_t k=1;k<3;++k) {
                const auto target=bucket(item,k);
                if(bins[target]==empty) {bins[target]=Bins::pack(item,k);inserted=true;break;}
            }
            if(inserted) continue;
            uint32_t cur=item,choice=0,target=initial,kicks=0;
            for(;;) {
                const auto displaced=bins[target];
                if(displaced==empty) {bins[target]=Bins::pack(cur,choice);break;}
                if(kicks>=maxkick) return true;
                bins[target]=Bins::pack(cur,choice);
                cur=Bins::ordinal(displaced);choice=Bins::choice(displaced);
                const auto previous=target;bool found=false;
                for(unsigned k=0;k<3;++k) {
                    choice=(choice+1)%3;const auto next=bucket(cur,choice);
                    if(next!=previous) {target=next;found=true;break;}
                }
                if(!found) return true;
                ++kicks;
            }
        }
    }
    return false;
}
