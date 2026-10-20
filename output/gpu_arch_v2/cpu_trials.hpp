#pragma once
#include "cuckoo_core.hpp"
#include <wmmintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif

// Included after cuckoo_bench.cu's host definitions. No CUDA dependency.
static bool cpu_has_aesni() {
#ifdef _MSC_VER
    int info[4]; __cpuid(info,1); return (info[2]&(1<<25))!=0;
#else
    return __builtin_cpu_supports("aes");
#endif
}
static uint32_t cpu_bswap(uint32_t x) {
    return (x<<24)|((x&0xff00u)<<8)|((x>>8)&0xff00u)|(x>>24);
}
class CpuAes {
    __m128i rounds_[11];
public:
    explicit CpuAes(const AesKey& key) {
        for(int r=0;r<11;++r)
            rounds_[r]=_mm_set_epi32((int)cpu_bswap(key.w[4*r+3]),
                (int)cpu_bswap(key.w[4*r+2]),(int)cpu_bswap(key.w[4*r+1]),
                (int)cpu_bswap(key.w[4*r]));
    }
    __m128i encrypt(uint32_t a,uint32_t b,uint32_t c,uint32_t d) const {
        auto state=_mm_set_epi32((int)cpu_bswap(d),(int)cpu_bswap(c),
                                (int)cpu_bswap(b),(int)cpu_bswap(a));
        state=_mm_xor_si128(state,rounds_[0]);
        for(int r=1;r<10;++r) state=_mm_aesenc_si128(state,rounds_[r]);
        return _mm_aesenclast_si128(state,rounds_[10]);
    }
    void words(uint32_t a,uint32_t b,uint32_t c,uint32_t d,uint32_t out[4]) const {
        uint32_t little[4];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(little),encrypt(a,b,c,d));
        for(int i=0;i<4;++i) out[i]=cpu_bswap(little[i]);
    }
    uint32_t last_word(uint32_t a,uint32_t b,uint32_t c,uint32_t d) const {
        return cpu_bswap((uint32_t)_mm_cvtsi128_si32(_mm_srli_si128(encrypt(a,b,c,d),12)));
    }
};
static AesKey cpu_key_from_words(const uint32_t words[4]) {
    const uint64_t lo=uint64_t(cpu_bswap(words[0]))|(uint64_t(cpu_bswap(words[1]))<<32);
    const uint64_t hi=uint64_t(cpu_bswap(words[2]))|(uint64_t(cpu_bswap(words[3]))<<32);
    return expand_key(lo,hi);
}
static uint64_t cpu_splitmix(uint64_t& state) {
    state+=0x9E3779B97F4A7C15ULL;
    uint64_t z=state;
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;
    z=(z^(z>>27))*0x94D049BB133111EBULL;
    return z^(z>>31);
}
static uint32_t cpu_bucket(uint64_t element,uint32_t which,const CpuAes& aes,const Params& p) {
    uint64_t y=element*3+which;
    do {
        auto left=y>>p.half_bits, right=y&p.mask;
        for(uint32_t r=0;r<8;++r) {
            const auto next=(left^(uint64_t(aes.last_word(0x43455350u,r,p.half_bits,(uint32_t)right))&p.mask))&p.mask;
            left=right; right=next;
        }
        y=(left<<p.half_bits)|right;
    } while(y>=p.domain);
    if(p.domain<=0xffffffffULL) return uint32_t(y)/uint32_t(p.bucket_size);
    return static_cast<uint32_t>(y/p.bucket_size);
}
static bool cpu_trial(const Options& o,const Params& p,const CpuAes& batch,
                      uint32_t offset,std::vector<uint32_t>& bins) {
    uint32_t words[4];
    batch.words(0x4b455931u,0,offset,0x50525031u,words);
    const CpuAes aes(cpu_key_from_words(words));
    batch.words(0x57414c4bu,0,offset,0x524e4731u,words);
    uint64_t walk=(uint64_t(words[0])<<32)|words[1];
    const auto range_mask=(uint64_t{1}<<o.ell_i)-1;
    const auto shift=((uint64_t(words[2])<<32)|words[3])&range_mask;
    uint64_t input_state=shift^0xd1b54a32d192ed03ULL;
    const auto step=(cpu_splitmix(input_state)|1ULL)&range_mask;
    auto randomness=cpu_splitmix(walk);
    const bool sequential=o.input_mode=="sequential";
    const bool bgh=o.mode=="bgh25";
    auto element=[&](uint32_t ordinal) {return sequential?uint64_t(ordinal):(shift+ordinal*step)&range_mask;};
    std::fill(bins.begin(),bins.end(),0xffffffffu);
    for(uint32_t item=0;item<o.pnum;++item) {
        if(bgh) {
            uint32_t cur=item,collisions=0;
            for(;;) {
                if(randomness<3) randomness=cpu_splitmix(walk);
                const uint32_t which=static_cast<uint32_t>(randomness%3);
                randomness/=3;
                const auto bucket=cpu_bucket(element(cur),which,aes,p);
                const auto old=bins[bucket];
                if(old==0xffffffffu) {bins[bucket]=cur|(which<<16); break;}
                if(++collisions>p.kick) return true;
                bins[bucket]=cur|(which<<16); cur=old&65535u;
            }
        } else {
            const auto initial0=cpu_bucket(element(item),0,aes,p);
            if(bins[initial0]==0xffffffffu) {bins[initial0]=item;continue;}
            bool inserted=false;
            for(uint32_t k=1;k<3;++k) {
                const auto bucket=cpu_bucket(element(item),k,aes,p);
                if(bins[bucket]==0xffffffffu) {bins[bucket]=item|(k<<16);inserted=true;break;}
            }
            if(inserted) continue;
            uint32_t cur=item,choice=0,bucket=initial0,kicks=0;
            for(;;) {
                const auto displaced=bins[bucket];
                if(displaced==0xffffffffu) {bins[bucket]=cur|(choice<<16);break;}
                if(kicks>=p.kick) return true;
                bins[bucket]=cur|(choice<<16);
                cur=displaced&65535u; choice=(displaced>>16)&3u;
                const auto previous=bucket;
                bool found=false;
                for(unsigned k=0;k<3;++k) {
                    choice=(choice+1)%3;
                    const auto next=cpu_bucket(element(cur),choice,aes,p);
                    if(next!=previous) {bucket=next;found=true;break;}
                }
                if(!found) return true;
                ++kicks;
            }
        }
    }
    return false;
}
static bool cpu_trial_cached(const Options& o,const Params& p,const CpuAes& batch,
                             uint32_t offset,std::vector<uint32_t>& bins,std::vector<uint16_t>& rounds) {
    uint32_t words[4];batch.words(0x4b455931u,0,offset,0x50525031u,words);
    const CpuAes aes(cpu_key_from_words(words));
    batch.words(0x57414c4bu,0,offset,0x524e4731u,words);
    uint64_t walk=(uint64_t(words[0])<<32)|words[1];
    const uint64_t range_mask=(uint64_t{1}<<o.ell_i)-1;
    const uint64_t shift=((uint64_t(words[2])<<32)|words[3])&range_mask;
    uint64_t input_state=shift^0xd1b54a32d192ed03ULL;
    const uint64_t step=(cpu_splitmix(input_state)|1ULL)&range_mask;
    const uint64_t randomness=cpu_splitmix(walk);
    const unsigned width=1u<<p.half_bits;
    for(unsigned r=0;r<8;++r)
        for(unsigned right=0;right<width;++right)
            rounds[r*width+right]=static_cast<uint16_t>(aes.last_word(0x43455350u,r,p.half_bits,right)&p.mask);
    std::fill(bins.begin(),bins.end(),0xffffffffu);
    const CachedBuckets bucket{rounds.data(),1,p.domain,p.bucket_size,p.mask,range_mask,shift,step,
                               p.half_bits,o.input_mode=="sequential"};
    const StridedBins table{bins.data(),1};
    return o.mode=="bgh25"?insert_core<true>(o.pnum,p.kick,table,bucket,walk,randomness):
                           insert_core<false>(o.pnum,p.kick,table,bucket,walk,randomness);
}
static void cpu_aes_self_test() {
    const CpuAes aes(expand_key(0x0706050403020100ULL,0x0f0e0d0c0b0a0908ULL));
    uint32_t words[4]; aes.words(0x00112233u,0x44556677u,0x8899aabbu,0xccddeeffu,words);
    const uint32_t expected[]={0x69c4e0d8u,0x6a7b0430u,0xd8cdb780u,0x70b4c55au};
    if(std::memcmp(words,expected,sizeof(expected))) throw std::runtime_error("CPU AES-NI NIST self-test failed");
}

struct CpuBatchPaused {};
// One coordinator owns the batch and its key. Helper CPU threads divide offsets
// inside that batch, so short tests can use multiple cores without changing BT.
// Only a fully completed batch is passed to the existing journal commit function.
static void cpu_batch_launch(const Options& o,const Params& p,const AesKey& key,
                             uint32_t count,Counters& s,std::vector<uint32_t>& failures) {
    const CpuAes batch(key);
    const bool cached=round_cache_enabled(o,p);
    const uint32_t grain=std::min(o.micro,256u);
    std::atomic<uint32_t> next{0},finished{0};
    std::atomic<bool> abort{false};
    std::vector<std::vector<uint32_t>> results(o.workers);
    std::vector<std::exception_ptr> errors(o.workers);
    std::vector<std::thread> threads;
    auto worker=[&](int id) {
        try {
            std::vector<uint32_t> bins(p.buckets);
            std::vector<uint16_t> rounds(cached?(size_t{8}<<p.half_bits):0);
            while(!abort.load() && !s.abort_inflight.load()) {
                if(pause_requested(o)) break;
                const auto first=next.fetch_add(grain);
                if(first>=count) break;
                const auto end=std::min(count,first+grain);
                for(auto j=first;j<end;++j) {
                    const bool failed=cached?cpu_trial_cached(o,p,batch,j,bins,rounds):cpu_trial(o,p,batch,j,bins);
                    if(failed) results[id].push_back(j);
                }
                finished.fetch_add(end-first);
            }
        } catch(...) {errors[id]=std::current_exception();abort.store(true);}
    };
    try {
        for(int id=1;id<o.workers;++id) threads.emplace_back(worker,id);
        worker(0);
    } catch(...) {
        abort.store(true); for(auto& thread:threads) thread.join(); throw;
    }
    for(auto& thread:threads) thread.join();
    for(const auto& error:errors) if(error) std::rethrow_exception(error);
    if(finished.load()!=count) throw CpuBatchPaused{};
    for(const auto& result:results) failures.insert(failures.end(),result.begin(),result.end());
}
