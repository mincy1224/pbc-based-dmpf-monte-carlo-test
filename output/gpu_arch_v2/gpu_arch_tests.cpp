#define CUCKOO_HOST_TEST
#include "cuckoo_bench.cu"
#include "cpu_trials.hpp"
#include "gpu_cached.cuh"
#include "gpu_shared.cuh"

static void ensure(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
static std::vector<uint32_t> replicated_table() {
    std::vector<uint32_t> table(8192);
    for(unsigned index=0;index<256;++index) {
        const uint32_t s=HOST_SBOX[index],a=xtime(static_cast<uint8_t>(s)),b=a^s;
        const uint32_t t=(a<<24)|(s<<16)|(s<<8)|b;
        for(unsigned lane=0;lane<32;++lane) table[index*32+lane]=t;
    }
    return table;
}
static void check_lane_aes(const std::vector<uint32_t>& table) {
    uint64_t state=123;
    for(unsigned lane=0;lane<32;++lane)
        for(unsigned sample=0;sample<16;++sample) {
            uint32_t words[4];for(auto& value:words)value=static_cast<uint32_t>(cpu_splitmix(state));
            AesKey key;lane_expand(words[0],words[1],words[2],words[3],table.data(),lane,key);
            const auto expected_key=cpu_key_from_words(words);
            ensure(std::memcmp(&key,&expected_key,sizeof(key))==0,"replicated-table key expansion differs from AES-NI");
            for(auto& value:words)value=static_cast<uint32_t>(cpu_splitmix(state));
            uint32_t actual[4],expected[4];
            lane_aes(key,table.data(),lane,words[0],words[1],words[2],words[3],actual);
            CpuAes(expected_key).words(words[0],words[1],words[2],words[3],expected);
            ensure(std::memcmp(actual,expected,sizeof(actual))==0,"replicated-table AES differs from AES-NI");
        }
}
static void emulate_cached_trials(Options o,const std::vector<uint32_t>& table,unsigned count) {
    const auto p=parameters(o);
    const unsigned width=1u<<p.half_bits;
    const size_t pitch=(count+31)/32*32;
    std::vector<uint16_t> rounds(pitch*8*width,0xbeef);
    std::vector<uint32_t> bins(pitch*p.buckets,0x12345678u),expected_bins(p.buckets);
    const auto bk=batch_key(123,7);const CpuAes batch(bk);
    for(unsigned tid=0;tid<count;++tid) {
        const unsigned lane=tid%32;uint32_t seed[4];
        lane_aes(bk,table.data(),lane,0x4b455931u,0,tid,0x50525031u,seed);
        AesKey key;lane_expand(seed[0],seed[1],seed[2],seed[3],table.data(),lane,key);
        lane_aes(bk,table.data(),lane,0x57414c4bu,0,tid,0x524e4731u,seed);
        uint64_t walk=(uint64_t(seed[0])<<32)|seed[1];
        const uint64_t range_mask=(uint64_t{1}<<o.ell_i)-1;
        const uint64_t shift=((uint64_t(seed[2])<<32)|seed[3])&range_mask;
        uint64_t input=shift^0xd1b54a32d192ed03ULL;
        const uint64_t step=(core_splitmix(input)|1ULL)&range_mask,randomness=core_splitmix(walk);
        for(unsigned r=0;r<8;++r) for(unsigned right=0;right<width;++right) {
            uint32_t output[4];lane_aes(key,table.data(),lane,0x43455350u,r,p.half_bits,right,output);
            rounds[(r*width+right)*pitch+tid]=static_cast<uint16_t>(output[3]&p.mask);
        }
        const CachedBuckets bucket{rounds.data()+tid,pitch,p.domain,p.bucket_size,p.mask,range_mask,
                                    shift,step,p.half_bits,o.input_mode=="sequential"};
        // Verify bucket positions themselves, including cycle walking.
        const CpuAes reference_key(key);
        for(uint32_t ordinal=0;ordinal<o.pnum;ordinal+=std::max(1u,o.pnum/17))
            for(unsigned which=0;which<3;++which) {
                const uint64_t element=o.input_mode=="sequential"?uint64_t(ordinal):(shift+uint64_t(ordinal)*step)&range_mask;
                ensure(bucket(ordinal,which)==cpu_bucket(element,which,reference_key,p),"cached PRP bucket differs from direct AES-NI");
            }
        const StridedBins view{bins.data()+tid,pitch};
        for(unsigned i=0;i<p.buckets;++i) view[i]=0xffffffffu;
        const bool actual=o.mode=="bgh25"?insert_core<true>(o.pnum,p.kick,view,bucket,walk,randomness):
                                            insert_core<false>(o.pnum,p.kick,view,bucket,walk,randomness);
        const bool expected=cpu_trial(o,p,batch,tid,expected_bins);
        ensure(actual==expected,"cached GPU host emulation has different insertion outcome");
        for(unsigned i=0;i<p.buckets;++i) ensure(view[i]==expected_bins[i],"cached insertion table differs from direct CPU table");
    }
    for(size_t tid=count;tid<pitch;++tid) for(unsigned i=0;i<p.buckets;++i)
        ensure(bins[i*pitch+tid]==0x12345678u,"padded trial column was overwritten");
}
// Host emulation of the warp-private shared layout: uint16 packed bins, a
// per-trial round cache and the same insert_core used on the device.  Verifies
// both the failure decision and every final table slot against the direct
// AES-NI reference, including a non-zero launch offset and a short tail.
static void emulate_shared_trials(Options o,unsigned first,unsigned count) {
    const auto p=parameters(o);
    ensure(shared_layout_supported(o.pnum,p.half_bits),"shared layout unexpectedly unsupported");
    const unsigned width=1u<<p.half_bits,entries=8u<<p.half_bits;
    const CpuAes batch(batch_key(o.seed,7));
    std::vector<uint16_t> bins(p.buckets),rounds(entries);
    std::vector<uint32_t> reference_bins(p.buckets);
    for(unsigned index=0;index<count;++index) {
        const unsigned tid=first+index;
        uint32_t words[4];
        batch.words(0x4b455931u,0,tid,0x50525031u,words);
        const CpuAes aes(cpu_key_from_words(words));
        batch.words(0x57414c4bu,0,tid,0x524e4731u,words);
        uint64_t walk=(uint64_t(words[0])<<32)|words[1];
        const uint64_t range_mask=(uint64_t{1}<<o.ell_i)-1;
        const uint64_t shift=((uint64_t(words[2])<<32)|words[3])&range_mask;
        uint64_t input_state=shift^0xd1b54a32d192ed03ULL;
        const uint64_t step=(cpu_splitmix(input_state)|1ULL)&range_mask;
        const uint64_t randomness=cpu_splitmix(walk);
        for(unsigned e=0;e<entries;++e)
            rounds[e]=static_cast<uint16_t>(
                aes.last_word(0x43455350u,e>>p.half_bits,p.half_bits,e&(width-1))&p.mask);
        std::fill(bins.begin(),bins.end(),SharedBins::empty());
        const CachedBucketsT<uint16_t> bucket{rounds.data(),1,p.domain,p.bucket_size,
            p.mask,range_mask,shift,step,p.half_bits,o.input_mode=="sequential"};
        const SharedBins table{bins.data(),1};
        const bool actual=o.mode=="bgh25"?
            insert_core<true>(o.pnum,p.kick,table,bucket,walk,randomness):
            insert_core<false>(o.pnum,p.kick,table,bucket,walk,randomness);
        const bool expected=cpu_trial(o,p,batch,tid,reference_bins);
        ensure(actual==expected,"shared-layout outcome differs from direct CPU AES-NI");
        for(unsigned i=0;i<p.buckets;++i) {
            if(reference_bins[i]==0xffffffffu) {
                ensure(bins[i]==SharedBins::empty(),"shared-layout empty slot mismatch");
            } else {
                ensure(SharedBins::ordinal(bins[i])==(reference_bins[i]&65535u) &&
                       SharedBins::choice(bins[i])==((reference_bins[i]>>16)&3u),
                       "shared-layout table differs from direct CPU AES-NI");
            }
        }
    }
}
// The kernel maps one warp to one trial.  This checks the same host/device
// helper used by the kernel: every offset in [first, first+count) is launched
// exactly once, including short tail blocks and a non-zero micro offset.
static void check_shared_schedule(unsigned count,unsigned first,unsigned warps) {
    std::vector<unsigned> hits(count,0);
    const unsigned blocks=(count+warps-1)/warps;
    for(unsigned block=0;block<blocks;++block)
        for(unsigned warp=0;warp<warps;++warp) {
            uint32_t trial=0;
            if(shared_trial_index(block,warp,warps,count,first,trial)) {
                ensure(trial>=first && trial<first+count,"shared launch index out of range");
                ++hits[trial-first];
            }
        }
    for(unsigned hit:hits) ensure(hit==1,"shared launch index missing or duplicated");
}
// Host policy checks for --table-layout/--prp-cache selection.
static void check_layout_selection() {
    auto planned=[&](Options o,uint32_t pnum,uint32_t ell) {
        o.pnum=pnum;o.ell_i=ell;o.mode="bgh25";
        return std::string(table_layout_name(o,parameters(o)));
    };
    Options o;
    ensure(planned(o,4096,16)=="shared-trial","auto should pick shared-trial for m=4096");
    ensure(planned(o,1024,16)=="shared-trial","auto should pick shared-trial for m=1024");
    ensure(planned(o,512,16)=="cached-global","auto should keep cached-global for m=512");
    ensure(planned(o,16,8)=="direct-global","auto should keep the per-thread direct path at m=16");
    ensure(planned(o,16,32)=="direct-global","large half_bits should disable the round cache");
    o.table_layout="global";
    ensure(planned(o,4096,16)=="cached-global","--table-layout global must not use shared");
    o=Options{};o.table_layout="shared";o.prp_cache="off";
    bool threw=false;
    try { (void)planned(o,4096,16); } catch(const std::runtime_error&) { threw=true; }
    ensure(threw,"shared layout with --prp-cache off must be rejected");
    o=Options{};o.table_layout="shared";o.force_global=true;
    threw=false;
    try { (void)planned(o,4096,16); } catch(const std::runtime_error&) { threw=true; }
    ensure(threw,"shared layout with --global-table must be rejected");
}
int main() {
    if(!cpu_has_aesni()) return 77;
    try {
        check_layout_selection();
        for(unsigned count:{1u,5u,7u,64u,65u,257u})
            for(unsigned warps:{1u,2u,4u,7u,8u})
                for(unsigned first:{0u,7u,123u}) check_shared_schedule(count,first,warps);
        const auto table=replicated_table();check_lane_aes(table);
        for(const auto& mode:{std::string("bgh25"),std::string("drrt18")})
            for(const auto& input:{std::string("sequential"),std::string("affine")}) {
                Options o;o.mode=mode;o.input_mode=input;o.pnum=16;o.ell_i=8;
                emulate_cached_trials(o,table,65);
                emulate_shared_trials(o,0,65);
                emulate_shared_trials(o,7,5);
                o.pnum=4096;o.ell_i=16;emulate_cached_trials(o,table,2);
                emulate_shared_trials(o,0,5);
                // Largest supported 16-bit table: ordinal 15999 uses all 14
                // ordinal bits while choice still fits in bits 14-15.
                o.pnum=16000;o.ell_i=14;emulate_shared_trials(o,3,2);
            }
        std::cout<<"PASS replicated AES/key expansion, cached PRP positions, strided/tail layout, "
                   "shared-trial layout and exact insertion tables (host emulation)\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
