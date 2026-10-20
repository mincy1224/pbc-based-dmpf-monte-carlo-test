#define CUCKOO_HOST_TEST
#include "cuckoo_bench.cu"
#include "cpu_trials.hpp"
#include "cpu_golden.hpp"

static void check(bool condition,const char* message) {
    if(!condition) throw std::runtime_error(message);
}
static void check_reference(const char* mode,const std::vector<uint32_t>& expected) {
    Options o; o.mode=mode;o.seed=123;o.pnum=16;o.ell_i=8;o.et=15;o.bt=8;
    const auto p=parameters(o); std::vector<uint32_t> bins(p.buckets),cached_bins(p.buckets),failed;
    std::vector<uint16_t> rounds(size_t{8}<<p.half_bits);
    for(uint32_t batch=0;batch<128;++batch) {
        const CpuAes aes(batch_key(o.seed,batch));
        for(uint32_t offset=0;offset<256;++offset) {
            const bool direct=cpu_trial(o,p,aes,offset,bins);
            const bool cached=cpu_trial_cached(o,p,aes,offset,cached_bins,rounds);
            check(direct==cached && bins==cached_bins,"cached CPU differs from direct CPU outcome/table");
            if(direct) failed.push_back(batch*256+offset);
        }
    }
    check(failed==expected,"CPU AES-NI decisions differ from independent 32768-trial Python reference");
    std::cout<<"PASS "<<mode<<" all 32768 trials against Python reference; failed="<<failed.size()<<'\n';
}
int main(int argc,char** argv) {
    if(!cpu_has_aesni()) {std::cout<<"SKIP: AES-NI unavailable\n";return 77;}
    try {
        cpu_aes_self_test();
        check_reference("bgh25",golden_bgh25);check_reference("drrt18",golden_drrt18);
        Options o;o.seed=123;o.pnum=3;o.ell_i=8;o.input_mode="affine";o.workers=4;o.micro=3;
        const auto root=fs::absolute(argc>1?argv[1]:"cpu_test_data")/std::to_string(steady_ns());
        fs::create_directories(root);o.out=root.string();Counters s;
        for(const auto& mode:{std::string("bgh25"),std::string("drrt18")}) {
            o.mode=mode;const auto p=parameters(o);std::vector<uint32_t> failed;
            cpu_batch_launch(o,p,batch_key(123,0),256,s,failed);
            std::sort(failed.begin(),failed.end());
            const std::vector<uint32_t> expected=mode=="bgh25"?std::vector<uint32_t>{4,174,197}:std::vector<uint32_t>{166,227};
            check(failed==expected,"parallel affine CPU decisions differ from independent Python reference");
        }
        std::ofstream(root/"PAUSE").close();bool paused=false;
        try {std::vector<uint32_t> failed;cpu_batch_launch(o,parameters(o),batch_key(123,0),256,s,failed);}
        catch(const CpuBatchPaused&) {paused=true;}
        check(paused,"incomplete CPU batch was accepted after pause");
        std::cout<<"PASS NIST AES vector, parallel affine trials and CPU batch discard\n";return 0;
    } catch(const std::exception& e) {std::cerr<<"CPU TEST FAILURE: "<<e.what()<<'\n';return 1;}
}
