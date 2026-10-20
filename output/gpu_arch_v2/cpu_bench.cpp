#define CUCKOO_HOST_TEST
#include "cuckoo_bench.cu"
#include "cpu_trials.hpp"

int main(int argc,char** argv) {
    try {
        auto o=parse(argc,argv); o.backend="cpu-aesni";
        if(o.workers==0) o.workers=1;
        const auto p=parameters(o);
        fs::create_directories(o.out); RunLock run_lock(o.out); PauseSignals signals;
        Counters s; const auto done=resume(o,p,s);
        const auto planned=uint64_t{1}<<o.et;
        const auto start=std::chrono::steady_clock::now();
        std::cout<<"backend=cpu-aesni mode="<<o.mode<<" workers="<<o.workers
                 <<" PNUM="<<o.pnum<<" buckets="<<p.buckets<<" BT="<<o.bt
                 <<"\nResume: "<<s.completed<<" completed / "<<s.failed<<" failed\n";
        if(o.status_only || s.completed.load()==planned || pause_requested(o) || s.failed.load()>o.stop_after) {
            const std::string status=s.completed.load()==planned?"complete":pause_requested(o)?"paused":
                                      s.failed.load()>o.stop_after?"threshold":"ready";
            final_outputs(o,p,s,0,status); return 0;
        }
        if(!cpu_has_aesni()) throw std::runtime_error("CPU does not support AES-NI");
        cpu_aes_self_test();
        std::exception_ptr error;
        s.active.store(1); const auto work_start=steady_ns();
        std::thread coordinator([&] {
            try {
                std::unique_ptr<FILE,decltype(&fclose)> journal(open_journal(o,p,0),fclose);
                for(uint64_t id=0;id<done.size() && !s.stop.load();++id) {
                    if(done[id]) continue;
                    try {
                        if(!process_batch(o,id,static_cast<uint32_t>(batch_length(o,id)),s,
                            [&](auto count,auto offset,const auto& key,auto& failed) {
                                if(offset!=0) throw std::runtime_error("CPU launch must cover the whole batch");
                                cpu_batch_launch(o,p,key,count,s,failed);
                            },[&](auto batch,auto count,const auto& failed) {
                                append_record(journal.get(),batch,count,failed);
                            })) break;
                    } catch(const CpuBatchPaused&) {s.stop.store(true);break;}
                }
            } catch(...) {error=std::current_exception();s.stop.store(true);}
            mark_finished(s); s.active.store(0);
        });
        print_progress(o,s,start); coordinator.join();
        Counters recovered; resume(o,p,recovered);
        const std::string status=error?"error":recovered.completed.load()==planned?"complete":
                                  pause_requested(o)?"paused":"threshold";
        const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        const double work_seconds=std::max<int64_t>(0,s.finished_ns.load()-work_start)*1e-9;
        final_outputs(o,p,recovered,elapsed,status,work_seconds);
        std::cout<<"RESULT backend=cpu-aesni status="<<status<<" completed="<<recovered.completed
                 <<" failed="<<recovered.failed<<" elapsed_s="<<elapsed<<" work_s="<<work_seconds<<'\n';
        if(error) std::rethrow_exception(error);
        return 0;
    } catch(const std::exception& e) {std::cerr<<"FATAL: "<<e.what()<<'\n';return 1;}
}
