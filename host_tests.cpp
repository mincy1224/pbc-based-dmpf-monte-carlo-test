#define CUCKOO_HOST_TEST
#include "cuckoo_bench.cu"
#include <array>
#include <set>

static void require(bool ok,const char* what) {
    if(!ok) throw std::runtime_error(what);
}
template<class F> static void must_throw(F f,const char* what) {
    bool threw=false; try { f(); } catch(const std::exception&) { threw=true; }
    require(threw,what);
}
using Journal=std::unique_ptr<FILE,decltype(&fclose)>;
static Journal journal(const Options& o,const Params& p,int worker=0) {
    return Journal(open_journal(o,p,worker),fclose);
}
static void mock_launch(uint64_t id,uint32_t n,uint32_t offset,const AesKey& key,
                        std::vector<uint32_t>& failures) {
    const auto expected=batch_key(123,id);
    require(std::memcmp(&key,&expected,sizeof(key))==0,"batch seed changed with scheduling");
    for(uint32_t j=offset;j<offset+n;++j) if((id*8+j)%7==0) failures.push_back(j);
}
static bool run_batch(const Options& o,uint64_t id,uint32_t micro,Counters& s,FILE* f) {
    return process_batch(o,id,micro,s,
        [&](auto n,auto offset,const auto& key,auto& failures){mock_launch(id,n,offset,key,failures);},
        [&](auto batch,auto count,const auto& failures){append_record(f,batch,count,failures);});
}
static Options make_options(const fs::path& root,const char* name) {
    Options o; o.et=6; o.bt=3; o.seed=123; o.pnum=16; o.ell_i=8;
    o.out=(root/name).string(); fs::create_directories(o.out); return o;
}
static std::set<std::string> csv_rows(const Options& o) {
    std::ifstream file(fs::path(o.out)/"failures.csv");
    std::string line; std::getline(file,line);
    std::set<std::string> rows; while(std::getline(file,line)) rows.insert(line); return rows;
}
static void check_keys() {
    const std::array<uint64_t,2> ids{0,7};
    const uint32_t expected[2][4]={{0x992dc0b2,0xecc207f0,0x5f9a7017,0x1fb84a2f},
                                  {0x58f85b99,0xfc3cbd02,0x022bf19c,0x10dad53d}};
    for(size_t i=0;i<ids.size();++i) {
        uint32_t out[4]; host_aes128(batch_key(123,ids[i]),0,0,0,0,out);
        require(std::memcmp(out,expected[i],sizeof(out))==0,"AES batch-key golden vector changed");
    }
    std::cout<<"PASS batch-key vectors against independent Python AES oracle\n";
}
static void check_pause(const fs::path& root) {
    auto o=make_options(root,"pause"); const auto p=parameters(o);
    auto f=journal(o,p); Counters s; PauseSignals signals;
    unsigned launches=0;
    bool result=process_batch(o,0,2,s,[&](auto n,auto offset,const auto& key,auto& failed){
        mock_launch(0,n,offset,key,failed); ++launches; std::raise(SIGINT);
    },[&](auto id,auto n,const auto& failed){append_record(f.get(),id,n,failed);});
    require(!result && launches==1 && s.completed==0 && s.failed==0,"partial batch was counted");
    f.reset(); Counters r; auto done=resume(o,p,r);
    require(!done[0] && r.completed==0,"partial batch was persisted");
    pause_signal.store(false); s.stop.store(false); f=journal(o,p);
    require(run_batch(o,0,3,s,f.get()),"unfinished batch did not restart");
    require(s.completed==8 && s.failed==2,"restart counts wrong");
    f.reset(); Counters saved; done=resume(o,p,saved);
    require(done[0] && saved.completed==8,"completed batch not recovered");

    auto full=make_options(root,"pause_after_last_micro"); const auto fp=parameters(full);
    auto ff=journal(full,fp); Counters fsaved;
    result=process_batch(full,0,8,fsaved,[&](auto n,auto offset,const auto& key,auto& failed){
        mock_launch(0,n,offset,key,failed); pause_signal.store(true);
    },[&](auto id,auto n,const auto& failed){append_record(ff.get(),id,n,failed);});
    require(result && fsaved.completed==8,"fully finished batch was discarded on pause");
    pause_signal.store(false);

    auto filecase=make_options(root,"pause_file"); const auto filep=parameters(filecase);
    std::ofstream(fs::path(filecase.out)/"PAUSE").close(); Counters filec;
    auto filef=journal(filecase,filep);
    require(!run_batch(filecase,0,2,filec,filef.get()) && filec.completed==0,"PAUSE file ignored");
    fs::remove(fs::path(filecase.out)/"PAUSE");
    auto aborted=make_options(root,"worker_error"); auto ap=parameters(aborted); Counters ac;
    {auto af=journal(aborted,ap); unsigned calls=0;
     must_throw([&]{process_batch(aborted,0,2,ac,[&](auto n,auto off,const auto& key,auto& failed){
        if(++calls==2) throw std::runtime_error("mock GPU error");
        mock_launch(0,n,off,key,failed);
     },[&](auto id,auto n,const auto& failed){append_record(af.get(),id,n,failed);});},"GPU error was ignored");
     ac.abort_inflight.store(true);
     require(!run_batch(aborted,1,2,ac,af.get()),"another worker's abort was ignored");}
    Counters ar; resume(aborted,ap,ar); require(ar.completed==0,"failed batch was committed");
    std::cout<<"PASS SIGINT, PAUSE file, discard partial batch, and commit finished batch\n";
}
static void check_journals(const fs::path& root) {
    auto o=make_options(root,"holes"); const auto p=parameters(o); Counters s;
    { auto f=journal(o,p,3); run_batch(o,6,2,s,f.get()); run_batch(o,2,2,s,f.get()); }
    const auto path=journal_name(o,3); const auto safe=fs::file_size(path);
    { std::ofstream f(path,std::ios::binary|std::ios::app); RecordHeader h{1,8,1}; f.write((char*)&h,sizeof(h)); f.write("x",1); }
    Counters restored; auto done=resume(o,p,restored);
    require(done[6] && done[2] && !done[1] && restored.completed==16,"out-of-order restore failed");
    require(fs::file_size(path)==safe,"torn tail was not removed");
    // Finish holes with a different worker count and micro size.
    std::vector<std::thread> workers; std::array<std::exception_ptr,3> errors{};
    for(int w=0;w<3;++w) workers.emplace_back([&,w]{try {
        auto f=journal(o,p,w);
        for(;;) {
            auto id=restored.next_batch.fetch_add(1); if(id>=done.size()) break;
            if(done[id]) continue;
            run_batch(o,id,5,restored,f.get());
        }
    } catch(...) { errors[w]=std::current_exception(); }});
    for(auto& w:workers) w.join(); for(auto e:errors) if(e) std::rethrow_exception(e);
    Counters recovered; done=resume(o,p,recovered);
    require(recovered.completed==64 && recovered.batches==8 && recovered.failed==10,"resume double-counted or omitted work");
    final_outputs(o,p,recovered,0,"complete");
    auto one=make_options(root,"one_shot"); const auto onep=parameters(one); Counters ones;
    { auto f=journal(one,onep); for(uint64_t id=0;id<8;++id) run_batch(one,id,1,ones,f.get()); }
    final_outputs(one,onep,ones,0,"complete");
    require(csv_rows(o)==csv_rows(one),"pause/resume differs from uninterrupted run");

    auto torn=make_options(root,"torn_file_header"); const auto tp=parameters(torn);
    {std::ofstream f(journal_name(torn,0),std::ios::binary); f.write("CUK",3);}
    Counters tc; resume(torn,tp,tc);
    {auto f=journal(torn,tp); require(run_batch(torn,0,2,tc,f.get()),"header repair failed");}
    Counters tr; resume(torn,tp,tr); require(tr.completed==8,"repaired header not usable");

    auto crash=make_options(root,"commit_before_counter_crash"); auto cp=parameters(crash); Counters cs;
    {auto f=journal(crash,cp); must_throw([&]{process_batch(crash,0,4,cs,
        [&](auto n,auto off,const auto& key,auto& failures){mock_launch(0,n,off,key,failures);},
        [&](auto id,auto n,const auto& failed){append_record(f.get(),id,n,failed); throw std::runtime_error("crash");});},"mock crash did not throw");}
    require(cs.completed==0,"mock crash unexpectedly published counters");
    Counters cr; resume(crash,cp,cr); require(cr.completed==8,"durable commit lost before counter update");

    auto corrupt=make_options(root,"corrupt"); auto xp=parameters(corrupt); Counters xc;
    {auto f=journal(corrupt,xp); run_batch(corrupt,0,8,xc,f.get());}
    const auto xfile=journal_name(corrupt,0); const auto xsize=fs::file_size(xfile);
    {std::fstream f(xfile,std::ios::binary|std::ios::in|std::ios::out); f.seekp(-1,std::ios::end); f.put('\0');}
    Counters xr; must_throw([&]{resume(corrupt,xp,xr);},"corrupt complete record was accepted");
    require(fs::file_size(xfile)==xsize,"corrupt complete record was silently truncated");

    auto dup=make_options(root,"duplicate"); auto dp=parameters(dup); Counters dc;
    {auto f=journal(dup,dp,0); run_batch(dup,0,8,dc,f.get());}
    {auto f=journal(dup,dp,1); run_batch(dup,0,8,dc,f.get());}
    Counters dr; must_throw([&]{resume(dup,dp,dr);},"duplicate batch was accepted");
    auto different=o; different.seed=124; Counters mismatch;
    must_throw([&]{resume(different,p,mismatch);},"mismatched seed was accepted");
    std::cout<<"PASS out-of-order journals, worker/micro changes, torn tails/headers, crash boundary, corruption, duplicates, parameter mismatch\n";
}
static void check_locks_and_parameters(const fs::path& root) {
    auto o=make_options(root,"locks");
    { RunLock lock(o.out); must_throw([&]{RunLock another(o.out);},"concurrent directory lock accepted"); }
    { RunLock lock(o.out); }
    must_throw([]{parse_u64("-1");},"negative unsigned argument accepted");
    must_throw([]{parse_u32("4294967296");},"overflowing argument accepted");
    auto p=parameters(o); require(p.buckets==23,"DRRT18 parameter derivation unexpectedly changed");
    o.mode="bgh25"; require(parameters(o).buckets==32,"BGH25 parameter derivation unexpectedly changed");
    std::cout<<"PASS directory lock release, argument bounds, and parameter compatibility\n";
}
int main(int argc,char** argv) {
    try {
        const auto base=fs::absolute(argc>1?argv[1]:"checkpoint_test_data");
        const auto root=base/("run_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(root);
        check_keys(); check_pause(root); check_journals(root); check_locks_and_parameters(root);
        std::cout<<"ALL HOST CHECKPOINT TESTS PASSED\n"; return 0;
    } catch(const std::exception& e) {std::cerr<<"TEST FAILURE: "<<e.what()<<'\n'; return 1;}
}
