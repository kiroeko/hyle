#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "cpu_paged_reference.h"
#include "../common/perf_process.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/batch_execution.h"
#include <nanobench.h>
#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {
#ifndef HYLE_RESIDENT_PLACEMENT
#define HYLE_RESIDENT_PLACEMENT 0
#endif
namespace c=hyle::compute;namespace nb=ankerl::nanobench;
[[noreturn]] void fail(hyle::Status status){std::fprintf(stderr,"CPU paged: %.*s\n",int(status.message().size()),status.message().data());std::exit(2);}
void check(hyle::Status status){if(!status.ok())fail(status);}
template<class T>T take(hyle::Result<T> value){if(!value)fail(value.error());return std::move(*value);}
bool set(const char *name){const auto *p=std::getenv(name);return p&&*p;}
float decode(std::uint16_t x){return std::bit_cast<float>(std::uint32_t(x)<<16);}
std::uint16_t encode(float x){return std::uint16_t(std::bit_cast<std::uint32_t>(x)>>16);}
struct Data {
  c::BatchCapacity capacity;
  std::array<std::int32_t,3> qp,kp;
  std::vector<std::int32_t> ids;std::array<std::int32_t,2> tails;
  std::array<std::vector<std::uint16_t>,3> values;
  float scale;bool causal;
  Data(int q1,int q2,int k1,int k2,int depth,bool masked):
    capacity{.storage=c::BatchStorage::kPaged,.data_type=c::DataType::kBFloat16,.max_requests=2,
      .max_query_rows=unsigned(q1+q2+3),.max_kv_rows=unsigned(k1+k2+13),.max_pages=unsigned((k1+15)/16+(k2+15)/16+2),.max_page_refs=unsigned((k1+15)/16+(k2+15)/16),
      .query_heads=8,.kv_heads=2,.head_dim=unsigned(depth),.page_size=16},
    qp{0,q1,q1+q2},kp{0,(k1+15)/16,(k1+15)/16+(k2+15)/16},scale(1.F/std::sqrt(float(depth))),causal(masked){
    tails={(k1-1)%16+1,(k2-1)%16+1};
    for(unsigned i=0;i<capacity.max_page_refs;++i)ids.push_back(static_cast<std::int32_t>(capacity.max_pages-1-i));
    ids[static_cast<unsigned>(kp[1])]=ids[0];
    values[0].resize(std::size_t(capacity.max_query_rows)*8*depth);
    for(int port=1;port<3;++port)values[port].resize(std::size_t(capacity.max_pages)*capacity.page_size*2*depth);
    for(int port=0;port<3;++port)for(std::size_t i=0;i<values[port].size();++i)
      values[port][i]=encode(float(int((i*13+std::size_t(port)*17)%31)-15)*.015625F);
    for(int port=1;port<3;++port)for(unsigned page=0;page<capacity.max_pages;++page){
      unsigned valid=0;for(unsigned request=0;request<2;++request)for(auto ref=kp[request];ref<kp[request+1];++ref)
        if(ids[static_cast<unsigned>(ref)]==static_cast<std::int32_t>(page))valid=std::max(valid,ref+1==kp[request+1]?static_cast<unsigned>(tails[request]):capacity.page_size);
      std::fill(values[port].begin()+std::size_t(page*16+valid)*2*depth,values[port].begin()+std::size_t((page+1)*16)*2*depth,encode(128.F));
    }
  }
  std::vector<double> oracle()const{
    const int depth=int(capacity.head_dim);std::vector<double> output(std::size_t(qp[2])*8*depth),weights;
    for(unsigned r=0;r<2;++r){const int nq=qp[r+1]-qp[r],nk=(kp[r+1]-kp[r]-1)*16+tails[r];weights.resize(nk);
      for(int row=0;row<nq;++row)for(int h=0;h<8;++h){double sum=0;
        for(int j=0;j<nk;++j){double dot=0;for(int d=0;d<depth;++d)dot+=double(decode(values[0][((qp[r]+row)*8+h)*depth+d]))*decode(values[1][((ids[static_cast<unsigned>(kp[r]+j/16)]*16+j%16)*2+h/4)*depth+d]);
          weights[j]=causal&&j>nk-nq+row?0.:std::exp(dot*scale);sum+=weights[j];}
        for(int d=0;d<depth;++d)for(int j=0;j<nk;++j)output[((qp[r]+row)*8+h)*depth+d]+=weights[j]/sum*decode(values[2][((ids[static_cast<unsigned>(kp[r]+j/16)]*16+j%16)*2+h/4)*depth+d]);
      }}return output;
  }
};
struct Harness {
  std::shared_ptr<const hyle::bench::resident::CpuPagedReferenceData> reference;
  c::Context context;c::Queue queue;c::BatchSchema schema;c::BatchLayoutVersion layout;
  c::BatchProgram program;std::array<c::BatchInvocation,2> slots;c::LeasePool pool;
  c::PagedArena keys,values;c::PagedArenaRead key_frame,value_frame;
  std::vector<c::ReadLease> key_pages,value_pages;std::vector<unsigned> physical_ids;c::ReadLease query;std::size_t next=0;
  Harness(const Data &data,bool original){
    schema=take(c::BatchSchema::create(data.capacity));layout=take(schema.version({data.qp,data.kp,data.ids,data.tails}));
    const auto page_bytes=std::size_t(16)*2*data.capacity.head_dim*2;
    keys=take(c::PagedArena::create(schema,hyle::mem::system_resource(),{.max_backing_bytes=page_bytes*data.capacity.max_pages+1}));
    values=take(c::PagedArena::create(schema,hyle::mem::system_resource(),{.max_backing_bytes=page_bytes*data.capacity.max_pages+1}));
    for(unsigned page=0;page<data.capacity.max_pages;++page)if(std::find(data.ids.begin(),data.ids.end(),static_cast<std::int32_t>(page))!=data.ids.end()){
      auto kw=take(keys.acquire(page)),vw=take(values.acquire(page));
      check(kw.lease().view()->write(std::as_bytes(std::span(data.values[1])).subspan(page*page_bytes,page_bytes)));
      check(vw.lease().view()->write(std::as_bytes(std::span(data.values[2])).subspan(page*page_bytes,page_bytes)));
      physical_ids.push_back(page);key_pages.push_back(take(kw.publish()));value_pages.push_back(take(vw.publish()));
    }
    key_frame=take(keys.read(layout,key_pages));value_frame=take(values.read(layout,value_pages));
    if(original)reference=take(hyle::bench::resident::cpu_paged_reference_data(layout,key_frame,value_frame));
    context=original?take(c::Context::create(c::cpu::device(),hyle::bench::resident::cpu_paged_reference_provider(reference.get()))):take(c::cpu::create_context({.native_threads=1}));
    queue=take(context.create_queue());
    program=take(c::BatchProgram::prepare(context,schema,{.resource=hyle::mem::system_resource(),.max_float_workspace_bytes=32u<<20,.max_int_workspace_bytes=4096,.scale=data.scale,.causal=data.causal}));
    for(auto &slot:slots){slot=take(c::BatchInvocation::create(program,queue));auto prepared=take(slot.prepare_batch(layout));}
    pool=take(c::LeasePool::create(hyle::mem::system_resource(),data.values[0].size()*2+2048));
    auto writer=take(pool.allocate(data.values[0].size()*2));check(writer.view()->write(std::as_bytes(std::span(data.values[0]))));query=take(writer.publish());
    std::fprintf(stderr,"# CPU_PAGED_COLD original=%d known_native_bytes=%zu private_pool_upper=UNKNOWN requests=2 immutable_history=1 native_threads=1\n",original,slots[0].report().float_workspace_bytes);
  }
  ~Harness(){for(auto &slot:slots)check(slot.close());check(context.close());}
  c::ReadLease execute(){auto &slot=slots[next++%slots.size()];auto prepared=take(slot.update_batch(layout));if(!slot.report().reused_native_plan)fail(hyle::Status::Internal("paged warm topology rebuilt"));
    auto completion=take(prepared.run(query,key_frame,value_frame));check(completion.wait());return take(slot.output());}
  void verify(const Data &data,const std::vector<double> &expected){auto output=execute();std::vector<std::uint16_t> actual(data.values[0].size());check(output.view()->read(std::as_writable_bytes(std::span(actual))));
    double worst=0;for(std::size_t i=0;i<expected.size();++i){if(!std::isfinite(decode(actual[i])))fail(hyle::Status::Internal("paged nonfinite output"));worst=std::max(worst,std::abs(double(decode(actual[i]))-expected[i]));}
    if(worst>8e-3)fail(hyle::Status::Internal("paged independent double oracle exceeded 8e-3"));
    for(unsigned port=1;port<3;++port)for(unsigned index=0;index<key_pages.size();++index){const auto &lease=port==1?key_pages[index]:value_pages[index];auto view=take(lease.view());
      std::vector<std::uint16_t> after(view.size()/2);check(view.read(std::as_writable_bytes(std::span(after))));
      const auto first=data.values[port].begin()+std::size_t(physical_ids[index])*16*2*data.capacity.head_dim;
      if(!std::equal(after.begin(),after.end(),first))fail(hyle::Status::Internal("paged readonly KV changed"));
    }
    std::fprintf(stderr,"# CPU_PAGED_CORRECTNESS max_abs_error=%.9g elements=%zu\n",worst,expected.size());
  }
};
void measure(int q1,int q2,int k1,int k2,int depth,bool causal){Data data(q1,q2,k1,k2,depth,causal);std::unique_ptr<Harness> candidate,peer;
  const auto *aa=std::getenv("HYLE_BENCH_AA_ROUTE");const std::string route=aa?aa:"";if(!route.empty()&&route!="native"&&route!="selected")fail(hyle::Status::InvalidArgument("ragged AA route"));
  const auto make_candidate=[&]{candidate=std::make_unique<Harness>(data,route=="native");};const auto make_peer=[&]{peer=std::make_unique<Harness>(data,route!="selected");};
  if(set("HYLE_BENCH_PEER_ALLOCATION_FIRST")){make_peer();make_candidate();}else{make_candidate();make_peer();}
  const auto expected=data.oracle();candidate->verify(data,expected);peer->verify(data,expected);
  for(int i=0;i<4;++i){auto a=candidate->execute();auto b=peer->execute();nb::doNotOptimizeAway(a);nb::doNotOptimizeAway(b);}
  if(set("HYLE_BENCH_VERIFY_ONLY"))return;
  nb::Bench bench;bench.title("CPU paged BF16 Q="+std::to_string(q1)+","+std::to_string(q2)+" KV="+std::to_string(k1)+","+std::to_string(k2)+" D="+std::to_string(depth)+" causal="+std::to_string(causal)).batch(1).epochs(16).minEpochIterations(1).minEpochTime(std::chrono::milliseconds(4)).performanceCounters(false);
  const auto selected=[&]{auto result=candidate->execute();nb::doNotOptimizeAway(result);};const auto original=[&]{auto result=peer->execute();nb::doNotOptimizeAway(result);};
  if(set("HYLE_BENCH_RIVAL_FIRST"))(void)bench.compare("Original CPU contiguous immutable lifecycle",original,"HYLE CPU paged",selected);
  else (void)bench.compare("HYLE CPU paged",selected,"Original CPU contiguous immutable lifecycle",original);
}
}
int main(int,char **argv){const auto qos=hyle::bench::perf::configure_thread_qos();const auto affinity=hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
  const bool verify_only=set("HYLE_BENCH_VERIFY_ONLY");
  if(!verify_only&&!set("HYLE_BENCH_AA_ROUTE")&&set("HYLE_PERF_RUN_NONCE")){
    std::string ready;std::ifstream(std::filesystem::absolute(argv[0]).parent_path()/"g01_cpu_paged_aa.ready")>>ready;
    if(ready!="PASS")fail(hyle::Status::FailedPrecondition("ragged finite workload AA not passed for this build"));
  }
  std::string parent_mask="not-applicable",priority="not-applicable";
#if defined(_WIN32)
  DWORD_PTR mask=0,system_mask=0;if(!GetProcessAffinityMask(GetCurrentProcess(),&mask,&system_mask))fail(hyle::Status::Internal("ragged process affinity observation"));
  char text[40]{};std::snprintf(text,sizeof(text),"0x%llx",static_cast<unsigned long long>(mask));parent_mask=text;
  const auto actual_priority=GetPriorityClass(GetCurrentProcess());if(!actual_priority)fail(hyle::Status::Internal("ragged process priority observation"));priority=std::to_string(actual_priority);
#endif
  const auto *aa=std::getenv("HYLE_BENCH_AA_ROUTE");
  std::fprintf(stderr,"# HYLE_CPU_PAGED_RUN run_nonce=%s apple_qos=%s qos_ok=%s affinity_topology=%s caller_cpu=%s affinity_ok=%s verify_only=%d parent_mask=%s priority=%s placement=%d execution_order=%s allocation_order=%s aa_route=%s paired_mode=nanobench-compare paired_rounds=16\n",hyle::bench::perf::run_nonce(),qos.name,qos.ok,affinity.topology,affinity.cpu,affinity.ok,verify_only,parent_mask.c_str(),priority.c_str(),HYLE_RESIDENT_PLACEMENT,set("HYLE_BENCH_RIVAL_FIRST")?"peer-first":"candidate-first",set("HYLE_BENCH_PEER_ALLOCATION_FIRST")?"peer-first":"candidate-first",aa&&*aa?aa:"none");
  measure(1,17,1024,768,256,true);measure(64,128,1024,768,256,true);measure(1,17,31,47,64,false);
}
