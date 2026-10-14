#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "cpu_ragged_reference.h"
#include "../common/perf_process.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/batch_execution.h"
#include <nanobench.h>
#include <array>
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
[[noreturn]] void fail(hyle::Status status){std::fprintf(stderr,"CPU ragged: %.*s\n",int(status.message().size()),status.message().data());std::exit(2);}
void check(hyle::Status status){if(!status.ok())fail(status);}
template<class T>T take(hyle::Result<T> value){if(!value)fail(value.error());return std::move(*value);}
bool set(const char *name){const auto *p=std::getenv(name);return p&&*p;}
float decode(std::uint16_t x){return std::bit_cast<float>(std::uint32_t(x)<<16);}
std::uint16_t encode(float x){return std::uint16_t(std::bit_cast<std::uint32_t>(x)>>16);}
struct Data {
  c::BatchCapacity capacity;
  std::array<std::int32_t,3> qp,kp;
  std::array<std::vector<std::uint16_t>,3> values;
  float scale;bool causal;
  Data(int q1,int q2,int k1,int k2,int depth,bool masked):
    capacity{.storage=c::BatchStorage::kRagged,.query_type=c::DataType::kBFloat16,.output_type=c::DataType::kBFloat16,.key_storage=c::CacheStorage::plain(c::DataType::kBFloat16).value(),.value_storage=c::CacheStorage::plain(c::DataType::kBFloat16).value(),.max_requests=2,
      .max_query_rows=unsigned(q1+q2+3),.max_kv_rows=unsigned(k1+k2+13),.max_pages=0,.max_page_refs=0,
      .query_heads=8,.kv_heads=2,.qk_head_dim=unsigned(depth),.value_head_dim=unsigned(depth),.page_size=1},
    qp{0,q1,q1+q2},kp{0,k1,k1+k2},scale(1.F/std::sqrt(float(depth))),causal(masked){
    values[0].resize(std::size_t(capacity.max_query_rows)*8*depth);
    for(int port=1;port<3;++port)values[port].resize(std::size_t(capacity.max_kv_rows)*2*depth);
    for(int port=0;port<3;++port)for(std::size_t i=0;i<values[port].size();++i)
      values[port][i]=encode(float(int((i*13+std::size_t(port)*17)%31)-15)*.015625F);
    for(int port=1;port<3;++port)std::fill(values[port].begin()+std::size_t(kp[2])*2*depth,values[port].end(),encode(128.F));
  }
  std::vector<double> oracle()const{
    const int depth=int(capacity.qk_head_dim);std::vector<double> output(std::size_t(qp[2])*8*depth),weights;
    for(unsigned r=0;r<2;++r){const int nq=qp[r+1]-qp[r],nk=kp[r+1]-kp[r];weights.resize(nk);
      for(int row=0;row<nq;++row)for(int h=0;h<8;++h){double sum=0;
        for(int j=0;j<nk;++j){double dot=0;for(int d=0;d<depth;++d)dot+=double(decode(values[0][((qp[r]+row)*8+h)*depth+d]))*decode(values[1][((kp[r]+j)*2+h/4)*depth+d]);
          weights[j]=causal&&j>nk-nq+row?0.:std::exp(dot*scale);sum+=weights[j];}
        for(int d=0;d<depth;++d)for(int j=0;j<nk;++j)output[((qp[r]+row)*8+h)*depth+d]+=weights[j]/sum*decode(values[2][((kp[r]+j)*2+h/4)*depth+d]);
      }}return output;
  }
};
struct Harness {
  c::Context context;c::Queue queue;c::BatchSchema schema;c::BatchLayoutVersion layout;
  c::BatchProgram program;std::array<c::BatchInvocation,2> slots;c::LeasePool pool;
  std::array<c::ReadLease,3> inputs;std::size_t next=0;
  Harness(const Data &data,bool original){
    context=original?take(c::Context::create(c::cpu::device(),hyle::bench::resident::cpu_ragged_reference_provider())):take(c::cpu::create_context({.native_threads=1}));
    queue=take(context.create_queue());schema=take(c::BatchSchema::create(data.capacity));layout=take(schema.version({data.qp,data.kp,{},{}}));
    program=take(c::BatchProgram::prepare(context,schema,{.resource=hyle::mem::system_resource(),.max_float_workspace_bytes=16u<<20,.max_int_workspace_bytes=4096,.scale=data.scale,.causal=data.causal}));
    for(auto &slot:slots){slot=take(c::BatchInvocation::create(program,queue));auto prepared=take(slot.prepare_batch(layout));}
    std::size_t bytes=2048;for(const auto &values:data.values)bytes+=values.size()*2;pool=take(c::LeasePool::create(hyle::mem::system_resource(),bytes));
    for(unsigned port=0;port<3;++port){auto writer=take(pool.allocate(data.values[port].size()*2));check(writer.view()->write(std::as_bytes(std::span(data.values[port]))));inputs[port]=take(writer.publish());}
    std::fprintf(stderr,"# CPU_RAGGED_COLD original=%d known_native_bytes=%zu private_pool_upper=UNKNOWN requests=2\n",original,slots[0].report().float_workspace_bytes);
  }
  ~Harness(){for(auto &slot:slots)check(slot.close());check(context.close());}
  c::ReadLease execute(){auto &slot=slots[next++%slots.size()];auto prepared=take(slot.update_batch(layout));if(!slot.report().reused_native_plan)fail(hyle::Status::Internal("ragged warm topology rebuilt"));
    auto completion=take(prepared.run(inputs));check(completion.wait());return take(slot.output());}
  void verify(const Data &data,const std::vector<double> &expected){auto output=execute();std::vector<std::uint16_t> actual(data.values[0].size());check(output.view()->read(std::as_writable_bytes(std::span(actual))));
    double worst=0;for(std::size_t i=0;i<expected.size();++i){if(!std::isfinite(decode(actual[i])))fail(hyle::Status::Internal("ragged nonfinite output"));worst=std::max(worst,std::abs(double(decode(actual[i]))-expected[i]));}
    if(worst>8e-3)fail(hyle::Status::Internal("ragged independent double oracle exceeded 8e-3"));
    for(unsigned port=1;port<3;++port){std::vector<std::uint16_t> after(data.values[port].size());check(inputs[port].view()->read(std::as_writable_bytes(std::span(after))));if(after!=data.values[port])fail(hyle::Status::Internal("ragged readonly KV changed"));}
    std::fprintf(stderr,"# CPU_RAGGED_CORRECTNESS max_abs_error=%.9g elements=%zu\n",worst,expected.size());
  }
};
void measure(int q1,int q2,int k1,int k2,int depth,bool causal){Data data(q1,q2,k1,k2,depth,causal);std::unique_ptr<Harness> candidate,peer;
  const auto *aa=std::getenv("HYLE_BENCH_AA_ROUTE");const std::string route=aa?aa:"";if(!route.empty()&&route!="native"&&route!="selected")fail(hyle::Status::InvalidArgument("ragged AA route"));
  const auto make_candidate=[&]{candidate=std::make_unique<Harness>(data,route=="native");};const auto make_peer=[&]{peer=std::make_unique<Harness>(data,route!="selected");};
  if(set("HYLE_BENCH_PEER_ALLOCATION_FIRST")){make_peer();make_candidate();}else{make_candidate();make_peer();}
  const auto expected=data.oracle();candidate->verify(data,expected);peer->verify(data,expected);
  for(int i=0;i<4;++i){auto a=candidate->execute();auto b=peer->execute();nb::doNotOptimizeAway(a);nb::doNotOptimizeAway(b);}
  if(set("HYLE_BENCH_VERIFY_ONLY"))return;
  nb::Bench bench;bench.title("CPU ragged BF16 Q="+std::to_string(q1)+","+std::to_string(q2)+" KV="+std::to_string(k1)+","+std::to_string(k2)+" D="+std::to_string(depth)+" causal="+std::to_string(causal)).batch(1).epochs(16).minEpochIterations(1).minEpochTime(std::chrono::milliseconds(4)).performanceCounters(false);
  const auto selected=[&]{auto result=candidate->execute();nb::doNotOptimizeAway(result);};const auto original=[&]{auto result=peer->execute();nb::doNotOptimizeAway(result);};
  if(set("HYLE_BENCH_RIVAL_FIRST"))(void)bench.compare("Original CPU ragged lifecycle",original,"HYLE CPU ragged",selected);
  else (void)bench.compare("HYLE CPU ragged",selected,"Original CPU ragged lifecycle",original);
}
}
int main(int,char **argv){const auto qos=hyle::bench::perf::configure_thread_qos();const auto affinity=hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
  const bool verify_only=set("HYLE_BENCH_VERIFY_ONLY");
  if(!verify_only&&!set("HYLE_BENCH_AA_ROUTE")&&set("HYLE_PERF_RUN_NONCE")){
    std::string ready;std::ifstream(std::filesystem::absolute(argv[0]).parent_path()/"g01_cpu_ragged_aa.ready")>>ready;
    if(ready!="PASS")fail(hyle::Status::FailedPrecondition("ragged finite workload AA not passed for this build"));
  }
  std::string parent_mask="not-applicable",priority="not-applicable";
#if defined(_WIN32)
  DWORD_PTR mask=0,system_mask=0;if(!GetProcessAffinityMask(GetCurrentProcess(),&mask,&system_mask))fail(hyle::Status::Internal("ragged process affinity observation"));
  char text[40]{};std::snprintf(text,sizeof(text),"0x%llx",static_cast<unsigned long long>(mask));parent_mask=text;
  const auto actual_priority=GetPriorityClass(GetCurrentProcess());if(!actual_priority)fail(hyle::Status::Internal("ragged process priority observation"));priority=std::to_string(actual_priority);
#endif
  const auto *aa=std::getenv("HYLE_BENCH_AA_ROUTE");
  std::fprintf(stderr,"# HYLE_CPU_RAGGED_RUN run_nonce=%s apple_qos=%s qos_ok=%s affinity_topology=%s caller_cpu=%s affinity_ok=%s verify_only=%d parent_mask=%s priority=%s placement=%d execution_order=%s allocation_order=%s aa_route=%s paired_mode=nanobench-compare paired_rounds=16\n",hyle::bench::perf::run_nonce(),qos.name,qos.ok,affinity.topology,affinity.cpu,affinity.ok,verify_only,parent_mask.c_str(),priority.c_str(),HYLE_RESIDENT_PLACEMENT,set("HYLE_BENCH_RIVAL_FIRST")?"peer-first":"candidate-first",set("HYLE_BENCH_PEER_ALLOCATION_FIRST")?"peer-first":"candidate-first",aa&&*aa?aa:"none");
  measure(1,17,1024,768,256,true);measure(64,128,1024,768,256,true);measure(1,17,31,47,64,false);
}
