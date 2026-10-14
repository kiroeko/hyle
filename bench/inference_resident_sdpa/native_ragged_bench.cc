#if defined(HYLE_RAGGED_APPLE_BENCH)
#include "apple_ragged_reference.h"
#include "hyle/compute/apple_memory.h"
#else
#include "native_ragged_reference.h"
#endif
#include "../common/perf_process.h"
#include <nanobench.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {
#ifndef HYLE_RESIDENT_PLACEMENT
#define HYLE_RESIDENT_PLACEMENT 0
#endif
namespace c=hyle::compute;namespace nb=ankerl::nanobench;
#if defined(HYLE_RAGGED_APPLE_BENCH)
namespace v=c::apple;
constexpr const char *backend="Apple";
#else
namespace v=c::vulkan;
constexpr const char *backend="Vulkan";
#endif
void check(hyle::Status status){if(!status.ok()){std::fprintf(stderr,"native ragged: %.*s\n",int(status.message().size()),status.message().data());std::exit(2);}}
template<class T>T take(hyle::Result<T> value){if(!value)check(value.error());return std::move(*value);}
bool set(const char *key){const auto *p=std::getenv(key);return p&&*p;}
std::uint16_t encode(float x){if(x==0)return 0;const auto bits=std::bit_cast<std::uint32_t>(x);return std::uint16_t(((bits>>16)&0x8000u)|(((bits>>23&255u)-112u)<<10)|((bits&0x7fffffu)>>13));}
double decode(std::uint16_t bits){const auto exponent=bits>>10&31u,fraction=bits&1023u;const auto sign=bits&0x8000u?-1.:1.;return sign*(exponent?std::ldexp(1.+double(fraction)/1024.,int(exponent)-15):std::ldexp(double(fraction),-24));}
struct Data{
  c::BatchCapacity cap;std::array<std::int32_t,3> qp,kp;std::array<std::vector<std::uint16_t>,3> values;float scale;bool causal;
  Data(int q1,int q2,int k1,int k2,int depth,bool masked):cap{.storage=c::BatchStorage::kRagged,.max_requests=2,.max_query_rows=unsigned(q1+q2+3),.max_kv_rows=unsigned(k1+k2+13),.max_pages=0,.max_page_refs=0,.query_heads=8,.kv_heads=2,.qk_head_dim=unsigned(depth),.value_head_dim=unsigned(depth),.page_size=1},qp{0,q1,q1+q2},kp{0,k1,k1+k2},scale(1.F/std::sqrt(float(depth))),causal(masked){
    values[0].resize(std::size_t(cap.max_query_rows)*8*depth);for(int port=1;port<3;++port)values[port].resize(std::size_t(cap.max_kv_rows)*2*depth);
    for(int port=0;port<3;++port)for(std::size_t i=0;i<values[port].size();++i)values[port][i]=encode(float(int((i*13+std::size_t(port)*17)%31)-15)*.015625F);
    for(int port=1;port<3;++port)std::fill(values[port].begin()+std::size_t(kp[2])*2*depth,values[port].end(),encode(128.F));
  }
  std::vector<double> oracle()const{std::vector<double> result(std::size_t(qp[2])*8*cap.qk_head_dim),weights;const auto depth=cap.qk_head_dim;
    for(unsigned r=0;r<2;++r){const auto nq=qp[r+1]-qp[r],nk=kp[r+1]-kp[r];weights.resize(nk);
      for(int row=0;row<nq;++row)for(int head=0;head<8;++head){double sum=0;
        for(int j=0;j<nk;++j){double dot=0;for(unsigned d=0;d<depth;++d)dot+=decode(values[0][((qp[r]+row)*8+head)*depth+d])*decode(values[1][((kp[r]+j)*2+head/4)*depth+d]);weights[j]=causal&&j>nk-nq+row?0.:std::exp(dot*scale);sum+=weights[j];}
        for(unsigned d=0;d<depth;++d)for(int j=0;j<nk;++j)result[((qp[r]+row)*8+head)*depth+d]+=weights[j]/sum*decode(values[2][((kp[r]+j)*2+head/4)*depth+d]);
      }}return result;
  }
};
struct Harness{
  v::Context context;v::Program program;std::array<v::Invocation,2> slots;std::size_t next=0;
#if defined(HYLE_RAGGED_APPLE_BENCH)
  v::MemoryResource memory;c::LeasePool pool;std::array<std::optional<mlx::core::array>,3> inputs;
#else
  std::array<v::Tensor,3> inputs;
#endif
  Harness(const Data&data,bool original)
#if defined(HYLE_RAGGED_APPLE_BENCH)
    :context(take(v::Context::create())),memory(take(v::MemoryResource::create(context)))
#endif
  {
#if defined(HYLE_RAGGED_APPLE_BENCH)
    // Construct the original opaque resource directly; no inline default
    // construction of its incomplete private unique_ptr payload is required.
    auto environment=context.environment();if(!environment)check(environment.error());const auto &facts=*environment;
    std::fprintf(stderr,"# NATIVE_RAGGED_ENV backend=Apple device=%s architecture=%s runtime=%s mlx=%s revision=%s\n",facts.device_name.c_str(),facts.architecture.c_str(),facts.os_runtime.c_str(),facts.mlx_version.c_str(),facts.mlx_revision.c_str());
#else
    v::ContextOptions options;
#if defined(HYLE_RESIDENT_VULKAN_LOADER)
    options.loader_path=HYLE_RESIDENT_VULKAN_LOADER;
#endif
    context=take(v::Context::create(options));if(context.environment().software_device)check(hyle::Status::Unimplemented("native ragged requires actual GPU"));
    const auto &facts=context.environment();std::fprintf(stderr,"# NATIVE_RAGGED_ENV backend=Vulkan device=%s driver=%s version=%u ggml=%s\n",facts.device_name.c_str(),facts.driver_name.c_str(),facts.driver_version,facts.ggml_revision.c_str());
#endif
    auto schema=take(c::BatchSchema::create(data.cap));auto layout=take(schema.version({data.qp,data.kp,{},{}}));const c::RaggedAttention attention{data.cap,data.scale,data.causal};
    program=original?hyle::bench::resident::ragged_reference_program(context,layout,attention):take(v::prepare_ragged_batch(context,layout,attention));
    const auto qd=take(c::TensorDesc::dense(data.cap.query_type,std::array<std::int64_t,3>{data.cap.max_query_rows,8,data.cap.qk_head_dim}));
    const auto kd=take(c::TensorDesc::dense(data.cap.query_type,std::array<std::int64_t,3>{data.cap.max_kv_rows,2,data.cap.qk_head_dim}));
#if defined(HYLE_RAGGED_APPLE_BENCH)
    std::size_t bytes=2048;for(const auto &value:data.values)bytes+=value.size()*2;pool=take(c::LeasePool::create(memory.ref(),bytes));
    for(unsigned port=0;port<3;++port){auto writer=take(pool.allocate(data.values[port].size()*2));check(writer.view()->write(std::as_bytes(std::span(data.values[port]))));inputs[port]=take(memory.binding(take(writer.publish()),port?kd:qd));}
#else
    for(unsigned port=0;port<3;++port)inputs[port]=take(context.upload(port?kd:qd,std::as_bytes(std::span(data.values[port]))));
#endif
    for(auto &slot:slots)slot=take(program.create_invocation());
    std::fprintf(stderr,"# NATIVE_RAGGED_COLD backend=%s original=%d requests=2 private_pool_upper=UNKNOWN\n",backend,original);
  }
  ~Harness(){
#if defined(HYLE_RAGGED_APPLE_BENCH)
    for(auto &slot:slots)slot={};for(auto &input:inputs)input.reset();program={};
#endif
    check(context.close());
  }
  void execute(std::vector<std::uint16_t>*host=nullptr){auto &slot=slots[next++%slots.size()];
#if defined(HYLE_RAGGED_APPLE_BENCH)
    const std::array bound{*inputs[0],*inputs[1],*inputs[2]};check(slot.bind(bound));
#else
    check(slot.bind(inputs));
#endif
    auto submitted=take(slot.submit());auto result=take(submitted.wait());
#if defined(HYLE_RAGGED_APPLE_BENCH)
    const auto output=result->arrays()[0];if(host){const auto desc=take(c::TensorDesc::dense(c::DataType::kFloat16,std::array<std::int64_t,3>{output.shape(0),8,output.shape(2)}));auto lease=take(memory.import_readonly(output,desc));check(lease.view()->read(std::as_writable_bytes(std::span(*host))));}
#else
    const auto output=result->tensors()[0];if(host)check(context.download(output,std::as_writable_bytes(std::span(*host))));
#endif
    nb::doNotOptimizeAway(output);
  }
  void verify(const Data&data,const std::vector<double>&expected){std::vector<std::uint16_t> actual(expected.size());execute(&actual);double worst=0;
    for(std::size_t i=0;i<actual.size();++i){if((actual[i]&0x7c00u)==0x7c00u)check(hyle::Status::Internal("native ragged nonfinite output"));worst=std::max(worst,std::abs(decode(actual[i])-expected[i]));}
    if(worst>2e-3)check(hyle::Status::Internal("native ragged original F16 oracle bound exceeded"));
    for(unsigned port=1;port<3;++port){std::vector<std::uint16_t> after(data.values[port].size());
#if defined(HYLE_RAGGED_APPLE_BENCH)
      const auto desc=take(c::TensorDesc::dense(c::DataType::kFloat16,std::array<std::int64_t,3>{data.cap.max_kv_rows,2,data.cap.qk_head_dim}));auto lease=take(memory.import_readonly(*inputs[port],desc));check(lease.view()->read(std::as_writable_bytes(std::span(after))));
#else
      check(context.download(inputs[port],std::as_writable_bytes(std::span(after))));
#endif
      if(after!=data.values[port])check(hyle::Status::Internal("native ragged readonly KV changed"));}
    std::fprintf(stderr,"# NATIVE_RAGGED_CORRECTNESS max_abs_error=%.9g elements=%zu\n",worst,expected.size());
  }
};
void measure(int q1,int q2,int k1,int k2,int depth,bool causal){Data data(q1,q2,k1,k2,depth,causal);std::unique_ptr<Harness> candidate,peer;
  const auto *value=std::getenv("HYLE_BENCH_AA_ROUTE");const std::string route=value?value:"";if(!route.empty()&&route!="native"&&route!="selected")check(hyle::Status::InvalidArgument("native ragged AA route"));
  const auto make_candidate=[&]{candidate=std::make_unique<Harness>(data,route=="native");};const auto make_peer=[&]{peer=std::make_unique<Harness>(data,route!="selected");};
  if(set("HYLE_BENCH_PEER_ALLOCATION_FIRST")){make_peer();make_candidate();}else{make_candidate();make_peer();}
  const auto expected=data.oracle();candidate->verify(data,expected);peer->verify(data,expected);
  for(int i=0;i<4;++i){candidate->execute();peer->execute();}
  if(set("HYLE_BENCH_VERIFY_ONLY"))return;
  nb::Bench bench;bench.title(std::string(backend)+" ragged F16 Q="+std::to_string(q1)+","+std::to_string(q2)+" KV="+std::to_string(k1)+","+std::to_string(k2)+" D="+std::to_string(depth)+" causal="+std::to_string(causal)).batch(1).epochs(16).minEpochIterations(1).minEpochTime(std::chrono::milliseconds(4)).performanceCounters(false);
  const auto selected=[&]{candidate->execute();};const auto original=[&]{peer->execute();};
  const auto candidate_name=std::string("HYLE ")+backend+" ragged",peer_name=std::string("Original ")+backend+" ragged lifecycle";
  if(set("HYLE_BENCH_RIVAL_FIRST"))(void)bench.compare(peer_name,original,candidate_name,selected);
  else (void)bench.compare(candidate_name,selected,peer_name,original);
}
}
int main(int,char **argv){const auto qos=hyle::bench::perf::configure_thread_qos();
  if(!set("HYLE_BENCH_VERIFY_ONLY")&&!set("HYLE_BENCH_AA_ROUTE")&&set("HYLE_PERF_RUN_NONCE")){
    std::string ready;std::ifstream(std::filesystem::absolute(argv[0]).parent_path()/"g01_native_ragged_aa.ready")>>ready;
    if(ready!="PASS")check(hyle::Status::FailedPrecondition("native ragged A/A has not passed for this build"));
  }
  const auto *aa=std::getenv("HYLE_BENCH_AA_ROUTE");
  std::fprintf(stderr,"# HYLE_NATIVE_RAGGED_RUN run_nonce=%s apple_qos=%s qos_ok=%s backend=%s verify_only=%d paired_rounds=16 placement=%d execution_order=%s allocation_order=%s aa_route=%s\n",hyle::bench::perf::run_nonce(),qos.name,qos.ok,backend,set("HYLE_BENCH_VERIFY_ONLY"),HYLE_RESIDENT_PLACEMENT,set("HYLE_BENCH_RIVAL_FIRST")?"peer-first":"candidate-first",set("HYLE_BENCH_PEER_ALLOCATION_FIRST")?"peer-first":"candidate-first",aa&&*aa?aa:"none");
  measure(1,17,1024,768,256,true);measure(64,128,1024,768,256,true);measure(1,17,31,47,64,false);
}
