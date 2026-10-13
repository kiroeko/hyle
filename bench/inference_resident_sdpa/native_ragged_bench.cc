#include "native_ragged_reference.h"
#include "../common/perf_process.h"
#include <nanobench.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {
namespace c=hyle::compute;namespace v=c::vulkan;namespace nb=ankerl::nanobench;
void check(hyle::Status status){if(!status.ok()){std::fprintf(stderr,"native ragged: %.*s\n",int(status.message().size()),status.message().data());std::exit(2);}}
template<class T>T take(hyle::Result<T> value){if(!value)check(value.error());return std::move(*value);}
bool set(const char *key){const auto *p=std::getenv(key);return p&&*p;}
std::uint16_t encode(float x){if(x==0)return 0;const auto bits=std::bit_cast<std::uint32_t>(x);return std::uint16_t(((bits>>16)&0x8000u)|(((bits>>23&255u)-112u)<<10)|((bits&0x7fffffu)>>13));}
double decode(std::uint16_t bits){const auto exponent=bits>>10&31u,fraction=bits&1023u;const auto sign=bits&0x8000u?-1.:1.;return sign*(exponent?std::ldexp(1.+double(fraction)/1024.,int(exponent)-15):std::ldexp(double(fraction),-24));}
struct Data{
  c::BatchCapacity cap;std::array<std::int32_t,3> qp,kp;std::array<std::vector<std::uint16_t>,3> values;float scale;bool causal;
  Data(int q1,int q2,int k1,int k2,int depth,bool masked):cap{.storage=c::BatchStorage::kRagged,.max_requests=2,.max_query_rows=unsigned(q1+q2+3),.max_kv_rows=unsigned(k1+k2+13),.max_pages=0,.max_page_refs=0,.query_heads=8,.kv_heads=2,.head_dim=unsigned(depth),.page_size=1},qp{0,q1,q1+q2},kp{0,k1,k1+k2},scale(1.F/std::sqrt(float(depth))),causal(masked){
    values[0].resize(std::size_t(cap.max_query_rows)*8*depth);for(int port=1;port<3;++port)values[port].resize(std::size_t(cap.max_kv_rows)*2*depth);
    for(int port=0;port<3;++port)for(std::size_t i=0;i<values[port].size();++i)values[port][i]=encode(float(int((i*13+std::size_t(port)*17)%31)-15)*.015625F);
    for(int port=1;port<3;++port)std::fill(values[port].begin()+std::size_t(kp[2])*2*depth,values[port].end(),encode(128.F));
  }
  std::vector<double> oracle()const{std::vector<double> result(std::size_t(qp[2])*8*cap.head_dim),weights;const auto depth=cap.head_dim;
    for(unsigned r=0;r<2;++r){const auto nq=qp[r+1]-qp[r],nk=kp[r+1]-kp[r];weights.resize(nk);
      for(int row=0;row<nq;++row)for(int head=0;head<8;++head){double sum=0;
        for(int j=0;j<nk;++j){double dot=0;for(unsigned d=0;d<depth;++d)dot+=decode(values[0][((qp[r]+row)*8+head)*depth+d])*decode(values[1][((kp[r]+j)*2+head/4)*depth+d]);weights[j]=causal&&j>nk-nq+row?0.:std::exp(dot*scale);sum+=weights[j];}
        for(unsigned d=0;d<depth;++d)for(int j=0;j<nk;++j)result[((qp[r]+row)*8+head)*depth+d]+=weights[j]/sum*decode(values[2][((kp[r]+j)*2+head/4)*depth+d]);
      }}return result;
  }
};
struct Harness{
  v::Context context;v::Program program;std::array<v::Invocation,2> slots;std::array<v::Tensor,3> inputs;std::size_t next=0;
  Harness(const Data&data,bool original){
    v::ContextOptions options;
#if defined(HYLE_RESIDENT_VULKAN_LOADER)
    options.loader_path=HYLE_RESIDENT_VULKAN_LOADER;
#endif
    context=take(v::Context::create(options));if(context.environment().software_device)check(hyle::Status::Unimplemented("native ragged requires actual GPU"));
    auto schema=take(c::BatchSchema::create(data.cap));auto layout=take(schema.version({data.qp,data.kp,{},{}}));const c::RaggedAttention attention{data.cap,data.scale,data.causal};
    program=original?hyle::bench::resident::ragged_reference_program(context,layout,attention):take(v::prepare_ragged_batch(context,layout,attention));
    const auto qd=take(c::TensorDesc::dense(data.cap.data_type,std::array<std::int64_t,3>{data.cap.max_query_rows,8,data.cap.head_dim}));
    const auto kd=take(c::TensorDesc::dense(data.cap.data_type,std::array<std::int64_t,3>{data.cap.max_kv_rows,2,data.cap.head_dim}));
    for(unsigned port=0;port<3;++port)inputs[port]=take(context.upload(port?kd:qd,std::as_bytes(std::span(data.values[port]))));
    for(auto &slot:slots)slot=take(program.create_invocation());
    std::fprintf(stderr,"# NATIVE_RAGGED_COLD original=%d requests=2 private_pool_upper=UNKNOWN device=%s\n",original,context.environment().device_name.c_str());
  }
  ~Harness(){check(context.close());}
  void execute(std::vector<std::uint16_t>*host=nullptr){auto &slot=slots[next++%slots.size()];check(slot.bind(inputs));auto submitted=take(slot.submit());auto result=take(submitted.wait());const auto output=result->tensors()[0];if(host)check(context.download(output,std::as_writable_bytes(std::span(*host))));nb::doNotOptimizeAway(output);}
  void verify(const Data&data,const std::vector<double>&expected){std::vector<std::uint16_t> actual(expected.size());execute(&actual);double worst=0;
    for(std::size_t i=0;i<actual.size();++i){if((actual[i]&0x7c00u)==0x7c00u)check(hyle::Status::Internal("native ragged nonfinite output"));worst=std::max(worst,std::abs(decode(actual[i])-expected[i]));}
    if(worst>2e-3)check(hyle::Status::Internal("native ragged original F16 oracle bound exceeded"));
    for(unsigned port=1;port<3;++port){std::vector<std::uint16_t> after(data.values[port].size());check(context.download(inputs[port],std::as_writable_bytes(std::span(after))));if(after!=data.values[port])check(hyle::Status::Internal("native ragged readonly KV changed"));}
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
  nb::Bench bench;bench.title("Vulkan ragged F16 Q="+std::to_string(q1)+","+std::to_string(q2)+" KV="+std::to_string(k1)+","+std::to_string(k2)+" D="+std::to_string(depth)+" causal="+std::to_string(causal)).batch(1).epochs(16).minEpochIterations(1).minEpochTime(std::chrono::milliseconds(4)).performanceCounters(false);
  const auto selected=[&]{candidate->execute();};const auto original=[&]{peer->execute();};
  if(set("HYLE_BENCH_RIVAL_FIRST"))(void)bench.compare("Original Vulkan ragged lifecycle",original,"HYLE Vulkan ragged",selected);
  else (void)bench.compare("HYLE Vulkan ragged",selected,"Original Vulkan ragged lifecycle",original);
}
}
int main(){const auto qos=hyle::bench::perf::configure_thread_qos();std::fprintf(stderr,"# HYLE_NATIVE_RAGGED_RUN run_nonce=%s qos=%s qos_ok=%s verify_only=%d paired_rounds=16\n",hyle::bench::perf::run_nonce(),qos.name,qos.ok,set("HYLE_BENCH_VERIFY_ONLY"));
  measure(1,17,1024,768,256,true);measure(64,128,1024,768,256,true);measure(1,17,31,47,64,false);
}
