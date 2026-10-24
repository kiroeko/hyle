#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <nanobench.h>
#include "perf_process.h"
#include <hyle/compute/model_ops.h>
#if HYLE_G07_CUDA
#include <hyle/compute/backends/cuda.h>
#include <hyle/compute/prepared_operation.h>
#include <hyle/compute/providers/torch_cuda.h>
#include <hyle/mem/buffer.h>
namespace hyle::compute::torch_cuda { Provider g07_original_provider() noexcept; }
#else
#include <hyle/compute/apple.h>
#include <hyle/compute/low_precision.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>
namespace hyle::compute::apple::detail {
Result<std::vector<mlx::core::array>> g07_original_lower_mlx_float(
    const Operation&, std::span<const TensorDesc>, std::span<const mlx::core::array>, mlx::core::Stream) noexcept;
}
#endif

namespace {
namespace c=hyle::compute;
namespace nb=ankerl::nanobench;
template<class T> T take(hyle::Result<T> r) {
  if(!r)throw std::runtime_error(std::string(r.error().message()));
  return std::move(*r);
}
void check(hyle::Status s) {if(!s.ok())throw std::runtime_error(std::string(s.message()));}
bool set(const char* n) {const auto* p=std::getenv(n);return p&&*p&&std::strcmp(p,"0")!=0;}
constexpr int k=2048;
#if HYLE_G07_CUDA
constexpr int n=6144,group=64;
constexpr int paired_rounds=104;
constexpr const char* route="CUDA";
struct Data {
  int m;std::vector<float> x,scales,bias,expected;std::vector<std::uint32_t> words;
  explicit Data(int rows):m(rows),x(m*k),scales(n*k/group),bias(scales.size()),expected(m*n),words(n*k/8) {
    for(int i=0;i<m*k;++i)x[i]=float((i*7+i/k*3)%31-15)*.0625F;
    for(int row=0;row<n;++row)for(int j=0;j<k/8;++j){
      std::uint32_t word=0;for(int q=0;q<8;++q)word|=std::uint32_t((row*3+j*5+q*7)%16)<<(q*4);words[row*(k/8)+j]=word;
    }
    for(int row=0;row<n;++row)for(int b=0;b<k/group;++b){const auto i=row*(k/group)+b;scales[i]=float(1+(row+b)%7)*.015625F;bias[i]=float((row+3*b)%9-4)*.03125F;}
    for(int row=0;row<m;++row)for(int col=0;col<n;++col){
      double sum=0;for(int j=0;j<k;++j){const auto code=(words[col*(k/8)+j/8]>>((j%8)*4))&15u;const auto i=col*(k/group)+j/group;sum+=double(x[row*k+j])*(double(code)*scales[i]+bias[i]);}
      expected[row*n+col]=float(sum);
    }
  }
};
struct Harness {
  c::Context context;c::cuda::MemoryResource memory;c::Queue queue;c::PreparedOperation program;
  c::OperationInvocation invocation;std::array<hyle::mem::Buffer,4> buffers;hyle::mem::Buffer output;
  std::array<c::ConstTensorView,4> views;std::array<c::TensorView,1> outputs;bool original;
  Harness(const Data& d,bool old):original(old){
    const std::array providers{c::ProviderBinding{old?c::torch_cuda::g07_original_provider():c::torch_cuda::provider(),0}};
    context=take(c::Context::create(take(c::cuda::device(0)),{.providers=providers}));
    memory=take(c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice));queue=take(context.create_queue());
    const std::array descriptions{
      take(c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,2>{d.m,k})),
      take(c::TensorDesc::dense(c::DataType::kUInt32,std::array<std::int64_t,2>{n,k/8})),
      take(c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,2>{n,k/group})),
      take(c::TensorDesc::dense(c::DataType::kFloat32,std::array<std::int64_t,2>{n,k/group}))};
    const std::array bytes{std::as_bytes(std::span(d.x)),std::as_bytes(std::span(d.words)),std::as_bytes(std::span(d.scales)),std::as_bytes(std::span(d.bias))};
    for(unsigned i=0;i<4;++i){buffers[i]=take(hyle::mem::Buffer::create(memory.ref(),take(descriptions[i].required_bytes()),256));check(buffers[i].view().write(bytes[i]));views[i]=take(c::ConstTensorView::create(descriptions[i],buffers[i].view()));}
    const auto op=c::model_ops::mlx_quantized_matmul(c::model_ops::MlxWeightFormat::kAffine,4,group);
    std::array<c::TensorDesc,1> desc;check(c::infer_output_descriptions(op,descriptions,desc));
    program=take(c::prepare(context,op,descriptions,desc,memory.ref()));invocation=take(program.create_invocation());
    output=take(hyle::mem::Buffer::create(memory.ref(),take(desc[0].required_bytes()),256));outputs[0]=take(c::TensorView::create(desc[0],output.view()));
  }
  void execute(){auto done=take(invocation.submit(queue,views,outputs));check(done.wait());nb::doNotOptimizeAway(outputs[0]);}
  void verify(const Data& d){
    std::vector<float> actual(d.expected.size());float worst=0;
    for(int i=0;i<3;++i){execute();check(output.view().read(std::as_writable_bytes(std::span(actual))));
      for(std::size_t j=0;j<actual.size();++j){if(!std::isfinite(actual[j]))throw std::runtime_error("nonfinite packed result");worst=std::max(worst,std::abs(actual[j]-d.expected[j]));}}
    // This dyadic finite input has an exact F32 sum; allow the existing F32
    // arithmetic contract, never TF32 or a precision change for this benchmark.
    if(worst>1.e-4F)throw std::runtime_error("CUDA packed independent double oracle differs");
    std::fprintf(stderr,"# G07_ORACLE route=%s original=%d M=%d max_abs_error=%.9g replays=3\n",route,original,d.m,worst);
  }
};
#else
namespace mx=mlx::core;
namespace a=c::apple;
constexpr int n=2048;
constexpr int paired_rounds=16;
constexpr const char* route="Apple";
float fp8(std::uint8_t code){const int e=(code>>3)&15,f=code&7;const float v=e?std::ldexp(1.F+float(f)/8,e-7):std::ldexp(float(f),-9);return code&128?-v:v;}
std::uint16_t bf16(float value){auto bits=std::bit_cast<std::uint32_t>(value);return std::uint16_t((bits+0x7fff+((bits>>16)&1))>>16);}
struct Data {
  int m;std::vector<float> x;std::vector<std::uint8_t> carrier;std::vector<std::uint16_t> expected;
  explicit Data(int rows):m(rows),x(m*k),carrier(n*k+(n/128)*(k/128)*4),expected(m*n){
    constexpr std::array<std::uint8_t,16> codes{0x38,0x40,0x44,0x48,0x4c,0x50,0x54,0x58,0xb8,0xc0,0xc4,0xc8,0xcc,0xd0,0xd4,0xd8};
    // Every activation block has absmax28, scale1/16, and exact finite E4M3
    // values. The complete dynamic quantizer still executes on every call.
    for(int row=0;row<m;++row)for(int j=0;j<k;++j)x[row*k+j]=j%128==0?28.F:fp8(codes[(j*5+row*3)%16])*.0625F;
    for(int col=0;col<n;++col)for(int j=0;j<k;++j)carrier[col*k+j]=codes[(col*7+j*3+j/128)%16];
    std::vector<float> scales((n/128)*(k/128));
    for(std::size_t i=0;i<scales.size();++i)scales[i]=float(1+i%4)*.015625F;
    std::memcpy(carrier.data()+n*k,scales.data(),scales.size()*4);
    for(int row=0;row<m;++row)for(int col=0;col<n;++col){
      double sum=0;for(int j=0;j<k;++j)sum+=double(x[row*k+j])*fp8(carrier[col*k+j])*scales[(col/128)*(k/128)+j/128];
      expected[row*n+col]=bf16(float(sum));
    }
  }
};
struct Harness {
  a::Context context;a::Program program;a::Invocation invocation;std::vector<mx::array> inputs;bool original;
  Harness(const Data& d,bool old):context(take(a::Context::create())),original(old){
    const auto op=c::model_ops::float8_block_linear(c::model_ops::Float8ScaleRounding::kNone,1.e-10F);
    const std::array descriptions{
      take(c::TensorDesc::dense(c::DataType::kBFloat16,std::array<std::int64_t,2>{d.m,k})),
      take(c::float8_weight_description(k,n,c::Float8ScaleMode::kFloat32Block128))};
    const std::array specs{take(a::input_spec(descriptions[0])),take(a::input_spec(descriptions[1]))};
    if(old)program=take(context.compile([op,descriptions](const auto& values,mx::Stream stream){
      return a::detail::g07_original_lower_mlx_float(op,descriptions,values,stream);},specs));
    else program=take(context.prepare(op,specs));
    invocation=take(program.create_invocation());
    inputs.push_back(mx::astype(mx::array(d.x.data(),{d.m,k},mx::float32),mx::bfloat16));
    inputs.emplace_back(d.carrier.data(),mx::Shape{int(d.carrier.size())},mx::uint8);
    mx::async_eval(inputs);mx::synchronize(mx::default_stream(mx::Device::gpu));for(auto& input:inputs)input.wait();
    check(invocation.bind(inputs));
  }
  void execute(){auto done=take(invocation.submit());auto result=take(done.wait());nb::doNotOptimizeAway(result->arrays()[0]);}
  void verify(const Data& d){
    for(int replay=0;replay<3;++replay){auto done=take(invocation.submit());auto result=take(done.wait());const auto& out=result->arrays()[0];
      if(out.dtype()!=mx::bfloat16||out.size()!=d.expected.size())throw std::runtime_error("Apple output contract differs");
      const auto* bits=reinterpret_cast<const std::uint16_t*>(out.data<mx::bfloat16_t>());
      for(std::size_t i=0;i<d.expected.size();++i)if(bits[i]!=d.expected[i])throw std::runtime_error("Apple packed independent stored-byte double/BF16 oracle differs");}
    std::fprintf(stderr,"# G07_ORACLE route=%s original=%d M=%d bit_exact_bf16=PASS replays=3\n",route,original,d.m);
  }
};
#endif

void measure(int m){
  Data data(m);const auto* p=std::getenv("HYLE_BENCH_AA_ROUTE");const std::string aa=p?p:"";
  if(!aa.empty()&&aa!="native"&&aa!="selected")throw std::runtime_error("invalid A/A route");
  std::unique_ptr<Harness> candidate,peer;
  if(set("HYLE_BENCH_PEER_ALLOCATION_FIRST")){peer=std::make_unique<Harness>(data,aa!="selected");candidate=std::make_unique<Harness>(data,aa=="native");}
  else{candidate=std::make_unique<Harness>(data,aa=="native");peer=std::make_unique<Harness>(data,aa!="selected");}
  candidate->verify(data);peer->verify(data);
  for(int i=0;i<4;++i){candidate->execute();peer->execute();}
  if(set("HYLE_BENCH_VERIFY_ONLY"))return;
#if HYLE_G07_CUDA
  // The desktop WSL M1 A/A was not stable with only16short rounds. Warm both
  // complete public calls through one second before sampling, then use the
  // existing104-round instrument recipe. No work moves out of either call.
  const auto warmed=std::chrono::steady_clock::now()+std::chrono::seconds(1);
  do{candidate->execute();peer->execute();}while(std::chrono::steady_clock::now()<warmed);
#endif
  nb::Bench bench;bench.title(std::string(route)+" resident packed M="+std::to_string(m)+" K=2048 N="+std::to_string(n))
    .batch(1).epochs(paired_rounds).minEpochIterations(1).minEpochTime(std::chrono::milliseconds(4)).performanceCounters(false);
  const auto a=[&]{candidate->execute();};const auto b=[&]{peer->execute();};
  const auto cn=std::string("HYLE ")+route+" resident packed",pn=std::string("Original669 ")+route+" packed";
  if(set("HYLE_BENCH_RIVAL_FIRST"))(void)bench.compare(pn,b,cn,a);else(void)bench.compare(cn,a,pn,b);
}
}
int main(int,char **){
  try{
    const auto qos=hyle::bench::perf::configure_thread_qos();
    std::fprintf(stderr,"# HYLE_PACKED_RUN run_nonce=%s apple_qos=%s qos_ok=%s route=%s verify_only=%d paired_rounds=%d placement=%d execution_order=%s allocation_order=%s aa_route=%s\n",
      hyle::bench::perf::run_nonce(),qos.name,qos.ok,route,set("HYLE_BENCH_VERIFY_ONLY"),paired_rounds,HYLE_G07_PLACEMENT,
      set("HYLE_BENCH_RIVAL_FIRST")?"peer-first":"candidate-first",set("HYLE_BENCH_PEER_ALLOCATION_FIRST")?"peer-first":"candidate-first",
      std::getenv("HYLE_BENCH_AA_ROUTE")?std::getenv("HYLE_BENCH_AA_ROUTE"):"none");
    measure(1);measure(64);return 0;
  }catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
