#include <nanobench.h>
#include <mlx/memory.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <pthread.h>
#include <mach-o/dyld.h>
#include <filesystem>
#include <fstream>

#include "../common/perf_process.h"
#include "hyle/compute/apple.h"
#include "hyle/compute/model_ops.h"

namespace {
namespace mx = mlx::core;
namespace a = hyle::compute::apple;
namespace c = hyle::compute;
namespace nb = ankerl::nanobench;
template<class T> T take(hyle::Result<T> result) {
  if (!result) throw std::runtime_error(std::string(result.error().message()));
  return std::move(*result);
}
void require(hyle::Status status) {
  if (!status.ok()) throw std::runtime_error(std::string(status.message()));
}
bool enabled(const char *name) {const auto *value = std::getenv(name); return value && *value;}
mx::array ready(mx::array x) {
  mx::async_eval({x}); mx::synchronize(mx::default_stream(mx::Device::gpu)); x.wait(); return x;
}
mx::array bf16(std::span<const float> x, mx::Shape shape) {
  std::vector<std::uint16_t> bits;
  for (float value : x) {
    auto word = std::bit_cast<std::uint32_t>(value);
    word += 0x7fff + ((word >> 16) & 1);
    bits.push_back(std::uint16_t(word >> 16));
  }
  return ready(mx::view(mx::array(bits.data(), std::move(shape)), mx::bfloat16,
      mx::default_stream(mx::Device::gpu)));
}
struct Shape {const char *title; bool grouped, bf16_weight; int rows, slots, e, n, k;};
constexpr std::array shapes{
    Shape{"BF16 router M1 N512 K8192", false, true, 1, 1, 1, 512, 8192},
    Shape{"BF16 router M16 N512 K8192", false, true, 16, 1, 1, 512, 8192},
    Shape{"BF16 grouped T5 S8 E8 N1024 K4096", true, true, 5, 8, 8, 1024, 4096},
    Shape{"F32 grouped T5 S8 E8 N1024 K4096", true, false, 5, 8, 8, 1024, 4096}};
struct Data {
  std::array<std::vector<mx::array>, 2> inputs;
  std::array<std::vector<float>, 2> expected;
  std::vector<a::InputSpec> specs;
  explicit Data(const Shape &s) {
    const auto wt = s.bf16_weight ? mx::bfloat16 : mx::float32;
    specs = s.grouped ? std::vector<a::InputSpec>{a::InputSpec{{s.e,s.n,s.k},wt},
        a::InputSpec{{s.rows,s.slots,s.k},mx::float32},a::InputSpec{{s.rows,s.slots},mx::int32}} :
        std::vector<a::InputSpec>{a::InputSpec{{s.rows,s.k},mx::bfloat16},a::InputSpec{{s.n,s.k},wt}};
    for (int bank = 0; bank < 2; ++bank) {
      const int seed = bank ? 19 : 3, count = s.rows * s.slots;
      std::vector<float> x(count*s.k), w(s.e*s.n*s.k);
      std::vector<std::int32_t> ids(count);
      for (int token = 0; token < count; ++token) {
        ids[token] = (token*3+seed)%s.e;
        for (int col = 0; col < s.k; ++col) x[token*s.k+col] = float((col*13+token*5+seed)%31-15)/16;
      }
      expected[bank].resize(count*s.n);
      for (int expert = 0; expert < s.e; ++expert) for (int row = 0; row < s.n; ++row) {
        const int first = (row*17+expert*13+seed)%s.k, second = (first+s.k/2+7)%s.k;
        const float p = float(1+(row+expert)%7)/8, q = -float(1+(row+expert)%5)/16;
        w[(expert*s.n+row)*s.k+first]=p; w[(expert*s.n+row)*s.k+second]=q;
        for (int token=0; token<count; ++token) if (ids[token]==expert)
          expected[bank][token*s.n+row]=float(double(x[token*s.k+first])*p+double(x[token*s.k+second])*q);
      }
      if (s.grouped) {
        auto weights = s.bf16_weight ? bf16(w,{s.e,s.n,s.k}) : mx::array(w.data(),{s.e,s.n,s.k});
        inputs[bank]={weights,mx::array(x.data(),{s.rows,s.slots,s.k}),mx::array(ids.data(),{s.rows,s.slots})};
      } else inputs[bank]={bf16(x,{s.rows,s.k}),bf16(w,{s.n,s.k})};
    }
    if (expected[0] == expected[1]) throw std::runtime_error("resident banks do not differ");
  }
};
a::Program native_mlx(a::Context &context, const Shape &shape, const Data &data) {
  return take(context.compile([grouped=shape.grouped](const auto &inputs, mx::Stream s)
      -> hyle::Result<std::vector<mx::array>> {
    if (!grouped) return std::vector{mx::matmul(mx::astype(inputs[0],mx::float32,s),
        mx::transpose(mx::astype(inputs[1],mx::float32,s),s),s)};
    const std::array index_inputs{mx::arange(inputs[0].shape(0),mx::int32,s),inputs[2]};
    HYLE_ASSIGN_OR_RETURN(auto checked,a::lower(c::Operation(c::Gather{}),index_inputs,s));
    auto ids=mx::astype(mx::reshape(checked[0],{-1},s),mx::uint32,s);
    const int tokens=inputs[1].shape(0),slots=inputs[1].shape(1),k=inputs[1].shape(2);
    auto x=mx::reshape(inputs[1],{tokens*slots,1,k},s);
    auto w=mx::transpose(mx::astype(inputs[0],mx::float32,s),{0,2,1},s);
    return std::vector{mx::reshape(mx::gather_mm(x,w,std::nullopt,ids,false,s),
        {tokens,slots,inputs[0].shape(1)},s)};
  },data.specs));
}
struct Harness {
  a::Context context;
  a::Program program;
  std::array<a::Invocation,2> slots;
  const Data &data;
  int next=0;
  Harness(const Shape &shape,const Data &input,bool mlx) : context(take(a::Context::create())),data(input) {
    const auto observed_environment=context.environment();
    if(!observed_environment) throw std::runtime_error(std::string(observed_environment.error().message()));
    const auto &environment=*observed_environment;
    std::fprintf(stderr,"# OVERLAP_ENV gpu=%s arch=%s os=%s mlx=%s ggml=%s sdk=%u\n",
        environment.device_name.c_str(),environment.architecture.c_str(),environment.os_runtime.c_str(),
        environment.mlx_revision.c_str(),environment.ggml_revision.c_str(),environment.build_sdk);
    // Cold telemetry on the actual native owner, never inferred from caller QoS.
    const std::array telemetry_spec{a::InputSpec{{1},mx::float32}};
    auto telemetry=take(context.compile([](const auto &input,mx::Stream) {
      qos_class_t qos=QOS_CLASS_UNSPECIFIED;
      if(pthread_get_qos_class_np(pthread_self(),&qos,nullptr)!=0)
        throw std::runtime_error("cannot observe native owner QoS");
      std::fprintf(stderr,"# OVERLAP_OWNER_QOS %u\n",unsigned(qos));
      return input;
    },telemetry_spec));
    program=mlx?native_mlx(context,shape,data):take(context.prepare(shape.grouped?
        c::model_ops::grouped_matmul():c::model_ops::bfloat_linear_f32(),data.specs));
    for(auto &slot:slots) slot=take(program.create_invocation());
  }
  void run() {
    const int bank=next++%2;
    require(slots[bank].bind(data.inputs[bank]));
    auto work=take(slots[bank].submit());auto result=take(work.wait());
    nb::doNotOptimizeAway(result.get()->arrays()[0].id());
  }
  void verify() {
    for(int bank=0;bank<2;++bank) {
      require(slots[bank].bind(data.inputs[bank]));
      auto work=take(slots[bank].submit());auto result=take(work.wait());
      auto out=ready(mx::contiguous(result.get()->arrays()[0],false,mx::default_stream(mx::Device::gpu)));
      if(out.size()!=data.expected[bank].size()) throw std::runtime_error("full output size differs");
      for(std::size_t i=0;i<out.size();++i) if(out.data<float>()[i]!=data.expected[bank][i])
        throw std::runtime_error("independent full output differs at "+std::to_string(i));
    }
  }
};
struct Runner {Harness *h;void operator()() const {h->run();}};
} // namespace

int main() {
  try {
    const bool verify=enabled("HYLE_BENCH_VERIFY_ONLY");
    const char *raw=std::getenv("HYLE_BENCH_AA_ROUTE");const std::string aa=raw?raw:"";
    if(!aa.empty()&&aa!="selected"&&aa!="native") throw std::runtime_error("invalid AA route");
    if(!verify&&aa.empty()&&!enabled("HYLE_PERF_RUN_NONCE"))
      throw std::runtime_error("formal measurement requires the unified runner");
    if(!verify&&aa.empty()) {
      std::uint32_t size=0;(void)_NSGetExecutablePath(nullptr,&size);std::string path(size,'\0');
      if(_NSGetExecutablePath(path.data(),&size)!=0) throw std::runtime_error("cannot locate AA receipt");
      std::ifstream ready_file(std::filesystem::path(path.c_str()).parent_path()/"hyle_apple_overlap_aa.ready");
      std::string state;ready_file>>state;if(state!="PASS") throw std::runtime_error("overlap AA not qualified");
    }
    const auto qos=hyle::bench::perf::configure_thread_qos();
    const bool rival=enabled("HYLE_BENCH_RIVAL_FIRST"),allocate=enabled("HYLE_BENCH_PEER_ALLOCATION_FIRST");
    std::fprintf(stderr,"# HYLE_APPLE_OVERLAP_RUN qos=%s qos_ok=%s aa=%s verify=%d execution=%s allocation=%s run_nonce=%s\n",
        qos.name,qos.ok,aa.empty()?"none":aa.c_str(),verify,rival?"peer-first":"candidate-first",
        allocate?"peer-first":"candidate-first",hyle::bench::perf::run_nonce());
    for(const auto &shape:shapes) {
      Data data(shape);std::unique_ptr<Harness> candidate,peer;
      const auto begin=std::chrono::steady_clock::now();
      if(allocate) {peer=std::make_unique<Harness>(shape,data,aa!="selected");candidate=std::make_unique<Harness>(shape,data,aa=="native");}
      else {candidate=std::make_unique<Harness>(shape,data,aa=="native");peer=std::make_unique<Harness>(shape,data,aa!="selected");}
      std::fprintf(stderr,"# OVERLAP_COLD title=%s setup_ms=%.3f active=%zu peak=%zu cache=%zu\n",shape.title,
          std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count(),
          mx::get_active_memory(),mx::get_peak_memory(),mx::get_cache_memory());
      candidate->verify();peer->verify();
      if(verify) {std::fprintf(stderr,"# OVERLAP_VERIFY PASS %s\n",shape.title);continue;}
      for(int i=0;i<8;++i) {if(i%2){peer->run();candidate->run();}else{candidate->run();peer->run();}}
      const std::string cname="Original Metal "+std::string(shape.title),pname="Original MLX "+std::string(shape.title);
      nb::Bench bench;bench.title(shape.title).batch(1).epochs(104).minEpochIterations(2)
          .minEpochTime(std::chrono::milliseconds(4)).performanceCounters(false);
      if(rival)(void)bench.compare(pname,Runner{peer.get()},cname,Runner{candidate.get()});
      else (void)bench.compare(cname,Runner{candidate.get()},pname,Runner{peer.get()});
    }
  } catch(const std::exception &e) {std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
