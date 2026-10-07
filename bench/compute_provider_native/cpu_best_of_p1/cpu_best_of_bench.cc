#include "../../common/perf_process.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/compute/providers/native.h"
#include "hyle/compute/providers/onednn.h"
#include "hyle/mem/buffer.h"
#include "hyle/simd/dtype.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <nanobench.h>
#include <span>
#include <string>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace m = hyle::mem;
namespace nb = ankerl::nanobench;
template <class T> T take(hyle::Result<T> r) {
  if (!r) {
    std::fprintf(stderr, "ERROR %.*s\n", int(r.error().message().size()),
                 r.error().message().data());
    std::exit(2);
  }
  return std::move(*r);
}
void require(hyle::Status s) {
  if (!s.ok()) {
    std::fprintf(stderr, "ERROR %.*s\n", int(s.message().size()),
                 s.message().data());
    std::exit(2);
  }
}
c::TensorDesc dense(c::DataType t, std::initializer_list<std::int64_t> dims) {
  return take(c::TensorDesc::dense(t, {dims.begin(), dims.size()}));
}
enum class Kind {
  Unary,
  Binary,
  Reduce,
  Norm,
  Gate,
  MatMul,
  Rope,
  Softmax,
  Attention,
  Gather,
  Scatter,
  Concat,
  Cast,
  Copy,
  Fill,
  Select
};
struct Case {
  std::string id;
  Kind kind;
  c::DataType type;
  int variant = 0;
  int rows = 1;
  c::DataType other = c::DataType::kFloat32;
  int width = 2048;
  int columns = 24;
};
const std::array floats{c::DataType::kFloat32, c::DataType::kFloat16,
                        c::DataType::kBFloat16};
const std::array storage{c::DataType::kFloat32, c::DataType::kFloat16,
                         c::DataType::kBFloat16, c::DataType::kFloat8E4M3FN,
                         c::DataType::kFloat8E5M2};
// Fixed real-width normal/holdout and ordinary tensor roles. Provider admission
// is observed for every exact request; this catalog grants no performance WIN.
std::vector<Case> catalog() {
  std::vector<Case> result;
  auto add = [&](Kind kind, c::DataType type, int variant,
                 const std::string &name, c::DataType other = c::DataType::kFloat32) {
    for (const auto point : {std::pair{2048, 1}, std::pair{2048, 32},
                             std::pair{4096, 1}})
      result.push_back({name + "-t" + std::to_string(int(type)) + "-r" +
                        std::to_string(point.second) + "-h" +
                        std::to_string(point.first) + "-o" + std::to_string(int(other)),
                        kind, type, variant, point.second, other, point.first});
  };
  for (auto t : floats) {
    for (int i=0;i<13;++i) add(Kind::Unary,t,i,"unary-"+std::to_string(i));
    for (int i=0;i<6;++i) add(Kind::Binary,t,i,"binary-"+std::to_string(i));
    for (int i=0;i<4;++i) add(Kind::Reduce,t,i,"reduce-"+std::to_string(i));
    for (int i=0;i<2;++i) add(Kind::Norm,t,i,"norm-"+std::to_string(i));
    add(Kind::Select,t,0,"select"); add(Kind::Concat,t,0,"concat");
    add(Kind::Fill,t,0,"fill");
    for (auto index : {c::DataType::kInt32,c::DataType::kInt64}) {
      add(Kind::Gather,t,0,"gather",index); add(Kind::Scatter,t,0,"assign",index);
      if(t==c::DataType::kFloat32) add(Kind::Scatter,t,1,"scatter-add",index);
    }
    for(int i=0;i<4;++i) add(Kind::Softmax,t,i,"softmax-"+std::to_string(i));
    for(int i=0;i<2;++i) add(Kind::Gate,t,i,"silu-gate-"+std::to_string(i));
  }
  for (auto t : storage) {
    add(Kind::Copy,t,0,"copy");
    for (auto other : storage) add(Kind::Cast,t,0,"cast",other);
  }
  add(Kind::Rope,c::DataType::kFloat32,0,"rope-f32-phase");
  // F2 official F32 projection ports retain K=16384/N=4 or 24. Dense and
  // transposed physical layouts are distinct requests, with no free repacking.
  for(int n : {4,24}) for(int rows : {1,32}) for(int layout : {0,1})
    result.push_back({"matmul-f32-k16384-n"+std::to_string(n)+"-r"+
        std::to_string(rows)+"-l"+std::to_string(layout),Kind::MatMul,
        c::DataType::kFloat32,layout,rows,c::DataType::kFloat32,16384,n});
  for(int rows : {1,32})
    result.push_back({"matmul-qwen-bf16-k2048-n6144-r"+std::to_string(rows),
        Kind::MatMul,c::DataType::kBFloat16,1,rows,c::DataType::kFloat32,2048,6144});
  for(int rows : {1,32})
    result.push_back({"sdpa-f32-q"+std::to_string(rows),Kind::Attention,
        c::DataType::kFloat32,0,rows});
  return result;
}

template <class T> void store(std::byte *p, T v) {
  std::memcpy(p, &v, sizeof(v));
}
template <class T> T load(const std::byte *p) {
  T v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}
void encode(std::byte *p, c::DataType type, float x) {
  switch (type) {
  case c::DataType::kFloat32:
    store(p, x);
    break;
  case c::DataType::kFloat16:
    store(p, hyle::simd::f32_to_f16(x));
    break;
  case c::DataType::kBFloat16:
    store(p, hyle::simd::f32_to_bf16(x));
    break;
  case c::DataType::kFloat8E4M3FN:
    store(p, hyle::simd::f32_to_f8e4m3(x));
    break;
  case c::DataType::kFloat8E5M2:
    store(p, hyle::simd::f32_to_f8e5m2(x));
    break;
  default:
    std::abort();
  }
}
double decode(const std::byte *p, c::DataType type) {
  switch (type) {
  case c::DataType::kFloat32:
    return load<float>(p);
  case c::DataType::kFloat16:
    return hyle::simd::f16_to_f32(load<hyle::simd::float16_t>(p));
  case c::DataType::kBFloat16:
    return hyle::simd::bf16_to_f32(load<hyle::simd::bfloat16_t>(p));
  case c::DataType::kFloat8E4M3FN:
    return hyle::simd::f8e4m3_to_f32(load<hyle::simd::float8_e4m3_t>(p));
  case c::DataType::kFloat8E5M2:
    return hyle::simd::f8e5m2_to_f32(load<hyle::simd::float8_e5m2_t>(p));
  default:
    std::abort();
  }
}
struct Data {
  Case spec;
  c::Operation op;
  std::vector<c::TensorDesc> desc;
  c::TensorDesc output;
  std::vector<m::Buffer> buffers;
  m::Buffer result;
  std::vector<c::ConstTensorView> inputs;
  std::array<c::TensorView, 1> outputs;
  explicit Data(Case s) : spec(std::move(s)) {
    const auto t = spec.type;
    const auto r = std::int64_t(spec.rows);
    const std::int64_t h = spec.width;
    auto d = dense(t, {r, h});
    switch (spec.kind) {
    case Kind::Unary:
      op = c::Operation(c::Unary{c::UnaryKind(spec.variant)});
      desc = {d};
      break;
    case Kind::Binary:
      op = c::Operation(c::Binary{c::BinaryKind(spec.variant)});
      desc = {d, d};
      break;
    case Kind::Reduce:
      op = c::Operation(
          c::Reduce{.kind = c::ReductionKind(spec.variant),
                    .axes = take(c::Axes::from(std::array<std::int32_t, 1>{1})),
                    .accumulation_type = c::DataType::kFloat32});
      desc = {d};
      break;
    case Kind::Norm:
      op = spec.variant ? c::Operation(c::LayerNorm{})
                        : c::Operation(c::RmsNorm{});
      desc = {d, dense(t, {h})};
      break;
    case Kind::Gate:
      op = c::Operation(c::GatedActivation{
          .activation = c::GatedActivationKind::kSilu,
          .inputs = spec.variant ? c::GatedInputKind::kPackedHalves
                                 : c::GatedInputKind::kSeparate});
      desc =
          spec.variant ? std::vector{dense(t, {r, 2 * h})} : std::vector{d, d};
      break;
    case Kind::MatMul: {
      op = c::Operation(c::MatMul{});
      desc = {dense(t, {r, h}),
              spec.variant ? take(dense(t, {spec.columns, h}).transposed())
                           : dense(t, {h, spec.columns})};
      break;
    }
    case Kind::Rope:
      op = c::Operation(c::RoPE{.rotary_dimensions = 64, .sequence_axis = 1});
      desc = {dense(t, {1, r, 8, 256}), dense(c::DataType::kInt32, {r}),
              dense(c::DataType::kFloat32, {32})};
      break;
    case Kind::Softmax: {
      const bool boolean = spec.variant >= 2;
      op = c::Operation(c::Softmax{
          .kind = (spec.variant % 2) ? c::SoftmaxKind::kLogSoftmax
                                     : c::SoftmaxKind::kSoftmax,
          .mask = boolean ? c::MaskKind::kBoolean : c::MaskKind::kNone});
      desc = {d};
      if (boolean)
        desc.push_back(dense(c::DataType::kBool, {r, h}));
      break;
    }
    case Kind::Attention:
      op = c::Operation(c::ScaledDotProductAttention{.scale = .125F});
      desc = {dense(t, {1, r, 4, 64}), dense(t, {1, 128, 4, 64}),
              dense(t, {1, 128, 4, 64})};
      break;
    case Kind::Gather:
      op = c::Operation(c::Gather{});
      desc = {dense(t, {64, h}), dense(spec.other, {r})};
      break;
    case Kind::Scatter:
      op = c::Operation(c::Scatter{.kind = spec.variant
                                               ? c::ScatterKind::kAdd
                                               : c::ScatterKind::kAssign});
      desc = {dense(t, {64, h}), dense(spec.other, {r}), d};
      break;
    case Kind::Concat:
      op = c::Operation(c::Concat{.axis = 1});
      desc = {d, d};
      break;
    case Kind::Cast:
      op = c::Operation(c::Cast{spec.other});
      desc = {d};
      break;
    case Kind::Copy:
      op = c::Operation(c::Copy{});
      desc = {d};
      break;
    case Kind::Fill: {
      std::uint64_t bits = 0;
      std::array<std::byte, 8> raw{};
      encode(raw.data(), t, 1.25F);
      std::memcpy(&bits, raw.data(), c::data_type_size(t));
      op = c::Operation(c::Fill{.output_shape = take(c::TensorShape::create(
                                    std::array<std::int64_t, 2>{r, h})),
                                .output_type = t,
                                .value = c::ScalarValue::from_bits(t, bits)});
      break;
    }
    case Kind::Select:
      op = c::Operation(c::Select{});
      desc = {dense(c::DataType::kBool, {r, h}), d, d};
      break;
    }
    output = take(c::infer_single_output(op, desc));
    for (std::size_t input = 0; input < desc.size(); ++input) {
      const auto &dd = desc[input];
      auto b = take(m::Buffer::create(m::system_resource(),
                                      take(dd.required_bytes()), 64));
      std::vector<std::byte> bytes(b.size());
      const auto width = c::data_type_size(dd.data_type());
      for (std::size_t i = 0; i < take(dd.element_count()); ++i) {
        auto *p = bytes.data() + i * width;
        const auto type = dd.data_type();
        if (type == c::DataType::kBool) {
          *p = i % 7 == 0 ? std::byte{0} : std::byte{1};
          continue;
        }
        if (type == c::DataType::kInt32) {
          store(p, static_cast<std::int32_t>(i % 64));
          continue;
        }
        if (type == c::DataType::kInt64) {
          store(p, static_cast<std::int64_t>(i % 64));
          continue;
        }
        float value = .5F + float((i + input * 11) % 29) / 64.F;
        if (spec.kind == Kind::MatMul)
          value = float(int((i + input * 7) % 23) - 11) / 64.F;
        if (spec.kind == Kind::Rope && input == 2)
          value = std::pow(10000000.F, -float(i) / 32.F);
        encode(p, type, value);
      }
      require(b.view().write(bytes));
      buffers.push_back(std::move(b));
    }
    result = take(m::Buffer::create(m::system_resource(),
                                    take(output.required_bytes()), 64));
    for (std::size_t i = 0; i < desc.size(); ++i)
      inputs.push_back(
          take(c::ConstTensorView::create(desc[i], buffers[i].view())));
    outputs[0] = take(c::TensorView::create(output, result.view()));
  }
  std::vector<std::byte> read() const {
    std::vector<std::byte> bytes(result.size());
    require(result.view().read(bytes));
    return bytes;
  }
};
enum class Family { Native, OneDnn, Ggml, Aten };
const char *family_name(Family family) {
  switch(family) {
  case Family::Native:return "native"; case Family::OneDnn:return "onednn";
  case Family::Ggml:return "ggml"; case Family::Aten:return "aten";
  }
  return "invalid";
}
c::Provider provider(Family family) {
  switch(family) {
  case Family::Native:return c::native::provider();
  case Family::OneDnn:
#if defined(HYLE_COMPUTE_ONEDNN)
    return c::onednn::provider();
#else
    return {};
#endif
  case Family::Ggml:
#if defined(HYLE_COMPUTE_GGML_CPU)
    return c::ggml_cpu::provider({.native_threads=1});
#else
    return {};
#endif
  case Family::Aten:return c::aten_cpu::provider();
  }
  return {};
}
struct Session {
  c::Context context;
  c::Queue queue;
  c::PreparedOperation prepared;
  m::Buffer workspace;
  c::OperationInvocation invocation;
  Data *data;
  Family family;
  bool available=false;
  Session(Data &d, Family selected) : data(&d),family(selected) {
    auto implementation=provider(selected);
    if(!implementation.valid()) {
      std::fprintf(stderr,"# ADMISSION id=%s provider=%s result=FEATURE_ABSENT\n",
                   d.spec.id.c_str(),family_name(selected)); return;
    }
    context=take(c::Context::create(c::cpu::device(),implementation));
    queue=take(context.create_queue());
    auto result=c::prepare(context,d.op,d.desc,std::span(&d.output,1),m::system_resource());
    if(!result && result.error().IsUnimplemented()) {
      std::fprintf(stderr,"# ADMISSION id=%s provider=%s result=DECLINED reason=%.*s\n",
                   d.spec.id.c_str(),family_name(selected),int(result.error().message().size()),
                   result.error().message().data()); return;
    }
    prepared=take(std::move(result));
    const auto w=prepared.workspace_requirement();
    if(w.size) workspace=take(m::Buffer::create(m::system_resource(),w.size,w.alignment));
    invocation=take(prepared.create_invocation(workspace.view()));
    available=true;
    const auto identity=prepared.provider();
    const auto runtime=prepared.runtime_properties();
    const auto private_bytes=prepared.invocation_private_bytes();
    std::fprintf(stderr,"# ADMISSION id=%s provider=%s actual=%.*s version=%.*s result=ACCEPTED workspace=%zu private=%s threads=%d replay=%d\n",
        d.spec.id.c_str(),family_name(selected),int(identity.name.size()),identity.name.data(),
        int(identity.version.size()),identity.version.data(),w.size,private_bytes?"bounded":"unknown",
        int(runtime.uses_private_threads),int(runtime.replay_class));
  }
  void run() {
    require(take(invocation.submit(queue,data->inputs,data->outputs)).wait());
    nb::doNotOptimizeAway(data->outputs.data());
  }
};
struct SessionRunner {
  Session *session;
  void operator()() const { session->run(); }
};
bool exact(Kind k) {
  return k == Kind::Copy || k == Kind::Cast || k == Kind::Fill ||
         k == Kind::Concat || k == Kind::Select || k == Kind::Gather ||
         k == Kind::Scatter;
}
bool equal(Data &d, const std::vector<std::byte> &a,
           const std::vector<std::byte> &b) {
  if (exact(d.spec.kind))
    return a == b;
  const auto type = d.output.data_type();
  const auto width = c::data_type_size(type);
  for (std::size_t i = 0; i < a.size(); i += width) {
    const double x = decode(a.data() + i, type), y = decode(b.data() + i, type);
    if (std::isnan(y)) {
      if (!std::isnan(x))
        return false;
      continue;
    }
    if (std::isinf(y)) {
      if (x != y)
        return false;
      continue;
    }
    double relative = type == c::DataType::kBFloat16  ? .008
                      : type == c::DataType::kFloat16 ? .001
                                                      : 3e-6;
    double absolute = type == c::DataType::kFloat32
                          ? 4 * double(std::numeric_limits<float>::denorm_min())
                          : 1e-8;
    if (d.spec.kind == Kind::Norm)
      absolute = 2e-5;
    if (d.spec.kind == Kind::MatMul && type == c::DataType::kFloat32) {
      absolute = 2e-5;
      relative = 3e-5;
    }
    if (d.spec.kind == Kind::Attention || d.spec.kind == Kind::Rope)
      absolute = 8e-5;
    if (std::abs(x - y) > absolute + relative * std::abs(y)) {
      std::fprintf(stderr,
                   "NUMERIC_DIFFERENCE %s i=%zu native=%.17g mature=%.17g\n",
                   d.spec.id.c_str(), i / width, x, y);
      return false;
    }
  }
  return true;
}
bool present(const char *name) { return std::getenv(name) != nullptr; }

constexpr std::array families{Family::Native,Family::OneDnn,Family::Ggml,Family::Aten};
void probe(const Case &spec) {
  Data data(spec);
  std::vector<std::unique_ptr<Session>> eligible;
  std::vector<std::vector<std::byte>> results;
  for(auto family:families) {
    auto session=std::make_unique<Session>(data,family);
    if(!session->available)continue;
    session->run(); results.push_back(data.read());eligible.push_back(std::move(session));
  }
  for(std::size_t i=0;i<eligible.size();++i) for(std::size_t j=i+1;j<eligible.size();++j) {
    const auto same=equal(data,results[i],results[j]);
    std::fprintf(stderr,"PAIR\t%s\t%s\t%s\t%s\n",spec.id.c_str(),family_name(eligible[i]->family),
        family_name(eligible[j]->family),same?"NUMERIC_MATCH":"NUMERIC_MISMATCH");
  }
  std::fprintf(stderr,"CARRIER_COUNT\t%s\t%zu\n",spec.id.c_str(),eligible.size());
}
void measure_pair(const Case &spec,Family left,Family right) {
  Data data(spec);
  std::unique_ptr<Session> candidate,peer;
  const bool allocation_first=present("HYLE_BENCH_PEER_ALLOCATION_FIRST");
  if(allocation_first){peer=std::make_unique<Session>(data,right);candidate=std::make_unique<Session>(data,left);}
  else{candidate=std::make_unique<Session>(data,left);peer=std::make_unique<Session>(data,right);}
  if(!candidate->available || !peer->available)std::exit(5);
  candidate->run();auto a=data.read();peer->run();auto b=data.read();
  if(!equal(data,a,b))std::exit(6);
  for(int i=0;i<8;++i){if(i%2){candidate->run();peer->run();}else{peer->run();candidate->run();}}
  const std::string id=spec.id+"-"+family_name(left)+"-vs-"+family_name(right);
  nb::Bench bench;bench.title(id).batch(1).epochs(16).minEpochIterations(1)
      .minEpochTime(std::chrono::milliseconds(2)).performanceCounters(false);
  const std::string cn="Candidate "+id,pn="Peer "+id;
  if(present("HYLE_BENCH_RIVAL_FIRST"))bench.compare(pn,SessionRunner{peer.get()},cn,SessionRunner{candidate.get()});
  else bench.compare(cn,SessionRunner{candidate.get()},pn,SessionRunner{peer.get()});
}
} // namespace
int main(int argc,char **argv) {
  const auto affinity=hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
  std::fprintf(stderr,"# CPU_BEST_OF affinity_topology=%s caller_cpu=%s affinity_ok=%s run_nonce=%s\n",
      affinity.topology,affinity.cpu,affinity.ok,hyle::bench::perf::run_nonce());
  const auto cases=catalog();
  if(argc==1 || std::strcmp(argv[1],"--probe")==0) {
    for(const auto &spec:cases)probe(spec);
    return 0;
  }
  if(argc!=5 || std::strcmp(argv[1],"--pair")!=0)return 64;
  const auto find_family=[](const char *name){for(auto f:families)if(std::strcmp(name,family_name(f))==0)return f;std::exit(64);};
  for(const auto &spec:cases) if(spec.id==argv[2]) {
    measure_pair(spec,find_family(argv[3]),find_family(argv[4]));return 0;
  }
  return 65;
}
