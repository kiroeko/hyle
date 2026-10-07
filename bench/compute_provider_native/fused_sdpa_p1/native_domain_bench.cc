#include "../../common/perf_process.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/compute/providers/native.h"
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
};
const std::array floats{c::DataType::kFloat32, c::DataType::kFloat16,
                        c::DataType::kBFloat16};
const std::array storage{c::DataType::kFloat32, c::DataType::kFloat16,
                         c::DataType::kBFloat16, c::DataType::kFloat8E4M3FN,
                         c::DataType::kFloat8E5M2};
std::vector<Case> catalog() {
  std::vector<Case> result;
  for (int variant : {0, 1})
    for (int rows : {1, 32})
      result.push_back({std::string(variant ? "causal-sdpa-d256" : "equal-head-sdpa") +
                         "-t6-r" + std::to_string(rows) + "-o6",
                        Kind::Attention, c::DataType::kFloat32, variant, rows});
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
    constexpr std::int64_t h = 2048;
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
      op = c::Operation(c::RmsNorm{.epsilon = spec.variant == 2 ? 1.0e-6F : 1.0e-5F});
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
      desc = {dense(t, {r, 1024}),
              spec.variant ? take(dense(t, {1024, 1024}).transposed())
                           : dense(t, {1024, 1024})};
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
    case Kind::Attention: {
      op = c::Operation(c::ScaledDotProductAttention{
          .scale = spec.variant ? .0625F : .125F,
          .causal = spec.variant != 0});
      const std::int64_t heads = spec.variant ? 8 : 4;
      const std::int64_t depth = spec.variant ? 256 : 64;
      desc = {dense(t, {1, r, heads, depth}), dense(t, {1, 128, heads, depth}),
              dense(t, {1, 128, heads, depth})};
      break;
    }
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
      desc = {spec.variant ? take(c::TensorDesc::strided_bytes(
          t, std::array{r, h}, std::array<std::int64_t, 2>{h * 8, 8})) : d};
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
      for (std::size_t i = 0; i < b.size() / width; ++i) {
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
struct Session {
  c::Context context;
  c::Queue queue;
  c::PreparedOperation prepared;
  m::Buffer workspace;
  c::OperationInvocation invocation;
  Data *data;
  Session(Data &d, bool native)
      : context(take(
            native ? c::Context::create(c::cpu::device(), c::native::provider())
                   : c::cpu::create_context({.native_threads = 1}))),
        queue(take(context.create_queue())), data(&d) {
    prepared = take(c::prepare(context, d.op, d.desc, std::span(&d.output, 1),
                               m::system_resource()));
    if (!native && prepared.provider().id == c::native::kProviderId) {
      std::fprintf(stderr, "NO_MATURE_PEER %s\n", d.spec.id.c_str());
      std::exit(5);
    }
    const auto w = prepared.workspace_requirement();
    if (w.size)
      workspace =
          take(m::Buffer::create(m::system_resource(), w.size, w.alignment));
    invocation = take(prepared.create_invocation(workspace.view()));
  }
  void run() {
    require(take(invocation.submit(queue, data->inputs, data->outputs)).wait());
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
    if (d.spec.kind == Kind::MatMul) {
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
void measure(const Case &s, bool verify_only, bool peer_first,
             bool allocate_peer_first, const char *aa) {
  Data data(s);
  std::unique_ptr<Session> native, mature;
  const bool native_arm = std::strcmp(aa, "selected") != 0,
             mature_arm = std::strcmp(aa, "native") == 0;
  if (allocate_peer_first) {
    mature = std::make_unique<Session>(data, mature_arm);
    native = std::make_unique<Session>(data, native_arm);
  } else {
    native = std::make_unique<Session>(data, native_arm);
    mature = std::make_unique<Session>(data, mature_arm);
  }
  native->run();
  const auto a = data.read();
  mature->run();
  const auto b = data.read();
  if (!equal(data, a, b))
    std::exit(6);
  const auto np = native->prepared.provider(), mp = mature->prepared.provider();
  std::fprintf(stderr,
               "# DOMAIN_CASE id=%s native=%.*s mature=%.*s "
               "mature_version=%.*s checked=1\n",
               s.id.c_str(), int(np.name.size()), np.name.data(),
               int(mp.name.size()), mp.name.data(), int(mp.version.size()),
               mp.version.data());
  if (verify_only)
    return;
  for (int i = 0; i < 8; ++i) {
    if (i % 2) {
      mature->run();
      native->run();
    } else {
      native->run();
      mature->run();
    }
  }
  nb::Bench bench;
  bench.title(s.id)
      .batch(1)
      .epochs(16)
      .minEpochIterations(1)
      .minEpochTime(std::chrono::milliseconds(5))
      .performanceCounters(false);
  const std::string cn = "Native " + s.id, pn = "Mature " + s.id;
  if (peer_first)
    bench.compare(pn, SessionRunner{mature.get()}, cn, SessionRunner{native.get()});
  else
    bench.compare(cn, SessionRunner{native.get()}, pn, SessionRunner{mature.get()});
}
} // namespace
int main() {
  const auto affinity =
      hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
  const bool peer_first = present("HYLE_BENCH_RIVAL_FIRST"),
             allocation_first = present("HYLE_BENCH_PEER_ALLOCATION_FIRST"),
             verify_only = present("HYLE_BENCH_VERIFY_ONLY");
  const char *aa = std::getenv("HYLE_BENCH_AA_ROUTE");
  if (!aa)
    aa = "none";
  std::fprintf(
      stderr,
      "# HYLE_NATIVE_DOMAIN_RUN execution_order=%s allocation_order=%s "
      "aa_route=%s paired_mode=nanobench-compare affinity_topology=%s "
      "caller_cpu=%s affinity_ok=%s run_nonce=%s\n",
      peer_first ? "peer-first" : "candidate-first",
      allocation_first ? "peer-first" : "candidate-first", aa,
      affinity.topology, affinity.cpu, affinity.ok,
      hyle::bench::perf::run_nonce());
  const auto cases = catalog();
  std::fprintf(stderr, "# DOMAIN_CATALOG cases=%zu\n", cases.size());
  for (const auto &s : cases)
    measure(s, verify_only, peer_first, allocation_first, aa);
}
