// Full actual Kimi axes, generated original-format weights. No publisher payload.
#include "../../engine/inference/artifact_digest.h"
#include "../../engine/inference/prepared_output.h"
#include "model_plane_resident_weights.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/model_ops.h"
#include "hyle/graph/cuda.h"
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <set>
#include <map>
#include <functional>

#include <gtest/gtest.h>
namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace supply = model_plane_resident_weights;
namespace inf = hyle::inference;
using T = c::DataType;
using V = g::ValueId;
using Bytes = std::vector<std::byte>;
template <class X> X take(hyle::Result<X> r) {
  if (!r)
    throw std::runtime_error(std::string(r.error().message()));
  return std::move(*r);
}
void check(hyle::Status s) {
  if (!s.ok())
    throw std::runtime_error(std::string(s.message()));
}
void require(bool b, const char *message) {
  if (!b)
    throw std::runtime_error(message);
}
c::TensorDesc dense(T t, std::initializer_list<std::int64_t> s) {
  return take(c::TensorDesc::dense(t, {s.begin(), s.size()}));
}
template <class X> Bytes raw(std::span<const X> x) {
  auto b = std::as_bytes(x);
  return {b.begin(), b.end()};
}
template <class X> Bytes raw(const std::vector<X> &x) {
  return raw(std::span<const X>(x));
}
std::uint16_t bf(float x) {
  const auto u = std::bit_cast<std::uint32_t>(x);
  return static_cast<std::uint16_t>((u + 0x7fffU + ((u >> 16U) & 1U)) >> 16U);
}
float wide(std::uint16_t x) { return std::bit_cast<float>(std::uint32_t(x) << 16U); }
float round_bf(double x) { return wide(bf(static_cast<float>(x))); }
float f32(double x) {
  volatile float y = static_cast<float>(x);
  return y;
}
constexpr std::int64_t H = 7168, E = 896, S = 16, L = 3584, I = 3072, SH = 6144;
enum class Codec { Quark, K512 };
struct Selection {
  std::vector<std::int32_t> globals, map;
};
Selection select(std::span<const std::int32_t> ids) {
  Selection r;
  r.map.assign(E, -1);
  r.globals.assign(ids.begin(), ids.end());
  std::ranges::sort(r.globals);
  r.globals.erase(std::unique(r.globals.begin(), r.globals.end()), r.globals.end());
  require(r.globals.size() == S, "true Top16 must contain16 distinct experts");
  for (std::size_t j = 0; j < r.globals.size(); ++j) {
    require(r.globals[j] >= 0 && r.globals[j] < E, "global expert outside896");
    r.map[std::size_t(r.globals[j])] = static_cast<std::int32_t>(j);
  }
  return r;
}
std::int64_t n_for(unsigned leg) { return leg == 2 ? L : I; }
std::int64_t k_for(unsigned leg) { return leg == 2 ? I : L; }
// Exactly one nonzero logical weight per row. All packed bytes still exist,
// upload and execute; this only makes an independent full-output oracle cheap.
std::int64_t column(std::int64_t row, std::int64_t k, unsigned global, unsigned leg) {
  return (row * 17 + global * 13 + leg * 7) % k;
}
unsigned code(unsigned global, unsigned leg) { return 1U + (global + leg) % 3U; }
float coefficient(unsigned global, unsigned leg) {
  return .25F * float(1U + (global + leg) % 3U);
}
Bytes plane(Codec codec, unsigned leg, bool scales, std::int32_t global,
            std::int64_t n = 0, std::int64_t k = 0) {
  if (!n) {
    n = n_for(leg);
    k = k_for(leg);
  }
  const auto group = codec == Codec::Quark ? 32 : 512;
  if (scales) {
    if (codec == Codec::Quark)
      return Bytes(std::size_t(n * (k / group)), std::byte{125});
    return raw(std::vector<float>(std::size_t(n * (k / group)),
                                  coefficient(unsigned(global), leg)));
  }
  Bytes data(std::size_t(n * k / 2),
             codec == Codec::Quark ? std::byte{0} : std::byte{0x88});
  for (std::int64_t row = 0; row < n; ++row) {
    const auto col = column(row, k, unsigned(global), leg);
    const auto at = std::size_t((row * k + col) / 2);
    const auto shift = unsigned(col % 2) * 4;
    const auto value = code(unsigned(global), leg) + (codec == Codec::Quark ? 0U : 8U);
    const auto old = std::to_integer<unsigned>(data[at]);
    data[at] = std::byte((old & ~(15U << shift)) | (value << shift));
  }
  return data;
}
struct Sources {
  std::map<std::pair<std::size_t, std::int32_t>, m::Buffer> storage;
  std::vector<std::string> hashes;
  Sources(Codec codec, const Selection &selected) {
    for (std::size_t p = 0; p < 6; ++p)
      for (auto id : selected.globals) {
        auto data = plane(codec, unsigned(p / 2), p % 2 != 0, id);
        hashes.push_back(inf::artifact::sha256(data));
        auto buffer = take(m::Buffer::create(m::system_resource(), data.size(), 256));
        check(buffer.view().write(data));
        storage.emplace(std::pair{p, id}, std::move(buffer));
      }
  }
  m::ConstBufferView plane_view(std::size_t p, std::int32_t id) const {
    return storage.at({p, id}).view();
  }
};
std::vector<c::TensorDesc> expert_planes(Codec codec) {
  std::vector<c::TensorDesc> d;
  for (unsigned leg = 0; leg < 3; ++leg) {
    const auto n = n_for(leg), k = k_for(leg);
    d.push_back(dense(codec == Codec::Quark ? T::kUInt8 : T::kInt32,
                      {n, k / (codec == Codec::Quark ? 2 : 8)}));
    d.push_back(dense(codec == Codec::Quark ? T::kUInt8 : T::kFloat32,
                      {n, k / (codec == Codec::Quark ? 32 : 512)}));
  }
  return d;
}
struct SharedMemory {
  std::shared_ptr<c::cuda::MemoryResource> resource;
  m::MemoryResourceRef ref() const noexcept { return resource->ref(); }
};
struct Device {
  c::Context context = take(g::cuda::create_context());
  std::shared_ptr<c::cuda::MemoryResource> memory =
      std::make_shared<c::cuda::MemoryResource>(
          take(c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice)));
  c::StorageOwner owner = take(c::StorageOwner::adopt(context, SharedMemory{memory}));
  c::Queue queue = take(context.create_queue()),
           transfer = take(context.create_queue());
  c::ReadLease upload(c::TensorDesc description, std::span<const std::byte> data) {
    require(take(description.required_bytes()) == data.size(),
            "exact source tensor byte extent");
    auto b = take(m::Buffer::create(memory->ref(), data.size(), 256));
    check(b.view().write(data));
    return take(owner.adopt_buffer(std::move(b)));
  }
};
struct Graph {
  Device &device;
  g::GraphBuilder b = take(g::GraphBuilder::create());
  std::vector<g::ConstantBinding> constants;
  std::vector<c::ReadLease> literals;
  std::vector<V> outputs;
  explicit Graph(Device &d) : device(d) {}
  V input(c::TensorDesc d) { return take(b.input(d)); }
  V one(c::Operation op, std::initializer_list<V> args) {
    return take(b.result(take(b.add_node(op, {args.begin(), args.size()}))));
  }
  template <class Op> V one(Op op, std::initializer_list<V> args) {
    return one(c::Operation(op), args);
  }
  V constant(c::TensorDesc d, Bytes data) {
    auto value = take(b.constant(d));
    literals.push_back(device.upload(d, data));
    constants.push_back({value, take(literals.back().tensor(d)),
                         g::ConstantPolicy::kBorrow, device.owner, literals.back()});
    return value;
  }
  V number(float x) {
    return one(c::Fill{.output_shape =
                           take(c::TensorShape::create(std::array<std::int64_t, 1>{1})),
                       .output_type = T::kFloat32,
                       .value = c::ScalarValue(x)},
               {});
  }
  V cast(V v, T t) { return one(c::Cast{t}, {v}); }
  V binary(c::BinaryKind kind, V a, V z) { return one(c::Binary{kind}, {a, z}); }
  V unary(c::UnaryKind kind, V x) { return one(c::Unary{kind}, {x}); }
  V reshape(V x, std::initializer_list<std::int64_t> s) {
    return one(
        take(c::View::reshape(take(c::TensorShape::create({s.begin(), s.size()})))),
        {x});
  }
  V sum(V x, std::int32_t axis) {
    return one(c::Reduce{.kind = c::ReductionKind::kSum,
                         .axes = take(c::Axes::from(std::array{axis})),
                         .accumulation_type = T::kFloat32},
               {x});
  }
  V situ(V gate, V up) {
    gate = cast(gate, T::kFloat32);
    up = cast(up, T::kFloat32);
    const auto a =
        binary(c::BinaryKind::kMultiply,
               binary(c::BinaryKind::kMultiply, number(4.F),
                      unary(c::UnaryKind::kTanh,
                            binary(c::BinaryKind::kMultiply, gate, number(.25F)))),
               unary(c::UnaryKind::kSigmoid, gate));
    const auto z = binary(
        c::BinaryKind::kMultiply, number(25.F),
        unary(c::UnaryKind::kTanh, binary(c::BinaryKind::kMultiply, up, number(.04F))));
    return cast(binary(c::BinaryKind::kMultiply, a, z), T::kBFloat16);
  }
  V dense_weight(std::int64_t n, std::int64_t k, unsigned seed) {
    std::vector<std::uint16_t> w(std::size_t(n * k));
    for (std::int64_t row = 0; row < n; ++row)
      w[std::size_t(row * k + column(row, k, seed, 0))] = bf(1.F);
    return constant(dense(T::kBFloat16, {n, k}), raw(w));
  }
  V dense_projection(V a, std::int64_t n, std::int64_t k, unsigned seed) {
    auto w = dense_weight(n, k, seed);
    return one(
        c::MatMul{.accumulation_type = T::kFloat32},
        {a, one(take(c::View::permute(std::array<std::int32_t, 2>{1, 0})), {w})});
  }
  V fixed_projection(Codec codec, V a, std::int64_t n, std::int64_t k, unsigned leg) {
    if (codec == Codec::K512)
      return dense_projection(a, n, k, 100 + leg);
    auto w = constant(dense(T::kUInt8, {n, k / 2}), plane(codec, leg, false, 0, n, k));
    auto sf = constant(dense(T::kUInt8, {n, k / 32}), plane(codec, leg, true, 0, n, k));
    return one(c::model_ops::mxfp4_linear(), {a, w, sf});
  }
  void publish(V v) {
    check(b.publish(v));
    outputs.push_back(v);
  }
  g::CompiledGraph compile() {
    auto graph = take(std::move(b).finish());
    return take(g::cuda::compile(graph, device.context, device.owner,
                                 {.constants = constants}));
  }
};
struct Router {
  Graph graph;
  V x, ids, weights, latent;
  g::CompiledGraph compiled;
  explicit Router(Device &device) : graph(device) {
    x = graph.input(dense(T::kBFloat16, {1, H}));
    std::vector<std::uint16_t> w(std::size_t(E * H));
    for (std::int64_t row = 0; row < 17; ++row)
      w[std::size_t(row * H + row)] = bf(64.F);
    const auto rw = graph.constant(dense(T::kBFloat16, {E, H}), raw(w));
    auto logits = graph.one(c::model_ops::dense_linear_f32(), {x, rw});
    auto scores = graph.unary(c::UnaryKind::kSigmoid, logits);
    std::vector<float> bias(E, -4.F);
    for (std::size_t j = 0; j < 17; ++j)
      bias[j] = float(int(j % 3) - 1) / 16.F;
    const auto select_scores =
        graph.binary(c::BinaryKind::kAdd, scores,
                     graph.constant(dense(T::kFloat32, {1, E}), raw(bias)));
    const auto node = take(graph.b.add_node(c::Operation(c::TopK{.count = S}),
                                            std::span(&select_scores, 1)));
    ids = take(graph.b.result(node, 1));
    auto chosen =
        graph.one(c::model_ops::gather_elements(-1, c::IndexBoundsPolicy::kUnchecked),
                  {scores, ids});
    weights = graph.binary(c::BinaryKind::kDivide, chosen,
                           graph.reshape(graph.sum(chosen, 1), {1, 1}));
    latent = graph.dense_projection(x, L, H, 33);
    for (auto out : {ids, weights, latent})
      graph.publish(out);
    compiled = graph.compile();
  }
};
struct Expert {
  Graph graph;
  V x, ids, weights, latent;
  std::array<V, 7> banks;
  std::vector<g::WeightBinding> bindings;
  g::CompiledGraph compiled;
  Codec codec;
  Expert(Device &device, Codec chosen, const supply::Layout &layout)
      : graph(device), codec(chosen) {
    x = graph.input(dense(T::kBFloat16, {1, H}));
    ids = graph.input(dense(T::kInt32, {1, S}));
    weights = graph.input(dense(T::kFloat32, {1, S}));
    latent = graph.input(dense(T::kBFloat16, {1, L}));
    for (std::size_t p = 0; p < 6; ++p) {
      banks[p] = graph.input(layout.resident[p]);
      bindings.push_back({banks[p], p + 1});
    }
    banks[6] = graph.input(layout.map_description());
    bindings.push_back({banks[6], 7});
    auto local =
        graph.one(c::Gather{.bounds = c::IndexBoundsPolicy::kChecked}, {banks[6], ids});
    const auto grouped = [&](V a, unsigned leg) {
      return graph.one(codec == Codec::Quark
                           ? c::model_ops::grouped_mxfp4_linear()
                           : c::model_ops::grouped_blockwise_w4a8_linear(),
                       {a, banks[2 * leg], banks[2 * leg + 1], local});
    };
    auto a = graph.reshape(latent, {1, 1, L});
    auto gate = grouped(a, 0), up = grouped(a, 1), activated = graph.situ(gate, up),
         down = grouped(activated, 2);
    auto combined = graph.cast(
        graph.sum(graph.binary(c::BinaryKind::kMultiply, graph.cast(down, T::kFloat32),
                               graph.reshape(weights, {1, S, 1})),
                  1),
        T::kBFloat16);
    const auto gamma =
        graph.constant(dense(T::kFloat32, {L}), raw(std::vector<float>(L, 1.F)));
    auto normalized = graph.cast(graph.one(c::RmsNorm{.epsilon = 1.e-5F},
                                           {graph.cast(combined, T::kFloat32), gamma}),
                                 T::kBFloat16);
    auto routed = graph.dense_projection(normalized, H, L, 44);
    auto sg = graph.fixed_projection(codec, x, SH, H, 0),
         su = graph.fixed_projection(codec, x, SH, H, 1);
    auto sa = graph.situ(sg, su), shared = graph.fixed_projection(codec, sa, H, SH, 2);
    auto result = graph.binary(c::BinaryKind::kAdd, routed, shared);
    for (auto out : {gate, up, activated, down, combined, normalized, routed, sg, su,
                     sa, shared, result})
      graph.publish(out);
    // The source E4 projection is an independent attention-frontier branch.
    // It is never inserted as a fictional extra FFN stage.
    if (codec == Codec::Quark) {
      std::vector<std::uint8_t> w(128 * H);
      for (std::int64_t row = 0; row < 128; ++row)
        w[std::size_t(row * H + row)] = 0x38;
      auto wf = graph.constant(dense(T::kFloat8E4M3FN, {128, H}), raw(w)),
           sf = graph.constant(dense(T::kFloat32, {1, 128}),
                               raw(std::vector<float>(128, 1.F)));
      auto xf = graph.cast(x, T::kFloat32);
      auto max = graph.one(
          c::Reduce{.kind = c::ReductionKind::kMaximum,
                    .axes = take(c::Axes::from(std::array<std::int32_t, 1>{1})),
                    .keep_dimensions = true,
                    .accumulation_type = T::kFloat32},
          {graph.unary(c::UnaryKind::kAbs, xf)});
      auto scale = graph.binary(c::BinaryKind::kMultiply, max,
                                graph.number(std::bit_cast<float>(0x3b124925U)));
      auto inverse = graph.binary(c::BinaryKind::kDivide, graph.number(1.F), scale);
      auto q = graph.cast(graph.binary(c::BinaryKind::kMultiply, xf, inverse),
                          T::kFloat8E4M3FN);
      auto dot = graph.one(
          c::MatMul{.accumulation_type = T::kFloat32},
          {graph.cast(q, T::kFloat32),
           graph.one(take(c::View::permute(std::array<std::int32_t, 2>{1, 0})),
                     {graph.cast(wf, T::kFloat32)})});
      graph.publish(graph.cast(
          graph.binary(c::BinaryKind::kMultiply,
                       graph.binary(c::BinaryKind::kMultiply, dot, sf), scale),
          T::kBFloat16));
    }
    compiled = graph.compile();
  }
};
struct Episode {
  g::Invocation invocation;
  std::vector<c::ReadLease> outputs;
  std::vector<c::LeasePool> caller_pools;
  std::vector<Bytes> saved;
};
struct Input {
  V id;
  c::TensorDesc description;
  c::ReadLease read;
};
Episode execute(Device &device, const g::CompiledGraph &compiled,
                std::span<const V> outputs, std::span<const Input> inputs,
                const std::function<void(g::Invocation &)> &external = {},
                const std::function<void(g::Invocation &)> &after_prepare = {}) {
  Episode episode;
  episode.invocation = take(compiled.executable.create_invocation());
  check(episode.invocation.prepare_resources());
  for (const auto &input : inputs)
    check(episode.invocation.bind(input.id, input.description, input.read));
  if (external)
    external(episode.invocation);
  std::vector<inf::PreparedOutputFacts> facts;
  for (auto id : outputs) {
    auto f = take(inf::prepare_output(compiled, id));
    facts.push_back(f);
    if (f.caller_owned()) {
      const auto size = take(f.description.required_bytes());
      episode.caller_pools.push_back(
          take(c::LeasePool::create(device.memory->ref(), size, {.max_blocks = 1})));
      check(episode.invocation.bind(id, f.description,
                                    take(episode.caller_pools.back().allocate(size))));
    }
  }
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &device.queue}};
  check(episode.invocation.prepare_replay(
      queues,
      {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kAllow}));
  if (after_prepare)
    after_prepare(episode.invocation);
  {
    auto done = take(episode.invocation.submit(queues));
    check(done.wait());
  }
  for (const auto &f : facts) {
    auto ready = take(inf::retain_output(episode.invocation, f));
    Bytes data(ready.size());
    check(take(ready.view()).read(data));
    episode.saved.push_back(std::move(data));
    episode.outputs.push_back(std::move(ready));
  }
  return episode;
}
template <class X> std::vector<X> values(const Bytes &bytes) {
  require(bytes.size() % sizeof(X) == 0, "typed output bytes");
  std::vector<X> result(bytes.size() / sizeof(X));
  std::memcpy(result.data(), bytes.data(), bytes.size());
  return result;
}
void unchanged(const Episode &episode) {
  for (std::size_t p = 0; p < episode.outputs.size(); ++p) {
    Bytes current(episode.outputs[p].size());
    check(take(episode.outputs[p].view()).read(current));
    require(current == episode.saved[p],
            "old read lease changed after other generation");
  }
}

std::vector<std::uint16_t> activation(unsigned epoch) {
  std::vector<std::uint16_t> x(H);
  constexpr std::array<float, 4> v{1.F, 2.F, -.75F, -1.F};
  for (std::size_t j = 0; j < x.size(); ++j)
    x[j] = bf(v[j % 4]);
  for (std::size_t j = 0; j < 17; ++j)
    x[j] = bf((2.F + float(epoch ? 16 - j : j) / 8.F) / 64.F);
  return x;
}
std::vector<std::int32_t> route_reference(std::span<const std::uint16_t> x) {
  std::vector<std::int32_t> ids(E);
  std::iota(ids.begin(), ids.end(), 0);
  const auto score = [&](std::int32_t id) {
    const auto logit = id < 17 ? f32(double(wide(x[std::size_t(id)])) * 64.) : 0.F;
    const auto sigmoid = f32(1. / (1. + std::exp(-double(logit))));
    return f32(double(sigmoid) + (id < 17 ? double(id % 3 - 1) / 16. : -4.));
  };
  std::stable_sort(ids.begin(), ids.end(),
                   [&](auto a, auto b) { return score(a) > score(b); });
  ids.resize(S);
  return ids;
}
std::vector<std::uint16_t> dense_reference(std::span<const std::uint16_t> x,
                                           std::int64_t n, unsigned seed) {
  std::vector<std::uint16_t> y(std::size_t(n), 0);
  for (std::int64_t row = 0; row < n; ++row)
    y[std::size_t(row)] = x[std::size_t(column(row, std::int64_t(x.size()), seed, 0))];
  return y;
}
float a32_scale(float maximum) {
  if (maximum == 0)
    return std::ldexp(1.F, -127);
  int exponent = 0;
  const auto fraction = std::frexp(double(maximum), &exponent);
  return std::ldexp(1.F, std::clamp(exponent - (fraction >= .875 ? 2 : 3), -127, 127));
}
float a32_round(float v, float maximum) {
  const auto scale = a32_scale(maximum);
  const auto normalized = round_bf(double(v) / double(scale));
  constexpr std::array<float, 8> mag{0, .5F, 1, 1.5F, 2, 3, 4, 6};
  unsigned best = 0;
  double distance = std::abs(double(normalized));
  for (unsigned i = 1; i < mag.size(); ++i) {
    const auto d = std::abs(std::abs(double(normalized)) - mag[i]);
    if (d < distance || (d == distance && !(i & 1U))) {
      best = i;
      distance = d;
    }
  }
  return round_bf(double(std::copysign(mag[best], normalized)) * scale);
}
std::vector<std::uint16_t> projection_reference(Codec codec,
                                                std::span<const std::uint16_t> input,
                                                std::span<const std::int32_t> ids,
                                                std::int64_t n, std::int64_t k,
                                                unsigned leg) {
  require(input.size() == std::size_t(k) || input.size() == ids.size() * std::size_t(k),
          "oracle activation rows");
  std::vector<std::uint16_t> out(ids.size() * std::size_t(n));
  for (std::size_t slot = 0; slot < ids.size(); ++slot) {
    const auto x = input.subspan(
        input.size() == std::size_t(k) ? 0 : slot * std::size_t(k), std::size_t(k));
    const auto group = codec == Codec::Quark ? 32 : 512;
    std::vector<float> maximum(std::size_t(k / group), 0), scales(maximum.size()),
        inverse(maximum.size());
    for (std::size_t col = 0; col < x.size(); ++col)
      maximum[col / std::size_t(group)] =
          std::max(maximum[col / std::size_t(group)], std::abs(wide(x[col])));
    if (codec == Codec::K512)
      for (std::size_t g0 = 0; g0 < maximum.size(); ++g0) {
        scales[g0] = round_bf(double(maximum[g0]) * .00787353515625);
        inverse[g0] = scales[g0] == 0 ? 0 : round_bf(1. / double(scales[g0]));
      }
    for (std::int64_t row = 0; row < n; ++row) {
      const auto col = column(row, k, unsigned(ids[slot]), leg);
      const auto groupid = std::size_t(col / group);
      float result = 0;
      if (codec == Codec::Quark) {
        constexpr std::array<float, 4> mag{0, .5F, 1.F, 1.5F};
        const auto qa = a32_round(wide(x[std::size_t(col)]), maximum[groupid]);
        const auto qw = round_bf(double(mag[code(unsigned(ids[slot]), leg)]) * .25);
        result = f32(0. + double(qa) * qw);
      } else {
        const auto v = round_bf(double(wide(x[std::size_t(col)])) * inverse[groupid]);
        require(std::isfinite(v) && std::trunc(v) >= -128 && std::trunc(v) <= 127,
                "generated source entered undefined original I8 domain");
        const auto dot = round_bf(double(std::int32_t(std::trunc(v))) *
                                  code(unsigned(ids[slot]), leg));
        const auto first = round_bf(double(dot) * scales[groupid]);
        const auto sf = round_bf(coefficient(unsigned(ids[slot]), leg));
        result = round_bf(double(first) * sf);
      }
      out[slot * std::size_t(n) + std::size_t(row)] = bf(result);
    }
  }
  return out;
}
// Independent scalar FP8 table/RNE writer copied verbatim from the existing
// static_fp8_reference_fixture.h oracle; no product codec is used.
inline float e4m3(std::uint8_t code) {
  const auto magnitude = unsigned(code & 127u), exponent = magnitude >> 3,
             fraction = magnitude & 7u;
  if (magnitude == 127u)
    return std::numeric_limits<float>::quiet_NaN();
  const auto value = exponent == 0
                         ? std::ldexp(float(fraction), -9)
                         : std::ldexp(1.F + float(fraction) / 8.F, int(exponent) - 7);
  return code & 128u ? -value : value;
}
inline std::uint8_t encode_e4m3(float value) {
  const auto sign = std::signbit(value) ? 128u : 0u;
  if (std::isnan(value))
    return static_cast<std::uint8_t>(sign | 127u);
  const auto magnitude = std::abs(value);
  if (magnitude >= 448.F)
    return static_cast<std::uint8_t>(sign | 126u);
  unsigned below = 0;
  while (below < 126u && e4m3(static_cast<std::uint8_t>(below + 1)) <= magnitude)
    ++below;
  if (below < 126u) {
    const double low =
        double(magnitude) - double(e4m3(static_cast<std::uint8_t>(below)));
    const double high =
        double(e4m3(static_cast<std::uint8_t>(below + 1))) - double(magnitude);
    if (high < low || (high == low && (below & 1u)))
      ++below;
  }
  return static_cast<std::uint8_t>(sign | below);
}

std::size_t situ_witnesses = 0, exact_values = 0, nonlinear_values = 0;
double maximum_error = 0;
void exact(const Bytes &actual, std::span<const std::uint16_t> expected,
           const char *where) {
  const auto v = values<std::uint16_t>(actual);
  require(v.size() == expected.size(), "oracle output extent");
  exact_values += v.size();
  if (!std::equal(v.begin(), v.end(), expected.begin())) {
    std::size_t first = v.size(), different = 0, signed_zero = 0;
    double max_abs = 0;
    for (std::size_t j = 0; j < v.size(); ++j)
      if (v[j] != expected[j]) {
        if (first == v.size())
          first = j;
        ++different;
        signed_zero += ((v[j] & 0x7fffU) == 0 && (expected[j] & 0x7fffU) == 0);
        max_abs = std::max(max_abs, std::abs(double(wide(v[j])) - wide(expected[j])));
      }
    std::cerr << "ORACLE_FIRST_DIFFERENCE where=" << where << " index=" << first
              << " actual_bits=" << v[first] << " expected_bits=" << expected[first]
              << " differing_values=" << different
              << " signed_zero_only=" << signed_zero
              << " maximum_absolute_difference=" << max_abs << '\n';
    throw std::runtime_error(std::string(where) +
                             " exact BF16 sparse projection mismatch");
  }
}
void nonlinear(const Bytes &actual, std::span<const float> expected,
               const char *where) {
  const auto v = values<std::uint16_t>(actual);
  require(v.size() == expected.size(), "nonlinear output extent");
  nonlinear_values += v.size();
  for (std::size_t j = 0; j < v.size(); ++j) {
    const auto error = std::abs(double(wide(v[j])) - expected[j]);
    maximum_error = std::max(maximum_error, error);
    const auto tolerance = 8.e-5 + (.00390625 + 4.e-5) * std::abs(double(expected[j]));
    if (!(error <= tolerance))
      throw std::runtime_error(
          std::string(where) +
          " outside independent F32/transcendental plus BF16 interval");
  }
}
void situ_check(const Bytes &actual, const Bytes &gate, const Bytes &up) {
  auto a = values<std::uint16_t>(gate), b = values<std::uint16_t>(up);
  std::vector<float> reference(a.size());
  for (std::size_t j = 0; j < a.size(); ++j) {
    const auto x = wide(a[j]), z = wide(b[j]);
    const auto sigmoid = f32(1. / (1. + std::exp(-double(x))));
    const auto g0 = f32(4. * f32(std::tanh(double(f32(double(x) * .25)))));
    const auto g1 = f32(double(g0) * sigmoid);
    const auto u0 = f32(25. * f32(std::tanh(double(f32(double(z) * double(.04F))))));
    reference[j] = f32(double(g1) * u0);
    const auto wrong =
        bf(round_bf(double(round_bf(double(g0))) * round_bf(sigmoid)) * round_bf(u0));
    situ_witnesses += wrong != bf(reference[j]);
  }
  nonlinear(actual, reference, "SiTU F32 then BF16");
}
void check_outputs(Codec codec, const Episode &episode, const Episode &router,
                   std::span<const std::uint16_t> x,
                   std::span<const std::int32_t> ids) {
  auto latent = values<std::uint16_t>(router.saved[2]);
  exact(router.saved[2], dense_reference(x, L, 33), "latent down");
  exact(episode.saved[0], projection_reference(codec, latent, ids, I, L, 0),
        "expert gate");
  exact(episode.saved[1], projection_reference(codec, latent, ids, I, L, 1),
        "expert up");
  situ_check(episode.saved[2], episode.saved[0], episode.saved[1]);
  exact(episode.saved[3],
        projection_reference(codec, values<std::uint16_t>(episode.saved[2]), ids, L, I,
                             2),
        "expert down");
  auto down = values<std::uint16_t>(episode.saved[3]);
  auto weights = values<float>(router.saved[1]);
  std::vector<float> combined(L);
  for (std::size_t row = 0; row < std::size_t(L); ++row) {
    double sum = 0;
    for (std::size_t slot = 0; slot < std::size_t(S); ++slot)
      sum += f32(double(wide(down[slot * std::size_t(L) + row])) * weights[slot]);
    combined[row] = f32(sum);
  }
  nonlinear(episode.saved[4], combined, "weighted combine");
  auto actual_combined = values<std::uint16_t>(episode.saved[4]);
  double square = 0;
  for (auto v : actual_combined)
    square += double(wide(v)) * wide(v);
  std::vector<float> norm(L);
  const auto denominator = std::sqrt(square / double(L) + 1.e-5);
  for (std::size_t row = 0; row < norm.size(); ++row)
    norm[row] = f32(wide(actual_combined[row]) / denominator);
  nonlinear(episode.saved[5], norm, "latent RMSNorm");
  exact(episode.saved[6],
        dense_reference(values<std::uint16_t>(episode.saved[5]), H, 44), "latent up");
  const std::array<std::int32_t, 1> zero{0};
  for (unsigned leg = 0; leg < 2; ++leg) {
    const auto expected = codec == Codec::Quark
                              ? projection_reference(codec, x, zero, SH, H, leg)
                              : dense_reference(x, SH, 100 + leg);
    exact(episode.saved[7 + leg], expected, "shared gate/up");
  }
  situ_check(episode.saved[9], episode.saved[7], episode.saved[8]);
  exact(episode.saved[10],
        codec == Codec::Quark
            ? projection_reference(codec, values<std::uint16_t>(episode.saved[9]), zero,
                                   H, SH, 2)
            : dense_reference(values<std::uint16_t>(episode.saved[9]), H, 102),
        "shared down");
  auto routed = values<std::uint16_t>(episode.saved[6]),
       shared = values<std::uint16_t>(episode.saved[10]);
  std::vector<std::uint16_t> final(H);
  for (std::size_t row = 0; row < final.size(); ++row)
    final[row] = bf(f32(double(wide(routed[row])) + wide(shared[row])));
  exact(episode.saved[11], final, "ordinary final add");
  if (codec == Codec::Quark) {
    float maximum = 0;
    for (auto v : x)
      maximum = std::max(maximum, std::abs(wide(v)));
    const auto scale = f32(double(maximum) * std::bit_cast<float>(0x3b124925U));
    const auto inverse = f32(1. / double(scale));
    std::vector<std::uint16_t> expected(128);
    for (std::size_t row = 0; row < 128; ++row) {
      const auto q = e4m3(encode_e4m3(f32(double(wide(x[row])) * inverse)));
      expected[row] = bf(f32(double(f32(double(q) * 1.F)) * scale));
    }
    exact(episode.saved[12], expected, "independent E4 F32 outer-scale projection");
  }
}
std::array<Input, 4> expert_inputs(const Expert &expert, const c::ReadLease &x,
                                   const Episode &route) {
  return {{{expert.x, dense(T::kBFloat16, {1, H}), x},
           {expert.ids, dense(T::kInt32, {1, S}), route.outputs[0]},
           {expert.weights, dense(T::kFloat32, {1, S}), route.outputs[1]},
           {expert.latent, dense(T::kBFloat16, {1, L}), route.outputs[2]}}};
}
void run(Codec codec) {
  Device device;
  const auto planes = expert_planes(codec);
  const auto layout = take(supply::Layout::create(planes, E, S));
  require(layout.logical_bytes == (codec == Codec::Quark ? 280759808u : 268373504u),
          "actual seven-bank bounded extent");
  Router router(device);
  Expert expert(device, codec, layout);
  auto resident = take(
      supply::CudaSupply::create(device.context, device.owner, *device.memory, layout));
  std::array<Episode, 2> routes, episodes;
  std::array<c::ReadLease, 2> xready;
  std::array<c::cuda::WeightBatch, 2> batches;
  std::array<Selection, 2> selections;
  std::array<std::unique_ptr<Sources>, 2> sources;
  std::vector<c::WeightUploadReport> reports;
  const auto start_exact = exact_values, start_nonlinear = nonlinear_values;
  std::vector<std::vector<std::int32_t>> actual_ids;
  for (unsigned epoch = 0; epoch < 2; ++epoch) {
    const auto x = activation(epoch);
    xready[epoch] = device.upload(dense(T::kBFloat16, {1, H}), raw(x));
    const std::array inputs{
        Input{router.x, dense(T::kBFloat16, {1, H}), xready[epoch]}};
    routes[epoch] = execute(device, router.compiled, router.graph.outputs, inputs);
    const auto ids = values<std::int32_t>(routes[epoch].saved[0]);
    require(ids == route_reference(x),
            "actual generated router Top16 differs from scalar route");
    actual_ids.push_back(ids);
    const auto actual_weights = values<float>(routes[epoch].saved[1]);
    std::vector<float> reference_weights;
    double denominator = 0;
    for (auto id : ids) {
      const auto logit = f32(double(wide(x[std::size_t(id)])) * 64.);
      const auto probability = f32(1. / (1. + std::exp(-double(logit))));
      reference_weights.push_back(probability);
      denominator += probability;
    }
    for (std::size_t j = 0; j < reference_weights.size(); ++j)
      require(std::abs(double(actual_weights[j]) -
                       f32(double(reference_weights[j]) / f32(denominator))) < 2.e-7,
              "true route weights differ from normalized original sigmoid");
    selections[epoch] = select(ids);
    sources[epoch] = std::make_unique<Sources>(codec, selections[epoch]);
    batches[epoch] = take(
        resident.upload(epoch, selections[epoch], *sources[epoch], device.transfer));
    const auto input = expert_inputs(expert, xready[epoch], routes[epoch]);
    episodes[epoch] = execute(
        device, expert.compiled, expert.graph.outputs, input, [&](auto &invocation) {
          check(g::cuda::bind(invocation, expert.bindings, batches[epoch]));
        });
    check_outputs(codec, episodes[epoch], routes[epoch], x, ids);
    check(batches[epoch].wait());
    reports.push_back(take(batches[epoch].report()));
    require(reports.back().logical_bytes == layout.logical_bytes &&
                reports.back().staged_bytes == layout.logical_bytes &&
                reports.back().completed_bytes == reports.back().submitted_bytes &&
                !reports.back().copied_bytes_unknown,
            "real seven-bank upload accounting");
  }
  require(actual_ids[0] != actual_ids[1],
          "two genuine Top16 generations did not change");
  unchanged(episodes[0]);
  unchanged(episodes[1]);
  require(!resident.upload(0, selections[0], *sources[0], device.transfer),
          "old immutable batch slot reused while still retained");
  const auto input = expert_inputs(expert, xready[0], routes[0]);
  auto probe = execute(
      device, expert.compiled, expert.graph.outputs, input,
      [&](auto &invocation) {
        check(g::cuda::bind(invocation, expert.bindings, batches[0]));
      },
      [&](auto &invocation) {
        auto bad = expert.bindings;
        bad.back().id = 999999;
        require(!g::cuda::bind(invocation, bad, batches[1]).ok(),
                "missing final bank binding accepted");
      });
  require(probe.saved == episodes[0].saved,
          "rejected final bank replaced part of original bindings/preparation");
  unchanged(episodes[0]);
  unchanged(episodes[1]);
  std::cout
      << "{\"codec\":\"" << (codec == Codec::Quark ? "QuarkA32" : "K512")
      << "\",\"status\":\"PASS_GENERATED_FULL_AXES_COMPONENT\",\"hidden\":7168,\"global_experts\":896,\"top_k\":16,\"latent\":3584,\"intermediate\":3072,\"shared\":6144,\"banks\":7,\"upload_generations\":2,\"graph_submissions\":5,\"logical_bytes_per_slot\":"
      << layout.logical_bytes
      << ",\"old_reader_unchanged\":true,\"rejected_last_binding_preserved_and_executed\":true,\"publisher_payload\":false,\"source_plane_sha256\":[";
  bool first = true;
  for (const auto &source : sources)
    for (const auto &hash : source->hashes) {
      if (!first)
        std::cout << ',';
      first = false;
      std::cout << '"' << hash << '"';
    }
  std::cout << "],\"exact_BF16_values\":" << (exact_values - start_exact)
            << ",\"nonlinear_BF16_interval_values\":"
            << (nonlinear_values - start_nonlinear) << ",\"actual_global_ids\":[";
  for (unsigned epoch = 0; epoch < 2; ++epoch) {
    if (epoch)
      std::cout << ',';
    std::cout << '[';
    for (std::size_t j = 0; j < actual_ids[epoch].size(); ++j) {
      if (j)
        std::cout << ',';
      std::cout << actual_ids[epoch][j];
    }
    std::cout << ']';
  }
  std::cout << "],\"output_SHA256\":[";
  for (unsigned epoch = 0; epoch < 2; ++epoch) {
    if (epoch)
      std::cout << ',';
    std::cout << '[';
    for (std::size_t j = 0; j < episodes[epoch].saved.size(); ++j) {
      if (j)
        std::cout << ',';
      std::cout << '"' << inf::artifact::sha256(episodes[epoch].saved[j]) << '"';
    }
    std::cout << ']';
  }
  std::cout << "],\"upload_reports\":[";
  for (std::size_t j = 0; j < reports.size(); ++j) {
    if (j)
      std::cout << ',';
    const auto &r = reports[j];
    std::cout << "{\"logical_bytes\":" << r.logical_bytes
              << ",\"staged_bytes\":" << r.staged_bytes
              << ",\"submitted_bytes\":" << r.submitted_bytes
              << ",\"completed_bytes\":" << r.completed_bytes
              << ",\"copied_bytes_unknown\":"
              << (r.copied_bytes_unknown ? "true" : "false") << '}';
  }
  std::cout << "]}\n";
  probe = Episode{};
  episodes = {};
  routes = {};
  batches = {};
  check(resident.close());
}

} // namespace

TEST(CudaKimiMixedWorkingSet, OriginalFormatsKeepTwoGenerationsAndAtomicBanks) {
  const auto devices = c::cuda::backend().device_count();
  if (!devices) {
    const auto native = devices.error().native_error();
    if (native.domain == hyle::NativeErrorDomain::CudaRuntime &&
        native.code == static_cast<std::int64_t>(cudaErrorNoDevice))
      GTEST_SKIP() << "CUDA runtime reports no device";
    FAIL() << devices.error().message();
  }
  if (*devices == 0)
    GTEST_SKIP() << "CUDA runtime reports no device";
  EXPECT_NO_THROW({
    run(Codec::Quark);
    run(Codec::K512);
    require(situ_witnesses > 0, "fixture did not discriminate BF16-intermediate SiTU");
  });
}
