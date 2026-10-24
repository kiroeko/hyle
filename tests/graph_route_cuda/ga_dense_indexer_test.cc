#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/graph/cuda.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
using BF16 = hyle::simd::bfloat16_t;
using F16 = hyle::simd::float16_t;
constexpr c::RoutePolicyId kScorePolicy{0x48594c454e445343ULL, 1, 1};
enum class Escape { kNone, kPublishedDot, kProductView, kControl };

c::TensorDesc dense(c::DataType type,
                    std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}
c::View reshape(std::initializer_list<std::int64_t> shape) {
  return c::View::reshape(c::TensorShape::create(
      {shape.begin(), shape.size()}).value()).value();
}
struct Shape final {
  std::int64_t b = 1, t = 8, h = 3, d = 32, l = 1, pitch = 1, gap_bytes = 0;
  bool masked = true, widen = true;
  std::int64_t mask_batches = 1;
};
struct Logical final {
  Shape shape;
  g::Graph graph;
  std::array<g::ValueId, 4> inputs{};
  std::array<c::TensorDesc, 4> descriptions{};
  g::ValueId kt{}, wr{}, dot{}, relu{}, product{}, sum{}, score{}, extra{};
  std::array<g::NodeId, 4> math{};
};
Logical build(Shape s, Escape escape = Escape::kNone, bool publish_query = false) {
  Logical f;
  f.shape = s;
  f.descriptions[0] = dense(c::DataType::kBFloat16, {s.b, s.t, s.h, s.d});
  f.descriptions[1] = c::TensorDesc::strided_bytes(c::DataType::kBFloat16,
      std::array{s.b, s.l, s.d}, std::array{s.pitch * s.d * 2 + s.gap_bytes, s.d * 2, std::int64_t{2}}).value();
  f.descriptions[2] = dense(c::DataType::kBFloat16, {s.b, s.t, s.h});
  f.descriptions[3] = dense(c::DataType::kFloat16, {s.mask_batches, s.t, s.l});
  auto builder = g::GraphBuilder::create().value();
  for (std::size_t i = 0; i < (s.masked ? 4U : 3U); ++i)
    f.inputs[i] = builder.input(f.descriptions[i]).value();
  const auto kr = builder.add(reshape({s.b, 1, s.l, s.d}), f.inputs[1]).value();
  f.kt = builder.add(c::View::permute(std::array<std::int32_t, 4>{0, 1, 3, 2}).value(), kr).value();
  f.dot = builder.add(c::MatMul{.accumulation_type = c::DataType::kFloat32}, f.inputs[0], f.kt).value();
  const auto relu_node = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kRelu}),
      std::array{f.dot}).value();
  f.relu = builder.result(relu_node).value();
  f.wr = builder.add(reshape({s.b, s.t, s.h, 1}), f.inputs[2]).value();
  f.product = builder.add(c::Binary{c::BinaryKind::kMultiply}, f.relu, f.wr).value();
  f.sum = builder.add(c::Reduce{.kind = c::ReductionKind::kSum,
      .axes = c::Axes::from(std::array<std::int32_t, 1>{2}).value(),
      .keep_dimensions = false, .accumulation_type = c::DataType::kFloat32}, f.product).value();
  f.score = f.sum;
  if (s.masked) {
    const auto mask = builder.add(c::Cast{c::DataType::kBFloat16}, f.inputs[3]).value();
    f.score = builder.add(c::Binary{c::BinaryKind::kAdd}, f.score, mask).value();
  }
  if (s.widen) f.score = builder.add(c::Cast{c::DataType::kFloat32}, f.score).value();
  if (publish_query) EXPECT_TRUE(builder.publish(f.inputs[0]).ok());
  if (escape == Escape::kPublishedDot) f.extra = f.dot;
  if (escape == Escape::kProductView) {
    const auto view = builder.add(reshape({s.b * s.t * s.h * s.l}), f.product).value();
    f.extra = builder.add(c::Copy{}, view).value();
  }
  if (escape == Escape::kControl) {
    const auto observer = builder.add_node(c::Operation(c::Copy{}), std::array{f.inputs[0]}).value();
    f.extra = builder.result(observer).value();
    EXPECT_TRUE(builder.depends_on(observer, relu_node).ok());
  }
  if (f.extra.valid()) EXPECT_TRUE(builder.publish(f.extra).ok());
  EXPECT_TRUE(builder.publish(f.score).ok());
  f.graph = std::move(builder).finish().value();
  f.math = {f.graph.producer(f.dot).value(), f.graph.producer(f.relu).value(),
            f.graph.producer(f.product).value(), f.graph.producer(f.sum).value()};
  return f;
}

struct Tensor final {
  m::Buffer owner;
  c::TensorView view;
  Tensor() = default;
  Tensor(c::TensorDesc desc, m::MemoryResourceRef resource)
      : owner(m::Buffer::create(resource, desc.required_bytes().value(), 256).value()),
        view(c::TensorView::create(desc, owner.view()).value()) {}
  template <class T> void write(const std::vector<T> &values) {
    ASSERT_EQ(values.size() * sizeof(T), view.storage().size());
    ASSERT_TRUE(view.storage().write(std::as_bytes(std::span(values))).ok());
  }
};
template <class T> std::vector<T> read(c::ConstTensorView view) {
  std::vector<T> result(view.storage().size() / sizeof(T));
  EXPECT_TRUE(view.storage().read(std::as_writable_bytes(std::span(result))).ok());
  return result;
}
std::vector<float> score_values(c::ConstTensorView view) {
  if (view.description().data_type() == c::DataType::kFloat32) return read<float>(view);
  const auto values = read<BF16>(view);
  std::vector<float> result;
  for (auto x : values) result.push_back(float(half_test::decode(x)));
  return result;
}
float rne(double x) { return float(half_test::decode(half_test::encode<BF16>(x))); }
struct Values final {
  std::vector<BF16> q, k, w, dot, product;
  std::vector<F16> mask;
  std::vector<float> expected;
};
Values values(const Logical &f, unsigned seed, bool witness = false) {
  const auto s = f.shape;
  Values v;
  v.q.resize(std::size_t(s.b * s.t * s.h * s.d));
  v.k.resize(f.descriptions[1].required_bytes().value() / 2,
             half_test::encode<BF16>(31.0)); // Poison padding is not a key row.
  v.w.resize(std::size_t(s.b * s.t * s.h));
  v.mask.resize(std::size_t(s.mask_batches * s.t * s.l));
  for (std::int64_t b = 0; b < s.b; ++b) {
    for (std::int64_t l = 0; l < s.l; ++l)
      for (std::int64_t d = 0; d < s.d; ++d) {
        const double key = witness ? (d == 0 ? 1.0 : d == 1 ? 1.0 / 256 : 0)
            : d == 0 ? (int(l % 7) - 3 + int(b) + int(seed)) * 0.125
            : d == 1 ? double(l % 3 + 1) * 0.25 : 0;
        v.k[std::size_t(b * (s.pitch * s.d + s.gap_bytes / 2) + l * s.d + d)] = half_test::encode<BF16>(key);
      }
    for (std::int64_t t = 0; t < s.t; ++t)
      for (std::int64_t h = 0; h < s.h; ++h) {
        double weight = h == 0 ? 1.0 : h == 1 ? -0.5 : 0.25;
        if (witness) {
          weight = h >= 2 ? 0 : t == 0 ? (h == 0 ? 1 : -1)
              : t == 1 ? (h == 0 ? 1.0078125 : -1.015625)
              : t == 3 ? (h == 0 ? -1 : 0) : 1;
        }
        v.w[std::size_t((b * s.t + t) * s.h + h)] = half_test::encode<BF16>(weight);
        for (std::int64_t d = 0; d < s.d; ++d) {
          double q = d == 0 ? double(h + 1 + b + t + seed) * 0.25 : d == 1 ? 0.5 : 0;
          if (witness) {
            q = 0;
            if (h < 2 && d < 2) {
              if (t == 0) q = d == 0 ? 1 : h == 0 ? 1 : 0;
              else if (t == 1) q = d == 0 ? 1 : h == 0 ? 2 : 0;
              else if (t == 2) q = h == d ? 1 : 0;
              else q = h == 0 && d == 0 ? 1 : 0;
            }
            if (h == 0 && d == 0 && t == 4) q = std::numeric_limits<double>::quiet_NaN();
            if (h == 0 && d == 0 && t == 5) q = std::numeric_limits<double>::infinity();
            if (h == 0 && d == 0 && t == 6) q = -std::numeric_limits<double>::infinity();
          }
          v.q[std::size_t(((b * s.t + t) * s.h + h) * s.d + d)] = half_test::encode<BF16>(q);
        }
      }
  }
  if (!witness)
    for (std::size_t i = 0; i < v.mask.size(); ++i)
      v.mask[i] = half_test::encode<F16>((int(i % 11) + int(seed)) * 0.03125);
  if (witness && s.masked)
    v.mask[3] = half_test::encode<F16>(1.0 + 1.0 / 1024);
  v.dot.resize(std::size_t(s.b * s.t * s.h * s.l));
  v.product.resize(v.dot.size());
  for (std::int64_t b = 0; b < s.b; ++b)
    for (std::int64_t t = 0; t < s.t; ++t)
      for (std::int64_t l = 0; l < s.l; ++l) {
        double total = 0;
        for (std::int64_t h = 0; h < s.h; ++h) {
          double dot = 0;
          for (std::int64_t d = 0; d < s.d; ++d)
            dot += half_test::decode(v.q[std::size_t(((b * s.t + t) * s.h + h) * s.d + d)]) *
                half_test::decode(v.k[std::size_t(b * (s.pitch * s.d + s.gap_bytes / 2) + l * s.d + d)]);
          const auto i = std::size_t(((b * s.t + t) * s.h + h) * s.l + l);
          v.dot[i] = half_test::encode<BF16>(dot);
          const float rounded = float(half_test::decode(v.dot[i]));
          const float relu = std::isnan(rounded) ? rounded : rounded > 0 ? rounded : 0;
          v.product[i] = half_test::encode<BF16>(double(relu) *
              half_test::decode(v.w[std::size_t((b * s.t + t) * s.h + h)]));
          total += half_test::decode(v.product[i]);
        }
        float result = rne(total);
        if (s.masked) {
          const auto mb = s.mask_batches == 1 ? 0 : b;
          result = rne(double(result) + rne(half_test::decode(v.mask[std::size_t((mb * s.t + t) * s.l + l)])));
        }
        v.expected.push_back(result);
      }
  return v;
}
void expect_scores(std::span<const float> actual, std::span<const float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (std::isnan(expected[i])) EXPECT_TRUE(std::isnan(actual[i])) << i;
    else EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),
                   std::bit_cast<std::uint32_t>(expected[i])) << i;
}
struct Bindings final {
  Values host;
  std::array<Tensor, 4> input;
  Tensor output, extra;
};
Bindings allocate(const Logical &f, m::MemoryResourceRef resource, unsigned seed,
             bool witness = false, bool upload = true) {
  Bindings r;
  r.host = values(f, seed, witness);
  for (std::size_t i = 0; i < (f.shape.masked ? 4U : 3U); ++i)
    r.input[i] = Tensor(f.descriptions[i], resource);
  if (upload) {
    r.input[0].write(r.host.q); r.input[1].write(r.host.k); r.input[2].write(r.host.w);
    if (f.shape.masked) r.input[3].write(r.host.mask);
  }
  r.output = Tensor(f.graph.value_description(f.score).value(), resource);
  if (f.extra.valid()) r.extra = Tensor(f.graph.value_description(f.extra).value(), resource);
  return r;
}
void bind(const Logical &f, Bindings &r, g::Invocation &invocation) {
  for (std::size_t i = 0; i < (f.shape.masked ? 4U : 3U); ++i)
    ASSERT_TRUE(invocation.bind(f.inputs[i], r.input[i].view.as_const()).ok());
  ASSERT_TRUE(invocation.bind(f.score, r.output.view).ok());
  if (f.extra.valid()) ASSERT_TRUE(invocation.bind(f.extra, r.extra.view).ok());
}
void selected(const Logical &f, const g::CompileReport &report, bool expected) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < report.step_count(); ++i) {
    const auto step = report.step(i);
    ASSERT_TRUE(step) << step.error().message();
    if (step->policy != kScorePolicy) continue;
    ++count;
    EXPECT_EQ(step->kind, g::CompileStepKind::kProviderRegion);
    EXPECT_EQ(step->provider.id, c::cuda_native::kProviderId);
    EXPECT_EQ(step->implementation, c::ProviderImplementationId{1});
    for (const auto node : f.math) {
      bool found = false;
      for (std::size_t n = 0; n < step->node_count; ++n)
        found |= report.step_node(i, n).value() == node;
      EXPECT_TRUE(found) << node.ordinal();
    }
  }
  EXPECT_EQ(count, expected ? 1U : 0U);
}
void submit(g::Invocation &slot, c::Queue &queue) {
  auto done = slot.submit(queue);
  ASSERT_TRUE(done) << done.error().message();
  const auto status = done->wait();
  ASSERT_TRUE(status.ok()) << status.message();
}
void stage(c::Context &context, c::Queue &queue, c::Operation op,
           std::span<const c::ConstTensorView> inputs, c::TensorView output) {
  auto prepared = c::prepare(context, op, inputs, std::span(&output, 1));
  ASSERT_TRUE(prepared) << prepared.error().message();
  const auto need = prepared->workspace_requirement();
  auto workspace = m::Buffer::create(output.storage().resource(), need.size, need.alignment).value();
  auto slot = prepared->create_invocation(workspace.view()).value();
  auto done = slot.submit(queue, inputs, std::span(&output, 1));
  ASSERT_TRUE(done) << done.error().message();
  const auto status = done->wait();
  ASSERT_TRUE(status.ok()) << status.message();
}
void ordinary_stages(const Logical &f, Bindings &r, c::Context &context, c::Queue &queue,
                     m::MemoryResourceRef resource) {
  Tensor dot(f.graph.value_description(f.dot).value(), resource),
      relu(f.graph.value_description(f.relu).value(), resource),
      product(f.graph.value_description(f.product).value(), resource),
      sum(f.graph.value_description(f.sum).value(), resource);
  const auto kt = c::ConstTensorView::create(f.graph.value_description(f.kt).value(),
      r.input[1].view.storage()).value();
  const auto wr = c::ConstTensorView::create(f.graph.value_description(f.wr).value(),
      r.input[2].view.storage()).value();
  stage(context, queue, c::Operation(c::MatMul{.accumulation_type = c::DataType::kFloat32}),
      std::array{r.input[0].view.as_const(), kt}, dot.view);
  stage(context, queue, c::Operation(c::Unary{c::UnaryKind::kRelu}), std::array{dot.view.as_const()}, relu.view);
  stage(context, queue, c::Operation(c::Binary{c::BinaryKind::kMultiply}), std::array{relu.view.as_const(), wr}, product.view);
  stage(context, queue, c::Operation(c::Reduce{.kind = c::ReductionKind::kSum,
      .axes = c::Axes::from(std::array<std::int32_t, 1>{2}).value(),
      .keep_dimensions = false, .accumulation_type = c::DataType::kFloat32}),
      std::array{product.view.as_const()}, sum.view);
  Tensor masked;
  auto final = sum.view.as_const();
  if (f.shape.masked) {
    Tensor mask(dense(c::DataType::kBFloat16, {f.shape.mask_batches, f.shape.t, f.shape.l}), resource);
    masked = Tensor(f.graph.value_description(f.sum).value(), resource);
    stage(context, queue, c::Operation(c::Cast{c::DataType::kBFloat16}), std::array{r.input[3].view.as_const()}, mask.view);
    stage(context, queue, c::Operation(c::Binary{c::BinaryKind::kAdd}), std::array{final, mask.view.as_const()}, masked.view);
    final = masked.view.as_const();
  }
  Tensor widened;
  if (f.shape.widen) {
    widened = Tensor(f.graph.value_description(f.score).value(), resource);
    stage(context, queue, c::Operation(c::Cast{c::DataType::kFloat32}), std::array{final}, widened.view);
    final = widened.view.as_const();
  }
  expect_scores(score_values(final), r.host.expected);
}

TEST(CudaDenseIndexer, DyadicStagesAndMaskKeepThreeBfloatRoundsAndNonfiniteValues) {
  auto context = g::cuda::create_context().value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  const auto f = build({});
  auto compiled = g::cuda::compile(f.graph, context, memory.ref());
  ASSERT_TRUE(compiled) << compiled.error().message();
  selected(f, compiled->report, true);
  auto r = allocate(f, memory.ref(), 0, true);
  ASSERT_EQ(r.host.expected[0], 0.F); // dot RNE: (1+2^-8)-1 -> 0.
  ASSERT_EQ(r.host.expected[1], 0.F); // product RNE: (1+2^-7)^2-(1+2^-6) -> 0.
  ASSERT_EQ(r.host.expected[2], 1.F); // head-sum RNE: 1+2^-8 -> 1.
  ASSERT_EQ(r.host.expected[3], 0.F); // mask first rounds 1+2^-10 to BF16 one.
  ASSERT_TRUE(std::isnan(r.host.expected[4]));
  ASSERT_EQ(r.host.expected[5], std::numeric_limits<float>::infinity());
  ASSERT_EQ(r.host.expected[6], 0.F);
  auto queue = context.create_queue().value();
  ordinary_stages(f, r, context, queue, memory.ref());
  auto run = compiled->executable.create_invocation().value();
  bind(f, r, run); submit(run, queue);
  expect_scores(score_values(r.output.view.as_const()), r.host.expected);
}

TEST(CudaDenseIndexer, PaddedBatchKeysKeepIndependentSlotsAndRetainedResults) {
  auto context = g::cuda::create_context().value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  for (const bool masked : {false, true}) {
    const auto f = build({.b = 2, .t = 2, .h = 3, .d = 32, .l = 257, .pitch = 260, .gap_bytes = 2,
                          .masked = masked, .widen = masked, .mask_batches = masked ? 2 : 1},
                         Escape::kNone, true);
    auto compiled = g::cuda::compile(f.graph, context, memory.ref());
    ASSERT_TRUE(compiled) << compiled.error().message();
    selected(f, compiled->report, true);
    std::array<std::array<Bindings, 2>, 2> runs;
    std::array queues{context.create_queue().value(), context.create_queue().value()};
    std::array slots{compiled->executable.create_invocation().value(), compiled->executable.create_invocation().value()};
    g::GraphOutput old;
    for (unsigned round = 0; round < 2; ++round) {
      if (!round) {
        // Install both independent mutable slots before either submission.
        for (unsigned lane = 0; lane < 2; ++lane) {
          auto &r = runs[0][lane]; r = allocate(f, memory.ref(), 1 + lane);
          bind(f, r, slots[lane]);
          const auto prepared = g::cuda::prepare_replay(slots[lane], queues[lane],
              {.mode = g::ReplayMode::kRequireCapture});
          ASSERT_TRUE(prepared.ok()) << prepared.message();
          EXPECT_EQ(slots[lane].replay_route(), g::ReplayRoute::kCaptured);
        }
        std::array<c::Completion, 2> done;
        for (unsigned lane = 0; lane < 2; ++lane) {
          auto started = slots[lane].submit(queues[lane]);
          ASSERT_TRUE(started) << started.error().message();
          done[lane] = std::move(*started);
        }
        ASSERT_TRUE(done[1].wait().ok()); ASSERT_TRUE(done[0].wait().ok());
        for (unsigned lane = 0; lane < 2; ++lane)
          expect_scores(score_values(runs[0][lane].output.view.as_const()), runs[0][lane].host.expected);
        old = slots[0].result(f.score).value(); // A real slot-retaining GraphOutput.
        EXPECT_TRUE(slots[0].reset().IsUnavailable());
        EXPECT_TRUE(slots[0].bind(f.inputs[0], runs[0][0].input[0].view.as_const()).IsUnavailable());
        continue;
      }
      for (const unsigned lane : {1U, 0U}) {
        if (lane == 0) old.reset();
        ASSERT_TRUE(slots[lane].reset().ok());
        auto &r = runs[round][lane]; r = allocate(f, memory.ref(), 1 + round * 2 + lane);
        EXPECT_NE(r.input[0].owner.native_handle(), runs[0][lane].input[0].owner.native_handle());
        EXPECT_NE(r.output.owner.native_handle(), runs[0][lane].output.owner.native_handle());
        bind(f, r, slots[lane]);
        const auto prepared = g::cuda::prepare_replay(slots[lane], queues[lane],
            {.mode = g::ReplayMode::kRequireCapture});
        ASSERT_TRUE(prepared.ok()) << prepared.message();
        EXPECT_EQ(slots[lane].replay_route(), g::ReplayRoute::kCaptured);
        submit(slots[lane], queues[lane]);
        expect_scores(score_values(r.output.view.as_const()), r.host.expected);
        expect_scores(score_values(runs[0][lane].output.view.as_const()), runs[0][lane].host.expected);
        if (lane == 1) {
          ASSERT_TRUE(old.valid());
          expect_scores(score_values(old.tensor().value()), runs[0][0].host.expected);
        }
      }
    }
  }
}

TEST(CudaDenseIndexer, EscapingCoreIntermediatesKeepOrdinaryBoundaries) {
  auto context = g::cuda::create_context().value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  for (const auto escape : {Escape::kPublishedDot, Escape::kProductView, Escape::kControl}) {
    const auto f = build({.b = 1, .t = 2, .h = 3, .d = 32, .l = 4, .pitch = 4,
                          .masked = false, .widen = true}, escape);
    auto compiled = g::cuda::compile(f.graph, context, memory.ref());
    ASSERT_TRUE(compiled) << compiled.error().message();
    selected(f, compiled->report, false);
    auto r = allocate(f, memory.ref(), 2);
    auto run = compiled->executable.create_invocation().value();
    auto queue = context.create_queue().value(); bind(f, r, run); submit(run, queue);
    expect_scores(score_values(r.output.view.as_const()), r.host.expected);
    const auto expected = escape == Escape::kControl ? r.host.q
        : escape == Escape::kPublishedDot ? r.host.dot : r.host.product;
    const auto actual = read<BF16>(r.extra.view.as_const());
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) EXPECT_EQ(actual[i].bits, expected[i].bits) << i;
  }
}

TEST(CudaDenseIndexer, EmptyBatchTokenAndHistoryProduceNoWork) {
  auto context = g::cuda::create_context().value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  for (const auto axes : {std::array<std::int64_t, 3>{0, 2, 3}, {2, 0, 3}, {2, 2, 0}}) {
    const auto f = build({.b = axes[0], .t = axes[1], .h = 3, .d = 32,
                          .l = axes[2], .pitch = axes[2], .masked = false, .widen = false});
    auto compiled = g::cuda::compile(f.graph, context, memory.ref());
    ASSERT_TRUE(compiled) << compiled.error().message();
    selected(f, compiled->report, true);
    auto r = allocate(f, memory.ref(), 0, false, false); // No input payload upload.
    auto run = compiled->executable.create_invocation().value();
    auto queue = context.create_queue().value(); bind(f, r, run); submit(run, queue);
    EXPECT_TRUE(score_values(r.output.view.as_const()).empty());
    EXPECT_TRUE(run.output(f.score).has_value());
  }
}
} // namespace
