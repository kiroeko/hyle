#include "hyle/graph/cuda.h"
#include "hyle/graph/providers/ggml_cuda.h"
#include "hyle/compute/providers/cublas.h"
#include "hyle/compute/storage_owner.h"
#include "../compute_provider_ggml_cuda/tensor_test_support.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
using BF16 = hyle::simd::bfloat16_t;
using Type = c::DataType;
using ggml_cuda_tensor_test::dense;
constexpr c::RoutePolicyId kPolicy{0x48594c4547474355ULL, 5, 1};
constexpr c::RoutePolicyId kFfnPolicy{0x48594c4547474355ULL, 3, 1};

template <class T> T take(hyle::Result<T> result, std::string_view where) {
  if (!result) throw std::runtime_error(std::string(where) + ": " + std::string(result.error().message()));
  return std::move(*result);
}
void check(hyle::Status status, std::string_view where) {
  if (!status.ok()) throw std::runtime_error(std::string(where) + ": " + std::string(status.message()));
}
struct Shape final {
  std::int64_t m = 1, k = 128;
  std::array<std::int64_t, 2> n{8, 8};
  bool foreign_a = false, transposed = true, constants = false, same_weight = false;
  bool reader = false, reentry = false, ffn = false;
  bool ffn_up_first = false;
};
constexpr std::array kPairedShapes{Shape{}, Shape{.m = 2, .n = {16, 16}}};
struct Logical final {
  Shape shape;
  g::Graph graph;
  std::array<g::ValueId, 2> a{}, w{}, y{};
  std::array<g::NodeId, 2> mm{};
  c::TensorDesc activation;
  std::array<c::TensorDesc, 2> stored{}, logical_weights{}, outputs{};
  g::ValueId reader, result;
};
Logical build(Shape shape = {}) {
  Logical f{.shape = shape};
  auto b = g::GraphBuilder::create().value();
  f.activation = dense(Type::kBFloat16, {shape.m, shape.k});
  f.a[0] = b.input(f.activation).value();
  f.a[1] = shape.foreign_a ? b.input(f.activation).value() : f.a[0];
  for (std::size_t p = 0; p < 2; ++p) {
    f.stored[p] = shape.transposed ? dense(Type::kBFloat16, {shape.n[p], shape.k})
                                  : dense(Type::kBFloat16, {shape.k, shape.n[p]});
    f.logical_weights[p] = shape.transposed ? f.stored[p].transposed().value() : f.stored[p];
    f.outputs[p] = dense(Type::kBFloat16, {shape.m, shape.n[p]});
    f.w[p] = shape.same_weight && p ? f.w[0] : shape.constants
        ? b.constant(f.stored[p]).value() : b.input(f.stored[p]).value();
  }
  std::array<g::ValueId, 2> rhs;
  for (std::size_t p = 0; p < 2; ++p)
    rhs[p] = shape.same_weight && p ? rhs[0] : shape.transposed
        ? b.add(c::View::permute(std::array<std::int32_t, 2>{1, 0}).value(), f.w[p]).value() : f.w[p];
  const auto first = b.add_node(c::Operation(c::MatMul{.accumulation_type = Type::kFloat32}), std::array{f.a[0], rhs[0]}).value();
  f.y[0] = b.result(first).value();
  if (shape.reader) f.reader = b.add(c::Unary{c::UnaryKind::kNeg}, f.y[0]).value();
  g::NodeId control;
  if (shape.reentry) {
    control = b.add_node(c::Operation(c::Copy{}), std::array{f.a[0]}).value();
    check(b.depends_on(control, first), "control leaves first projection");
  }
  auto silu = shape.ffn && !shape.ffn_up_first
      ? b.add(c::Unary{c::UnaryKind::kSilu}, f.y[0]).value() : g::ValueId{};
  const auto second = b.add_node(c::Operation(c::MatMul{.accumulation_type = Type::kFloat32}), std::array{f.a[1], rhs[1]}).value();
  f.y[1] = b.result(second).value();
  if (shape.ffn && shape.ffn_up_first)
    silu = b.add(c::Unary{c::UnaryKind::kSilu}, f.y[0]).value();
  if (shape.reentry)
    check(b.depends_on(second, control), "control re-enters second projection");
  if (shape.ffn) {
    f.result = b.add(c::Binary{c::BinaryKind::kMultiply}, silu, f.y[1]).value();
    check(b.publish(f.result), "publish complete FFN");
  } else {
    check(b.publish(f.y[1]), "publish second output first");
    check(b.publish(f.y[0]), "publish first output second");
    if (shape.reader) check(b.publish(f.reader), "publish external first-output reader");
  }
  f.graph = std::move(b).finish().value();
  for (std::size_t p = 0; p < 2; ++p) f.mm[p] = f.graph.producer(f.y[p]).value();
  return f;
}

struct Data final {
  std::vector<BF16> a;
  std::array<std::vector<BF16>, 2> rows;
};
Data data(const Logical &f, int seed = 0) {
  Data d;
  for (std::int64_t i = 0; i < f.shape.m * f.shape.k; ++i)
    d.a.push_back(half_test::encode<BF16>((int(i % 7) - 3) / 4.0 + (seed + 1) / 128.0));
  for (std::size_t p = 0; p < 2; ++p)
    for (std::int64_t i = 0; i < f.shape.n[p] * f.shape.k; ++i)
      d.rows[p].push_back(half_test::encode<BF16>((int(i % 11) - 5) / 32.0 + (seed + int(p)) / 256.0));
  if (f.shape.same_weight) d.rows[1] = d.rows[0];
  return d;
}
std::vector<BF16> stored(const Logical &f, const Data &d, std::size_t p) {
  if (f.shape.transposed) return d.rows[p];
  std::vector<BF16> result(d.rows[p].size());
  for (std::int64_t n = 0; n < f.shape.n[p]; ++n)
    for (std::int64_t k = 0; k < f.shape.k; ++k)
      result[std::size_t(k * f.shape.n[p] + n)] = d.rows[p][std::size_t(n * f.shape.k + k)];
  return result;
}
c::ReadLease upload(const c::StorageOwner &owner, std::span<const BF16> values) {
  auto buffer = take(m::Buffer::create(owner.ref(), values.size_bytes(), 256), "input Buffer::create");
  check(buffer.view().write(std::as_bytes(values)), "input upload");
  return take(owner.adopt_buffer(std::move(buffer)), "input StorageOwner::adopt_buffer");
}
std::vector<BF16> read(c::ConstTensorView view) {
  std::vector<BF16> values(view.description().element_count().value());
  check(view.storage().read(std::as_writable_bytes(std::span(values))), "output read");
  return values;
}
double decoded(BF16 value) {
  // Integer storage decode avoids host DAZ/FTZ when checking BF16 subnormals.
  const auto exponent = (value.bits >> 7) & 255U, fraction = value.bits & 127U;
  const double sign = value.bits & 0x8000U ? -1. : 1.;
  if (exponent == 255U)
    return fraction ? std::numeric_limits<double>::quiet_NaN()
                    : sign * std::numeric_limits<double>::infinity();
  return sign * (exponent ? std::ldexp(double(128U + fraction), int(exponent) - 134)
                         : std::ldexp(double(fraction), -133));
}
void bits(std::span<const BF16> actual, std::span<const BF16> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const auto x = decoded(actual[i]), y = decoded(expected[i]);
    if (std::isnan(y)) EXPECT_TRUE(std::isnan(x)) << i;
    else EXPECT_EQ(actual[i].bits, expected[i].bits) << i;
  }
}
struct Inputs final {
  c::ReadLease a;
  std::array<c::ReadLease, 2> w;
};
Inputs inputs(const Logical &f, const Data &d, const c::StorageOwner &owner) {
  Inputs result{.a = upload(owner, d.a)};
  for (std::size_t p = 0; p < 2; ++p)
    result.w[p] = p && f.shape.same_weight ? result.w[0] : upload(owner, stored(f, d, p));
  return result;
}
std::array<std::vector<BF16>, 2> leaf(const Logical &f, const Inputs &in,
    c::Context &context, c::Queue &queue, const c::StorageOwner &owner) {
  std::array<std::vector<BF16>, 2> result;
  for (std::size_t p = 0; p < 2; ++p) {
    const std::array operands{take(in.a.tensor(f.activation), "original leaf A"),
        take(in.w[p].tensor(f.logical_weights[p]), "original leaf transposed W")};
    auto buffer = take(m::Buffer::create(owner.ref(), f.outputs[p].required_bytes().value(), 256), "leaf output buffer");
    auto output = c::TensorView::create(f.outputs[p], buffer.view()).value();
    auto prepared = take(c::prepare(context, c::Operation(c::MatMul{.accumulation_type = Type::kFloat32}),
        operands, std::span(&output, 1)), "original leaf prepare");
    EXPECT_EQ(prepared.provider().id, c::cublas::kProviderId);
    const auto need = prepared.workspace_requirement();
    auto scratch = take(m::Buffer::create(owner.ref(), need.size, need.alignment), "original leaf workspace");
    auto call = take(prepared.create_invocation(scratch.view()), "original leaf Invocation");
    auto done = take(call.submit(queue, operands, std::span(&output, 1)), "original leaf submit");
    check(done.wait(), "original leaf Completion::wait");
    result[p] = read(output.as_const());
  }
  return result;
}
void route(const Logical &f, const g::CompiledGraph &program, bool paired) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < program.report.step_count(); ++i) {
    const auto step = program.report.step(i).value();
    if (step.policy != kPolicy) continue;
    ++count;
    EXPECT_EQ(step.provider.id, c::ggml_cuda::kProviderId);
    EXPECT_EQ(step.implementation, c::ProviderImplementationId{6});
    EXPECT_EQ(step.node_count, 2U); EXPECT_EQ(step.output_count, 2U);
    EXPECT_EQ(step.input_count, f.shape.same_weight ? 2U : 3U);
    EXPECT_EQ(step.copy_bytes, 0U); EXPECT_EQ(step.dispatch_count, 1U);
    for (const auto node : f.mm) {
      bool found = false;
      for (std::size_t j = 0; j < step.node_count; ++j) found |= program.report.step_node(i, j).value() == node;
      EXPECT_TRUE(found);
    }
  }
  EXPECT_EQ(count, paired ? 1U : 0U);
}
std::vector<c::LeasePool> bind_outputs(const Logical &f, g::Invocation &slot,
    const Inputs &in, const c::StorageOwner &owner,
    std::array<c::TensorView, 2> *output_views = nullptr) {
  check(slot.bind(f.a[0], f.activation, in.a), "bind activation");
  if (f.a[1] != f.a[0]) check(slot.bind(f.a[1], f.activation, in.a), "bind distinct logical A to same storage");
  if (!f.shape.constants)
    for (std::size_t p = 0; p < 2; ++p)
      if (!p || f.w[p] != f.w[0]) check(slot.bind(f.w[p], f.stored[p], in.w[p]), "bind weight");
  std::vector<c::LeasePool> pools;
  const auto bind_output = [&](g::ValueId id, const c::TensorDesc &desc, c::TensorView *view) {
    const auto bytes = desc.required_bytes().value();
    pools.push_back(take(c::LeasePool::create(owner.ref(), bytes,
        {.max_blocks = 1, .backing_alignment = 256}), "output LeasePool"));
    auto writer = take(pools.back().allocate(bytes, 256), "output WriteLease");
    if (view) *view = take(writer.tensor(desc), "original output writer view");
    check(slot.bind(id, desc, std::move(writer)), "bind output");
  };
  for (std::size_t p = 0; p < 2; ++p)
    bind_output(f.y[p], f.outputs[p], output_views ? &(*output_views)[p] : nullptr);
  if (f.reader.valid()) bind_output(f.reader, f.outputs[0], nullptr);
  return pools;
}
enum class CaptureCheck { kRoute, kRecord, kReuse };
void run(g::Invocation &slot, c::Queue &queue, CaptureCheck capture = CaptureCheck::kRoute) {
  check(g::cuda::prepare_replay(slot, queue, {.mode = g::ReplayMode::kRequireCapture}), "RequireCapture prepare");
  EXPECT_EQ(slot.replay_route(), g::ReplayRoute::kCaptured);
  const auto report = take(slot.replay_report(), "paired capture report");
  EXPECT_EQ(report.host_wait_dependency_count, 0U);
  if (capture == CaptureCheck::kRecord) {
    EXPECT_GT(report.recorded_capture_range_count, 0U);
    EXPECT_EQ(report.reused_capture_range_count, 0U);
  } else if (capture == CaptureCheck::kReuse) {
    EXPECT_EQ(report.recorded_capture_range_count, 0U);
    EXPECT_GT(report.reused_capture_range_count, 0U);
  }
  auto done = take(slot.submit(queue), "captured paired submit"); check(done.wait(), "captured paired wait");
}
void mathematical_bound(const Logical &f, const Data &d, std::size_t p,
                        std::span<const BF16> actual, bool exact_sum = false) {
  const double u = std::ldexp(1., -24);
  const double terms = double(2 * f.shape.k + 32), gamma = terms * u / (1 - terms * u);
  for (std::int64_t m0 = 0; m0 < f.shape.m; ++m0)
    for (std::int64_t n = 0; n < f.shape.n[p]; ++n) {
      double sum = 0, sumabs = 0;
      for (std::int64_t k = 0; k < f.shape.k; ++k) {
        const auto product = decoded(d.a[std::size_t(m0 * f.shape.k + k)]) *
            decoded(d.rows[p][std::size_t(n * f.shape.k + k)]);
        sum += product; sumabs += std::abs(product);
      }
      const auto value = decoded(actual[std::size_t(m0 * f.shape.n[p] + n)]);
      if (std::isnan(sum)) { EXPECT_TRUE(std::isnan(value)); continue; }
      if (std::isinf(sum)) { EXPECT_EQ(value, sum); continue; }
      ASSERT_TRUE(std::isfinite(value));
      if (exact_sum) {
        // The normal dyadic corpus's products and all partial sums fit F32.
        EXPECT_EQ(actual[std::size_t(m0 * f.shape.n[p] + n)].bits,
                  half_test::encode<BF16>(sum).bits);
        continue;
      }
      const auto bound = gamma * sumabs + double(f.shape.k) * std::numeric_limits<float>::denorm_min();
      EXPECT_GE(value, decoded(half_test::encode<BF16>(sum - bound)));
      EXPECT_LE(value, decoded(half_test::encode<BF16>(sum + bound)));
    }
}

TEST(GgmlCudaPairedProjection, PublishedReverseOutputsAndOutsideReaderRetainOriginalConstantBanks) {
  auto context = g::cuda::create_context().value();
  auto owner = c::StorageOwner::adopt(context,
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value()).value();
  auto queue = context.create_queue().value();
  for (const Shape profile : kPairedShapes) for (const bool same_weight : {false, true}) {
    SCOPED_TRACE(::testing::Message() << profile.m << '/' << profile.n[0] << '/' << same_weight);
    auto shape = profile;
    shape.constants = true; shape.same_weight = same_weight; shape.reader = true;
    const auto f = build(shape);
    const auto d = data(f); const auto in = inputs(f, d, owner);
    std::vector<g::ConstantBinding> constants;
    for (std::size_t p = 0; p < 2; ++p)
      if (!p || f.w[p] != f.w[0]) constants.push_back({.value = f.w[p], .tensor = take(in.w[p].tensor(f.stored[p]), "constant weight tensor"), .owner = owner, .lease = in.w[p]});
    auto program = take(g::cuda::compile(f.graph, context, owner, {.constants = constants}), "constant pair compile");
    route(f, program, true);
    EXPECT_EQ(program.report.packed_constant_count(), 0U);
    EXPECT_EQ(program.report.packed_constant_bytes(), 0U);
    auto slot = take(program.executable.create_invocation({.resources = g::ResourceOptions{}}), "constant pair Invocation");
    auto pools = bind_outputs(f, slot, in, owner); const auto expected = leaf(f, in, context, queue, owner);
    run(slot, queue);
    for (std::size_t p = 0; p < 2; ++p) {
      auto result = take(slot.result(f.y[p]), "original paired result");
      const auto actual = read(take(result.tensor(), "original paired result tensor"));
      bits(actual, expected[p]); mathematical_bound(f, d, p, actual, true);
    }
    auto reader = take(slot.result(f.reader), "outside reader result");
    auto negated = expected[0];
    for (auto &value : negated) value.bits ^= 0x8000U;
    bits(read(take(reader.tensor(), "outside reader tensor")), negated);
  }
}

TEST(GgmlCudaPairedProjection, TwoOwnedSlotsRebindBothOutputsAndRespectOldResultLeases) {
  auto context = g::cuda::create_context().value();
  auto owner = c::StorageOwner::adopt(context,
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value()).value();
  for (const Shape shape : kPairedShapes) {
    SCOPED_TRACE(::testing::Message() << shape.m << '/' << shape.n[0]);
    const auto f = build(shape); auto program = take(g::cuda::compile(f.graph, context, owner), "dynamic pair compile");
    route(f, program, true);
    std::array slots{program.executable.create_invocation({.resources = g::ResourceOptions{}}).value(),
                    program.executable.create_invocation({.resources = g::ResourceOptions{}}).value()};
    std::array queues{context.create_queue().value(), context.create_queue().value()};
    std::array<Inputs, 2> retained;
    std::array<g::GraphOutput, 2> old;
    std::array<std::array<std::vector<BF16>, 2>, 2> first;
    for (unsigned round = 0; round < 2; ++round)
      for (const unsigned lane : {1U, 0U}) {
        if (round) { if (!lane) { old[0].reset(); old[1].reset(); } check(slots[lane].reset(), "reset completed pair slot"); }
        const auto d = data(f, int(round * 2 + lane)); auto in = inputs(f, d, owner);
        if (round) {
          const auto before = take(retained[lane].a.view(), "previous source"), after = take(in.a.view(), "rebound source");
          EXPECT_TRUE(before.native_handle() != after.native_handle() || before.offset() != after.offset());
        }
        const auto expected = leaf(f, in, context, queues[lane], owner);
        auto pools = bind_outputs(f, slots[lane], in, owner); retained[lane] = std::move(in);
        run(slots[lane], queues[lane], CaptureCheck::kRecord);
        // Identical physical bindings must reuse the original recording, before
        // any GraphOutput publishes or pins these original output writers.
        run(slots[lane], queues[lane], CaptureCheck::kReuse);
        std::array results{take(slots[lane].result(f.y[0]), "first GraphOutput"), take(slots[lane].result(f.y[1]), "second GraphOutput")};
        const auto a = take(results[0].tensor(), "first output tensor"), b = take(results[1].tensor(), "second output tensor");
        EXPECT_TRUE(a.storage().native_handle() != b.storage().native_handle() || a.storage().offset() != b.storage().offset());
        for (std::size_t p = 0; p < 2; ++p) {
          const auto actual = read(take(results[p].tensor(), "paired GraphOutput tensor")); bits(actual, expected[p]);
          mathematical_bound(f, d, p, actual, true);
          if (!round) first[lane][p] = actual;
          else {
            bool changed = false;
            for (std::size_t i = 0; i < actual.size(); ++i) changed |= actual[i].bits != first[lane][p][i].bits;
            EXPECT_TRUE(changed);
            if (lane) bits(read(take(old[p].tensor(), "other slot original result")), first[0][p]);
          }
        }
        EXPECT_TRUE(slots[lane].reset().IsUnavailable());
        if (!round && !lane) old = std::move(results);
      }
  }
}

TEST(GgmlCudaPairedProjection, LateM2PhysicalAliasesPreserveOriginalBindingsAndCapture) {
  auto context = g::cuda::create_context().value();
  auto owner = c::StorageOwner::adopt(context,
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value()).value();
  auto queue = context.create_queue().value();
  const auto f = build({.m = 2, .n = {16, 16}}); const auto d = data(f);
  auto activation = take(m::Buffer::create(owner.ref(), d.a.size() * sizeof(BF16), 256), "alias activation buffer");
  check(activation.view().write(std::as_bytes(std::span(d.a))), "alias activation upload");
  // Keep only a raw view of the real backing before transferring its owner.
  // Neither alias attempt may turn this immutable input into output storage.
  const auto activation_storage = activation.view();
  Inputs in{.a = take(owner.adopt_buffer(std::move(activation)), "alias activation owner")};
  for (std::size_t p = 0; p < in.w.size(); ++p) in.w[p] = upload(owner, stored(f, d, p));
  auto program = take(g::cuda::compile(f.graph, context, owner), "alias pair compile");
  route(f, program, true);
  auto slot = take(program.executable.create_invocation({.resources = g::ResourceOptions{}}), "alias pair Invocation");
  std::array<c::TensorView, 2> output_views;
  auto pools = bind_outputs(f, slot, in, owner, &output_views);
  const auto expected = leaf(f, in, context, queue, owner);
  run(slot, queue, CaptureCheck::kRecord);
  const auto read_write = take(c::TensorView::create(f.outputs[0], activation_storage), "overlapping activation output view");
  const auto rejected_input = slot.bind(f.y[0], read_write);
  EXPECT_TRUE(rejected_input.IsInvalidArgument()) << rejected_input.message();
  const auto rejected_output = slot.bind(f.y[1], output_views[0]);
  EXPECT_TRUE(rejected_output.IsInvalidArgument()) << rejected_output.message();
  EXPECT_EQ(slot.replay_route(), g::ReplayRoute::kCaptured);
  // Rejected late bindings leave both original writers and the complete
  // physical capture tuple intact. No overlapping write is submitted.
  run(slot, queue, CaptureCheck::kReuse);
  for (std::size_t p = 0; p < f.y.size(); ++p) {
    auto output = take(slot.result(f.y[p]), "alias-rejection original result");
    bits(read(take(output.tensor(), "alias-rejection original output tensor")), expected[p]);
    bits(read(take(in.w[p].tensor(f.stored[p]), "unchanged original weight tensor")), stored(f, d, p));
  }
  bits(read(take(in.a.tensor(f.activation), "unchanged original activation tensor")), d.a);
}

TEST(GgmlCudaPairedProjection, OriginalLeavesAndIndependentMathCoverCancellationSubnormalAndNonfinite) {
  auto context = g::cuda::create_context().value();
  auto owner = c::StorageOwner::adopt(context,
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value()).value();
  auto queue = context.create_queue().value();
  for (const Shape shape : kPairedShapes) {
    SCOPED_TRACE(::testing::Message() << shape.m << '/' << shape.n[0]);
    const auto f = build(shape);
    auto program = take(g::cuda::compile(f.graph, context, owner), "numerical pair compile"); route(f, program, true);
    for (unsigned corpus = 0; corpus < 4; ++corpus) {
      SCOPED_TRACE(corpus); auto d = data(f);
      if (corpus) {
        d.a.assign(d.a.size(), BF16::from_bits(0));
        for (std::int64_t row = 0; row < f.shape.m; ++row) {
          const auto base = std::size_t(row * f.shape.k);
          const auto sign = row % 2 ? -1. : 1.;
          d.a[base] = half_test::encode<BF16>(sign);
          d.a[base + 1] = half_test::encode<BF16>(-sign);
          d.a[base + 2] = half_test::encode<BF16>(sign);
        }
        for (std::size_t p = 0; p < 2; ++p) {
          d.rows[p].assign(d.rows[p].size(), BF16::from_bits(0));
          for (std::int64_t n = 0; n < f.shape.n[p]; ++n) {
            const auto base = std::size_t(n * f.shape.k);
            d.rows[p][base] = half_test::encode<BF16>(std::ldexp(1., 40));
            d.rows[p][base + 1] = d.rows[p][base];
            d.rows[p][base + 2] = half_test::encode<BF16>(std::ldexp(1., -20));
          }
        }
        if (corpus == 2) {
          d.a.assign(d.a.size(), BF16::from_bits(0));
          for (std::int64_t row = 0; row < f.shape.m; ++row)
            d.a[std::size_t(row * f.shape.k)] = BF16::from_bits(static_cast<std::uint16_t>(row + 1));
          for (std::size_t p = 0; p < d.rows.size(); ++p) {
            auto &w = d.rows[p]; std::ranges::fill(w, BF16::from_bits(0));
            for (std::int64_t n = 0; n < f.shape.n[p]; ++n)
              w[std::size_t(n * f.shape.k)] = half_test::encode<BF16>(std::ldexp(1., 120));
          }
        }
        if (corpus == 3) {
          for (std::int64_t row = 0; row < f.shape.m; ++row)
            std::fill_n(d.a.begin() + row * f.shape.k, std::size_t(f.shape.k),
                half_test::encode<BF16>(row ? .5 : 1.));
          for (std::size_t p = 0; p < 2; ++p) {
            auto &w = d.rows[p];
            std::ranges::fill(w, BF16::from_bits(0));
            if (p) {
              for (std::int64_t n = 0; n < f.shape.n[p]; ++n)
                w[std::size_t(n * f.shape.k)] = half_test::encode<BF16>(.5 + double(n) / 8);
              continue; // A nonfinite first bank must not contaminate its finite peer.
            }
            w[0] = BF16::from_bits(0x7f80); w[std::size_t(f.shape.k)] = BF16::from_bits(0xff80);
            w[std::size_t(2 * f.shape.k)] = BF16::from_bits(0x7fc1);
            w[std::size_t(3 * f.shape.k)] = BF16::from_bits(0x7f80); w[std::size_t(3 * f.shape.k + 1)] = BF16::from_bits(0xff80);
          }
        }
      }
      const auto in = inputs(f, d, owner); const auto expected = leaf(f, in, context, queue, owner);
      auto slot = program.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
      auto pools = bind_outputs(f, slot, in, owner); run(slot, queue);
      std::size_t differences = 0;
      for (std::size_t p = 0; p < 2; ++p) {
        auto output = take(slot.result(f.y[p]), "numerical paired GraphOutput"); const auto actual = read(take(output.tensor(), "numerical output tensor"));
        mathematical_bound(f, d, p, expected[p], corpus == 0); mathematical_bound(f, d, p, actual, corpus == 0);
        for (std::size_t i = 0; i < actual.size(); ++i) differences += actual[i].bits != expected[p][i].bits;
        if (corpus != 1) bits(actual, expected[p]);
        if (corpus == 2)
          for (std::int64_t row = 0; row < f.shape.m; ++row)
            for (std::int64_t n = 0; n < f.shape.n[p]; ++n)
              EXPECT_EQ(actual[std::size_t(row * f.shape.n[p] + n)].bits,
                  half_test::encode<BF16>(std::ldexp(double(row + 1), -13)).bits);
      }
      RecordProperty((f.shape.m == 1 ? std::string{} : "m" + std::to_string(f.shape.m) + "_") +
          "corpus_" + std::to_string(corpus) + "_raw_leaf_differences", std::to_string(differences));
    }
  }
}

TEST(GgmlCudaPairedProjection, DistinctLogicalActivationGeometryAndControlReentryKeepOrdinaryLeaves) {
  auto context = g::cuda::create_context().value();
  auto owner = c::StorageOwner::adopt(context,
      c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value()).value();
  auto queue = context.create_queue().value();
  for (const Shape shape : {Shape{.foreign_a = true}, Shape{.m = 2, .n = {16, 16}, .foreign_a = true},
                           Shape{.k = 127}, Shape{.m = 3},
                           Shape{.n = {8, 9}}, Shape{.transposed = false}, Shape{.reentry = true}}) {
    const auto f = build(shape); const auto d = data(f); const auto in = inputs(f, d, owner);
    auto program = take(g::cuda::compile(f.graph, context, owner), "ordinary fallback compile"); route(f, program, false);
    const auto expected = leaf(f, in, context, queue, owner);
    auto slot = program.executable.create_invocation({.resources = g::ResourceOptions{}}).value();
    auto pools = bind_outputs(f, slot, in, owner); run(slot, queue);
    for (std::size_t p = 0; p < 2; ++p) {
      auto output = take(slot.result(f.y[p]), "ordinary fallback GraphOutput"); bits(read(take(output.tensor(), "ordinary fallback output tensor")), expected[p]);
    }
  }
}

TEST(GgmlCudaPairedProjection, CompletePrivateFfnRemainsTheEarlierFullCandidate) {
  auto context = g::cuda::create_context().value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  for (const bool up_first : {false, true}) {
    SCOPED_TRACE(up_first);
    const auto f = build({.ffn = true, .ffn_up_first = up_first});
    auto program = take(g::cuda::compile(f.graph, context, memory.ref()), "complete FFN compile"); route(f, program, false);
    std::size_t ffn = 0;
    for (std::size_t i = 0; i < program.report.step_count(); ++i)
      if (program.report.step(i)->policy == kFfnPolicy) {
        ++ffn; EXPECT_EQ(program.report.step(i)->implementation, c::ProviderImplementationId{4});
        EXPECT_EQ(program.report.step(i)->node_count, 4U);
      }
    EXPECT_EQ(ffn, 1U); // Selection proof only; existing FFN numerical tests remain the execution neighbor.
  }
}
} // namespace
