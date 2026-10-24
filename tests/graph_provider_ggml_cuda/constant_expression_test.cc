#include "hyle/graph/providers/ggml_cuda.h"
#include "hyle/graph/executable.h"
#include "hyle/compute/storage_owner.h"
#include "../compute_provider_ggml_cuda/tensor_test_support.h"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
using hyle::mem::Buffer;
using BF16 = hyle::simd::bfloat16_t;
using ggml_cuda_tensor_test::dense;
using ggml_cuda_tensor_test::output_buffer;
using ggml_cuda_tensor_test::upload;
constexpr std::int64_t kWidth = 17;

c::Context context() {
  return c::Context::create(c::cuda::device(0).value(), g::ggml_cuda::provider()).value();
}
c::StorageOwner storage(c::Context &ctx) {
  return c::StorageOwner::adopt(ctx, c::cuda::MemoryResource::create(
      ctx, c::cuda::MemoryKind::kDevice).value()).value();
}
template <class T>
g::ConstantBinding constant(const c::StorageOwner &owner, g::ValueId id,
                            const c::TensorDesc &description,
                            const std::vector<T> &values) {
  auto bytes = upload<T>(owner.ref(), values);
  auto ready = owner.adopt_buffer(std::move(bytes)).value();
  return {id, ready.tensor(description).value(), g::ConstantPolicy::kBorrow,
          owner, std::move(ready)};
}
std::vector<BF16> weights(int seed = 0) {
  std::vector<BF16> result(kWidth);
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = half_test::encode<BF16>(0.25 + double(i) / 32 + seed / 8.0);
  return result;
}
std::vector<float> factor(const std::vector<BF16> &weight, float bias) {
  std::vector<float> result;
  for (const auto value : weight)
    result.push_back(static_cast<float>(half_test::decode(value)) + bias);
  return result;
}
void expect_bits(const Buffer &output, const std::vector<BF16> &expected) {
  const auto actual = half_test::read<BF16>(output);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_EQ(actual[i].bits, expected[i].bits) << i;
}
std::vector<BF16> multiply(const std::vector<float> &input,
                           const std::vector<float> &scale) {
  std::vector<BF16> result;
  for (std::size_t i = 0; i < input.size(); ++i) {
    const float product = input[i] * scale[i % scale.size()];
    result.push_back(half_test::encode<BF16>(product));
  }
  return result;
}
std::size_t step_for(const g::CompileReport &report, g::NodeId node) {
  for (std::size_t step = 0; step < report.step_count(); ++step)
    for (std::size_t n = 0; n < report.step(step)->node_count; ++n)
      if (report.step_node(step, n).value() == node) return step;
  return SIZE_MAX;
}

// Two independent slots are rebound twice. All old allocations stay alive,
// including each old public result, while new addresses and values are used.
void exercise_slots(g::CompiledGraph &compiled, c::Context &queue_context,
                    g::ExecutionLaneId lane, const c::StorageOwner &owner,
                    g::ValueId x, const std::vector<g::ValueId> &outputs,
                    const std::vector<std::vector<float>> &scales,
                    c::TensorDesc input_desc = dense(c::DataType::kFloat32, {2, kWidth})) {
  const auto out_desc = c::TensorDesc::dense(c::DataType::kBFloat16,
                                           input_desc.dimensions()).value();
  std::vector<Buffer> retained;
  std::array queues{queue_context.create_queue().value(), queue_context.create_queue().value()};
  std::array runs{compiled.executable.create_invocation().value(),
                  compiled.executable.create_invocation().value()};
  std::array<std::vector<std::vector<BF16>>, 2> old;
  const auto buffers_per_slot = outputs.size() + 1;
  for (int round = 0; round < 2; ++round) {
    std::array<Buffer, 2> inputs;
    std::array<std::vector<Buffer>, 2> results;
    std::array<std::vector<float>, 2> host;
    std::array<c::Completion, 2> done;
    for (std::size_t slot = 0; slot < 2; ++slot) {
      host[slot].resize(input_desc.element_count().value());
      for (std::size_t i = 0; i < host[slot].size(); ++i)
        host[slot][i] = float(int(i % 23) - 11 + round * 3 + int(slot)) / 128;
      inputs[slot] = upload<float>(owner.ref(), host[slot]);
      if (round)
        EXPECT_NE(inputs[slot].native_handle(), retained[slot * buffers_per_slot].native_handle());
      ASSERT_TRUE(runs[slot].bind(x, c::ConstTensorView::create(
          input_desc, inputs[slot].view()).value()).ok());
      for (const auto output : outputs) {
        results[slot].push_back(output_buffer(owner.ref(), out_desc));
        if (round)
          EXPECT_NE(results[slot].back().native_handle(),
              retained[slot * buffers_per_slot + results[slot].size()].native_handle());
        ASSERT_TRUE(runs[slot].bind(output, c::TensorView::create(
            out_desc, results[slot].back().view()).value()).ok());
      }
      const std::array bindings{g::QueueBinding{lane, &queues[slot]}};
      ASSERT_TRUE(runs[slot].prepare_replay(bindings, {.mode = g::ReplayMode::kCommands}).ok());
      auto submitted = runs[slot].submit(bindings);
      ASSERT_TRUE(submitted) << submitted.error().message();
      done[slot] = std::move(*submitted);
    }
    ASSERT_TRUE(done[1].wait().ok()); ASSERT_TRUE(done[0].wait().ok());
    for (std::size_t slot = 0; slot < 2; ++slot) {
      retained.push_back(std::move(inputs[slot]));
      for (std::size_t port = 0; port < outputs.size(); ++port) {
        expect_bits(results[slot][port], multiply(host[slot], scales[port]));
        if (!round) old[slot].push_back(half_test::read<BF16>(results[slot][port]));
        retained.push_back(std::move(results[slot][port]));
      }
    }
    if (round)
      for (std::size_t slot = 0; slot < 2; ++slot)
        for (std::size_t port = 0; port < outputs.size(); ++port)
          expect_bits(retained[slot * buffers_per_slot + port + 1], old[slot][port]);
  }
}

TEST(GgmlCudaConstantExpression, RankOneKeysAndNativeExpPreserveSlotsAndRounding) {
  auto ctx = context(); auto owner = storage(ctx);
  const auto bf = dense(c::DataType::kBFloat16, {kWidth});
  const auto f = dense(c::DataType::kFloat32, {kWidth});
  const auto scalar = dense(c::DataType::kFloat32, {1});
  auto b = g::GraphBuilder::create().value();
  const auto x = b.input(dense(c::DataType::kFloat32, {2, kWidth})).value();
  const auto w = b.constant(bf).value(), one = b.constant(scalar).value();
  const auto two = b.constant(scalar).value(), log = b.constant(f).value();
  const auto cast = b.add(c::Cast{c::DataType::kFloat32}, w).value();
  const auto first = b.add(c::Binary{}, cast, one).value();
  const auto second = b.add(c::Binary{}, cast, two).value();
  const auto exp = b.add(c::Unary{c::UnaryKind::kExp}, log).value();
  const auto neg = b.add(c::Unary{c::UnaryKind::kNeg}, exp).value();
  std::vector<g::ValueId> outputs;
  for (const auto scale : {first, second, neg}) {
    const auto product = b.add(c::Binary{c::BinaryKind::kMultiply}, x, scale).value();
    outputs.push_back(b.add(c::Cast{c::DataType::kBFloat16}, product).value());
    ASSERT_TRUE(b.publish(outputs.back()).ok());
  }
  const auto graph = std::move(b).finish().value();
  std::vector<float> logs(kWidth);
  for (std::size_t i = 0; i < logs.size(); ++i) logs[i] = float(int(i % 7) - 3) / 4;
  std::vector<g::ConstantBinding> constants{
      constant(owner, w, bf, weights()), constant(owner, one, scalar, std::vector{1.F}),
      constant(owner, two, scalar, std::vector{2.F}), constant(owner, log, f, logs)};
  // Native Exp/Neg is the reference: host exp must not change native flags or
  // approximation. Only the subsequent ordinary F32 multiply is done on CPU.
  auto exponential = output_buffer(owner.ref(), f), negative = output_buffer(owner.ref(), f);
  const std::array exp_input{constants.back().tensor};
  ASSERT_TRUE(ggml_cuda_tensor_test::execute(ctx, c::Unary{c::UnaryKind::kExp}, exp_input,
      c::TensorView::create(f, exponential.view()).value()).ok());
  const std::array neg_input{c::ConstTensorView::create(f, exponential.view()).value()};
  ASSERT_TRUE(ggml_cuda_tensor_test::execute(ctx, c::Unary{c::UnaryKind::kNeg}, neg_input,
      c::TensorView::create(f, negative.view()).value()).ok());
  const std::vector scales{factor(weights(), 1), factor(weights(), 2), half_test::read<float>(negative)};
  const std::array domains{g::StorageDomain{.resource = owner.ref(), .owner = owner}};
  const std::array targets{g::ExecutionTarget{.source = g::ExecutionSource::acquire(ctx).value(),
      .default_storage = g::StorageDomainId{0}}};
  const std::array lanes{g::ExecutionLane{.target = g::ExecutionTargetId{0}}};
  auto compiled = g::compile(graph, {.storage_domains = domains, .targets = targets, .lanes = lanes},
                              {.constants = constants});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_GE(compiled->report.packed_constant_count(), 1U);
  EXPECT_GE(compiled->report.packed_constant_bytes(), 3U * kWidth * sizeof(float));
  constants.clear(); // The selected artifacts, not these caller handles, own cold results.
  exercise_slots(*compiled, ctx, g::ExecutionLaneId{0}, owner, x, outputs, scales);
}

TEST(GgmlCudaConstantExpression, RankOneRootPreparesWholeRegionWithoutWarmQueue) {
  auto cold = context(), warm = context(); auto owner = storage(cold);
  const auto bf = dense(c::DataType::kBFloat16, {kWidth});
  const auto scalar = dense(c::DataType::kFloat32, {1});
  auto b = g::GraphBuilder::create().value();
  const auto x = b.input(dense(c::DataType::kFloat32, {2, kWidth})).value();
  const auto w = b.constant(bf).value(), one = b.constant(scalar).value();
  const auto cast_node = b.add_node(c::Operation(c::Cast{c::DataType::kFloat32}),
      std::array{w}).value();
  const auto add_node = b.add_node(c::Operation(c::Binary{}),
      std::array{b.result(cast_node).value(), one}).value();
  const auto product_node = b.add_node(c::Operation(c::Binary{c::BinaryKind::kMultiply}),
      std::array{x, b.result(add_node).value()}).value();
  const auto narrow_node = b.add_node(c::Operation(c::Cast{c::DataType::kBFloat16}),
      std::array{b.result(product_node).value()}).value();
  const auto y = b.result(narrow_node).value(); ASSERT_TRUE(b.publish(y).ok());
  const auto graph = std::move(b).finish().value();
  const std::array constants{constant(owner, w, bf, weights()),
      constant(owner, one, scalar, std::vector{1.F})};
  const std::array domains{g::StorageDomain{.resource = owner.ref(), .owner = owner}};
  const std::array targets{
      g::ExecutionTarget{.source = g::ExecutionSource::acquire(cold).value(), .default_storage = g::StorageDomainId{0}},
      g::ExecutionTarget{.source = g::ExecutionSource::acquire(warm).value(), .default_storage = g::StorageDomainId{0}}};
  const std::array lanes{g::ExecutionLane{.target = g::ExecutionTargetId{0}},
                         g::ExecutionLane{.target = g::ExecutionTargetId{1}}};
  std::vector<g::NodePlacement> placements;
  for (auto node : {cast_node, add_node})
    placements.push_back({node, g::ExecutionTargetId{0}, g::PlacementStrength::kRequired});
  for (auto node : {product_node, narrow_node})
    placements.push_back({node, g::ExecutionTargetId{1}, g::PlacementStrength::kRequired});
  auto compiled = g::compile(graph, {.storage_domains = domains, .targets = targets, .lanes = lanes},
      {.constants = constants, .node_placements = placements});
  ASSERT_TRUE(compiled) << compiled.error().message();
  const auto step = step_for(compiled->report, add_node); ASSERT_NE(step, SIZE_MAX);
  EXPECT_EQ(compiled->report.step(step)->kind, g::CompileStepKind::kProviderRegion);
  EXPECT_EQ(compiled->report.step(step)->dispatch_count, 0U);
  EXPECT_EQ(compiled->report.step(step)->output_count, 0U);
  EXPECT_GT(compiled->report.packed_constant_count(), 0U);
  EXPECT_FALSE(compiled->executable.requires_queue(g::ExecutionLaneId{0}).value());
  EXPECT_TRUE(compiled->executable.requires_queue(g::ExecutionLaneId{1}).value());
  // No warm Queue is supplied for the cold target. This checks the public
  // no-command carrier/report, not an inferred physical GPU kernel count.
  exercise_slots(*compiled, warm, g::ExecutionLaneId{1}, owner, x, {y}, {factor(weights(), 1)});
}

TEST(GgmlCudaConstantExpression, ImmutableViewsKeepOffsetsAndRepeatedExpressionsDistinct) {
  auto ctx = context(); auto owner = storage(ctx);
  const auto root_desc = dense(c::DataType::kBFloat16, {2, 1, kWidth});
  auto original = weights(), other = weights(3);
  original.insert(original.end(), other.begin(), other.end());
  auto b = g::GraphBuilder::create().value();
  const auto x = b.input(dense(c::DataType::kFloat32, {2, kWidth})).value();
  const auto w = b.constant(root_desc).value();
  const auto reshape = b.add_node(c::Operation(c::View::reshape(
      c::TensorShape::create(std::array{std::int64_t{2}, kWidth}).value()).value()),
      std::array{w}).value();
  const auto flat = b.result(reshape).value();
  const auto first = b.add_node(c::Operation(c::View::slice(0, 0, 1).value()),
      std::array{flat}).value();
  const auto second = b.add_node(c::Operation(c::View::slice(0, 1, 1).value()),
      std::array{flat}).value();
  const auto a = b.result(first).value(), z = b.result(second).value();
  std::vector<g::ValueId> casts, outputs;
  // Repeated Cast(a) may share its result. Cast(z) has the same description
  // and physical root, but a different byte offset and different values.
  for (const auto source : {a, a, z, flat})
    casts.push_back(b.add(c::Cast{c::DataType::kFloat32}, source).value());
  for (const auto cast : casts) {
    const auto product = b.add(c::Binary{c::BinaryKind::kMultiply}, x, cast).value();
    outputs.push_back(b.add(c::Cast{c::DataType::kBFloat16}, product).value());
    ASSERT_TRUE(b.publish(outputs.back()).ok());
  }
  const auto graph = std::move(b).finish().value();
  const auto left = graph.value_description(a).value();
  ASSERT_EQ(left, graph.value_description(z).value());
  ASSERT_TRUE(left.is_dense());
  std::vector<g::ConstantBinding> constants{constant(owner, w, root_desc, original)};
  const std::array domains{g::StorageDomain{.resource = owner.ref(), .owner = owner}};
  const std::array targets{g::ExecutionTarget{.source = g::ExecutionSource::acquire(ctx).value(),
      .default_storage = g::StorageDomainId{0}}};
  const std::array lanes{g::ExecutionLane{.target = g::ExecutionTargetId{0}}};
  auto compiled = g::compile(graph, {.storage_domains = domains, .targets = targets, .lanes = lanes},
      {.constants = constants});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_GT(compiled->report.packed_constant_count(), 0U);
  EXPECT_GE(compiled->report.packed_constant_bytes(), 4U * kWidth * sizeof(float));
  for (const auto node : {reshape, first, second}) {
    const auto step = step_for(compiled->report, node); ASSERT_NE(step, SIZE_MAX);
    EXPECT_EQ(compiled->report.step(step)->kind, g::CompileStepKind::kView);
    EXPECT_EQ(compiled->report.step(step)->dispatch_count, 0U);
  }
  constants.clear();
  exercise_slots(*compiled, ctx, g::ExecutionLaneId{0}, owner, x, outputs,
      {factor(weights(), 0), factor(weights(), 0), factor(other, 0), factor(original, 0)});
}

TEST(GgmlCudaConstantExpression, ExternalViewSingletonCastNeedsNoWarmQueue) {
  for (const bool permuted : {false, true}) {
    SCOPED_TRACE(permuted);
    auto cold = context(), warm = context(); auto owner = storage(cold);
    const auto root_desc = dense(c::DataType::kBFloat16, {2, 1, kWidth});
    auto original = weights(), other = weights(3);
    original.insert(original.end(), other.begin(), other.end());
    const auto flat_desc = dense(c::DataType::kBFloat16, {2, kWidth});
    const auto view_desc = permuted ? flat_desc.transposed().value() : flat_desc;
    const auto activation_desc = c::TensorDesc::dense(c::DataType::kFloat32,
                                                     view_desc.dimensions()).value();
    if (permuted) {
      ASSERT_FALSE(view_desc.is_dense());
      // Admission stays with the original Cast provider. This optional domain
      // probe cannot manufacture support for a strided source or a new View op.
      auto admitted = c::prepare(cold, c::Operation(c::Cast{c::DataType::kFloat32}),
          std::array{view_desc}, std::array{activation_desc}, owner.ref());
      if (!admitted) {
        ASSERT_TRUE(admitted.error().IsUnimplemented()) << admitted.error().message();
        RecordProperty("permuted_cast_domain", "unsupported");
        continue;
      }
      RecordProperty("permuted_cast_domain", "supported");
    }
    auto b = g::GraphBuilder::create().value();
    const auto x = b.input(activation_desc).value(), w = b.constant(root_desc).value();
    const auto reshape = b.add_node(c::Operation(c::View::reshape(
        c::TensorShape::create(flat_desc.dimensions()).value()).value()), std::array{w}).value();
    std::vector<g::NodeId> views{reshape};
    auto source = b.result(reshape).value();
    if (permuted) {
      const auto node = b.add_node(c::Operation(c::View::permute(
          std::array<std::int32_t, 2>{1, 0}).value()), std::array{source}).value();
      views.push_back(node); source = b.result(node).value();
    }
    const auto cast = b.add_node(c::Operation(c::Cast{c::DataType::kFloat32}),
        std::array{source}).value();
    const auto product = b.add_node(c::Operation(c::Binary{c::BinaryKind::kMultiply}),
        std::array{x, b.result(cast).value()}).value();
    const auto narrow = b.add_node(c::Operation(c::Cast{c::DataType::kBFloat16}),
        std::array{b.result(product).value()}).value();
    const auto y = b.result(narrow).value(); ASSERT_TRUE(b.publish(y).ok());
    const auto graph = std::move(b).finish().value();
    std::vector<g::ConstantBinding> constants{constant(owner, w, root_desc, original)};
    const std::array domains{g::StorageDomain{.resource = owner.ref(), .owner = owner}};
    const std::array targets{
        g::ExecutionTarget{.source = g::ExecutionSource::acquire(cold).value(), .default_storage = g::StorageDomainId{0}},
        g::ExecutionTarget{.source = g::ExecutionSource::acquire(warm).value(), .default_storage = g::StorageDomainId{0}}};
    const std::array lanes{g::ExecutionLane{.target = g::ExecutionTargetId{0}},
                           g::ExecutionLane{.target = g::ExecutionTargetId{1}}};
    std::vector<g::NodePlacement> placements;
    for (const auto node : views)
      placements.push_back({node, g::ExecutionTargetId{0}, g::PlacementStrength::kRequired});
    placements.push_back({cast, g::ExecutionTargetId{0}, g::PlacementStrength::kRequired});
    for (const auto node : {product, narrow})
      placements.push_back({node, g::ExecutionTargetId{1}, g::PlacementStrength::kRequired});
    auto compiled = g::compile(graph, {.storage_domains = domains, .targets = targets, .lanes = lanes},
        {.constants = constants, .node_placements = placements});
    ASSERT_TRUE(compiled) << compiled.error().message();
    const auto step = step_for(compiled->report, cast); ASSERT_NE(step, SIZE_MAX);
    EXPECT_EQ(compiled->report.step(step)->kind, g::CompileStepKind::kProviderRegion);
    EXPECT_EQ(compiled->report.step(step)->node_count, 1U);
    EXPECT_EQ(compiled->report.step_node(step, 0).value(), cast);
    EXPECT_EQ(compiled->report.step(step)->dispatch_count, 0U);
    EXPECT_EQ(compiled->report.step(step)->output_count, 0U);
    EXPECT_GT(compiled->report.packed_constant_count(), 0U);
    EXPECT_GE(compiled->report.packed_constant_bytes(), 2U * kWidth * sizeof(float));
    for (const auto node : views) {
      const auto metadata = step_for(compiled->report, node); ASSERT_NE(metadata, SIZE_MAX);
      EXPECT_NE(metadata, step);
      EXPECT_EQ(compiled->report.step(metadata)->kind, g::CompileStepKind::kView);
      EXPECT_EQ(compiled->report.step(metadata)->dispatch_count, 0U);
    }
    EXPECT_FALSE(compiled->executable.requires_queue(g::ExecutionLaneId{0}).value());
    EXPECT_TRUE(compiled->executable.requires_queue(g::ExecutionLaneId{1}).value());
    auto scale = factor(original, 0);
    if (permuted)
      for (std::size_t i = 0; i < std::size_t(kWidth); ++i) {
        scale[2 * i] = static_cast<float>(half_test::decode(original[i]));
        scale[2 * i + 1] = static_cast<float>(half_test::decode(original[kWidth + i]));
      }
    constants.clear();
    // Only the downstream lane receives queues; View remains a real alias
    // step, while the singleton Cast is represented by its ready artifact.
    exercise_slots(*compiled, warm, g::ExecutionLaneId{1}, owner, x, {y}, {scale}, activation_desc);
  }
}

TEST(GgmlCudaConstantExpression, PublishedCastViewsAndDynamicWeightsKeepTheirContracts) {
  for (const int mode : {0, 1, 2}) {
    SCOPED_TRACE(mode); // Published Cast/View and a dynamic weight through View.
    auto ctx = context(); auto owner = storage(ctx);
    const auto bf = dense(c::DataType::kBFloat16, {kWidth});
    const auto f = dense(c::DataType::kFloat32, {kWidth});
    const auto input_desc = dense(c::DataType::kFloat32, {2, kWidth});
    const auto out_desc = dense(c::DataType::kBFloat16, {2, kWidth});
    const auto scalar = dense(c::DataType::kFloat32, {1});
    auto b = g::GraphBuilder::create().value();
    const auto x = b.input(input_desc).value();
    const auto w = (mode == 2 ? b.input(bf) : b.constant(bf)).value();
    const auto one = b.constant(scalar).value();
    auto source_weight = w;
    if (mode == 2) source_weight = b.add(c::View::reshape(
        c::TensorShape::create(std::array{std::int64_t{1}, kWidth}).value()).value(), w).value();
    const auto cast_node = b.add_node(c::Operation(c::Cast{c::DataType::kFloat32}), std::array{source_weight}).value();
    const auto cast = b.result(cast_node).value();
    auto published = cast;
    if (mode == 1) published = b.add(c::View::reshape(
        c::TensorShape::create(std::array{std::int64_t{1}, kWidth}).value()).value(), cast).value();
    if (mode < 2) ASSERT_TRUE(b.publish(published).ok());
    const auto scale = b.add(c::Binary{}, cast, one).value();
    const auto product = b.add(c::Binary{c::BinaryKind::kMultiply}, x, scale).value();
    const auto y = b.add(c::Cast{c::DataType::kBFloat16}, product).value();
    ASSERT_TRUE(b.publish(y).ok());
    const auto control_node = b.add_node(c::Operation(c::View::reshape(
        c::TensorShape::create(input_desc.dimensions()).value()).value()), std::array{x}).value();
    ASSERT_TRUE(b.depends_on(control_node, cast_node).ok());
    const auto control = b.result(control_node).value(); ASSERT_TRUE(b.publish(control).ok());
    const auto graph = std::move(b).finish().value();
    std::vector<g::ConstantBinding> constants{constant(owner, one, scalar, std::vector{1.F})};
    if (mode != 2) constants.push_back(constant(owner, w, bf, weights()));
    const std::array domains{g::StorageDomain{.resource = owner.ref(), .owner = owner}};
    const std::array targets{g::ExecutionTarget{.source = g::ExecutionSource::acquire(ctx).value(),
        .default_storage = g::StorageDomainId{0}}};
    const std::array lanes{g::ExecutionLane{.target = g::ExecutionTargetId{0}}};
    auto compiled = g::compile(graph, {.storage_domains = domains, .targets = targets, .lanes = lanes},
        {.constants = constants});
    ASSERT_TRUE(compiled) << compiled.error().message();
    // A metadata View cannot turn a rebindable input into an immutable root.
    if (mode == 2) EXPECT_EQ(compiled->report.packed_constant_count(), 0U);
    const auto step = step_for(compiled->report, cast_node); ASSERT_NE(step, SIZE_MAX);
    EXPECT_GT(compiled->report.step(step)->dispatch_count, 0U);
    std::vector<Buffer> retained;
    auto queue = ctx.create_queue().value();
    auto run = compiled->executable.create_invocation().value();
    std::vector<BF16> old;
    for (int round = 0; round < 2; ++round) {
      std::vector<float> input(2 * kWidth);
      for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 19) - 9 + round) / 64;
      auto source = upload<float>(owner.ref(), input);
      auto weight = upload<BF16>(owner.ref(), weights(mode == 2 ? round : 0));
      auto output = output_buffer(owner.ref(), out_desc), exposed = output_buffer(owner.ref(), f);
      ASSERT_TRUE(run.bind(x, c::ConstTensorView::create(input_desc, source.view()).value()).ok());
      if (mode == 2) ASSERT_TRUE(run.bind(w, c::ConstTensorView::create(bf, weight.view()).value()).ok());
      if (mode == 0) ASSERT_TRUE(run.bind(cast, c::TensorView::create(f, exposed.view()).value()).ok());
      ASSERT_TRUE(run.bind(y, c::TensorView::create(out_desc, output.view()).value()).ok());
      auto done = run.submit(queue); ASSERT_TRUE(done) << done.error().message(); ASSERT_TRUE(done->wait().ok());
      expect_bits(output, multiply(input, factor(weights(mode == 2 ? round : 0), 1)));
      if (mode < 2) {
        auto view = run.output(published); ASSERT_TRUE(view) << view.error().message();
        std::vector<float> actual(kWidth);
        ASSERT_TRUE(view->storage().read(std::as_writable_bytes(std::span(actual))).ok());
        EXPECT_EQ(actual, factor(weights(), 0));
      }
      auto alias = run.output(control); ASSERT_TRUE(alias) << alias.error().message();
      std::vector<float> actual(input.size());
      ASSERT_TRUE(alias->storage().read(std::as_writable_bytes(std::span(actual))).ok());
      EXPECT_EQ(actual, input);
      if (!round) old = half_test::read<BF16>(output);
      retained.push_back(std::move(source)); retained.push_back(std::move(weight));
      retained.push_back(std::move(output)); retained.push_back(std::move(exposed));
      if (round) expect_bits(retained[2], old);
    }
  }
}
} // namespace
