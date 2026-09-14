#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <utility>

#include "hyle/graph/graph.h"

namespace {

using hyle::Status;
using hyle::compute::AliasPolicy;
using hyle::compute::DataType;
using hyle::compute::EffectKind;
using hyle::compute::GatedActivation;
using hyle::compute::LayerNorm;
using hyle::compute::MatMul;
using hyle::compute::Operation;
using hyle::compute::OperationId;
using hyle::compute::OperationKind;
using hyle::compute::OperationTraits;
using hyle::compute::PrecisionPolicy;
using hyle::compute::RmsNorm;
using hyle::compute::StorageFormat;
using hyle::compute::StorageFormatId;
using hyle::compute::TensorDesc;
using hyle::compute::TopK;
using hyle::graph::Dimension;
using hyle::graph::DimensionBounds;
using hyle::graph::Graph;
using hyle::graph::GraphBuilder;
using hyle::graph::GraphBuilderOptions;
using hyle::graph::TensorSpec;
using hyle::graph::TensorSpecKind;
using hyle::graph::ValueKind;
using hyle::mem::AllocatorRef;

TensorDesc row(std::int64_t size) {
  const std::array shape{size};
  return TensorDesc::dense(DataType::kFloat32, shape).value();
}

TensorDesc matrix(std::int64_t rows, std::int64_t columns) {
  const std::array shape{rows, columns};
  return TensorDesc::dense(DataType::kFloat32, shape).value();
}

struct AllocatorProbe final {
  AllocatorRef backing = hyle::mem::system_allocator();
  std::size_t allocations = 0;
  std::size_t deallocations = 0;
  std::size_t attempts = 0;
  std::size_t fail_at = SIZE_MAX;
  bool fail = false;
};

void *probe_allocate(void *opaque, std::size_t size,
                     std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(opaque);
  if (probe.fail || probe.attempts++ == probe.fail_at)
    return nullptr;
  void *result = probe.backing.allocate(size, alignment);
  if (result != nullptr)
    ++probe.allocations;
  return result;
}

void probe_free(void *opaque, void *pointer, std::size_t size,
                std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(opaque);
  if (pointer != nullptr)
    ++probe.deallocations;
  probe.backing.deallocate(pointer, size, alignment);
}

void *probe_reallocate(void *opaque, void *pointer, std::size_t old_size,
                       std::size_t new_size, std::size_t alignment) noexcept {
  auto &probe = *static_cast<AllocatorProbe *>(opaque);
  if (probe.fail || probe.attempts++ == probe.fail_at)
    return nullptr;
  return probe.backing.reallocate(pointer, old_size, new_size, alignment);
}

AllocatorRef probe_allocator(AllocatorProbe &probe) noexcept {
  return AllocatorRef(&probe_allocate, &probe_free, &probe_reallocate, &probe);
}

struct ExternalEffect final {
  static constexpr OperationId operation_id{0x48594c45U, 0x45464654U};
  static constexpr std::string_view operation_name = "test-external-effect";

  OperationTraits operation_traits() const noexcept {
    return {{0, 0},
            0,
            AliasPolicy::kNone,
            EffectKind::kExternalSideEffect,
            PrecisionPolicy::kExact,
            DataType::kUnknown,
            false};
  }

  Status
  infer_output_descriptions(std::span<const TensorDesc> inputs,
                            std::span<TensorDesc> outputs) const noexcept {
    return inputs.empty() && outputs.empty()
               ? Status::Ok()
               : Status::InvalidArgument("effect operation has no tensors");
  }
};

TEST(GraphBuilder, BuildsNaturalMultiNodeTransformerFragment) {
  GraphBuilder builder = GraphBuilder::create().value();
  const TensorDesc activation = matrix(2, 4);
  const TensorDesc norm_weight = row(4);
  const TensorDesc stored_gate_up = matrix(8, 4);
  const TensorDesc logical_gate_up = stored_gate_up.transposed().value();

  const auto input = builder.input(activation).value();
  const auto weight = builder.input(norm_weight).value();
  const auto matrix_weight = builder.input(logical_gate_up).value();
  const auto normalized = builder.add(RmsNorm{}, input, weight).value();
  const auto projected =
      builder.add(MatMul{}, normalized, matrix_weight).value();
  const auto activated = builder.add(GatedActivation{}, projected).value();
  ASSERT_TRUE(builder.publish(activated).ok());

  Graph graph = std::move(builder).finish().value();
  EXPECT_EQ(graph.input_count(), 3u);
  EXPECT_EQ(graph.node_count(), 3u);
  EXPECT_EQ(graph.output_count(), 1u);
  EXPECT_EQ(graph.value_count(), 6u);
  EXPECT_EQ(graph.input_id(0).value(), input);
  EXPECT_EQ(graph.input_id(1).value(), weight);
  EXPECT_EQ(graph.input_id(2).value(), matrix_weight);
  EXPECT_EQ(graph.output_id().value(), activated);
  EXPECT_EQ(graph.output_description().value(), activation);
  EXPECT_EQ(graph.value_description(projected).value(), matrix(2, 8));

  const auto norm_node = graph.node_id(0).value();
  const auto matmul_node = graph.node_id(1).value();
  const auto activation_node = graph.node_id(2).value();
  EXPECT_EQ(graph.operation(norm_node)->kind(), OperationKind::kRmsNorm);
  EXPECT_EQ(graph.operation(matmul_node)->kind(), OperationKind::kMatMul);
  EXPECT_EQ(graph.operation(activation_node)->kind(),
            OperationKind::kGatedActivation);
  EXPECT_EQ(graph.node_input_count(norm_node).value(), 2u);
  EXPECT_EQ(graph.node_input(norm_node, 0).value(), input);
  EXPECT_EQ(graph.node_input(norm_node, 1).value(), weight);
  EXPECT_EQ(graph.result(norm_node).value(), normalized);
  EXPECT_EQ(graph.result(matmul_node).value(), projected);
  EXPECT_EQ(graph.result(activation_node).value(), activated);
  EXPECT_EQ(graph.producer(normalized).value(), norm_node);
  EXPECT_EQ(input.kind(), ValueKind::kInput);
  EXPECT_EQ(normalized.kind(), ValueKind::kNodeOutput);
}

TEST(GraphBuilder, BuildsBoundedSymbolicGraphWithExplicitOutputSpecs) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto batch = builder.dimension({.minimum = 1, .maximum = 64}).value();
  const auto tokens =
      builder.dimension({.minimum = 8, .maximum = 4096, .multiple_of = 8})
          .value();
  const std::array activation_shape{Dimension(batch), Dimension(tokens),
                                    Dimension::fixed(128)};
  const TensorSpec activation =
      TensorSpec::dense(DataType::kFloat16, activation_shape).value();
  const auto input = builder.input(activation).value();
  const auto output = builder.add(RmsNorm{}, activation, input).value();
  ASSERT_TRUE(builder.publish(output).ok());

  Graph graph = std::move(builder).finish().value();
  EXPECT_EQ(graph.dimension_count(), 2u);
  EXPECT_EQ(graph.dimension_id(0).value(), batch);
  EXPECT_EQ(graph.dimension_id(1).value(), tokens);
  EXPECT_EQ(graph.dimension_bounds(tokens).value(),
            (DimensionBounds{.minimum = 8, .maximum = 4096, .multiple_of = 8}));
  EXPECT_EQ(graph.input_spec().value(), activation);
  EXPECT_EQ(graph.output_spec().value(), activation);
  EXPECT_EQ(graph.value_spec(output)->kind(), TensorSpecKind::kDense);
  EXPECT_TRUE(graph.input_description().error().IsFailedPrecondition());
  EXPECT_TRUE(graph.output_description().error().IsFailedPrecondition());
}

TEST(GraphBuilder, RejectsInvalidAndForeignSymbolicDimensions) {
  GraphBuilder first = GraphBuilder::create().value();
  EXPECT_TRUE(first.dimension({.minimum = -1, .maximum = 4})
                  .error()
                  .IsInvalidArgument());
  EXPECT_TRUE(first.dimension({.minimum = 9, .maximum = 15, .multiple_of = 8})
                  .error()
                  .IsInvalidArgument());

  GraphBuilder second = GraphBuilder::create().value();
  const auto foreign = second.dimension({.minimum = 1, .maximum = 8}).value();
  const std::array shape{Dimension(foreign), Dimension::fixed(4)};
  const TensorSpec spec = TensorSpec::dense(DataType::kFloat32, shape).value();
  EXPECT_TRUE(first.input(spec).error().IsInvalidArgument());

  const auto local = first.dimension({.minimum = 1, .maximum = 8}).value();
  const TensorSpec local_spec =
      TensorSpec::dense(DataType::kFloat32,
                        std::array{Dimension(local), Dimension::fixed(4)})
          .value();
  const auto input = first.input(local_spec).value();
  EXPECT_TRUE(first.add(RmsNorm{}, input).error().IsFailedPrecondition());
}

TEST(GraphBuilder, DerivedDimensionsShareFiniteProgramAndRemainGraphLocal) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto tokens = builder.dimension({0, 8, 2}).value();
  const auto doubled =
      builder.dimension_product(Dimension(tokens), Dimension::fixed(2)).value();
  // Input declarations may be interleaved with derived values.
  const auto extra = builder.dimension({1, 4, 1}).value();
  const auto sum =
      builder.dimension_sum(Dimension(doubled), Dimension(extra)).value();
  const auto split =
      builder.dimension_quotient(Dimension(doubled), Dimension::fixed(2))
          .value();
  const auto broadcast =
      builder.dimension_broadcast(Dimension(split), Dimension::fixed(1))
          .value();
  ASSERT_TRUE(
      builder.require_dimension_equal(Dimension(broadcast), Dimension(tokens))
          .ok());
  ASSERT_TRUE(
      builder
          .require_dimension_less_equal(Dimension(extra), Dimension::fixed(4))
          .ok());
  GraphBuilder foreign = GraphBuilder::create().value();
  const auto other = foreign.dimension({1, 4, 1}).value();
  EXPECT_TRUE(builder.dimension_sum(Dimension(other), Dimension(tokens))
                  .error()
                  .IsInvalidArgument());
  EXPECT_TRUE(builder.dimension_product(Dimension{}, Dimension(tokens))
                  .error()
                  .IsInvalidArgument());
  EXPECT_TRUE(
      builder.require_dimension_equal(Dimension(other), Dimension(tokens))
          .IsInvalidArgument());
  EXPECT_TRUE(builder.dimension_quotient(Dimension(tokens), Dimension::fixed(0))
                  .error()
                  .IsInvalidArgument());
  constexpr auto maximum = (std::numeric_limits<std::int64_t>::max)();
  EXPECT_TRUE(
      builder.dimension_sum(Dimension::fixed(maximum), Dimension::fixed(1))
          .error()
          .IsOutOfRange());
  const auto input =
      builder
          .input(
              TensorSpec::dense(DataType::kFloat32,
                                std::array{Dimension(sum), Dimension::fixed(4)})
                  .value())
          .value();
  ASSERT_TRUE(builder.publish(input).ok());
  Graph graph = std::move(builder).finish().value();
  EXPECT_EQ(graph.dimension_count(), 6u);
  EXPECT_EQ(graph.independent_dimension_count(), 2u);
  EXPECT_TRUE(graph.dimension_is_independent(tokens).value());
  EXPECT_TRUE(graph.dimension_is_independent(extra).value());
  EXPECT_FALSE(graph.dimension_is_independent(sum).value());
  EXPECT_TRUE(
      graph.dimension_is_independent(other).error().IsInvalidArgument());
  EXPECT_EQ(graph.dimension_bounds(doubled).value(),
            (DimensionBounds{0, 16, 4}));
  EXPECT_EQ(graph.dimension_bounds(sum).value(), (DimensionBounds{1, 20, 1}));
  EXPECT_TRUE(
      Graph{}.dimension_is_independent(tokens).error().IsFailedPrecondition());
  EXPECT_EQ(Graph{}.independent_dimension_count(), 0u);
}

TEST(GraphBuilder, ShapeGrowthFailureLeavesNoHalfDimensionOrGuard) {
  AllocatorProbe probe;
  {
    auto builder =
        GraphBuilder::create({.metadata_allocator = probe_allocator(probe)})
            .value();
    const auto input = builder.input(row(4)).value();
    ASSERT_TRUE(builder.publish(input).ok());
    const auto first = builder.dimension({0, 8, 1}).value();
    for (int i = 0; i < 3; ++i)
      ASSERT_TRUE(builder.dimension_sum(Dimension(first), Dimension::fixed(i)));
    for (int i = 0; i < 4; ++i)
      ASSERT_TRUE(
          builder.require_dimension_equal(Dimension(first), Dimension(first))
              .ok());
    probe.fail = true;
    EXPECT_TRUE(builder.dimension_product(Dimension(first), Dimension::fixed(2))
                    .error()
                    .IsResourceExhausted());
    EXPECT_TRUE(
        builder.require_dimension_equal(Dimension(first), Dimension::fixed(7))
            .IsResourceExhausted());
    probe.fail = false;
    const auto final =
        builder.dimension_sum(Dimension(first), Dimension::fixed(4)).value();
    EXPECT_EQ(final.ordinal(), 4u);
    Graph graph = std::move(builder).finish().value();
    EXPECT_EQ(graph.dimension_count(), 5u);
    EXPECT_EQ(graph.independent_dimension_count(), 1u);
  }
  EXPECT_EQ(probe.allocations, probe.deallocations);
}

TEST(GraphBuilder, FixedSpecsShareConcreteValidationAndInference) {
  const TensorSpec fixed =
      TensorSpec::dense(DataType::kFloat32,
                        std::array{Dimension::fixed(2), Dimension::fixed(4)})
          .value();
  EXPECT_FALSE(fixed.is_symbolic());
  EXPECT_EQ(fixed.concrete_description().value(), matrix(2, 4));

  GraphBuilder builder = GraphBuilder::create().value();
  const auto input = builder.input(fixed).value();
  const auto output = builder.add(RmsNorm{}, input).value();
  EXPECT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();
  EXPECT_EQ(graph.output_description().value(), matrix(2, 4));

  GraphBuilder mismatch = GraphBuilder::create().value();
  const auto mismatch_input = mismatch.input(fixed).value();
  const TensorSpec wrong =
      TensorSpec::dense(DataType::kFloat32,
                        std::array{Dimension::fixed(2), Dimension::fixed(5)})
          .value();
  EXPECT_TRUE(mismatch.add(RmsNorm{}, wrong, mismatch_input)
                  .error()
                  .IsInvalidArgument());
}

TEST(GraphBuilder, SupportsBranchesSharedValuesAndMultipleOutputs) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto input = builder.input(matrix(2, 4)).value();
  const auto first_weight = builder.input(matrix(4, 3)).value();
  const auto second_weight = builder.input(matrix(4, 5)).value();
  const auto first = builder.add(MatMul{}, input, first_weight).value();
  const auto second = builder.add(MatMul{}, input, second_weight).value();
  ASSERT_TRUE(builder.publish(first).ok());
  ASSERT_TRUE(builder.publish(second).ok());

  Graph graph = std::move(builder).finish().value();
  EXPECT_EQ(graph.input_count(), 3u);
  EXPECT_EQ(graph.node_count(), 2u);
  EXPECT_EQ(graph.output_count(), 2u);
  EXPECT_EQ(graph.node_input(graph.node_id(0).value(), 0).value(), input);
  EXPECT_EQ(graph.node_input(graph.node_id(1).value(), 0).value(), input);
  EXPECT_EQ(graph.output_id(0).value(), first);
  EXPECT_EQ(graph.output_id(1).value(), second);
  EXPECT_EQ(graph.output_description(0).value(), matrix(2, 3));
  EXPECT_EQ(graph.output_description(1).value(), matrix(2, 5));
}

TEST(GraphBuilder, RepresentsConstantsGeneralArityAndMultiResultNodes) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto input = builder.input(matrix(2, 4)).value();
  const auto scale = builder.constant(row(4)).value();
  const auto bias = builder.constant(row(4)).value();
  const std::array normalization_inputs{input, scale, bias};
  const auto normalization =
      builder.add_node(Operation(LayerNorm{}), normalization_inputs).value();
  const auto normalized = builder.result(normalization).value();
  const std::array top_k_inputs{normalized};
  const auto top_k =
      builder.add_node(Operation(TopK{.count = 2}), top_k_inputs).value();
  const auto values = builder.result(top_k, 0).value();
  const auto indices = builder.result(top_k, 1).value();
  EXPECT_TRUE(builder.result(top_k, 2).error().IsOutOfRange());
  ASSERT_TRUE(builder.publish(values).ok());
  ASSERT_TRUE(builder.publish(indices).ok());

  Graph graph = std::move(builder).finish().value();
  EXPECT_EQ(graph.input_count(), 1u);
  EXPECT_EQ(graph.constant_count(), 2u);
  EXPECT_EQ(graph.node_count(), 2u);
  EXPECT_EQ(graph.value_count(), 6u);
  EXPECT_EQ(graph.constant_id(0).value(), scale);
  EXPECT_EQ(graph.constant_id(1).value(), bias);
  EXPECT_EQ(graph.constant_description(0).value(), row(4));
  EXPECT_EQ(scale.kind(), ValueKind::kConstant);
  EXPECT_EQ(graph.node_input_count(normalization).value(), 3u);
  EXPECT_EQ(graph.node_output_count(normalization).value(), 1u);
  EXPECT_EQ(graph.node_output_count(top_k).value(), 2u);
  EXPECT_EQ(graph.result(top_k, 0).value(), values);
  EXPECT_EQ(graph.result(top_k, 1).value(), indices);
  EXPECT_EQ(graph.producer(values).value(), top_k);
  EXPECT_TRUE(graph.producer(scale).error().IsFailedPrecondition());
}

TEST(GraphBuilder, PublishesInputAndConstantWithoutManufacturingNodes) {
  GraphBuilder input_builder = GraphBuilder::create().value();
  const auto input = input_builder.input(row(4)).value();
  ASSERT_TRUE(input_builder.publish(input).ok());
  Graph input_graph = std::move(input_builder).finish().value();
  EXPECT_EQ(input_graph.node_count(), 0u);
  EXPECT_EQ(input_graph.output_id().value(), input);

  GraphBuilder constant_builder = GraphBuilder::create().value();
  const auto constant = constant_builder.constant(row(4)).value();
  ASSERT_TRUE(constant_builder.publish(constant).ok());
  Graph constant_graph = std::move(constant_builder).finish().value();
  EXPECT_EQ(constant_graph.node_count(), 0u);
  EXPECT_EQ(constant_graph.constant_count(), 1u);
  EXPECT_EQ(constant_graph.output_id().value(), constant);
}

TEST(GraphBuilder, RequiresExplicitOrderingBetweenExternalEffects) {
  constexpr std::array<hyle::graph::ValueId, 0> no_inputs{};
  GraphBuilder unordered = GraphBuilder::create().value();
  ASSERT_TRUE(
      unordered.add_node(Operation::custom(ExternalEffect{}), no_inputs));
  ASSERT_TRUE(
      unordered.add_node(Operation::custom(ExternalEffect{}), no_inputs));
  EXPECT_TRUE(std::move(unordered).finish().error().IsFailedPrecondition());

  GraphBuilder builder = GraphBuilder::create().value();
  const auto first =
      builder.add_node(Operation::custom(ExternalEffect{}), no_inputs).value();
  const auto second =
      builder.add_node(Operation::custom(ExternalEffect{}), no_inputs).value();
  EXPECT_TRUE(builder.depends_on(first, second).IsInvalidArgument());
  ASSERT_TRUE(builder.depends_on(second, first).ok());
  ASSERT_TRUE(builder.depends_on(second, first).ok());
  Graph graph = std::move(builder).finish().value();
  EXPECT_EQ(graph.node_count(), 2u);
  EXPECT_EQ(graph.output_count(), 0u);
  EXPECT_EQ(graph.node_output_count(first).value(), 0u);
  EXPECT_EQ(graph.node_dependency_count(first).value(), 0u);
  EXPECT_EQ(graph.node_dependency_count(second).value(), 1u);
  EXPECT_EQ(graph.node_dependency(second, 0).value(), first);
  EXPECT_TRUE(graph.node_dependency(second, 1).error().IsOutOfRange());
}

struct ResourceEffect final {
  static constexpr OperationId operation_id{0x48594c45U, 0x52455304U};
  static constexpr std::string_view operation_name = "graph-resource-effect";
  std::uint8_t resource_count = 1;
  bool has_tensor = false;
  hyle::compute::ResourceAccess access =
      hyle::compute::ResourceAccess::kReadWrite;
  OperationTraits operation_traits() const noexcept {
    const auto arity = static_cast<std::uint16_t>(has_tensor ? 1 : 0);
    return {.inputs = {arity, arity},
            .outputs = static_cast<std::uint8_t>(arity),
            .effect = EffectKind::kResourceAccess,
            .resource_count = resource_count};
  }
  hyle::compute::ResourceAccess resource_access(std::size_t) const noexcept {
    return access;
  }
  Status
  infer_output_descriptions(std::span<const TensorDesc> inputs,
                            std::span<TensorDesc> outputs) const noexcept {
    if (has_tensor)
      outputs[0] = inputs[0];
    return Status::Ok();
  }
};

TEST(GraphBuilder, ResourceDomainsMustOrderUnknownEffectsInBothDirections) {
  for (const bool unknown_first : {false, true}) {
    auto builder = GraphBuilder::create().value();
    const Operation resource = Operation::custom(ResourceEffect{});
    const Operation unknown = Operation::custom(ExternalEffect{});
    const auto first =
        builder.add_node(unknown_first ? unknown : resource, {}).value();
    const auto second =
        builder.add_node(unknown_first ? resource : unknown, {}).value();
    EXPECT_TRUE(std::move(builder).finish().error().IsFailedPrecondition());
    const auto domain = builder.effect_resource().value();
    ASSERT_TRUE(
        builder
            .bind_resources(unknown_first ? second : first, std::array{domain})
            .ok());
    EXPECT_TRUE(std::move(builder).finish().error().IsFailedPrecondition());
    ASSERT_TRUE(builder.depends_on(second, first).ok());
    EXPECT_TRUE(std::move(builder).finish().has_value());
  }
  auto builder = GraphBuilder::create().value();
  ASSERT_TRUE(builder.add_node(Operation::custom(ResourceEffect{}), {}));
  ASSERT_TRUE(builder.add_node(Operation::custom(ResourceEffect{}), {}));
  EXPECT_TRUE(std::move(builder).finish().error().IsFailedPrecondition());
}

TEST(GraphBuilder,
     IndependentDomainsAndSameDomainReadersNeedNoArtificialEdges) {
  using hyle::compute::ResourceAccess;
  for (bool independent : {false, true}) {
    auto builder = GraphBuilder::create().value();
    const auto first_resource = builder.effect_resource().value();
    const auto second_resource =
        independent ? builder.effect_resource().value() : first_resource;
    ResourceEffect effect;
    effect.access =
        independent ? ResourceAccess::kReadWrite : ResourceAccess::kRead;
    const auto first = builder.add_node(Operation::custom(effect), {}).value();
    const auto second = builder.add_node(Operation::custom(effect), {}).value();
    ASSERT_TRUE(builder.bind_resources(first, std::array{first_resource}).ok());
    ASSERT_TRUE(
        builder.bind_resources(second, std::array{second_resource}).ok());
    auto result = std::move(builder).finish();
    ASSERT_TRUE(result.has_value()) << result.error().message();
    Graph graph = *result;
    EXPECT_EQ(graph.effect_resource_count(), independent ? 2u : 1u);
    EXPECT_EQ(graph.effect_resource_id(0).value(), first_resource);
    EXPECT_EQ(graph.node_resource(first, 0).value(), first_resource);
    EXPECT_EQ(graph.node_resource(second, 0).value(), second_resource);
    EXPECT_EQ(graph.node_dependency_count(first).value(), 0u);
    EXPECT_EQ(graph.node_dependency_count(second).value(), 0u);
    EXPECT_EQ(graph.node_resource_count(second).value(), 1u);
    EXPECT_TRUE(graph.node_resource(second, 1).error().IsOutOfRange());
    EXPECT_TRUE(graph.effect_resource_id(2).error().IsOutOfRange());
  }
}

TEST(GraphBuilder,
     EveryConflictingAccessPairRequiresOrderAndBindingsAreFallible) {
  using hyle::compute::ResourceAccess;
  constexpr std::array modes{ResourceAccess::kRead, ResourceAccess::kWrite,
                             ResourceAccess::kReadWrite};
  for (auto left : modes) {
    for (auto right : modes) {
      auto builder = GraphBuilder::create().value();
      auto foreign = GraphBuilder::create().value();
      const auto resource = builder.effect_resource().value();
      const auto first =
          builder
              .add_node(Operation::custom(ResourceEffect{.access = left}), {})
              .value();
      const auto second =
          builder
              .add_node(Operation::custom(ResourceEffect{.access = right}), {})
              .value();
      EXPECT_TRUE(builder.bind_resources(first, {}).IsInvalidArgument());
      EXPECT_TRUE(builder
                      .bind_resources(
                          first, std::array{foreign.effect_resource().value()})
                      .IsInvalidArgument());
      EXPECT_TRUE(
          builder.bind_resources({}, std::array{resource}).IsInvalidArgument());
      ASSERT_TRUE(builder.bind_resources(first, std::array{resource}).ok());
      ASSERT_TRUE(builder.bind_resources(second, std::array{resource}).ok());
      if (left == ResourceAccess::kRead && right == ResourceAccess::kRead) {
        EXPECT_TRUE(std::move(builder).finish().has_value());
      } else {
        EXPECT_TRUE(std::move(builder).finish().error().IsFailedPrecondition());
        ASSERT_TRUE(builder.depends_on(second, first).ok());
        EXPECT_TRUE(std::move(builder).finish().has_value());
      }
    }
  }
}

TEST(GraphBuilder, WriterMustJoinAllReadersNotJustTheLatestOrdinal) {
  using hyle::compute::ResourceAccess;
  auto builder = GraphBuilder::create().value();
  const auto resource = builder.effect_resource().value();
  std::array<hyle::graph::NodeId, 3> nodes{};
  for (std::size_t index = 0; index < nodes.size(); ++index) {
    nodes[index] =
        builder
            .add_node(Operation::custom(ResourceEffect{
                          .access = index == 2 ? ResourceAccess::kWrite
                                               : ResourceAccess::kRead}),
                      {})
            .value();
    ASSERT_TRUE(
        builder.bind_resources(nodes[index], std::array{resource}).ok());
  }
  ASSERT_TRUE(builder.depends_on(nodes[2], nodes[1]).ok());
  EXPECT_TRUE(std::move(builder).finish().error().IsFailedPrecondition());
  ASSERT_TRUE(builder.depends_on(nodes[2], nodes[0]).ok());
  EXPECT_TRUE(std::move(builder).finish().has_value());
}

TEST(GraphBuilder, ResourceRebindingAndRepeatedPortsDoNotCreateNewDomains) {
  auto builder = GraphBuilder::create().value();
  const auto first_domain = builder.effect_resource().value();
  const auto second_domain = builder.effect_resource().value();
  const auto first =
      builder
          .add_node(Operation::custom(ResourceEffect{.resource_count = 2}), {})
          .value();
  const auto second =
      builder.add_node(Operation::custom(ResourceEffect{}), {}).value();
  ASSERT_TRUE(
      builder.bind_resources(first, std::array{first_domain, first_domain})
          .ok());
  ASSERT_TRUE(builder.bind_resources(second, std::array{first_domain}).ok());
  EXPECT_TRUE(std::move(builder).finish().error().IsFailedPrecondition());
  ASSERT_TRUE(builder.bind_resources(second, std::array{second_domain}).ok());
  auto graph = std::move(builder).finish();
  ASSERT_TRUE(graph.has_value());
  EXPECT_EQ(graph->node_resource(first, 0), graph->node_resource(first, 1));
  EXPECT_EQ(graph->node_resource(second, 0).value(), second_domain);
}

TEST(GraphBuilder, ResourceOrderingMatchesPairwiseDagOracle) {
  using hyle::compute::ResourceAccess;
  std::uint32_t random = 0x94a4015u;
  auto next = [&]() {
    random = random * 1664525u + 1013904223u;
    return random;
  };
  constexpr std::size_t count = 8;
  for (std::size_t trial = 0; trial < 512; ++trial) {
    auto builder = GraphBuilder::create().value();
    const std::array domains{builder.effect_resource().value(),
                             builder.effect_resource().value(),
                             builder.effect_resource().value()};
    std::array<hyle::graph::NodeId, count> nodes{};
    std::array<std::uint32_t, count> kind{}, domain{};
    std::array<std::array<bool, count>, count> reach{};
    for (std::size_t node = 0; node < count; ++node) {
      kind[node] = (next() >> 16) % 4; // pure, read, write, unknown
      domain[node] = (next() >> 16) % domains.size();
      if (kind[node] == 0) {
        const auto shape =
            hyle::compute::TensorShape::create(std::array<std::int64_t, 1>{1})
                .value();
        nodes[node] = builder
                          .add_node(Operation(hyle::compute::Fill{
                                        shape, DataType::kFloat32,
                                        hyle::compute::ScalarValue(0.0F)}),
                                    {})
                          .value();
      } else if (kind[node] == 3) {
        nodes[node] =
            builder.add_node(Operation::custom(ExternalEffect{}), {}).value();
      } else {
        nodes[node] =
            builder
                .add_node(
                    Operation::custom(ResourceEffect{
                        .access = kind[node] == 1 ? ResourceAccess::kRead
                                                  : ResourceAccess::kWrite}),
                    {})
                .value();
        ASSERT_TRUE(
            builder
                .bind_resources(nodes[node], std::array{domains[domain[node]]})
                .ok());
      }
      for (std::size_t prior = 0; prior < node; ++prior) {
        if ((next() >> 16) % 3 == 0) {
          reach[prior][node] = true;
          ASSERT_TRUE(builder.depends_on(nodes[node], nodes[prior]).ok());
        }
      }
    }
    for (std::size_t middle = 0; middle < count; ++middle)
      for (std::size_t left = 0; left < count; ++left)
        for (std::size_t right = 0; right < count; ++right)
          reach[left][right] = reach[left][right] ||
                               (reach[left][middle] && reach[middle][right]);
    bool legal = true;
    for (std::size_t left = 0; left < count; ++left) {
      for (std::size_t right = left + 1; right < count; ++right) {
        const bool conflict = kind[left] && kind[right] &&
                              (kind[left] == 3 || kind[right] == 3 ||
                               (domain[left] == domain[right] &&
                                (kind[left] == 2 || kind[right] == 2)));
        if (conflict && !reach[left][right])
          legal = false;
      }
    }
    const auto graph = std::move(builder).finish();
    EXPECT_EQ(graph.has_value(), legal) << "trial " << trial;
    if (!graph)
      EXPECT_TRUE(graph.error().IsFailedPrecondition());
  }
}

TEST(GraphBuilder, ResourceBindingAllocationFailureLeavesNodeRetryable) {
  AllocatorProbe probe;
  {
    auto builder =
        GraphBuilder::create({.metadata_allocator = probe_allocator(probe)})
            .value();
    const auto resource = builder.effect_resource().value();
    const auto node =
        builder
            .add_node(Operation::custom(ResourceEffect{.resource_count = 255}),
                      {})
            .value();
    std::array<hyle::graph::EffectResourceId, 255> resources{};
    resources.fill(resource);
    probe.fail = true;
    EXPECT_TRUE(builder.bind_resources(node, resources).IsResourceExhausted());
    probe.fail = false;
    EXPECT_TRUE(std::move(builder).finish().error().IsFailedPrecondition());
    ASSERT_TRUE(builder.bind_resources(node, resources).ok());
    auto graph = std::move(builder).finish();
    ASSERT_TRUE(graph.has_value());
    EXPECT_EQ(graph->node_resource(node, 254).value(), resource);
  }
  EXPECT_EQ(probe.allocations, probe.deallocations);
}

TEST(GraphBuilder, EffectAnalysisAllocationFailuresLeaveTheBuilderRetryable) {
  bool reached_success = false;
  for (std::size_t fail_offset = 0; fail_offset < 64; ++fail_offset) {
    SCOPED_TRACE(fail_offset);
    AllocatorProbe probe;
    {
      auto builder =
          GraphBuilder::create({.metadata_allocator = probe_allocator(probe)})
              .value();
      std::array<hyle::graph::EffectResourceId, 24> resources;
      for (auto &resource : resources) {
        resource = builder.effect_resource().value();
        auto node =
            builder.add_node(Operation::custom(ResourceEffect{}), {}).value();
        ASSERT_TRUE(builder.bind_resources(node, std::array{resource}).ok());
      }
      probe.fail_at = probe.attempts + fail_offset;
      auto finished = std::move(builder).finish();
      if (finished) {
        EXPECT_GT(fail_offset, 0u);
        reached_success = true;
      } else {
        EXPECT_TRUE(finished.error().IsResourceExhausted());
        EXPECT_TRUE(builder.valid());
        probe.fail_at = SIZE_MAX;
        finished = std::move(builder).finish();
        ASSERT_TRUE(finished.has_value());
      }
      EXPECT_EQ(finished->effect_resource_count(), resources.size());
      for (std::size_t index = 0; index < resources.size(); ++index) {
        EXPECT_EQ(finished->node_resource(finished->node_id(index).value(), 0)
                      .value(),
                  resources[index]);
      }
    }
    EXPECT_EQ(probe.allocations, probe.deallocations);
    if (reached_success)
      break;
  }
  EXPECT_TRUE(reached_success);
}

TEST(GraphBuilder, SymbolicOutputCannotBypassResourceSchemaValidation) {
  auto builder = GraphBuilder::create().value();
  const auto tokens = builder.dimension({1, 8, 1}).value();
  const auto spec =
      TensorSpec::dense(DataType::kFloat32, std::array{Dimension(tokens)})
          .value();
  const auto input = builder.input(spec).value();
  ResourceEffect descriptor{.resource_count = 0, .has_tensor = true};
  EXPECT_TRUE(builder
                  .add_node(Operation::custom(descriptor), std::array{input},
                            std::array{spec})
                  .error()
                  .IsInvalidArgument());
  descriptor.resource_count = 1;
  descriptor.access = static_cast<hyle::compute::ResourceAccess>(255);
  EXPECT_TRUE(builder
                  .add_node(Operation::custom(descriptor), std::array{input},
                            std::array{spec})
                  .error()
                  .IsInvalidArgument());
  descriptor.access = hyle::compute::ResourceAccess::kRead;
  const auto valid = builder.add_node(Operation::custom(descriptor),
                                      std::array{input}, std::array{spec});
  ASSERT_TRUE(valid.has_value());
  EXPECT_EQ(valid->ordinal(), 0u);
}

TEST(GraphBuilder, PreservesProviderNeutralBlockedWeightMetadata) {
  constexpr StorageFormatId codec{0x48594c45U, 41U};
  const StorageFormat blocked =
      StorageFormat::blocked(codec, 32, 18, 2).value();
  const TensorDesc activation = matrix(2, 64);
  const std::array<std::int64_t, 2> stored_shape{8, 64};
  const TensorDesc weight =
      TensorDesc::blocked_dense(DataType::kFloat32, stored_shape, blocked)
          ->transposed()
          .value();

  GraphBuilder builder = GraphBuilder::create().value();
  const auto input = builder.input(activation).value();
  const auto encoded_weight = builder.input(weight).value();
  const auto product = builder.add(MatMul{}, input, encoded_weight).value();
  ASSERT_TRUE(builder.publish(product).ok());
  Graph graph = std::move(builder).finish().value();

  EXPECT_EQ(graph.value_description(encoded_weight).value(), weight);
  EXPECT_EQ(graph.value_description(product).value(), matrix(2, 8));
}

TEST(GraphBuilder, RejectsForeignIdsAndShapeErrorsWithoutLogicalMutation) {
  GraphBuilder first = GraphBuilder::create().value();
  GraphBuilder second = GraphBuilder::create().value();
  const auto input = first.input(matrix(2, 4)).value();
  const auto foreign = second.input(matrix(2, 4)).value();

  auto cross = first.add(RmsNorm{}, foreign);
  ASSERT_FALSE(cross.has_value());
  EXPECT_TRUE(cross.error().IsInvalidArgument());

  const auto wrong_weight = first.input(row(5)).value();
  auto mismatch = first.add(RmsNorm{}, input, wrong_weight);
  ASSERT_FALSE(mismatch.has_value());
  EXPECT_TRUE(mismatch.error().IsInvalidArgument());

  auto bad_epsilon = first.add(
      RmsNorm{.epsilon = (std::numeric_limits<float>::quiet_NaN)()}, input);
  ASSERT_FALSE(bad_epsilon.has_value());
  EXPECT_TRUE(bad_epsilon.error().IsInvalidArgument());

  const auto recovered = first.add(RmsNorm{}, input).value();
  EXPECT_TRUE(first.publish(recovered).ok());
  EXPECT_EQ(std::move(first).finish()->node_count(), 1u);
}

TEST(GraphBuilder, OutputRoleAndPublicationAreExplicit) {
  GraphBuilder builder = GraphBuilder::create().value();
  const auto input = builder.input(row(4)).value();
  EXPECT_TRUE(builder.publish(input).ok());
  const auto result = builder.add(RmsNorm{}, input).value();
  EXPECT_TRUE(builder.publish(result).ok());
  EXPECT_TRUE(builder.publish(result).IsAlreadyExists());

  GraphBuilder unfinished = GraphBuilder::create().value();
  EXPECT_TRUE(std::move(unfinished).finish().error().IsFailedPrecondition());
}

TEST(GraphBuilder, MoveAndGraphCopiesPreserveOriginAndLifetime) {
  GraphBuilder original = GraphBuilder::create().value();
  const auto input = original.input(row(8)).value();
  GraphBuilder moved = std::move(original);
  EXPECT_FALSE(original.valid()); // NOLINT(bugprone-use-after-move)
  const auto output = moved.add(RmsNorm{}, input).value();
  ASSERT_TRUE(moved.publish(output).ok());
  Graph first = std::move(moved).finish().value();
  Graph copy = first;
  Graph relocated = std::move(first);
  EXPECT_FALSE(first.valid()); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(copy.output_id().value(), output);
  EXPECT_EQ(relocated.input_id().value(), input);
}

TEST(GraphBuilder, AllocationFailureDoesNotPublishPartialLogicalState) {
  AllocatorProbe probe;
  probe.fail = true;
  auto unavailable = GraphBuilder::create(
      GraphBuilderOptions{.metadata_allocator = probe_allocator(probe)});
  ASSERT_FALSE(unavailable.has_value());
  EXPECT_TRUE(unavailable.error().IsResourceExhausted());

  probe.fail = false;
  GraphBuilder builder =
      GraphBuilder::create(
          GraphBuilderOptions{.metadata_allocator = probe_allocator(probe)})
          .value();
  std::array<hyle::graph::ValueId, 8> inputs{};
  for (auto &input : inputs)
    input = builder.input(row(4)).value();
  probe.fail = true;
  auto ninth = builder.input(row(4));
  ASSERT_FALSE(ninth.has_value());
  EXPECT_TRUE(ninth.error().IsResourceExhausted());

  probe.fail = false;
  ninth = builder.input(row(4));
  ASSERT_TRUE(ninth.has_value());
  const auto output = builder.add(RmsNorm{}, inputs[0]).value();
  ASSERT_TRUE(builder.publish(output).ok());

  probe.fail = true;
  auto failed_finish = std::move(builder).finish();
  ASSERT_FALSE(failed_finish.has_value());
  EXPECT_TRUE(failed_finish.error().IsResourceExhausted());
  EXPECT_TRUE(builder.valid());

  probe.fail = false;
  Graph graph = std::move(builder).finish().value();
  EXPECT_EQ(graph.input_count(), 9u);
  EXPECT_EQ(graph.node_count(), 1u);
}

TEST(Graph, InvalidOutOfRangeAndForeignQueriesAreRecoverable) {
  Graph invalid;
  EXPECT_TRUE(invalid.input_id().error().IsFailedPrecondition());
  EXPECT_TRUE(invalid.node_id().error().IsFailedPrecondition());
  EXPECT_TRUE(invalid.output_id().error().IsFailedPrecondition());
  EXPECT_EQ(invalid.effect_resource_count(), 0u);
  EXPECT_TRUE(invalid.effect_resource_id(0).error().IsFailedPrecondition());
  EXPECT_TRUE(invalid.node_resource_count({}).error().IsFailedPrecondition());
  EXPECT_TRUE(invalid.node_resource({}, 0).error().IsFailedPrecondition());

  GraphBuilder builder = GraphBuilder::create().value();
  const auto input = builder.input(row(4)).value();
  const auto builder_node =
      builder.add_node(Operation(RmsNorm{}), std::array{input}).value();
  const auto output = builder.result(builder_node).value();
  ASSERT_TRUE(builder.publish(output).ok());
  Graph graph = std::move(builder).finish().value();
  EXPECT_TRUE(graph.input_id(1).error().IsOutOfRange());
  EXPECT_TRUE(graph.node_id(1).error().IsOutOfRange());
  const auto node = graph.node_id().value();
  EXPECT_TRUE(graph.node_input(node, 1).error().IsOutOfRange());
  EXPECT_TRUE(graph.effect_resource_id(0).error().IsOutOfRange());
  EXPECT_EQ(graph.node_resource_count(node).value(), 0u);
  EXPECT_TRUE(graph.node_resource(node, 0).error().IsOutOfRange());
  EXPECT_TRUE(graph.producer(input).error().IsFailedPrecondition());
  EXPECT_TRUE(graph.output_id(1).error().IsOutOfRange());

  GraphBuilder other = GraphBuilder::create().value();
  const auto foreign = other.input(row(4)).value();
  const auto foreign_output = other.add(RmsNorm{}, foreign).value();
  ASSERT_TRUE(other.publish(foreign_output).ok());
  Graph other_graph = std::move(other).finish().value();
  const auto foreign_node = other_graph.node_id().value();
  EXPECT_TRUE(graph.value_description(foreign).error().IsInvalidArgument());
  EXPECT_TRUE(graph.operation(foreign_node).error().IsInvalidArgument());
  EXPECT_TRUE(
      graph.node_resource_count(foreign_node).error().IsInvalidArgument());
  EXPECT_TRUE(graph.node_resource(foreign_node, 0).error().IsInvalidArgument());
  EXPECT_TRUE(graph.producer(foreign_output).error().IsInvalidArgument());
}

} // namespace
