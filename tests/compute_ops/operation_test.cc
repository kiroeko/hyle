#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <utility>

#include "custom_operation_fixture.h"
#include "hyle/compute/operation.h"

namespace {

using namespace hyle::compute;

template <class... Dimensions>
TensorDesc tensor(DataType type, Dimensions... dimensions) {
  const std::array<std::int64_t, sizeof...(Dimensions)> shape{
      static_cast<std::int64_t>(dimensions)...};
  return TensorDesc::dense(type, shape).value();
}

template <class... Dimensions> TensorShape shape(Dimensions... dimensions) {
  const std::array<std::int64_t, sizeof...(Dimensions)> values{
      static_cast<std::int64_t>(dimensions)...};
  return TensorShape::create(values).value();
}

template <std::size_t InputCount>
TensorDesc infer_one(const Operation &operation,
                     const std::array<TensorDesc, InputCount> &inputs) {
  auto inferred = infer_single_output(operation, inputs);
  if (!inferred) {
    ADD_FAILURE() << operation_name(operation) << ": "
                  << inferred.error().message();
    return {};
  }
  return *inferred;
}

TEST(OperationCatalog, HasNineteenTypedBuiltinFamiliesAndInvalidDefault) {
  const std::array<Operation, 19> operations{
      Operation(View::reshape(shape(2, 3)).value()),
      Operation(Copy{}),
      Operation(Fill{shape(2), DataType::kFloat32, ScalarValue(0.0F)}),
      Operation(Cast{DataType::kFloat16}),
      Operation(Unary{}),
      Operation(Binary{}),
      Operation(Select{}),
      Operation(Reduce{}),
      Operation(RmsNorm{}),
      Operation(LayerNorm{}),
      Operation(MatMul{}),
      Operation(GatedActivation{}),
      Operation(Gather{}),
      Operation(Scatter{}),
      Operation(Concat{}),
      Operation(RoPE{.rotary_dimensions = 8}),
      Operation(Softmax{}),
      Operation(TopK{}),
      Operation(ScaledDotProductAttention{})};

  Operation invalid;
  EXPECT_FALSE(invalid.valid());
  EXPECT_EQ(invalid.kind(), OperationKind::kUnknown);
  EXPECT_EQ(operation_name(invalid), "unknown");

  for (std::size_t index = 0; index < operations.size(); ++index) {
    EXPECT_TRUE(operations[index].valid());
    EXPECT_EQ(static_cast<std::size_t>(operations[index].kind()), index + 1);
    EXPECT_NE(operation_name(operations[index]), "unknown");
    EXPECT_GT(operation_traits(operations[index]).outputs, 0u);
    EXPECT_EQ(operation_traits(operations[index]).resource_count, 0u);
    EXPECT_TRUE(validate_operation_effects(operations[index]).ok());
  }
  EXPECT_EQ(operation_traits(operations[17]).outputs, 2u);
  EXPECT_EQ(operation_traits(operations[0]).alias,
            AliasPolicy::kMustAliasInput0);
  EXPECT_FALSE(operation_traits(operations[0]).materializes);
  EXPECT_EQ(operation_traits(operations[3]).alias, AliasPolicy::kNone);
}

TEST(OperationViews, InferReshapeSlicePermutationAndBroadcastWithoutKernel) {
  const TensorDesc input = tensor(DataType::kFloat32, 2, 3, 4);
  EXPECT_EQ(infer_one(Operation(View::reshape(shape(6, 4)).value()),
                      std::array{input}),
            tensor(DataType::kFloat32, 6, 4));

  const TensorDesc sliced =
      infer_one(Operation(View::slice(1, 0, 2, 2).value()), std::array{input});
  const std::array<std::int64_t, 3> slice_shape{2, 2, 4};
  const std::array<std::int64_t, 3> slice_strides{48, 32, 4};
  EXPECT_TRUE(std::ranges::equal(sliced.dimensions(), slice_shape));
  EXPECT_TRUE(std::ranges::equal(sliced.byte_strides(), slice_strides));

  const std::array<std::int32_t, 3> axes{2, 0, 1};
  const TensorDesc permuted =
      infer_one(Operation(View::permute(axes).value()), std::array{input});
  const std::array<std::int64_t, 3> permuted_shape{4, 2, 3};
  EXPECT_TRUE(std::ranges::equal(permuted.dimensions(), permuted_shape));

  const TensorDesc row = tensor(DataType::kFloat32, 1, 3);
  const TensorDesc broadcast = infer_one(
      Operation(View::broadcast_to(shape(5, 3)).value()), std::array{row});
  const std::array<std::int64_t, 2> broadcast_strides{0, 4};
  EXPECT_TRUE(std::ranges::equal(broadcast.byte_strides(), broadcast_strides));
}

TEST(OperationMaterialization, CoversCopyFillAndCastContracts) {
  const std::array<std::int64_t, 2> dimensions{2, 3};
  const std::array<std::int64_t, 2> padded_strides{16, 4};
  const TensorDesc padded =
      TensorDesc::strided_bytes(DataType::kFloat32, dimensions, padded_strides)
          .value();
  EXPECT_EQ(infer_one(Operation(Copy{}), std::array{padded}),
            tensor(DataType::kFloat32, 2, 3));
  EXPECT_EQ(infer_one(Operation(Cast{DataType::kFloat16}), std::array{padded}),
            tensor(DataType::kFloat16, 2, 3));

  const Operation fill(
      Fill{shape(4, 7), DataType::kInt32, ScalarValue(std::int32_t{9})});
  EXPECT_EQ(infer_single_output(fill, {}).value(),
            tensor(DataType::kInt32, 4, 7));
  EXPECT_TRUE(
      infer_single_output(Operation(Fill{shape(1), DataType::kInt32,
                                         ScalarValue(std::uint32_t{1})}),
                          {})
          .error()
          .IsInvalidArgument());
}

TEST(OperationElementwise, AppliesBroadcastAndSelectTypeRules) {
  const TensorDesc matrix = tensor(DataType::kFloat32, 2, 3);
  const TensorDesc row = tensor(DataType::kFloat32, 3);
  EXPECT_EQ(infer_one(Operation(Unary{UnaryKind::kGelu}), std::array{matrix}),
            matrix);
  EXPECT_EQ(
      infer_one(Operation(Binary{BinaryKind::kAdd}), std::array{matrix, row}),
      matrix);

  const TensorDesc condition = tensor(DataType::kBool, 2, 1);
  EXPECT_EQ(infer_one(Operation(Select{}), std::array{condition, matrix, row}),
            matrix);
  EXPECT_TRUE(infer_single_output(Operation(Unary{UnaryKind::kExp}),
                                  std::array{tensor(DataType::kInt32, 4)})
                  .error()
                  .IsInvalidArgument());
}

TEST(OperationReductionAndNorm, PreserveExplicitAxesAndParameterShapes) {
  const TensorDesc input = tensor(DataType::kFloat16, 2, 3, 4);
  const std::array<std::int32_t, 2> requested{0, -1};
  const Reduce reduce{.kind = ReductionKind::kSum,
                      .axes = Axes::from(requested).value(),
                      .keep_dimensions = true,
                      .accumulation_type = DataType::kFloat32};
  EXPECT_EQ(infer_one(Operation(reduce), std::array{input}),
            tensor(DataType::kFloat16, 1, 3, 1));

  const TensorDesc scale = tensor(DataType::kFloat16, 4);
  const TensorDesc bias = tensor(DataType::kFloat16, 4);
  EXPECT_EQ(infer_one(Operation(RmsNorm{}), std::array{input, scale}), input);
  EXPECT_EQ(infer_one(Operation(LayerNorm{}), std::array{input, scale, bias}),
            input);
}

TEST(OperationMatMul, BroadcastsBatchDimensionsWithoutLayoutFlags) {
  const TensorDesc left = tensor(DataType::kFloat16, 2, 1, 3, 4);
  const TensorDesc right = tensor(DataType::kFloat16, 1, 5, 4, 7);
  EXPECT_EQ(infer_one(Operation(MatMul{}), std::array{left, right}),
            tensor(DataType::kFloat16, 2, 5, 3, 7));

  EXPECT_TRUE(infer_single_output(
                  Operation(MatMul{}),
                  std::array{left, tensor(DataType::kFloat16, 1, 5, 6, 7)})
                  .error()
                  .IsInvalidArgument());
}

TEST(OperationGatedActivation, SupportsPackedAndSeparateConsumerShapes) {
  const TensorDesc packed = tensor(DataType::kFloat32, 8, 64);
  EXPECT_EQ(infer_one(Operation(GatedActivation{}), std::array{packed}),
            tensor(DataType::kFloat32, 8, 32));

  const TensorDesc gate = tensor(DataType::kFloat32, 8, 32);
  const GatedActivation separate{.activation = GatedActivationKind::kGeluTanh,
                                 .inputs = GatedInputKind::kSeparate};
  EXPECT_EQ(infer_one(Operation(separate), std::array{gate, gate}), gate);
}

TEST(OperationIndexing, CoversGatherScatterAndVariadicConcat) {
  const TensorDesc source = tensor(DataType::kFloat32, 2, 5, 3);
  const TensorDesc indices = tensor(DataType::kInt32, 4, 2);
  const TensorDesc gathered = tensor(DataType::kFloat32, 2, 4, 2, 3);
  EXPECT_EQ(
      infer_one(Operation(Gather{.axis = 1}), std::array{source, indices}),
      gathered);
  EXPECT_EQ(infer_one(Operation(Scatter{.axis = 1}),
                      std::array{source, indices, gathered}),
            source);

  EXPECT_EQ(infer_one(Operation(Concat{.axis = 1}),
                      std::array{tensor(DataType::kFloat32, 2, 3),
                                 tensor(DataType::kFloat32, 2, 7),
                                 tensor(DataType::kFloat32, 2, 1)}),
            tensor(DataType::kFloat32, 2, 11));
}

TEST(OperationTransformer, CoversRoPESoftmaxTopKAndAttentionShapes) {
  const TensorDesc hidden = tensor(DataType::kFloat32, 2, 7, 4, 64);
  const TensorDesc positions = tensor(DataType::kInt32, 2, 7);
  const TensorDesc frequencies = tensor(DataType::kFloat32, 7, 16);
  const RoPE rope{.rotary_dimensions = 32, .sequence_axis = 1};
  EXPECT_EQ(
      infer_one(Operation(rope), std::array{hidden, positions, frequencies}),
      hidden);

  const TensorDesc mask = tensor(DataType::kBool, 2, 1, 1, 64);
  const Softmax softmax{.axis = -1, .mask = MaskKind::kBoolean};
  EXPECT_EQ(infer_one(Operation(softmax), std::array{hidden, mask}), hidden);

  const Operation top_k(TopK{.count = 5, .axis = -1});
  std::array<TensorDesc, 2> top_k_outputs{};
  EXPECT_TRUE(
      infer_output_descriptions(top_k, std::array{hidden}, top_k_outputs).ok());
  EXPECT_EQ(top_k_outputs[0], tensor(DataType::kFloat32, 2, 7, 4, 5));
  EXPECT_EQ(top_k_outputs[1], tensor(DataType::kInt32, 2, 7, 4, 5));

  constexpr std::int64_t too_wide =
      static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 1;
  const TensorDesc wide = tensor(DataType::kFloat32, 1, too_wide);
  EXPECT_TRUE(infer_output_descriptions(
                  Operation(TopK{.count = 1, .axis = -1}),
                  std::array{wide}, top_k_outputs)
                  .IsOutOfRange());
  EXPECT_TRUE(infer_single_output(top_k, std::array{hidden})
                  .error()
                  .IsFailedPrecondition());

  const TensorDesc query = tensor(DataType::kFloat16, 2, 7, 4, 32);
  const TensorDesc key = tensor(DataType::kFloat16, 2, 11, 4, 32);
  const TensorDesc value = tensor(DataType::kFloat16, 2, 11, 4, 48);
  const TensorDesc attention_mask = tensor(DataType::kBool, 2, 1, 7, 11);
  const ScaledDotProductAttention attention{
      .layout = AttentionLayout::kBatchSequenceHeadDepth,
      .mask = MaskKind::kBoolean,
      .scale = 0.176776695F,
      .causal = true};
  EXPECT_EQ(infer_one(Operation(attention),
                      std::array{query, key, value, attention_mask}),
            tensor(DataType::kFloat16, 2, 7, 4, 48));
}

struct CustomIdentity final {
  static constexpr OperationId operation_id{0x48594c45U, 0x4944454eU};
  static constexpr std::string_view operation_name = "test-identity";

  std::uint32_t tag = 0;

  OperationTraits operation_traits() const noexcept {
    return {{1, 1},
            1,
            AliasPolicy::kMayAliasInput0,
            EffectKind::kPure,
            PrecisionPolicy::kExact,
            DataType::kUnknown,
            true};
  }

  hyle::Status
  infer_output_descriptions(std::span<const TensorDesc> inputs,
                            std::span<TensorDesc> outputs) const noexcept {
    if (inputs.size() != 1 || outputs.size() != 1 || tag != 7) {
      return hyle::Status::InvalidArgument("invalid custom identity schema");
    }
    outputs[0] = inputs[0];
    return hyle::Status::Ok();
  }
};

struct EquivalentCustomIdentity final {
  static constexpr OperationId operation_id = CustomIdentity::operation_id;
  static constexpr std::string_view operation_name = "test-identity";

  std::uint32_t tag = 0;

  OperationTraits operation_traits() const noexcept { return {}; }
  hyle::Status infer_output_descriptions(
      std::span<const TensorDesc>, std::span<TensorDesc>) const noexcept {
    return hyle::Status::Ok();
  }
};

struct WrongSizeCustomIdentity final {
  static constexpr OperationId operation_id = CustomIdentity::operation_id;
  static constexpr std::string_view operation_name = "test-identity";

  std::uint64_t tag = 0;

  OperationTraits operation_traits() const noexcept { return {}; }
  hyle::Status infer_output_descriptions(
      std::span<const TensorDesc>, std::span<TensorDesc>) const noexcept {
    return hyle::Status::Ok();
  }
};

struct SharedConsumerIdentity final {
  static constexpr OperationId operation_id{0x48594c4554455354ULL,
                                             0x435553544f4d0001ULL};
  static constexpr std::string_view operation_name = "shared-test-identity";

  std::uint32_t tag = 0;

  OperationTraits operation_traits() const noexcept { return {}; }
  hyle::Status infer_output_descriptions(
      std::span<const TensorDesc>, std::span<TensorDesc>) const noexcept {
    return hyle::Status::Ok();
  }
};

TEST(OperationCustom, UsesInlineTypedDescriptorWithoutGlobalRegistry) {
  const Operation operation = Operation::custom(CustomIdentity{.tag = 7});
  EXPECT_EQ(operation.kind(), OperationKind::kCustom);
  EXPECT_EQ(operation_name(operation), "test-identity");
  const auto descriptor = operation.custom_as<CustomIdentity>();
  ASSERT_TRUE(descriptor.has_value());
  EXPECT_EQ(descriptor->tag, 7u);
  ASSERT_TRUE(operation.custom_id().has_value());
  EXPECT_EQ(*operation.custom_id(), CustomIdentity::operation_id);

  const auto equivalent = operation.custom_as<EquivalentCustomIdentity>();
  ASSERT_TRUE(equivalent.has_value());
  EXPECT_EQ(equivalent->tag, 7u);
  EXPECT_FALSE(operation.custom_as<WrongSizeCustomIdentity>().has_value());
  EXPECT_FALSE(Operation(Unary{}).custom_id().has_value());

  const TensorDesc input = tensor(DataType::kUInt8, 19);
  EXPECT_EQ(infer_one(operation, std::array{input}), input);
}

struct ResourceDescriptor {
  static constexpr OperationId operation_id{0x48594c45U, 0x52455301U};
  static constexpr std::string_view operation_name = "test-resource-schema";

  EffectKind effect = EffectKind::kResourceAccess;
  std::uint8_t count = 3;
  std::array<ResourceAccess, 3> modes{ResourceAccess::kRead,
                                      ResourceAccess::kWrite,
                                      ResourceAccess::kReadWrite};
  std::size_t *access_calls = nullptr;
  std::size_t *infer_calls = nullptr;

  OperationTraits operation_traits() const noexcept {
    return {.inputs = {1, 1},
            .outputs = 1,
            .effect = effect,
            .resource_count = count};
  }
  ResourceAccess resource_access(std::size_t index) const noexcept {
    if (access_calls != nullptr)
      ++*access_calls;
    return modes[index % modes.size()];
  }
  hyle::Status
  infer_output_descriptions(std::span<const TensorDesc> inputs,
                            std::span<TensorDesc> outputs) const noexcept {
    if (infer_calls != nullptr)
      ++*infer_calls;
    outputs[0] = inputs[0];
    return hyle::Status::Ok();
  }
};

struct MissingResourceAccess final {
  static constexpr OperationId operation_id{0x48594c45U, 0x52455302U};
  static constexpr std::string_view operation_name = "test-missing-access";
  OperationTraits operation_traits() const noexcept {
    return {.effect = EffectKind::kResourceAccess, .resource_count = 1};
  }
  hyle::Status infer_output_descriptions(std::span<const TensorDesc>,
                                         std::span<TensorDesc>) const noexcept {
    return hyle::Status::Ok();
  }
};

struct ThrowingAccess final : ResourceDescriptor {
  static constexpr OperationId operation_id{0x48594c45U, 0x52455303U};
  ResourceAccess resource_access(std::size_t index) const {
    return ResourceDescriptor::resource_access(index);
  }
};

TEST(OperationResources, ReturnsOwnedAccessValuesFromCopiedInlineDescriptor) {
  ResourceDescriptor descriptor;
  const Operation operation = Operation::custom(descriptor);
  descriptor.modes.fill(ResourceAccess::kReadWrite);
  const Operation copied = operation;
  for (std::size_t index = 0; index < 3; ++index) {
    auto access = operation_resource_access(copied, index);
    ASSERT_TRUE(access.has_value());
    EXPECT_EQ(*access, static_cast<ResourceAccess>(index));
  }
  EXPECT_EQ(copied.custom_id(), ResourceDescriptor::operation_id);
  EXPECT_TRUE(validate_operation_effects(copied).ok());
  EXPECT_EQ(infer_one(copied, std::array{tensor(DataType::kFloat32, 4)}),
            tensor(DataType::kFloat32, 4));
}

TEST(OperationResources,
     ValidatesFullFiniteArityAndRejectsIndexBeforeCallback) {
  std::size_t calls = 0;
  ResourceDescriptor descriptor;
  descriptor.count = 255;
  descriptor.access_calls = &calls;
  const Operation operation = Operation::custom(descriptor);
  EXPECT_TRUE(validate_operation_effects(operation).ok());
  EXPECT_EQ(calls, 255u);
  EXPECT_EQ(operation_resource_access(operation, 254).value(),
            ResourceAccess::kReadWrite);
  EXPECT_EQ(calls, 256u);
  EXPECT_TRUE(operation_resource_access(operation, 255).error().IsOutOfRange());
  EXPECT_TRUE(operation_resource_access(
                  operation, (std::numeric_limits<std::size_t>::max)())
                  .error()
                  .IsOutOfRange());
  EXPECT_EQ(calls, 256u);
}

TEST(OperationResources, RejectsInvalidEffectAndCountBeforeInference) {
  const std::array inputs{tensor(DataType::kFloat32, 8)};
  std::array outputs{tensor(DataType::kFloat32, 3)};
  std::size_t inferred = 0, accessed = 0;
  for (const auto [effect, count] :
       std::array{std::pair{EffectKind::kPure, std::uint8_t{1}},
                  std::pair{EffectKind::kExternalSideEffect, std::uint8_t{1}},
                  std::pair{EffectKind::kResourceAccess, std::uint8_t{0}},
                  std::pair{static_cast<EffectKind>(255), std::uint8_t{0}}}) {
    ResourceDescriptor descriptor;
    descriptor.effect = effect;
    descriptor.count = count;
    descriptor.infer_calls = &inferred;
    descriptor.access_calls = &accessed;
    const auto operation = Operation::custom(descriptor);
    EXPECT_TRUE(validate_operation_effects(operation).IsInvalidArgument());
    EXPECT_TRUE(infer_output_descriptions(operation, inputs, outputs)
                    .IsInvalidArgument());
    EXPECT_EQ(outputs[0], tensor(DataType::kFloat32, 3));
  }
  EXPECT_EQ(inferred, 0u);
  EXPECT_EQ(accessed, 0u);
}

TEST(OperationResources, RejectsMissingOrPotentiallyThrowingAccessCallback) {
  const Operation missing = Operation::custom(MissingResourceAccess{});
  EXPECT_TRUE(validate_operation_effects(missing).IsInvalidArgument());
  EXPECT_TRUE(
      operation_resource_access(missing, 0).error().IsInvalidArgument());
  std::size_t calls = 0;
  ThrowingAccess throwing;
  throwing.access_calls = &calls;
  EXPECT_TRUE(validate_operation_effects(Operation::custom(throwing))
                  .IsInvalidArgument());
  EXPECT_EQ(calls, 0u);
}

TEST(OperationResources, RejectsMalformedAccessAtAnyPort) {
  for (std::size_t bad_index = 0; bad_index < 3; ++bad_index) {
    ResourceDescriptor descriptor;
    descriptor.modes[bad_index] = static_cast<ResourceAccess>(255);
    const Operation operation = Operation::custom(descriptor);
    EXPECT_TRUE(operation_resource_access(operation, bad_index)
                    .error()
                    .IsInvalidArgument());
    EXPECT_TRUE(validate_operation_effects(operation).IsInvalidArgument());
  }
}

TEST(OperationResources, PureAndUnknownEffectsDoNotRequireResourceCallbacks) {
  EXPECT_TRUE(
      validate_operation_effects(Operation::custom(CustomIdentity{})).ok());
  ResourceDescriptor descriptor;
  descriptor.effect = EffectKind::kExternalSideEffect;
  descriptor.count = 0;
  EXPECT_TRUE(validate_operation_effects(Operation::custom(descriptor)).ok());
  EXPECT_TRUE(operation_resource_access(Operation(Unary{}), 0)
                  .error()
                  .IsInvalidArgument());
  EXPECT_TRUE(validate_operation_effects(Operation{}).IsInvalidArgument());
  EXPECT_TRUE(
      operation_resource_access(Operation{}, 0).error().IsInvalidArgument());
}

#if !defined(HYLE_STATIC)
TEST(OperationCustom, DecodesSchemaIdentityAcrossSharedFactorBoundary) {
  const Operation operation =
      hyle::compute::test::make_shared_custom_operation(17);
  ASSERT_EQ(operation.custom_id(), SharedConsumerIdentity::operation_id);
  const auto descriptor = operation.custom_as<SharedConsumerIdentity>();
  ASSERT_TRUE(descriptor.has_value());
  EXPECT_EQ(descriptor->tag, 17u);
  EXPECT_EQ(operation_name(operation), "shared-test-identity");

  const TensorDesc input = tensor(DataType::kUInt8, 23);
  EXPECT_EQ(infer_one(operation, std::array{input}), input);
}

TEST(OperationResources, UsesProducerAccessCallbackAcrossSharedFactorBoundary) {
  for (const ResourceAccess access :
       {ResourceAccess::kRead, ResourceAccess::kWrite,
        ResourceAccess::kReadWrite}) {
    const Operation operation =
        hyle::compute::test::make_shared_resource_operation(access);
    EXPECT_EQ(operation_traits(operation).resource_count, 1u);
    EXPECT_EQ(operation_resource_access(operation, 0).value(), access);
    EXPECT_TRUE(validate_operation_effects(operation).ok());
    EXPECT_TRUE(infer_output_descriptions(operation, {}, {}).ok());
  }
  const auto invalid = hyle::compute::test::make_shared_resource_operation(
      static_cast<ResourceAccess>(255));
  EXPECT_TRUE(validate_operation_effects(invalid).IsInvalidArgument());
}
#endif

TEST(OperationValidation, RequiresExactInputAndOutputSpans) {
  const Operation operation(Binary{});
  const TensorDesc input = tensor(DataType::kFloat32, 4);
  TensorDesc output;
  EXPECT_TRUE(infer_output_descriptions(operation, std::array{input},
                                        std::span<TensorDesc>(&output, 1))
                  .IsInvalidArgument());
  EXPECT_TRUE(infer_output_descriptions(operation, std::array{input, input}, {})
                  .IsInvalidArgument());
}

TEST(OperationValidation, RejectsUnknownTypedModesBeforeShapeInference) {
  constexpr auto invalid = static_cast<std::uint8_t>(0xff);
  const TensorDesc value = tensor(DataType::kFloat32, 2, 4);
  const TensorDesc indices = tensor(DataType::kInt32, 1);
  const TensorDesc positions = tensor(DataType::kInt32, 2);
  const TensorDesc frequencies = tensor(DataType::kFloat32, 2, 2);

  const auto expect_invalid = [](const Operation &operation,
                                 std::span<const TensorDesc> inputs) {
    auto result = infer_single_output(operation, inputs);
    ASSERT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().IsInvalidArgument());
  };

  expect_invalid(Operation(Unary{static_cast<UnaryKind>(invalid)}),
                 std::array{value});
  expect_invalid(Operation(Binary{static_cast<BinaryKind>(invalid)}),
                 std::array{value, value});
  expect_invalid(Operation(Reduce{.kind = static_cast<ReductionKind>(invalid)}),
                 std::array{value});
  expect_invalid(Operation(GatedActivation{
                     .activation = static_cast<GatedActivationKind>(invalid)}),
                 std::array{value});
  expect_invalid(Operation(GatedActivation{
                     .inputs = static_cast<GatedInputKind>(invalid)}),
                 std::array{value});
  expect_invalid(
      Operation(Gather{.bounds = static_cast<IndexBoundsPolicy>(invalid)}),
      std::array{value, indices});
  expect_invalid(Operation(Scatter{.kind = static_cast<ScatterKind>(invalid)}),
                 std::array{value, indices, value});
  expect_invalid(Operation(RoPE{.rotary_dimensions = 4,
                                .layout = static_cast<RopeLayout>(invalid)}),
                 std::array{value, positions, frequencies});
  expect_invalid(Operation(Softmax{.kind = static_cast<SoftmaxKind>(invalid)}),
                 std::array{value});
  expect_invalid(Operation(Softmax{.mask = static_cast<MaskKind>(invalid)}),
                 std::array{value});
  expect_invalid(Operation(ScaledDotProductAttention{
                     .layout = static_cast<AttentionLayout>(invalid)}),
                 std::array{tensor(DataType::kFloat32, 1, 1, 1, 4),
                            tensor(DataType::kFloat32, 1, 1, 1, 4),
                            tensor(DataType::kFloat32, 1, 1, 1, 4)});
}

} // namespace
