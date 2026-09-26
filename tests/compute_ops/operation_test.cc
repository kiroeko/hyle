#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <utility>

#include "../../src/compute_ops/view_alias.h"
#include "custom_operation_fixture.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/operation.h"

namespace {

using namespace hyle::compute;

TEST(ModelClamp, InferenceRejectsMalformedDescriptionsAndBounds) {
  const auto shape = std::array<std::int64_t, 2>{3, 7};
  const auto input = TensorDesc::dense(DataType::kFloat32, shape).value();
  const std::array inputs{input};
  const auto infinity = std::numeric_limits<float>::infinity();
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  for (const auto operation : {model_ops::clamp(-10, 10),
                               model_ops::clamp(-infinity, 10),
                               model_ops::clamp(0, infinity),
                               model_ops::clamp(1, 1)}) {
    auto result = infer_single_output(operation, inputs);
    ASSERT_TRUE(result);
    EXPECT_EQ(*result, input);
    EXPECT_EQ(operation_name(operation), "model-clamp");
  }
  for (const auto operation : {model_ops::clamp(1, -1),
                               model_ops::clamp(nan, 1),
                               model_ops::clamp(0, nan)})
    EXPECT_FALSE(infer_single_output(operation, inputs));
  const auto operation = model_ops::clamp(-1, 1);
  for (auto invalid : {
         TensorDesc::dense(DataType::kBFloat16, shape).value(),
         TensorDesc::dense(DataType::kFloat32,
                           std::array<std::int64_t, 2>{0, 7}).value(),
         input.transposed().value(),
         TensorDesc::dense(DataType::kFloat32,
                           std::array<std::int64_t, 5>{1, 1, 1, 1, 1}).value(),
         TensorDesc{}})
    EXPECT_FALSE(infer_single_output(operation, std::array{invalid}));
  EXPECT_FALSE(infer_single_output(operation, std::span<const TensorDesc>{}));
  EXPECT_FALSE(infer_single_output(operation, std::array{input, input}));
}

TEST(SqrtSoftplusRouting, CompleteRegionMetadataAndRejections) {
  const auto logits = TensorDesc::dense(DataType::kFloat32,
                                        std::array<std::int64_t, 2>{513, 384}).value();
  const auto bias = TensorDesc::dense(DataType::kFloat32,
                                      std::array<std::int64_t, 1>{384}).value();
  const auto op = model_ops::sqrt_softplus_routing(2.5F);
  std::array<TensorDesc, 2> outputs;
  ASSERT_TRUE(infer_output_descriptions(op, std::array{logits, bias}, outputs).ok());
  EXPECT_EQ(outputs[0].data_type(), DataType::kFloat32);
  EXPECT_EQ(outputs[1].data_type(), DataType::kInt32);
  EXPECT_EQ(outputs[0].dimensions()[0], 513);
  EXPECT_EQ(outputs[0].dimensions()[1], 6);
  for (float bad : {0.0F, -1.0F, std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN()})
    EXPECT_FALSE(infer_output_descriptions(model_ops::sqrt_softplus_routing(bad),
                                            std::array{logits, bias}, outputs).ok());
  EXPECT_FALSE(infer_output_descriptions(op, std::array{logits}, outputs).ok());
  EXPECT_FALSE(infer_output_descriptions(op, std::array{logits, bias},
                                          std::span(outputs).first(1)).ok());
  auto bad_bias = TensorDesc::dense(DataType::kBFloat16, bias.dimensions()).value();
  EXPECT_FALSE(infer_output_descriptions(op, std::array{logits, bad_bias}, outputs).ok());
  auto unsupported = TensorDesc::dense(DataType::kFloat32,
                                       std::array<std::int64_t, 2>{1, 512}).value();
  EXPECT_FALSE(infer_output_descriptions(op, std::array{unsupported, bias}, outputs).ok());
}

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

TEST(OperationViewAlias, CheckedSliceOffsetsMatchAddressingAndEmptyRules) {
  const auto input = tensor(DataType::kFloat32, 7, 3);
  for (std::int64_t start = 0; start <= 7; ++start) {
    for (std::int64_t count = 0; count <= 7; ++count) {
      for (std::int64_t step = 1; step <= 3; ++step) {
        const auto view = View::slice(-2, start, count, step).value();
        auto output = infer_single_output(Operation(view), std::array{input});
        if (!output)
          continue; // The schema, not the offset helper, admits a slice.
        const auto offset = detail::view_alias_offset(view, input, *output);
        ASSERT_TRUE(offset.has_value()) << offset.error().message();
        EXPECT_EQ(*offset, count == 0 ? 0u : std::size_t(start) * 12);
        EXPECT_LE(*offset + output->required_bytes().value(),
                  input.required_bytes().value());
      }
    }
  }
  for (const auto &view :
       {View::reshape(shape(3, 7)).value(),
        View::permute(std::array<std::int32_t, 2>{1, 0}).value(),
        View::broadcast_to(shape(2, 7, 3)).value()}) {
    const auto output = infer_one(Operation(view), std::array{input});
    EXPECT_EQ(detail::view_alias_offset(view, input, output).value(), 0u);
  }
}

TEST(OperationViewAlias, RejectsUnrepresentableOffsetsAndInconsistentSpans) {
  const auto input = tensor(DataType::kFloat32, 3);
  const auto output = tensor(DataType::kFloat32, 2);
  EXPECT_TRUE(
      detail::view_alias_offset(View::slice(0, 2, 1).value(), input, output)
          .error()
          .IsOutOfRange());
  EXPECT_TRUE(
      detail::view_alias_offset(
          View::slice(0, (std::numeric_limits<std::int64_t>::max)(), 1).value(),
          input, output)
          .error()
          .IsOutOfRange());
  EXPECT_FALSE(detail::view_alias_offset(View{}, {}, {}).has_value());
}

TEST(OperationCatalog, HasTwentyTwoTypedBuiltinFamiliesAndInvalidDefault) {
  const std::array<Operation, 22> operations{
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
      Operation(Hadamard{}),
      Operation(Softmax{}),
      Operation(TopK{}),
      Operation(ScaledDotProductAttention{}),
      Operation(QuantizedMatMul{}),
      Operation(Scan{})};

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
  EXPECT_EQ(operation_traits(operations[18]).outputs, 2u);
  EXPECT_EQ(operation_traits(operations[0]).alias,
            AliasPolicy::kMustAliasInput0);
  EXPECT_FALSE(operation_traits(operations[0]).materializes);
  EXPECT_EQ(operation_traits(operations[3]).alias, AliasPolicy::kNone);
}

TEST(OperationScan, ChecksAxisAndInputDomainWithoutPayload) {
  for (const auto type : {DataType::kInt32, DataType::kFloat32}) {
    const auto input = tensor(type, 2, 0, 17);
    for (const auto axis : {-3, -1, 0, 1, 2})
      for (const bool exclusive : {false, true})
        EXPECT_EQ(
            infer_one(Operation(Scan{axis, exclusive}), std::array{input}),
            input);
    std::array<TensorDesc, 1> outputs{};
    for (const auto axis : {-4, 3})
      EXPECT_TRUE(infer_output_descriptions(Operation(Scan{.axis = axis}),
                                            std::array{input}, outputs)
                      .IsOutOfRange());
    EXPECT_TRUE(infer_output_descriptions(Operation(Scan{}),
                                          std::array{tensor(type)}, outputs)
                    .IsInvalidArgument());
  }
  for (const auto type : {DataType::kBool, DataType::kInt64, DataType::kFloat16,
                          DataType::kBFloat16}) {
    std::array<TensorDesc, 1> outputs{};
    EXPECT_TRUE(infer_output_descriptions(Operation(Scan{}),
                                          std::array{tensor(type, 3)}, outputs)
                    .IsInvalidArgument());
  }
  const auto traits = operation_traits(Operation(Scan{}));
  EXPECT_EQ(traits.alias, AliasPolicy::kMayAliasInput0);
  EXPECT_EQ(traits.precision, PrecisionPolicy::kAccumulateInOutputType);
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

TEST(OperationElementwise, SelectIncludesConditionInBroadcastShape) {
  const auto scalar = TensorDesc::dense(DataType::kUInt64, {}).value();
  const Operation operation(Select{});
  for (std::int64_t rows : {0, 2}) {
    const std::array<std::int64_t, 8> dimensions{1, 1, 1, 1, 1, 1, rows, 3};
    const auto condition =
        TensorDesc::dense(DataType::kBool, dimensions).value();
    const auto expected =
        TensorDesc::dense(DataType::kUInt64, dimensions).value();
    EXPECT_EQ(infer_one(operation, std::array{condition, scalar, scalar}),
              expected);
    EXPECT_EQ(
        infer_one(operation,
                  std::array{condition, tensor(DataType::kUInt64, 3), scalar}),
        expected);
  }
  EXPECT_TRUE(infer_single_output(
                  operation, std::array{tensor(DataType::kBool, 2, 4),
                                        tensor(DataType::kUInt64, 3), scalar})
                  .error()
                  .IsInvalidArgument());
  EXPECT_TRUE(
      infer_single_output(
          operation, std::array{tensor(DataType::kInt32, 2, 3), scalar, scalar})
          .error()
          .IsInvalidArgument());
}

TEST(OperationElementwise, SoftplusPreservesFloatingDomainAndRejectsInteger) {
  const Operation operation(Unary{UnaryKind::kSoftplus});
  for (DataType type :
       {DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16}) {
    const auto input = tensor(type, 2, 17);
    EXPECT_EQ(infer_one(operation, std::array{input}), input);
  }
  EXPECT_TRUE(
      infer_single_output(operation, std::array{tensor(DataType::kInt32, 17)})
          .error()
          .IsInvalidArgument());
  EXPECT_EQ(operation_traits(operation).effect, EffectKind::kPure);
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

TEST(OperationReductionAndNorm, HalfInputPreservesFloat32LearnedParameters) {
  for (const auto type : {DataType::kFloat16, DataType::kBFloat16}) {
    const auto input = tensor(type, 2, 17, 2048);
    const auto scale = tensor(DataType::kFloat32, 2048);
    const auto bias = tensor(type, 2048);
    EXPECT_EQ(infer_one(Operation(RmsNorm{}), std::array{input, scale}), input);
    EXPECT_EQ(infer_one(Operation(LayerNorm{}), std::array{input, scale, bias}),
              input);
    EXPECT_EQ(infer_one(Operation(LayerNorm{}), std::array{input, bias, scale}),
              input);
    for (const auto &invalid :
         {tensor(DataType::kFloat64, 2048), tensor(DataType::kInt32, 2048),
          tensor(DataType::kFloat32, 2047),
          tensor(DataType::kFloat32, 1, 2048)}) {
      EXPECT_TRUE(
          infer_single_output(Operation(RmsNorm{}), std::array{input, invalid})
              .error()
              .IsInvalidArgument());
    }
  }
  EXPECT_TRUE(infer_single_output(Operation(RmsNorm{}),
                                  std::array{tensor(DataType::kFloat32, 2, 7),
                                             tensor(DataType::kFloat16, 7)})
                  .error()
                  .IsInvalidArgument());
}

TensorDesc packed_q4(DataType type, std::int64_t channels,
                     std::int64_t reduction) {
  const auto format =
      StorageFormat::blocked({0x48594c4554455354ULL, 4}, 2, 1).value();
  return TensorDesc::blocked_dense(type, std::array{channels, reduction},
                                   format)
      .value();
}

TEST(OperationQuantizedMatMul, InfersSignedAndAffineGroupedLinearProjections) {
  for (const auto type : {DataType::kFloat16, DataType::kBFloat16,
                          DataType::kFloat32, DataType::kFloat64}) {
    const auto weight = packed_q4(type, 256, 256);
    const auto parameter = tensor(type, 256, 4);
    for (const auto batch : {1, 7}) {
      const auto activation = tensor(type, batch, 256);
      EXPECT_EQ(infer_one(Operation(QuantizedMatMul{}),
                          std::array{activation, weight, parameter}),
                tensor(type, batch, 256));
      EXPECT_EQ(infer_one(Operation(QuantizedMatMul{
                              .mode = QuantizedWeightMode::kUnsignedAffine4}),
                          std::array{activation, weight, parameter, parameter}),
                tensor(type, batch, 256));
      EXPECT_EQ(infer_one(Operation(QuantizedMatMul{
                              .mode = QuantizedWeightMode::kSignedSymmetric8,
                              .group_size = 32}),
                          std::array{activation, weight, tensor(type, 256, 8)}),
                tensor(type, batch, 256));
    }
    EXPECT_EQ(infer_one(Operation(QuantizedMatMul{.group_size = 128}),
                        std::array{tensor(type, 2, 7, 256), weight,
                                   tensor(type, 256, 2)}),
              tensor(type, 2, 7, 256));
  }
}

TEST(OperationHadamard, PreservesShapeAndRequiresPowerOfTwoFinalDimension) {
  for (const auto type : {DataType::kFloat16, DataType::kBFloat16,
                          DataType::kFloat32, DataType::kFloat64}) {
    const auto input = tensor(type, 2, 17, 256);
    for (const bool inverse : {false, true}) {
      const Operation operation(Hadamard{
          .inverse = inverse, .accumulation_type = DataType::kFloat32});
      EXPECT_EQ(infer_one(operation, std::array{input}), input);
      const auto traits = operation_traits(operation);
      EXPECT_EQ(traits.inputs.minimum, 1u);
      EXPECT_EQ(traits.inputs.maximum, 1u);
      EXPECT_EQ(traits.accumulation_type, DataType::kFloat32);
      EXPECT_EQ(traits.alias, AliasPolicy::kNone);
    }
  }
  for (const auto &invalid :
       {tensor(DataType::kFloat32), tensor(DataType::kFloat32, 0),
        tensor(DataType::kFloat32, 7), tensor(DataType::kInt32, 8)}) {
    EXPECT_TRUE(infer_single_output(Operation(Hadamard{}), std::array{invalid})
                    .error()
                    .IsInvalidArgument());
  }
  EXPECT_TRUE(infer_single_output(
                  Operation(Hadamard{.accumulation_type = DataType::kBool}),
                  std::array{tensor(DataType::kFloat32, 8)})
                  .error()
                  .IsInvalidArgument());
}

TEST(OperationQuantizedMatMul, RejectsIncompleteAndContradictoryContracts) {
  const auto type = DataType::kFloat32;
  const auto activation = tensor(type, 7, 128);
  const auto weight = packed_q4(type, 32, 128);
  const auto scales = tensor(type, 32, 2);
  const std::array inputs{activation, weight, scales};
  const Operation signed_op(QuantizedMatMul{});
  const Operation affine_op(
      QuantizedMatMul{.mode = QuantizedWeightMode::kUnsignedAffine4});
  EXPECT_TRUE(
      infer_single_output(affine_op, inputs).error().IsInvalidArgument());
  EXPECT_TRUE(infer_single_output(
                  signed_op, std::array{activation, weight, scales, scales})
                  .error()
                  .IsInvalidArgument());
  EXPECT_TRUE(infer_single_output(signed_op, std::array{activation, weight})
                  .error()
                  .IsInvalidArgument());
  for (const auto invalid :
       {QuantizedMatMul{.mode = static_cast<QuantizedWeightMode>(255)},
        QuantizedMatMul{.group_size = 0}, QuantizedMatMul{.group_size = 63},
        QuantizedMatMul{.group_size = 256},
        QuantizedMatMul{.accumulation_type = DataType::kUnknown},
        QuantizedMatMul{.accumulation_type = DataType::kInt32},
        QuantizedMatMul{.accumulation_type = DataType::kFloat8E4M3FN}}) {
    EXPECT_TRUE(infer_single_output(Operation(invalid), inputs)
                    .error()
                    .IsInvalidArgument());
  }
  for (const auto &bad_weight : {tensor(type, 32, 128), packed_q4(type, 32, 64),
                                 packed_q4(DataType::kFloat16, 32, 128)}) {
    EXPECT_TRUE(infer_single_output(signed_op,
                                    std::array{activation, bad_weight, scales})
                    .error()
                    .IsInvalidArgument());
  }
  for (const auto &bad_parameter :
       {tensor(type, 2, 32), tensor(type, 32, 1), tensor(type, 64),
        tensor(DataType::kFloat16, 32, 2)}) {
    EXPECT_TRUE(infer_single_output(
                    signed_op, std::array{activation, weight, bad_parameter})
                    .error()
                    .IsInvalidArgument());
    EXPECT_TRUE(
        infer_single_output(
            affine_op, std::array{activation, weight, scales, bad_parameter})
            .error()
            .IsInvalidArgument());
  }
  EXPECT_TRUE(infer_single_output(signed_op, std::array{tensor(type, 7, 0),
                                                        packed_q4(type, 32, 0),
                                                        tensor(type, 32, 0)})
                  .error()
                  .IsInvalidArgument());
  EXPECT_TRUE(infer_single_output(signed_op,
                                  std::array{tensor(type, 128), weight, scales})
                  .error()
                  .IsInvalidArgument());
}

TEST(OperationQuantizedMatMul, PreservesEmptyOutputsAndChecksOutputOverflow) {
  const auto type = DataType::kFloat32;
  const Operation operation(QuantizedMatMul{});
  EXPECT_EQ(infer_one(operation,
                      std::array{tensor(type, 0, 64), packed_q4(type, 32, 64),
                                 tensor(type, 32, 1)}),
            tensor(type, 0, 32));
  EXPECT_EQ(infer_one(operation,
                      std::array{tensor(type, 7, 64), packed_q4(type, 0, 64),
                                 tensor(type, 0, 1)}),
            tensor(type, 7, 0));
  constexpr std::int64_t huge = std::int64_t{1} << 32;
  const std::array inputs{tensor(type, huge, 64), packed_q4(type, huge, 64),
                          tensor(type, huge, 1)};
  std::array output{tensor(type, 1, 1)};
  const auto original = output;
  EXPECT_TRUE(
      infer_output_descriptions(operation, inputs, output).IsOutOfRange());
  EXPECT_EQ(output, original);
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

TEST(OperationIndexing, ScatterAssignmentAndAdditionHaveDistinctPrecision) {
  for (const auto kind : {ScatterKind::kAssign, ScatterKind::kAdd}) {
    const auto traits = operation_traits(Operation(Scatter{.kind = kind}));
    EXPECT_EQ(traits.precision, kind == ScatterKind::kAssign
                                    ? PrecisionPolicy::kExact
                                    : PrecisionPolicy::kAccumulateInOutputType);
    EXPECT_EQ(traits.accumulation_type, DataType::kUnknown);
    EXPECT_EQ(traits.alias, AliasPolicy::kMayAliasInput0);
    EXPECT_EQ(traits.effect, EffectKind::kPure);
    EXPECT_EQ(traits.outputs, 1U);
    EXPECT_TRUE(traits.inputs.accepts(3));
    EXPECT_FALSE(traits.inputs.accepts(2));
  }
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
  EXPECT_TRUE(infer_output_descriptions(Operation(TopK{.count = 1, .axis = -1}),
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

TEST(OperationTransformer, GroupedAttentionPreservesQueryHeadsAndMaskDomain) {
  for (const auto layout : {AttentionLayout::kBatchSequenceHeadDepth,
                            AttentionLayout::kBatchHeadSequenceDepth}) {
    const bool bshd = layout == AttentionLayout::kBatchSequenceHeadDepth;
    const auto query = bshd ? tensor(DataType::kBFloat16, 2, 17, 8, 256)
                            : tensor(DataType::kBFloat16, 2, 8, 17, 256);
    for (const auto heads : {1, 2, 8}) {
      const auto key = bshd ? tensor(DataType::kBFloat16, 2, 31, heads, 256)
                            : tensor(DataType::kBFloat16, 2, heads, 31, 256);
      const auto value = bshd ? tensor(DataType::kBFloat16, 2, 31, heads, 128)
                              : tensor(DataType::kBFloat16, 2, heads, 31, 128);
      const auto expected = bshd ? tensor(DataType::kBFloat16, 2, 17, 8, 128)
                                 : tensor(DataType::kBFloat16, 2, 8, 17, 128);
      const auto operation = Operation(ScaledDotProductAttention{
          .layout = layout, .mask = MaskKind::kBoolean, .causal = true});
      const auto mask = tensor(DataType::kBool, 1, 8, 17, 31);
      EXPECT_EQ(infer_one(operation, std::array{query, key, value, mask}),
                expected);
      const auto wrong_mask = tensor(DataType::kBool, 1, 3, 17, 31);
      EXPECT_FALSE(infer_single_output(
          operation, std::array{query, key, value, wrong_mask}));
    }
    for (const auto heads : {0, 3, 16}) {
      const auto key = bshd ? tensor(DataType::kBFloat16, 2, 31, heads, 256)
                            : tensor(DataType::kBFloat16, 2, heads, 31, 256);
      EXPECT_FALSE(infer_single_output(
          Operation(ScaledDotProductAttention{.layout = layout}),
          std::array{query, key, key}));
    }
    const auto key = bshd ? tensor(DataType::kBFloat16, 2, 31, 2, 256)
                          : tensor(DataType::kBFloat16, 2, 2, 31, 256);
    const auto value = bshd ? tensor(DataType::kBFloat16, 2, 31, 1, 256)
                            : tensor(DataType::kBFloat16, 2, 1, 31, 256);
    EXPECT_FALSE(infer_single_output(
        Operation(ScaledDotProductAttention{.layout = layout}),
        std::array{query, key, value}));
    const auto empty_heads = bshd ? tensor(DataType::kBFloat16, 2, 17, 0, 256)
                                  : tensor(DataType::kBFloat16, 2, 0, 17, 256);
    EXPECT_FALSE(infer_single_output(
        Operation(ScaledDotProductAttention{.layout = layout}),
        std::array{empty_heads, key, key}));
  }
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
  hyle::Status infer_output_descriptions(std::span<const TensorDesc>,
                                         std::span<TensorDesc>) const noexcept {
    return hyle::Status::Ok();
  }
};

struct WrongSizeCustomIdentity final {
  static constexpr OperationId operation_id = CustomIdentity::operation_id;
  static constexpr std::string_view operation_name = "test-identity";

  std::uint64_t tag = 0;

  OperationTraits operation_traits() const noexcept { return {}; }
  hyle::Status infer_output_descriptions(std::span<const TensorDesc>,
                                         std::span<TensorDesc>) const noexcept {
    return hyle::Status::Ok();
  }
};

#if !defined(HYLE_STATIC)
struct SharedConsumerIdentity final {
  static constexpr OperationId operation_id{0x48594c4554455354ULL,
                                            0x435553544f4d0001ULL};
  static constexpr std::string_view operation_name = "shared-test-identity";

  std::uint32_t tag = 0;

  OperationTraits operation_traits() const noexcept { return {}; }
  hyle::Status infer_output_descriptions(std::span<const TensorDesc>,
                                         std::span<TensorDesc>) const noexcept {
    return hyle::Status::Ok();
  }
};
#endif

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
  for (const auto &[effect, count] :
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
TEST(OperationQuantizedMatMul, CarriesTheWholeDescriptorAcrossSharedLibrary) {
  const auto operation = hyle::compute::test::make_shared_quantized_operation();
  ASSERT_EQ(operation.kind(), OperationKind::kQuantizedMatMul);
  const auto &descriptor = *operation.get_if<QuantizedMatMul>();
  EXPECT_EQ(descriptor.mode, QuantizedWeightMode::kUnsignedAffine4);
  EXPECT_EQ(descriptor.group_size, 128u);
  EXPECT_EQ(descriptor.accumulation_type, DataType::kFloat64);
  const auto type = DataType::kFloat32;
  EXPECT_EQ(infer_one(operation,
                      std::array{tensor(type, 7, 256), packed_q4(type, 32, 256),
                                 tensor(type, 32, 2), tensor(type, 32, 2)}),
            tensor(type, 7, 32));
}

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

TEST(OperationGatedDeltaNet, PreservesExplicitStateAndGroupedHeadContract) {
  const Operation operation(GatedDeltaNet{});
  EXPECT_EQ(operation.kind(), OperationKind::kGatedDeltaNet);
  EXPECT_EQ(operation_name(operation), "gated-delta-net");
  const auto traits = operation_traits(operation);
  EXPECT_EQ(traits.outputs, 2);
  EXPECT_EQ(traits.effect, EffectKind::kPure);
  EXPECT_EQ(traits.alias, AliasPolicy::kNone);
  EXPECT_EQ(traits.accumulation_type, DataType::kFloat32);
  for (auto type :
       {DataType::kFloat16, DataType::kBFloat16, DataType::kFloat32}) {
    // Non-square state and grouped heads are schema capabilities; no backend
    // qualification is implied by shape inference.
    const std::array inputs{tensor(type, 2, 17, 4, 64),
                            tensor(type, 2, 17, 4, 64),
                            tensor(type, 2, 17, 12, 128),
                            tensor(DataType::kFloat32, 2, 17, 12, 1),
                            tensor(DataType::kFloat32, 2, 17, 12, 1),
                            tensor(DataType::kFloat32, 2, 12, 128, 64)};
    std::array<TensorDesc, 2> outputs;
    ASSERT_TRUE(infer_output_descriptions(operation, inputs, outputs).ok());
    EXPECT_EQ(outputs[0], inputs[2]);
    EXPECT_EQ(outputs[1], inputs[5]);
    EXPECT_FALSE(infer_single_output(operation, inputs));

    auto per_channel = inputs;
    per_channel[3] = tensor(DataType::kFloat32, 2, 17, 12, 128);
    ASSERT_TRUE(
        infer_output_descriptions(operation, per_channel, outputs).ok());
    EXPECT_EQ(outputs[0], inputs[2]);
    EXPECT_EQ(outputs[1], inputs[5]);
  }
}

TEST(OperationGatedDeltaNet, RejectsWrongGatesStateTypesAndArityAtomically) {
  const Operation operation(GatedDeltaNet{});
  const auto f32 = DataType::kFloat32;
  const std::array valid{
      tensor(f32, 2, 17, 4, 64),   tensor(f32, 2, 17, 4, 64),
      tensor(f32, 2, 17, 12, 128), tensor(f32, 2, 17, 12, 1),
      tensor(f32, 2, 17, 12, 1),   tensor(f32, 2, 12, 128, 64)};
  const auto sentinel = tensor(f32, 7);
  const auto reject = [&](std::span<const TensorDesc> inputs) {
    std::array outputs{sentinel, sentinel};
    EXPECT_TRUE(infer_output_descriptions(operation, inputs, outputs)
                    .IsInvalidArgument());
    EXPECT_EQ(outputs[0], sentinel);
    EXPECT_EQ(outputs[1], sentinel);
  };
  for (std::size_t port = 0; port < valid.size(); ++port) {
    auto inputs = valid;
    for (std::size_t axis = 0; axis < 4; ++axis) {
      std::array<std::int64_t, 4> shape{};
      std::ranges::copy(valid[port].dimensions(), shape.begin());
      shape[axis] = 0;
      inputs[port] = TensorDesc::dense(f32, shape).value();
      reject(inputs);
    }
    inputs[port] = tensor(DataType::kInt32, 2, 17, 12, 1);
    reject(inputs);
    inputs[port] = tensor(f32, 2, 17, 12);
    reject(inputs);
  }
  auto inputs = valid;
  inputs[5] = tensor(f32, 2, 12, 64, 128); // State orientation matters.
  reject(inputs);
  inputs = valid;
  inputs[0] = inputs[1] = tensor(f32, 2, 17, 5, 64);
  reject(inputs); // Value heads must divide into complete key-head groups.
  inputs = valid;
  inputs[3] = inputs[4] = tensor(f32, 2, 17, 12, 64);
  reject(inputs); // Per-channel KDA gates must match the value depth.
  inputs = valid;
  inputs[5] = tensor(DataType::kBFloat16, 2, 12, 128, 64);
  reject(inputs);
  reject(std::span(valid).first(5));
  std::array<TensorDesc, 1> output;
  EXPECT_TRUE(
      infer_output_descriptions(operation, valid, output).IsInvalidArgument());
}

TEST(OperationModelOps, CausalConv1dKeepsNaturalSequenceLayout) {
  const auto operation = model_ops::causal_conv1d();
  EXPECT_EQ(operation.kind(), OperationKind::kCustom);
  EXPECT_EQ(operation_name(operation), "causal-conv1d");
  const auto descriptor = operation.custom_as<model_ops::CausalConv1d>();
  ASSERT_TRUE(descriptor.has_value());

  const std::array inputs{tensor(DataType::kFloat32, 2, 19, 8),
                          tensor(DataType::kFloat32, 8, 4)};
  EXPECT_EQ(infer_one(operation, inputs), tensor(DataType::kFloat32, 2, 16, 8));

  auto invalid = inputs;
  invalid[1] = tensor(DataType::kFloat32, 7, 4);
  EXPECT_FALSE(infer_single_output(operation, invalid));
  invalid = inputs;
  invalid[1] = tensor(DataType::kFloat16, 8, 4);
  EXPECT_FALSE(infer_single_output(operation, invalid));
}

TEST(OperationModelOps, RecurrentRegionKeepsQualifiedShapeAndPrivateGates) {
  const auto operation = model_ops::recurrent_gated_delta_net();
  EXPECT_EQ(operation_name(operation), "recurrent-gated-delta-net");
  EXPECT_EQ(operation_traits(operation).outputs, 2);
  EXPECT_EQ(operation_traits(operation).alias, AliasPolicy::kNone);
  const auto f32 = DataType::kFloat32;
  for (const auto heads : {16, 48, 128}) {
    for (const auto keys : {16, heads}) {
      const auto gate = tensor(f32, 1, 1, heads, 1);
      const std::array inputs{
          tensor(f32, 1, 1, keys, 128), tensor(f32, 1, 1, keys, 128),
          tensor(f32, 1, 1, heads, 128), gate, gate, gate, gate,
          tensor(f32, 1, heads, 128, 128)};
      std::array<TensorDesc, 2> outputs;
      ASSERT_TRUE(infer_output_descriptions(operation, inputs, outputs).ok());
      EXPECT_EQ(outputs[0], inputs[2]);
      EXPECT_EQ(outputs[1], inputs[7]);
      EXPECT_FALSE(infer_single_output(operation, inputs));
    }
  }
}

TEST(OperationModelOps, RecurrentRegionRejectsUnqualifiedPortsAtomically) {
  const auto operation = model_ops::recurrent_gated_delta_net();
  const auto f32 = DataType::kFloat32;
  const auto gate = tensor(f32, 1, 1, 48, 1);
  const std::array valid{
      tensor(f32, 1, 1, 16, 128), tensor(f32, 1, 1, 16, 128),
      tensor(f32, 1, 1, 48, 128), gate, gate, gate, gate,
      tensor(f32, 1, 48, 128, 128)};
  const auto sentinel = tensor(f32, 7);
  const auto reject = [&](std::span<const TensorDesc> inputs) {
    std::array outputs{sentinel, sentinel};
    EXPECT_TRUE(infer_output_descriptions(operation, inputs, outputs)
                    .IsInvalidArgument());
    EXPECT_EQ(outputs[0], sentinel);
    EXPECT_EQ(outputs[1], sentinel);
  };
  for (std::size_t port = 0; port < valid.size(); ++port) {
    auto inputs = valid;
    inputs[port] = TensorDesc::dense(DataType::kBFloat16,
                                      valid[port].dimensions()).value();
    reject(inputs);
    for (std::size_t axis = 0; axis < 4; ++axis) {
      std::array<std::int64_t, 4> shape;
      std::ranges::copy(valid[port].dimensions(), shape.begin());
      ++shape[axis];
      inputs[port] = TensorDesc::dense(f32, shape).value();
      reject(inputs);
    }
  }
  auto inputs = valid;
  inputs[3] = tensor(f32, 1, 1, 48, 128); // No per-channel gate domain.
  reject(inputs);
  inputs = valid;
  inputs[5] = tensor(f32, 48); // Explicit rank-four coefficient contract.
  reject(inputs);
  reject(std::span(valid).first(7));
  std::array outputs{sentinel};
  EXPECT_TRUE(infer_output_descriptions(operation, valid, outputs)
                  .IsInvalidArgument());
  EXPECT_EQ(outputs[0], sentinel);
}

TEST(OperationModelOps, BFloatLinearF32KeepsNativeScorePrecision) {
  const auto bf16 = DataType::kBFloat16;
  const auto f32 = DataType::kFloat32;
  const auto stored_weight = tensor(bf16, 512, 8192);
  const std::array inputs{tensor(bf16, 1, 8192), stored_weight};
  const auto operation = model_ops::bfloat_linear_f32();
  EXPECT_EQ(operation_name(operation), "bfloat-linear-f32");
  EXPECT_EQ(infer_one(operation, inputs), tensor(f32, 1, 512));

  auto invalid = inputs;
  invalid[0] = tensor(f32, 1, 8192);
  EXPECT_FALSE(infer_single_output(operation, invalid));
  invalid = inputs;
  invalid[1] = stored_weight.transposed().value();
  EXPECT_FALSE(infer_single_output(operation, invalid));
  invalid = inputs;
  invalid[1] = tensor(bf16, 512, 8191);
  EXPECT_FALSE(infer_single_output(operation, invalid));
}

TEST(OperationModelOps, GroupedExpertAndHyperConnectionKeepModelLayouts) {
  const auto f32 = DataType::kFloat32;
  const auto bf16 = DataType::kBFloat16;
  const auto i32 = DataType::kInt32;
  const std::array grouped{tensor(f32, 3, 5, 7), tensor(f32, 11, 2, 7),
                           tensor(i32, 11, 2)};
  EXPECT_EQ(infer_one(model_ops::grouped_matmul(), grouped),
            tensor(f32, 11, 2, 5));
  const std::array grouped_bf16{tensor(bf16, 3, 5, 7),
                                tensor(f32, 11, 2, 7), tensor(i32, 11, 2)};
  EXPECT_EQ(infer_one(model_ops::grouped_matmul(), grouped_bf16),
            tensor(f32, 11, 2, 5));
  auto invalid_grouped_type = grouped_bf16;
  invalid_grouped_type[1] = tensor(bf16, 11, 2, 7);
  EXPECT_FALSE(
      infer_single_output(model_ops::grouped_matmul(), invalid_grouped_type));
  invalid_grouped_type = grouped_bf16;
  invalid_grouped_type[0] = tensor(DataType::kFloat16, 3, 5, 7);
  EXPECT_FALSE(
      infer_single_output(model_ops::grouped_matmul(), invalid_grouped_type));
  invalid_grouped_type = grouped_bf16;
  invalid_grouped_type[0] =
      gguf_grouped_weight_description(GgufQuantizedType::kQ8_0, bf16, 3, 5,
                                      32)
          .value();
  invalid_grouped_type[1] = tensor(f32, 11, 2, 32);
  EXPECT_FALSE(
      infer_single_output(model_ops::grouped_matmul(), invalid_grouped_type));

  const std::array combine{tensor(f32, 11, 24), tensor(f32, 3),
                           tensor(f32, 24)};
  EXPECT_EQ(infer_one(model_ops::hyper_connection_combine(), combine),
            tensor(f32, 11, 4, 4));
  const std::array pre{tensor(f32, 11, 4, 7), tensor(f32, 11, 4)};
  EXPECT_EQ(infer_one(model_ops::hyper_connection_pre(), pre),
            tensor(f32, 11, 7));
  const std::array post{tensor(f32, 11, 7), tensor(f32, 11, 4, 7),
                        tensor(f32, 11, 4), tensor(f32, 11, 4, 4)};
  EXPECT_EQ(infer_one(model_ops::hyper_connection_post(), post),
            tensor(f32, 11, 4, 7));

  auto invalid_grouped = grouped;
  invalid_grouped[2] = tensor(i32, 11, 3);
  EXPECT_FALSE(infer_single_output(model_ops::grouped_matmul(), invalid_grouped));
  EXPECT_FALSE(infer_single_output(
      model_ops::hyper_connection_combine(-1.0F, 0), combine));
  auto invalid_post = post;
  invalid_post[3] = tensor(f32, 11, 4, 3);
  EXPECT_FALSE(
      infer_single_output(model_ops::hyper_connection_post(), invalid_post));
}

TEST(OperationModelOps, GroupedExpertAcceptsNativeGgufWeights) {
  const auto f32 = DataType::kFloat32;
  const auto i32 = DataType::kInt32;
  const auto activation = tensor(f32, 1, 10, 8192);
  const auto ids = tensor(i32, 1, 10);
  for (const auto type : {GgufQuantizedType::kIq4Xs,
                          GgufQuantizedType::kQ8_0}) {
    const auto weight =
        gguf_grouped_weight_description(type, f32, 11, 2048, 8192).value();
    const std::array inputs{weight, activation, ids};
    EXPECT_EQ(infer_one(model_ops::grouped_matmul(), inputs),
              tensor(f32, 1, 10, 2048));

    auto invalid = inputs;
    invalid[0] = gguf_grouped_weight_description(
                     type, DataType::kBFloat16, 11, 2048, 8192)
                     .value();
    EXPECT_FALSE(infer_single_output(model_ops::grouped_matmul(), invalid));
    invalid = inputs;
    invalid[1] = tensor(DataType::kBFloat16, 1, 10, 8192);
    EXPECT_FALSE(infer_single_output(model_ops::grouped_matmul(), invalid));
    invalid = inputs;
    invalid[2] = tensor(i32, 1, 9);
    EXPECT_FALSE(infer_single_output(model_ops::grouped_matmul(), invalid));
  }
  EXPECT_FALSE(gguf_grouped_weight_description(
      GgufQuantizedType::kIq4Xs, f32, 11, 2048, 8191));
}

TEST(OperationModelOps, LightningIndexerKeepsSparseModelLayout) {
  const auto f32 = DataType::kFloat32;
  const auto f16 = DataType::kFloat16;
  const std::array inputs{tensor(f32, 2, 3, 64, 128),
                          tensor(f32, 2, 7, 1, 128),
                          tensor(f32, 2, 1, 3, 64),
                          tensor(f16, 1, 1, 3, 7)};
  EXPECT_EQ(infer_one(model_ops::lightning_indexer(), inputs),
            tensor(f32, 2, 1, 3, 7));
  auto invalid = inputs;
  invalid[3] = tensor(f32, 1, 1, 3, 7);
  EXPECT_FALSE(infer_single_output(model_ops::lightning_indexer(), invalid));
  invalid = inputs;
  invalid[0] = tensor(f32, 2, 3, 32, 128);
  EXPECT_FALSE(infer_single_output(model_ops::lightning_indexer(), invalid));
  invalid = inputs;
  invalid[3] = tensor(f16, 3, 1, 3, 7);
  EXPECT_FALSE(infer_single_output(model_ops::lightning_indexer(), invalid));
}

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
