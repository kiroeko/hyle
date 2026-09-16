#include "hyle/compute/apple.h"

#include <array>
#include <cstdint>

#include <mlx/ops.h>

#include "apple_internal.h"

namespace hyle::compute::apple {
namespace mx = mlx::core;
namespace {
Result<TensorDesc> float_description(const mx::array &input) noexcept {
  if (input.dtype() != mx::float32 || input.ndim() > kMaxTensorRank)
    return std::unexpected(Status::Unimplemented(
        "Apple typed operations require F32 within the tensor rank limit"));
  std::array<std::int64_t, kMaxTensorRank> shape{};
  for (std::size_t axis = 0; axis < input.ndim(); ++axis) {
    if (input.shape(static_cast<int>(axis)) <= 0)
      return std::unexpected(Status::InvalidArgument(
          "Apple typed operations require nonempty dimensions"));
    shape[axis] = input.shape(static_cast<int>(axis));
  }
  // Native MLX owns any stride handling. This description validates logical
  // operation semantics; it does not import a HYLE buffer or claim zero copy.
  return TensorDesc::dense(DataType::kFloat32, {shape.data(), input.ndim()});
}
} // namespace

Result<TensorDesc> affine_q4_description(std::int64_t output_channels,
                                         std::int64_t input_channels) noexcept {
  if (output_channels <= 0 || input_channels <= 0 || input_channels % 64 != 0)
    return std::unexpected(Status::InvalidArgument(
        "Apple affine Q4 needs positive N and K divisible by 64"));
  const auto format = StorageFormat::blocked(kAffineQ4Format, 8, 4, 4);
  const std::array shape{output_channels, input_channels};
  return TensorDesc::blocked_dense(DataType::kFloat32, shape, *format);
}

Result<std::vector<mx::array>> lower(const Operation &operation,
                                     std::span<const mx::array> inputs,
                                     mx::Stream stream) noexcept {
  try {
    const auto *matmul = operation.get_if<MatMul>();
    const auto *qmm = operation.get_if<QuantizedMatMul>();
    const auto *gated = operation.get_if<GatedActivation>();
    if (matmul == nullptr && qmm == nullptr && gated == nullptr)
      return std::unexpected(Status::Unimplemented(
          "Apple typed operation has no admitted MLX lowering"));
    if (!operation_traits(operation).inputs.accepts(inputs.size()))
      return std::unexpected(Status::InvalidArgument(
          "Apple typed operation input count mismatch"));
    if (stream.device.type != mx::Device::gpu)
      return std::unexpected(Status::InvalidArgument(
          "Apple typed lowering requires a GPU stream"));
    if (matmul && (matmul->accumulation_type != DataType::kFloat32 ||
                   inputs[0].ndim() != 2 || inputs[1].ndim() != 2))
      return std::unexpected(Status::Unimplemented(
          "Apple typed MatMul admits 2-D F32 accumulation"));
    if (qmm && (qmm->mode != QuantizedWeightMode::kUnsignedAffine4 ||
                qmm->group_size != 64 ||
                qmm->accumulation_type != DataType::kFloat32 ||
                inputs[0].ndim() != 2 || inputs[1].ndim() != 2 ||
                inputs[1].dtype() != mx::uint32))
      return std::unexpected(Status::Unimplemented(
          "Apple Q4 admits 2-D affine group64 F32 with uint32 packed weights"));
    if (gated && gated->activation != GatedActivationKind::kSilu)
      return std::unexpected(
          Status::Unimplemented("Apple typed gated activation admits SiLU"));

    std::array<TensorDesc, 4> descriptions{};
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      auto description =
          qmm != nullptr && i == 1
              ? affine_q4_description(inputs[i].shape(0),
                                      std::int64_t{inputs[i].shape(1)} * 8)
              : float_description(inputs[i]);
      if (!description)
        return std::unexpected(std::move(description).error());
      descriptions[i] = *description;
    }
    auto output =
        infer_single_output(operation, {descriptions.data(), inputs.size()});
    if (!output)
      return std::unexpected(std::move(output).error());

    if (matmul)
      return std::vector<mx::array>{mx::matmul(inputs[0], inputs[1], stream)};
    if (qmm)
      return std::vector<mx::array>{
          mx::quantized_matmul(inputs[0], inputs[1], inputs[2], inputs[3], true,
                               64, 4, "affine", stream)};

    mx::array gate = inputs[0];
    mx::array up = inputs[0];
    if (gated->inputs == GatedInputKind::kSeparate) {
      up = inputs[1];
    } else {
      auto halves = mx::split(inputs[0], 2, -1, stream);
      gate = std::move(halves[0]);
      up = std::move(halves[1]);
    }
    return std::vector<mx::array>{mx::multiply(
        mx::multiply(gate, mx::sigmoid(gate, stream), stream), up, stream)};
  } catch (...) {
    return std::unexpected(detail::native_exception());
  }
}

Result<Program>
Context::prepare(const Operation &operation,
                 std::span<const mx::array> examples) const noexcept {
  try {
    return compile(
        [operation](const auto &inputs, mx::Stream stream) {
          return lower(operation, inputs, stream);
        },
        examples);
  } catch (...) {
    return std::unexpected(detail::native_exception());
  }
}
} // namespace hyle::compute::apple
