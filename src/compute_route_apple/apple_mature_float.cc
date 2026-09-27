#include "hyle/compute/detail/apple_plan.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"

#include <array>
#include <cstdint>
#include <limits>
#include <utility>

#include <mlx/ops.h>

#include "apple_checks.h"
#include "apple_internal.h"

namespace hyle::compute::apple::detail {
namespace mx = mlx::core;
namespace {
constexpr auto kMode = Float8ScaleMode::kFloat32Block128;

// These are original MLX conversion/arithmetic nodes, traced once as part of
// its native region. No HYLE conversion kernel or hot graph interpreter.
mx::array decode(const mx::array &codes, mx::Stream s) {
  auto value = mx::from_fp8(codes, mx::float32, s);
  // MLX's finite-only conversion maps the reserved E4M3FN NaN code to 480.
  // Retain the storage format's NaN using native selection, not that value.
  auto nan = mx::equal(mx::bitwise_and(codes, mx::array(std::uint8_t{127}), s),
                       mx::array(std::uint8_t{127}), s);
  return mx::where(nan, mx::array(std::numeric_limits<float>::quiet_NaN()),
                   value, s);
}

struct Quantized {
  mx::array codes;
  mx::array scales;
};

Quantized quantize(const mx::array &input, int group, float floor, bool power,
                   mx::Stream s) {
  auto shape = input.shape();
  shape.back() /= group;
  shape.push_back(group);
  auto x = mx::reshape(mx::astype(input, mx::float32, s), shape, s);
  auto maximum = mx::maximum(mx::max(mx::abs(x, s), -1, true, s), mx::array(floor), s);
  auto scale = power ? mx::multiply(maximum, mx::array(1.0F / 448.0F), s)
                     : mx::divide(maximum, mx::array(448.0F), s);
  if (power) {
    // DeepSeek fast_log2_ceil/fast_pow2: raise a normal positive F32 scale
    // to the next power of two, preserving an already exact power.
    auto bits = mx::view(scale, mx::uint32, s);
    auto exponent = mx::right_shift(bits, mx::array(std::uint32_t{23}), s);
    auto fraction = mx::not_equal(
        mx::bitwise_and(bits, mx::array(std::uint32_t{0x7fffff}), s),
        mx::array(std::uint32_t{0}), s);
    scale = mx::view(
        mx::left_shift(mx::add(exponent, mx::astype(fraction, mx::uint32, s), s),
                        mx::array(std::uint32_t{23}), s),
        mx::float32, s);
  }
  auto normalized = mx::clip(mx::divide(x, scale, s), mx::array(-448.0F),
                             mx::array(448.0F), s);
  return {mx::to_fp8(normalized, s), scale};
}

Quantized weight(const mx::array &carrier, int count, int n, int k,
                  mx::Stream s) {
  const int payload = n * k, blocks = k / 128;
  const int bytes = count * (payload + ((n + 127) / 128) * blocks * 4);
  auto codes = mx::reshape(mx::slice(carrier, {0}, {count * payload}, s),
                            {count, n, blocks, 128}, s);
  auto scales = mx::view(
      mx::contiguous(mx::slice(carrier, {count * payload}, {bytes}, s), false, s),
      mx::float32, s);
  scales = mx::reshape(scales, {count, (n + 127) / 128, blocks}, s);
  scales = mx::slice(mx::repeat(scales, 128, 1, s), {0, 0, 0},
                      {count, n, blocks}, s);
  return {codes, mx::expand_dims(scales, -1, s)};
}

mx::array block_dot(const Quantized &a, const Quantized &b, mx::Stream s) {
  // [batch,M,Kblocks,128] x [batch,N,Kblocks,128]. Apply scales
  // after each original F32 dot, before reducing Kblocks.
  auto lhs = mx::transpose(decode(a.codes, s), {0, 2, 1, 3}, s);
  auto rhs = mx::transpose(decode(b.codes, s), {0, 2, 3, 1}, s);
  auto dots = mx::matmul(lhs, rhs, s);
  auto scaled = mx::multiply(dots, mx::transpose(a.scales, {0, 2, 1, 3}, s), s);
  auto weights = mx::transpose(b.scales, {0, 2, 3, 1}, s);
  const int batch = dots.shape(0), rows = dots.shape(2), columns = dots.shape(3);
  auto accumulated = mx::zeros({batch, 1, rows, columns}, mx::float32, s);
  // SGLang applies the activation scale before the weight scale and the
  // ordered F32 accumulation. Combining scales or a tree sum can cross a
  // BF16 midpoint that a later dynamic quantizer makes observable. This loop
  // builds native MLX nodes only at prepare/compile time; MLX owns fusion.
  for (int block = 0; block < dots.shape(1); ++block) {
    auto value = mx::slice(scaled, {0, block, 0, 0},
                            {batch, block + 1, rows, columns}, s);
    auto scale = mx::slice(weights, {0, block, 0, 0},
                            {batch, block + 1, 1, columns}, s);
    accumulated = mx::add(accumulated, mx::multiply(value, scale, s), s);
  }
  return mx::reshape(accumulated, {batch, rows, columns}, s);
}
} // namespace

bool is_mlx_float_operation(const Operation &operation,
                            std::span<const TensorDesc> inputs) noexcept {
  return operation.custom_as<model_ops::Float8BlockLinear>() ||
         operation.custom_as<model_ops::Float8WeightDecode>() ||
         operation.custom_as<model_ops::GroupedFloat8Linear>() ||
         operation.custom_as<model_ops::BlockQuantizationRoundtrip>() ||
         (operation.get_if<MatMul>() && inputs.size() == 2 &&
          inputs[1].storage_format().id() == kFloat8WeightFloat32Block128);
}

Result<InputSpec> mlx_float_input_spec(const TensorDesc &description) noexcept {
  try {
    const auto id = description.storage_format().id();
    Result<TensorDesc> expected = std::unexpected(
        Status::Unimplemented("Apple native FP8 storage format is not admitted"));
    const auto dims = description.dimensions();
    if (id == kFloat8ActivationFloat32Block128 && dims.size() == 2)
      expected = float8_activation_description(dims[0], dims[1], kMode);
    else if (id == kFloat8WeightFloat32Block128 && dims.size() == 2)
      expected = float8_weight_description(dims[0], dims[1], kMode);
    else if (id == kFloat8GroupedWeightFloat32Block128 && dims.size() == 3)
      expected = float8_grouped_weight_description(dims[0], dims[1], dims[2]);
    if (!expected)
      return std::unexpected(std::move(expected).error());
    if (description != *expected)
      return std::unexpected(Status::InvalidArgument("Apple FP8 requires canonical bytes and layout"));
    HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
    if (bytes > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
      return std::unexpected(Status::OutOfRange("Apple FP8 carrier exceeds native int shape"));
    return InputSpec{{static_cast<int>(bytes)}, mx::uint8, description};
  } catch (...) {
    return std::unexpected(native_exception());
  }
}

Result<std::vector<mx::array>> lower_mlx_float(
    const Operation &operation, std::span<const TensorDesc> descriptions,
    std::span<const mx::array> inputs, mx::Stream s) noexcept {
  try {
    if (s.device.type != mx::Device::gpu || inputs.size() != descriptions.size())
      return std::unexpected(Status::InvalidArgument("Apple FP8 lowering requires GPU and matching ports"));
    HYLE_ASSIGN_OR_RETURN(auto output, infer_single_output(operation, descriptions));
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(descriptions[i]));
      if (inputs[i].dtype() != spec.dtype || inputs[i].shape() != spec.shape)
        return std::unexpected(Status::InvalidArgument("Apple FP8 carrier differs from logical input"));
      HYLE_ASSIGN_OR_RETURN(auto bytes, descriptions[i].required_bytes());
      if (bytes > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
        return std::unexpected(Status::OutOfRange("Apple FP8 input exceeds native index range"));
    }
    const auto type = output.data_type() == DataType::kBFloat16 ? mx::bfloat16 : mx::float32;
    if (const auto roundtrip = operation.custom_as<model_ops::BlockQuantizationRoundtrip>()) {
      if (roundtrip->format == model_ops::BlockQuantizationFormat::kFloat4E2M1Block32)
        return std::unexpected(Status::Unimplemented("Apple MXFP4 small-scale roundtrip is not qualified"));
      const auto group = roundtrip->format == model_ops::BlockQuantizationFormat::kFloat8E4M3Block64 ? 64 : 128;
      auto q = quantize(inputs[0], group, 1.0e-4F, true, s);
      return std::vector{mx::astype(mx::reshape(
          mx::multiply(decode(q.codes, s), q.scales, s), inputs[0].shape(), s), type, s)};
    }
    if (operation.custom_as<model_ops::Float8WeightDecode>()) {
      const auto dims = descriptions[0].dimensions();
      const int k = static_cast<int>(dims[0]), n = static_cast<int>(dims[1]);
      auto w = weight(inputs[0], 1, n, k, s);
      return std::vector{mx::astype(mx::reshape(
          mx::multiply(decode(w.codes, s), w.scales, s), {n, k}, s), type, s)};
    }
    if (const auto grouped = operation.custom_as<model_ops::GroupedFloat8Linear>()) {
      const auto dims = descriptions[0].dimensions();
      const int e = static_cast<int>(dims[0]), n = static_cast<int>(dims[1]), k = static_cast<int>(dims[2]);
      const int t = inputs[1].shape(0), a = inputs[1].shape(1), slots = inputs[2].shape(1);
      if (t > (std::numeric_limits<int>::max)() / slots)
        return std::unexpected(Status::OutOfRange("Apple grouped FP8 slot count exceeds native int"));
      const int rows = t * slots;
      auto invalid = mx::logical_or(mx::less(inputs[2], mx::array(0), s),
                                    mx::greater_equal(inputs[2], mx::array(e), s), s);
      auto ids = mx::reshape(mx::where(invalid, mx::array(0), inputs[2], s), {rows}, s);
      auto w = weight(inputs[0], e, n, k, s);
      w.codes = mx::take(w.codes, ids, 0, s);
      w.scales = mx::take(w.scales, ids, 0, s);
      auto q = quantize(inputs[1], 128, grouped->absmax_floor, false, s);
      if (a == 1 && slots != 1) {
        q.codes = mx::broadcast_to(q.codes, {t, slots, k / 128, 128}, s);
        q.scales = mx::broadcast_to(q.scales, {t, slots, k / 128, 1}, s);
      }
      q.codes = mx::reshape(q.codes, {rows, 1, k / 128, 128}, s);
      q.scales = mx::reshape(q.scales, {rows, 1, k / 128, 1}, s);
      auto value = mx::astype(mx::reshape(block_dot(q, w, s), {t, slots, n}, s), type, s);
      auto status = mx::where(mx::any(invalid, s), mx::array(kIndexOutOfRange), mx::array(0), s);
      HYLE_ASSIGN_OR_RETURN(auto checked, with_device_status(std::move(value), std::move(status), s));
      return std::vector{std::move(checked)};
    }
    const int m = static_cast<int>(descriptions[0].dimensions()[0]);
    const int k = static_cast<int>(descriptions[0].dimensions()[1]);
    const int n = static_cast<int>(descriptions[1].dimensions()[1]);
    auto w = weight(inputs[1], 1, n, k, s);
    Quantized q{inputs[0], inputs[0]};
    if (const auto linear = operation.custom_as<model_ops::Float8BlockLinear>()) {
      q = quantize(inputs[0], 128, linear->absmax_floor,
                   linear->scale_rounding == model_ops::Float8ScaleRounding::kPowerOfTwo, s);
      q.codes = mx::expand_dims(q.codes, 0, s);
      q.scales = mx::expand_dims(q.scales, 0, s);
    } else if (operation.get_if<MatMul>()) {
      HYLE_ASSIGN_OR_RETURN(auto expected, float8_activation_description(m, k, kMode));
      if (descriptions[0] != expected)
        return std::unexpected(Status::Unimplemented("Apple encoded FP8 MatMul requires block128 activation"));
      q.codes = mx::reshape(mx::slice(inputs[0], {0}, {m * k}, s), {1, m, k / 128, 128}, s);
      q.scales = mx::reshape(mx::view(mx::slice(inputs[0], {m * k},
          {static_cast<int>(inputs[0].size())}, s), mx::float32, s), {1, m, k / 128, 1}, s);
    } else {
      return std::unexpected(Status::Unimplemented("Apple FP8 operation is not bound"));
    }
    return std::vector{mx::astype(mx::reshape(block_dot(q, w, s), {m, n}, s), type, s)};
  } catch (...) {
    return std::unexpected(native_exception());
  }
}
} // namespace hyle::compute::apple::detail
