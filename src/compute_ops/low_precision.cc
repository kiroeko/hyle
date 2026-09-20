#include "hyle/compute/low_precision.h"

#include <array>
#include <utility>

#include "hyle/math/scalar.h"

namespace hyle::compute {
namespace {

Result<TensorDesc> float8_description(std::int64_t rows, std::int64_t k,
                                      Float8ScaleMode mode,
                                      bool weight) noexcept {
  const bool f32 = mode == Float8ScaleMode::kFloat32Block128;
  if (!f32 && mode != Float8ScaleMode::kUnsignedExponent8Group32)
    return std::unexpected(Status::InvalidArgument("unknown FP8 scale mode"));
  const std::int64_t block = f32 ? 128 : 32;
  if (rows <= 0 || k <= 0 || k % block != 0 ||
      (weight && f32 && rows % block != 0))
    return std::unexpected(Status::InvalidArgument(
        "FP8 matrices require positive dimensions and complete scale blocks"));

  const auto r = math::try_integer_cast<std::size_t>(rows);
  const auto width = math::try_integer_cast<std::size_t>(k);
  if (!r || !width)
    return std::unexpected(Status::OutOfRange("FP8 dimensions exceed size_t"));
  const auto payload = math::try_multiply(*r, *width);
  const auto scales = math::try_multiply(
      weight && f32 ? *r / 128 : *r, *width / static_cast<std::size_t>(block));
  const auto scale_bytes =
      scales ? math::try_multiply(*scales, std::size_t{f32 ? 4u : 1u})
             : std::nullopt;
  const auto bytes = payload && scale_bytes
                         ? math::try_add(*payload, *scale_bytes)
                         : std::nullopt;
  if (!bytes)
    return std::unexpected(Status::OutOfRange("FP8 storage size overflows"));

  const auto id = weight ? (f32 ? kFloat8WeightFloat32Block128
                                : kFloat8WeightUnsignedExponent8Group32)
                         : (f32 ? kFloat8ActivationFloat32Block128
                                : kFloat8ActivationUnsignedExponent8Group32);
  const auto format = StorageFormat::opaque(id, f32 ? 4 : 1);
  const std::array shape = weight ? std::array{k, rows} : std::array{rows, k};
  return TensorDesc::opaque(DataType::kFloat32, shape, *format, *bytes);
}

} // namespace

Result<TensorDesc>
float8_activation_description(std::int64_t m, std::int64_t k,
                              Float8ScaleMode mode) noexcept {
  return float8_description(m, k, mode, false);
}

Result<TensorDesc> float8_weight_description(std::int64_t k, std::int64_t n,
                                             Float8ScaleMode mode) noexcept {
  return float8_description(n, k, mode, true);
}

Result<TensorDesc> mxfp4_weight_description(std::int64_t k,
                                            std::int64_t n) noexcept {
  if (k <= 0 || n <= 0 || k % 32 != 0)
    return std::unexpected(Status::InvalidArgument(
        "MXFP4 weights require positive N and K divisible by 32"));
  const auto stride = math::try_multiply(k / 32, std::int64_t{17});
  if (!stride)
    return std::unexpected(Status::OutOfRange("MXFP4 row stride overflows"));
  const auto format = StorageFormat::blocked(kMxfp4Weight, 32, 17);
  const std::array shape{k, n};
  const std::array strides{std::int64_t{17}, *stride};
  return TensorDesc::blocked_strided(DataType::kFloat16, shape, *format, 0,
                                     strides);
}

} // namespace hyle::compute
