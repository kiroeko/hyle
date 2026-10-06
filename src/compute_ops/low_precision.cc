#include "hyle/compute/low_precision.h"

#include <array>
#include <utility>

#include "hyle/math/scalar.h"

namespace hyle::compute {
namespace {

struct GgufBlock final {
  StorageFormatId id;
  std::size_t elements;
  std::size_t bytes;
};

std::optional<GgufBlock> gguf_block(GgufQuantizedType type) noexcept {
  switch (type) {
  case GgufQuantizedType::kQ3K:
    return GgufBlock{kGgufQ3K, 256, 110};
  case GgufQuantizedType::kQ4K:
    return GgufBlock{kGgufQ4K, 256, 144};
  case GgufQuantizedType::kQ5K:
    return GgufBlock{kGgufQ5K, 256, 176};
  case GgufQuantizedType::kQ6K:
    return GgufBlock{kGgufQ6K, 256, 210};
  case GgufQuantizedType::kQ8_0:
    return GgufBlock{kGgufQ8_0, 32, 34};
  case GgufQuantizedType::kIq3Xxs:
    return GgufBlock{kGgufIq3Xxs, 256, 98};
  case GgufQuantizedType::kIq3S:
    return GgufBlock{kGgufIq3S, 256, 110};
  case GgufQuantizedType::kIq4Nl:
    return GgufBlock{kGgufIq4Nl, 32, 18};
  case GgufQuantizedType::kIq4Xs:
    return GgufBlock{kGgufIq4Xs, 256, 136};
  case GgufQuantizedType::kMxfp4:
    return GgufBlock{kMxfp4Weight, 32, 17};
  case GgufQuantizedType::kQ4_0:
    return GgufBlock{kGgufQ4_0, 32, 18};
  case GgufQuantizedType::kQ4_1:
    return GgufBlock{kGgufQ4_1, 32, 20};
  case GgufQuantizedType::kQ2K:
    return GgufBlock{kGgufQ2K, 256, 84};
  case GgufQuantizedType::kIq2Xxs:
    return GgufBlock{kGgufIq2Xxs, 256, 66};
  case GgufQuantizedType::kIq2Xs:
    return GgufBlock{kGgufIq2Xs, 256, 74};
  case GgufQuantizedType::kIq2S:
    return GgufBlock{kGgufIq2S, 256, 82};
  case GgufQuantizedType::kIq1S:
    return GgufBlock{kGgufIq1S, 256, 50};
  case GgufQuantizedType::kIq1M:
    return GgufBlock{kGgufIq1M, 256, 56};
  case GgufQuantizedType::kIq1Xxxs:
    return GgufBlock{kGgufIq1Xxxs, 256, 38};
  case GgufQuantizedType::kIq1Xxs:
    return GgufBlock{kGgufIq1Xxs, 256, 42};
  case GgufQuantizedType::kIq1Xs:
    return GgufBlock{kGgufIq1Xs, 256, 46};
  }
  return std::nullopt;
}

bool gguf_logical_type(DataType type) noexcept {
  return type == DataType::kFloat32 || type == DataType::kFloat16 ||
         type == DataType::kBFloat16;
}

Result<TensorDesc> float8_description(std::int64_t rows, std::int64_t k,
                                      Float8ScaleMode mode,
                                      bool weight) noexcept {
  const bool f32 = mode == Float8ScaleMode::kFloat32Block128;
  if (!f32 && mode != Float8ScaleMode::kUnsignedExponent8Group32)
    return std::unexpected(Status::InvalidArgument("unknown FP8 scale mode"));
  const std::int64_t block = f32 ? 128 : 32;
  if (rows <= 0 || k <= 0 || k % block != 0)
    return std::unexpected(Status::InvalidArgument(
        "FP8 matrices require positive dimensions and complete K scale blocks"));

  const auto r = math::try_integer_cast<std::size_t>(rows);
  const auto width = math::try_integer_cast<std::size_t>(k);
  if (!r || !width)
    return std::unexpected(Status::OutOfRange("FP8 dimensions exceed size_t"));
  const auto payload = math::try_multiply(*r, *width);
  const auto scales = math::try_multiply(
      weight && f32 ? *r / 128 + (*r % 128 != 0) : *r,
      *width / static_cast<std::size_t>(block));
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

Result<TensorDesc> float8_grouped_weight_description(std::int64_t experts,
                                                    std::int64_t n,
                                                    std::int64_t k) noexcept {
  if (experts <= 0)
    return std::unexpected(Status::InvalidArgument("FP8 expert count must be positive"));
  HYLE_ASSIGN_OR_RETURN(const auto matrix,
                       float8_description(n, k, Float8ScaleMode::kFloat32Block128, true));
  const auto count = math::try_integer_cast<std::size_t>(experts);
  const auto bytes = count ? math::try_multiply(*count, *matrix.required_bytes())
                           : std::nullopt;
  if (!bytes)
    return std::unexpected(Status::OutOfRange("grouped FP8 storage size overflows"));
  const auto format = StorageFormat::opaque(kFloat8GroupedWeightFloat32Block128, 4);
  return TensorDesc::opaque(DataType::kFloat32, std::array{experts, n, k},
                            *format, *bytes);
}

Result<TensorDesc> float4_grouped_weight_description(std::int64_t experts,
                                                    std::int64_t n,
                                                    std::int64_t k) noexcept {
  if (experts <= 0 || n <= 0 || k <= 0 || k % 32 != 0)
    return std::unexpected(Status::InvalidArgument("grouped FP4 requires positive dimensions and complete group32 K"));
  const auto e = math::try_integer_cast<std::size_t>(experts);
  const auto rows = math::try_integer_cast<std::size_t>(n);
  const auto blocks = math::try_integer_cast<std::size_t>(k / 32);
  const auto row_bytes = blocks ? math::try_multiply(*blocks, std::size_t{20}) : std::nullopt;
  const auto matrix = row_bytes && rows ? math::try_multiply(*row_bytes, *rows) : std::nullopt;
  const auto bytes = matrix && e ? math::try_multiply(*matrix, *e) : std::nullopt;
  if (!bytes)
    return std::unexpected(Status::OutOfRange("grouped FP4 storage size overflows"));
  const auto format = StorageFormat::opaque(kFloat4GroupedWeightFloat32Group32, 4);
  return TensorDesc::opaque(DataType::kFloat32, std::array{experts, n, k}, *format, *bytes);
}

std::optional<GgufQuantizedType>
gguf_quantized_type(StorageFormatId id) noexcept {
  constexpr std::array types{
      GgufQuantizedType::kQ3K,   GgufQuantizedType::kQ4K,
      GgufQuantizedType::kQ5K,   GgufQuantizedType::kQ6K,
      GgufQuantizedType::kQ8_0,  GgufQuantizedType::kIq3Xxs,
      GgufQuantizedType::kIq3S,  GgufQuantizedType::kIq4Nl,
      GgufQuantizedType::kIq4Xs, GgufQuantizedType::kMxfp4,
      GgufQuantizedType::kQ4_0, GgufQuantizedType::kQ4_1,
      GgufQuantizedType::kQ2K, GgufQuantizedType::kIq2Xxs,
      GgufQuantizedType::kIq2Xs, GgufQuantizedType::kIq2S,
      GgufQuantizedType::kIq1S, GgufQuantizedType::kIq1M,
      GgufQuantizedType::kIq1Xxxs, GgufQuantizedType::kIq1Xxs,
      GgufQuantizedType::kIq1Xs};
  for (const auto type : types)
    if (gguf_block(type)->id == id)
      return type;
  return std::nullopt;
}

std::optional<GgufQuantizedType>
gguf_quantized_type(const StorageFormat &format) noexcept {
  const auto type = gguf_quantized_type(format.id());
  if (!type || format.kind() != StorageFormatKind::kBlocked)
    return std::nullopt;
  const auto block = gguf_block(*type);
  if (format.elements_per_block() != block->elements ||
      format.bytes_per_block() != block->bytes)
    return std::nullopt;
  return type;
}

Result<TensorDesc> gguf_weight_description(GgufQuantizedType type,
                                           DataType logical_type,
                                           std::int64_t k,
                                           std::int64_t n) noexcept {
  const auto block = gguf_block(type);
  if (!block || !gguf_logical_type(logical_type))
    return std::unexpected(Status::InvalidArgument(
        "GGUF weight requires a known format and floating logical type"));
  if (k <= 0 || n <= 0 || k % static_cast<std::int64_t>(block->elements) != 0)
    return std::unexpected(Status::InvalidArgument(
        "GGUF weight dimensions do not contain complete K blocks"));
  const auto blocks = math::try_integer_cast<std::size_t>(
      k / static_cast<std::int64_t>(block->elements));
  const auto rows = math::try_integer_cast<std::size_t>(n);
  const auto row_bytes =
      blocks ? math::try_multiply(*blocks, block->bytes) : std::nullopt;
  const auto total_bytes =
      row_bytes && rows ? math::try_multiply(*row_bytes, *rows) : std::nullopt;
  if (!total_bytes)
    return std::unexpected(Status::OutOfRange("GGUF weight size overflows"));
  const auto stride = math::try_integer_cast<std::int64_t>(*row_bytes);
  if (!stride)
    return std::unexpected(Status::OutOfRange("GGUF weight stride overflows"));
  const auto format =
      StorageFormat::blocked(block->id, block->elements, block->bytes);
  const std::array shape{k, n};
  const std::array strides{static_cast<std::int64_t>(block->bytes), *stride};
  return TensorDesc::blocked_strided(logical_type, shape, *format, 0, strides);
}

Result<TensorDesc> gguf_grouped_weight_description(GgufQuantizedType type,
                                                   DataType logical_type,
                                                   std::int64_t experts,
                                                   std::int64_t n,
                                                   std::int64_t k) noexcept {
  const auto block = gguf_block(type);
  if (!block || !gguf_logical_type(logical_type))
    return std::unexpected(
        Status::InvalidArgument("grouped GGUF weight requires a known format "
                                "and floating logical type"));
  if (experts <= 0 || n <= 0 || k <= 0 ||
      k % static_cast<std::int64_t>(block->elements) != 0)
    return std::unexpected(Status::InvalidArgument(
        "grouped GGUF dimensions do not contain complete K blocks"));
  const auto format =
      StorageFormat::blocked(block->id, block->elements, block->bytes);
  const std::array shape{experts, n, k};
  return TensorDesc::blocked_dense(logical_type, shape, *format);
}

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
  return gguf_weight_description(GgufQuantizedType::kMxfp4, DataType::kFloat16,
                                 k, n);
}

} // namespace hyle::compute
