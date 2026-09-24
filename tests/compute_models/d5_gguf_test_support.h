#pragma once

#include <ggml.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "hyle/compute/low_precision.h"

namespace d5_gguf_test {
namespace c = hyle::compute;

inline ggml_type native_type(c::GgufQuantizedType type) {
  switch (type) {
  case c::GgufQuantizedType::kQ3K:
    return GGML_TYPE_Q3_K;
  case c::GgufQuantizedType::kQ4K:
    return GGML_TYPE_Q4_K;
  case c::GgufQuantizedType::kQ5K:
    return GGML_TYPE_Q5_K;
  case c::GgufQuantizedType::kQ6K:
    return GGML_TYPE_Q6_K;
  case c::GgufQuantizedType::kQ8_0:
    return GGML_TYPE_Q8_0;
  case c::GgufQuantizedType::kIq3Xxs:
    return GGML_TYPE_IQ3_XXS;
  case c::GgufQuantizedType::kIq3S:
    return GGML_TYPE_IQ3_S;
  case c::GgufQuantizedType::kIq4Nl:
    return GGML_TYPE_IQ4_NL;
  case c::GgufQuantizedType::kIq4Xs:
    return GGML_TYPE_IQ4_XS;
  case c::GgufQuantizedType::kMxfp4:
    return GGML_TYPE_MXFP4;
  }
  return GGML_TYPE_COUNT;
}

inline std::vector<float> values(std::int64_t rows, std::int64_t width,
                                 int seed) {
  std::vector<float> result(static_cast<std::size_t>(rows * width));
  for (std::size_t i = 0; i < result.size(); ++i) {
    const auto wave = std::sin(static_cast<double>(i + 97 * seed) * 0.013);
    result[i] = static_cast<float>(wave * 0.125 +
                                   double((i + seed) % 11) * 0.003 - 0.015);
  }
  return result;
}

struct Packed final {
  std::vector<std::byte> bytes;
  std::vector<float> decoded;
};

inline Packed pack(c::GgufQuantizedType type, std::span<const float> source,
                   std::int64_t rows, std::int64_t width,
                   std::size_t required_bytes) {
  const auto native = native_type(type);
  Packed result{std::vector<std::byte>(required_bytes),
                std::vector<float>(source.size())};
  std::vector<float> importance(static_cast<std::size_t>(width), 1.0F);
  const auto written = ggml_quantize_chunk(
      native, source.data(), result.bytes.data(), 0, rows, width,
      ggml_quantize_requires_imatrix(native) ? importance.data() : nullptr);
  EXPECT_EQ(written, required_bytes);
  const auto *traits = ggml_get_type_traits(native);
  EXPECT_NE(traits, nullptr);
  if (traits == nullptr || traits->to_float == nullptr)
    return result;
  const auto row_bytes = ggml_row_size(native, width);
  for (std::int64_t row = 0; row < rows; ++row)
    traits->to_float(result.bytes.data() + row * row_bytes,
                     result.decoded.data() + row * width, width);
  return result;
}

inline std::vector<float> selected_rows(std::span<const float> rows,
                                        std::int64_t width,
                                        std::span<const std::int32_t> ids) {
  std::vector<float> result(static_cast<std::size_t>(width) * ids.size());
  for (std::size_t i = 0; i < ids.size(); ++i)
    std::copy_n(rows.begin() + static_cast<std::ptrdiff_t>(ids[i] * width),
                width,
                result.begin() + static_cast<std::ptrdiff_t>(i * width));
  return result;
}

struct QuantizerCleanup final {
  ~QuantizerCleanup() { ggml_quantize_free(); }
};

} // namespace d5_gguf_test
