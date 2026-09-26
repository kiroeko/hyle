#pragma once

#include <ggml.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
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
                width, result.begin() + static_cast<std::ptrdiff_t>(i * width));
  return result;
}

struct Qwen35Projection final {
  std::string_view tensor;
  c::GgufQuantizedType type;
  std::int64_t reduction;
  std::int64_t columns;
  std::uint64_t region;
  int seed;
};

// Exact representative tensor roles from
// unsloth/Qwen3.5-2B-GGUF@f6d5376, UD-Q4_K_XL.  The batch/token extent is
// reduced, while every model reduction/output width and native storage format
// remains unchanged.
inline constexpr std::array kQwen35PrimaryProjectionCases{
    Qwen35Projection{"blk.0.ffn_gate.weight", c::GgufQuantizedType::kQ4K, 2048,
                     6144, 0x0203, 11},
    Qwen35Projection{"blk.0.ffn_down.weight", c::GgufQuantizedType::kQ6K, 6144,
                     2048, 0x0303, 13},
    Qwen35Projection{"blk.3.attn_k.weight", c::GgufQuantizedType::kQ5K, 2048,
                     512, 0x0403, 17},
    Qwen35Projection{"blk.8.ffn_gate.weight", c::GgufQuantizedType::kIq4Xs,
                     2048, 6144, 0x0803, 19},
    Qwen35Projection{"blk.0.ssm_out.weight", c::GgufQuantizedType::kQ8_0, 2048,
                     2048, 0x0503, 23}};

// One exact representative for every quantized storage subtype present in
// unsloth/Qwen3.8-27B-GGUF@4ca7207 UD-Q4_K_M. Token/expert capacity may be
// reduced elsewhere, but these projection reduction/output widths and tensor
// roles are copied from the retained 866-row GGUF header audit.
inline constexpr std::array kQwen38PrimaryProjectionCases{
    Qwen35Projection{"blk.0.ffn_up.weight", c::GgufQuantizedType::kQ3K, 5120,
                     17408, 0x0203, 61},
    Qwen35Projection{"blk.11.ffn_gate.weight", c::GgufQuantizedType::kIq3S,
                     5120, 17408, 0x0203, 67},
    Qwen35Projection{"blk.21.attn_qkv.weight", c::GgufQuantizedType::kIq4Nl,
                     5120, 10240, 0x0403, 71},
    Qwen35Projection{"blk.0.ffn_down.weight", c::GgufQuantizedType::kIq4Xs,
                     17408, 5120, 0x0303, 73},
    Qwen35Projection{"blk.1.attn_qkv.weight", c::GgufQuantizedType::kQ4K, 5120,
                     10240, 0x0403, 79},
    Qwen35Projection{"blk.0.ssm_out.weight", c::GgufQuantizedType::kQ5K, 6144,
                     5120, 0x0503, 83},
    Qwen35Projection{"blk.3.attn_output.weight", c::GgufQuantizedType::kQ6K,
                     6144, 5120, 0x0403, 89},
    Qwen35Projection{"blk.64.nextn.eh_proj.weight",
                     c::GgufQuantizedType::kQ6K, 10240, 5120, 0x0d03, 93},
    Qwen35Projection{"blk.0.ssm_alpha.weight", c::GgufQuantizedType::kQ8_0,
                     5120, 48, 0x0503, 97}};

// Exact Q8_0 projection roles from the same fixed repository revision's
// Qwen3.8-27B UD-Q8_K_XL artifact. Full-attention Q/K/V, the LM head and MTP
// projection remain BF16 and are exercised by the profile semantic chains.
inline constexpr std::array kQwen38FidelityQuantizedProjectionCases{
    Qwen35Projection{"blk.0.ffn_up.weight", c::GgufQuantizedType::kQ8_0, 5120,
                     17408, 0x0208, 101},
    Qwen35Projection{"blk.0.ffn_down.weight", c::GgufQuantizedType::kQ8_0,
                     17408, 5120, 0x0308, 103},
    Qwen35Projection{"blk.0.attn_qkv.weight", c::GgufQuantizedType::kQ8_0,
                     5120, 10240, 0x0408, 107},
    Qwen35Projection{"blk.3.attn_output.weight", c::GgufQuantizedType::kQ8_0,
                     6144, 5120, 0x0408, 109},
    Qwen35Projection{"blk.0.ssm_alpha.weight", c::GgufQuantizedType::kQ8_0,
                     5120, 48, 0x0508, 113},
    Qwen35Projection{"blk.0.ssm_out.weight", c::GgufQuantizedType::kQ8_0,
                     6144, 5120, 0x0508, 127}};

// Fixed Qwen3.8-2.4T GGUF headers at unsloth@567d3e6. A routed expert case
// holds one complete expert slice at the model's 8192/2048 inner widths; it
// qualifies native storage and MatMul only, not the 512-way grouped region.
inline constexpr std::array kQwen38_24TPrimaryProjectionCases{
    Qwen35Projection{"blk.0.ffn_gate_shexp.weight",
                     c::GgufQuantizedType::kQ5K, 8192, 2048, 0x0204, 131},
    Qwen35Projection{"blk.0.ffn_down_shexp.weight",
                     c::GgufQuantizedType::kQ6K, 2048, 8192, 0x0204, 137},
    Qwen35Projection{"blk.0.ffn_gate_exps.weight",
                     c::GgufQuantizedType::kIq4Xs, 8192, 2048, 0x0b04, 139}};

inline constexpr std::array kQwen38_24TFidelityProjectionCases{
    Qwen35Projection{"blk.0.ssm_alpha.weight",
                     c::GgufQuantizedType::kQ8_0, 8192, 128, 0x0508, 149},
    Qwen35Projection{"blk.0.ffn_gate_shexp.weight",
                     c::GgufQuantizedType::kQ8_0, 8192, 2048, 0x0208, 151},
    Qwen35Projection{"blk.0.ffn_gate_exps.weight",
                     c::GgufQuantizedType::kQ8_0, 8192, 2048, 0x0b08, 157}};

// Exact Q8_0 representatives from the same fixed repository revision's
// UD-Q8_K_XL artifact.  The artifact name is not treated as a global tensor
// type: the F16 embedding and F16 projection roles are described separately
// below.
inline constexpr std::array kQwen35FidelityQuantizedProjectionCases{
    Qwen35Projection{"blk.0.ffn_gate.weight", c::GgufQuantizedType::kQ8_0, 2048,
                     6144, 0x0208, 29},
    Qwen35Projection{"blk.0.ffn_down.weight", c::GgufQuantizedType::kQ8_0, 6144,
                     2048, 0x0308, 31},
    Qwen35Projection{"blk.3.attn_v.weight", c::GgufQuantizedType::kQ8_0, 2048,
                     512, 0x0408, 37},
    Qwen35Projection{"blk.3.attn_output.weight", c::GgufQuantizedType::kQ8_0,
                     2048, 2048, 0x0e08, 41}};

struct Qwen35DenseProjection final {
  std::string_view tensor;
  std::int64_t reduction;
  std::int64_t columns;
  std::uint64_t region;
  int seed;
};

// UD-Q8_K_XL retains these representative projection roles as native F16.
// Their exact names, shapes and formats come from the fixed GGUF header audit.
inline constexpr std::array kQwen35FidelityF16ProjectionCases{
    Qwen35DenseProjection{"blk.0.ssm_out.weight", 2048, 2048, 0x0508, 43},
    Qwen35DenseProjection{"blk.3.attn_k.weight", 2048, 512, 0x0408, 47},
    Qwen35DenseProjection{"blk.3.attn_q.weight", 2048, 4096, 0x0408, 53}};

inline std::vector<float> activation(std::int64_t width, int seed) {
  std::vector<float> result(static_cast<std::size_t>(width));
  for (std::size_t i = 0; i < result.size(); ++i) {
    const auto wave = std::cos(static_cast<double>(i + 31 * seed) * 0.017);
    result[i] = static_cast<float>(wave * 0.0625 +
                                   double((i + seed) % 7) * 0.002 - 0.006);
  }
  return result;
}

inline std::vector<float> matmul(std::span<const float> input,
                                 std::span<const float> decoded_weights,
                                 std::int64_t columns, std::int64_t reduction) {
  std::vector<float> result(static_cast<std::size_t>(columns));
  for (std::int64_t column = 0; column < columns; ++column) {
    double sum = 0.0;
    const auto offset = static_cast<std::size_t>(column * reduction);
    for (std::int64_t k = 0; k < reduction; ++k)
      sum += static_cast<double>(input[static_cast<std::size_t>(k)]) *
             decoded_weights[offset + static_cast<std::size_t>(k)];
    result[static_cast<std::size_t>(column)] = static_cast<float>(sum);
  }
  return result;
}

inline void expect_quantized_matmul_close(std::span<const float> actual,
                                          std::span<const float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  double squared_error = 0.0;
  float max_error = 0.0F;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const auto error = std::abs(actual[i] - expected[i]);
    max_error = std::max(max_error, error);
    squared_error += static_cast<double>(error) * error;
  }
  const auto rmse =
      std::sqrt(squared_error / static_cast<double>(actual.size()));
  // The independent oracle decodes only the stored GGUF weights. ggml's
  // native quantized MatMul additionally quantizes each activation row for its
  // Q8 dot-product path, so the implementation error is intentionally judged
  // separately from the already-applied weight quantization loss.
  EXPECT_LE(max_error, 8.0e-3F);
  EXPECT_LE(rmse, 2.0e-3);
}

inline void
expect_quantized_chain_matmul_close(std::span<const float> actual,
                                    std::span<const float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  double squared_error = 0.0;
  double squared_reference = 0.0;
  float max_error = 0.0F;
  float max_reference = 0.0F;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const auto error = std::abs(actual[index] - expected[index]);
    max_error = std::max(max_error, error);
    max_reference = std::max(max_reference, std::abs(expected[index]));
    squared_error += static_cast<double>(error) * error;
    squared_reference += static_cast<double>(expected[index]) * expected[index];
  }
  const auto rmse =
      std::sqrt(squared_error / static_cast<double>(actual.size()));
  const auto reference_rms =
      std::sqrt(squared_reference / static_cast<double>(actual.size()));
  // A complete GGUF FFN quantizes the normalized activation and then the
  // post-gate activation for their respective native dot-product types. The
  // decoded-weight oracle deliberately keeps those activations in F32. Judge
  // that declared extra quantization as a scale-relative error while retaining
  // an absolute floor for values near zero.
  EXPECT_LE(rmse, 2.0e-3 + 2.0e-2 * reference_rms)
      << "reference_rms=" << reference_rms;
  EXPECT_LE(max_error, 2.0e-2F + 2.0e-2F * max_reference)
      << "max_reference=" << max_reference;
}

struct QuantizerCleanup final {
  ~QuantizerCleanup() { ggml_quantize_free(); }
};

} // namespace d5_gguf_test
