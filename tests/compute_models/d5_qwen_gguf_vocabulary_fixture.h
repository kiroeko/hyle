#pragma once

#include <ggml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

#include "d5_gguf_test_support.h"
#include "d5_qwen_gguf_vocabulary_cases.h"
#include "d5_qwen_vocabulary_fixture.h"

namespace d5_qwen_gguf_vocabulary_fixture {
namespace c = hyle::compute;
namespace cases = d5_qwen_gguf_vocabulary_cases;
namespace vocab = d5_qwen_vocabulary_fixture;

inline constexpr std::int64_t kPatternPeriod = 448;

inline bool is_quantized(cases::Storage storage) {
  return storage == cases::Storage::kQ4K ||
         storage == cases::Storage::kQ6K ||
         storage == cases::Storage::kQ8_0;
}

inline c::GgufQuantizedType quantized_type(cases::Storage storage) {
  switch (storage) {
  case cases::Storage::kQ4K:
    return c::GgufQuantizedType::kQ4K;
  case cases::Storage::kQ6K:
    return c::GgufQuantizedType::kQ6K;
  case cases::Storage::kQ8_0:
    return c::GgufQuantizedType::kQ8_0;
  default:
    throw std::invalid_argument("dense GGUF vocabulary storage");
  }
}

inline c::DataType data_type(cases::Storage storage) {
  switch (storage) {
  case cases::Storage::kFloat16:
    return c::DataType::kFloat16;
  case cases::Storage::kBFloat16:
    return c::DataType::kBFloat16;
  default:
    return c::DataType::kFloat32;
  }
}

inline c::TensorDesc head_description(cases::Storage storage,
                                      std::int64_t hidden,
                                      std::int64_t rows) {
  if (is_quantized(storage))
    return c::gguf_weight_description(quantized_type(storage),
                                      c::DataType::kFloat32, hidden, rows)
        .value();
  return c::TensorDesc::dense(data_type(storage),
                              std::array{rows, hidden})
      .value()
      .transposed()
      .value();
}

inline c::TensorDesc table_description(cases::Storage storage,
                                       std::int64_t hidden,
                                       std::int64_t rows) {
  return head_description(storage, hidden, rows).transposed().value();
}

inline std::int64_t source_column(std::int64_t row, std::int64_t hidden) {
  return ((row % kPatternPeriod) * 131 + 17) % hidden;
}

inline float coefficient(std::int64_t row) {
  const auto sign = row % 2 == 0 ? 1 : -1;
  return static_cast<float>(sign * (1 + row % 7)) / 16.0F;
}

inline std::vector<float> activation(std::int64_t hidden, int seed) {
  std::vector<float> result(static_cast<std::size_t>(hidden));
  for (std::size_t column = 0; column < result.size(); ++column)
    result[column] = static_cast<float>(
        (static_cast<int>((column * 11 + seed * 19) % 47) - 23) / 64.0);
  return result;
}

inline std::vector<std::byte> encode_dense(cases::Storage storage,
                                            std::span<const float> source) {
  std::vector<std::byte> result(source.size() * sizeof(std::uint16_t));
  for (std::size_t i = 0; i < source.size(); ++i) {
    const auto bits = storage == cases::Storage::kFloat16
                          ? half_test::encode<hyle::simd::float16_t>(source[i])
                                .bits
                          : half_test::encode<hyle::simd::bfloat16_t>(source[i])
                                .bits;
    std::memcpy(result.data() + i * sizeof(bits), &bits, sizeof(bits));
  }
  return result;
}

inline std::vector<float> decode_values(cases::Storage storage,
                                        std::span<const std::byte> bytes) {
  const auto width = is_quantized(storage) ? sizeof(float)
                                            : sizeof(std::uint16_t);
  if (bytes.size() % width != 0)
    throw std::invalid_argument("unaligned vocabulary output");
  std::vector<float> result(bytes.size() / width);
  for (std::size_t i = 0; i < result.size(); ++i) {
    if (is_quantized(storage)) {
      std::memcpy(&result[i], bytes.data() + i * sizeof(float), sizeof(float));
    } else {
      std::uint16_t bits = 0;
      std::memcpy(&bits, bytes.data() + i * sizeof(bits), sizeof(bits));
      result[i] = static_cast<float>(
          storage == cases::Storage::kFloat16
              ? half_test::decode_bits<hyle::simd::float16_t>(bits)
              : half_test::decode_bits<hyle::simd::bfloat16_t>(bits));
    }
  }
  return result;
}

class SparseRows final {
public:
  SparseRows(cases::Storage storage, std::int64_t hidden)
      : storage_(storage), hidden_(hidden), templates_{} {
    if (hidden <= 0)
      throw std::invalid_argument("empty vocabulary hidden width");
    const auto native = is_quantized(storage_)
                            ? d5_gguf_test::native_type(quantized_type(storage_))
                            : GGML_TYPE_COUNT;
    const auto row_bytes = is_quantized(storage_)
                               ? ggml_row_size(native, hidden_)
                               : static_cast<std::size_t>(hidden_) *
                                     sizeof(std::uint16_t);
    row_bytes_ = row_bytes;
    for (std::int64_t pattern = 0; pattern < kPatternPeriod; ++pattern) {
      auto &entry = templates_[static_cast<std::size_t>(pattern)];
      std::vector<float> source(static_cast<std::size_t>(hidden_), 0.0F);
      source[static_cast<std::size_t>(source_column(pattern, hidden_))] =
          coefficient(pattern);
      entry.decoded.resize(source.size());
      if (is_quantized(storage_)) {
        entry.encoded.resize(row_bytes_);
        const auto written = ggml_quantize_chunk(
            native, source.data(), entry.encoded.data(), 0, 1, hidden_, nullptr);
        if (written != row_bytes_)
          throw std::runtime_error("GGUF vocabulary row quantization failed");
        const auto *traits = ggml_get_type_traits(native);
        if (traits == nullptr || traits->to_float == nullptr)
          throw std::runtime_error("GGUF vocabulary row decoder unavailable");
        traits->to_float(entry.encoded.data(), entry.decoded.data(), hidden_);
      } else {
        entry.encoded = encode_dense(storage_, source);
        entry.decoded = decode_values(storage_, entry.encoded);
      }
      if (std::abs(entry.decoded[static_cast<std::size_t>(
                       source_column(pattern, hidden_))]) < 0.01F)
        throw std::runtime_error("GGUF vocabulary oracle lost its nonzero row");
    }
  }

  std::vector<std::byte> shard(vocab::Shard shard) const {
    std::vector<std::byte> result(static_cast<std::size_t>(shard.rows) *
                                  row_bytes_);
    for (std::int64_t local = 0; local < shard.rows; ++local) {
      const auto &entry = template_for(shard.first + local);
      std::memcpy(result.data() + static_cast<std::size_t>(local) * row_bytes_,
                  entry.encoded.data(), row_bytes_);
    }
    return result;
  }

  std::span<const float> row(std::int64_t global_row) const {
    return template_for(global_row).decoded;
  }

  std::vector<float> logits(std::span<const float> input,
                            vocab::Shard shard) const {
    std::vector<float> result(static_cast<std::size_t>(shard.rows));
    std::array<double, kPatternPeriod> dot{};
    for (std::int64_t pattern = 0; pattern < kPatternPeriod; ++pattern) {
      const auto &decoded = templates_[static_cast<std::size_t>(pattern)].decoded;
      double sum = 0.0;
      for (std::size_t i = 0; i < decoded.size(); ++i)
        sum += static_cast<double>(decoded[i]) * input[i];
      dot[static_cast<std::size_t>(pattern)] = sum;
    }
    for (std::int64_t local = 0; local < shard.rows; ++local)
      result[static_cast<std::size_t>(local)] = static_cast<float>(
          dot[static_cast<std::size_t>((shard.first + local) % kPatternPeriod)]);
    return result;
  }

  std::size_t row_bytes() const noexcept { return row_bytes_; }

private:
  struct Row final {
    std::vector<std::byte> encoded;
    std::vector<float> decoded;
  };

  const Row &template_for(std::int64_t global_row) const {
    if (global_row < 0 || global_row >= vocab::kVocabularyRows)
      throw std::out_of_range("global vocabulary row");
    return templates_[static_cast<std::size_t>(global_row % kPatternPeriod)];
  }

  cases::Storage storage_;
  std::int64_t hidden_;
  std::size_t row_bytes_ = 0;
  std::array<Row, kPatternPeriod> templates_;
  d5_gguf_test::QuantizerCleanup cleanup_;
};

} // namespace d5_qwen_gguf_vocabulary_fixture
