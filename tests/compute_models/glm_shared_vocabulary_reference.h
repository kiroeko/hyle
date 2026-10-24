#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "glm_original_indexer_reference.h"

// Independent local checks for shared original BF16 vocabulary roles. The
// decoder remainder is still generated: these checks start at its ACTUAL F32
// residual/add boundary, not an alleged original BF16 decoder activation.
// Mapped weights stay as little-endian bytes; no second dense weight bank or
// reference executor is constructed. No ideal result crosses a BF16 store.
namespace glm_shared_vocabulary_reference {
namespace detail {
namespace scalar = glm_original_indexer_reference::detail;
inline std::size_t bank_bytes(std::size_t vocabulary, std::size_t hidden) {
  if (!vocabulary || !hidden || vocabulary >
      std::numeric_limits<std::size_t>::max() / hidden / 2)
    throw std::runtime_error("shared vocabulary geometry overflows its BF16 bank");
  return vocabulary * hidden * 2;
}
inline std::uint16_t read_bfloat(std::span<const std::byte> bank, std::size_t element) {
  const auto offset = element * 2;
  return std::uint16_t(std::to_integer<unsigned int>(bank[offset]) |
      (std::to_integer<unsigned int>(bank[offset + 1]) << 8U));
}
inline float finite_bfloat(std::uint16_t value) {
  const auto result = scalar::widen(value);
  if (!std::isfinite(result))
    throw std::runtime_error("shared original BF16 operand is nonfinite");
  return result;
}
inline void normal_bfloat(std::uint16_t value) {
  // Reject BF16 subnormals even when a large other operand could make their
  // product normal: a native F32 source conversion may otherwise be FTZ.
  if ((value & 0x7f80U) == 0 && (value & 0x007fU) != 0)
    throw std::runtime_error("shared LM needs separate subnormal-source analysis");
}
inline void rows(std::size_t size, std::size_t hidden) {
  if (!hidden || !size || size % hidden)
    throw std::runtime_error("shared vocabulary actual row geometry differs");
}
struct Dot final { double center = 0, bound = 0, sumabs = 0; };
inline Dot dot(std::span<const float> input, std::span<const std::byte> weight) {
  if (input.empty() || weight.size() != input.size() * 2)
    throw std::runtime_error("shared LM dot row extent differs");
  constexpr std::size_t block = 128;
  scalar::Sum blocks;
  double absolute = 0;
  for (std::size_t first = 0; first < input.size(); first += block) {
    double partial = 0;
    const auto last = std::min(input.size(), first + block);
    for (std::size_t column = first; column < last; ++column) {
      const auto bits = read_bfloat(weight, column);
      normal_bfloat(bits);
      const double product = double(input[column]) * finite_bfloat(bits);
      // BF16*BF16 has at most16 significant bits and is exact F32 when this
      // normal-quantum condition and the final absolute-sum bound hold.
      if (!std::isfinite(product) || (product != 0 && std::abs(product) < 0x1p-103))
        throw std::runtime_error("shared LM needs separate product/underflow analysis");
      partial += product;
      absolute += std::abs(product);
    }
    blocks.add(partial);
  }
  if (!std::isfinite(absolute))
    throw std::runtime_error("shared LM absolute-product sum is nonfinite");
  const auto width = input.size();
  const double g64 = scalar::gamma(width, 0x1p-53);
  const double upper_sum = absolute == 0 ? 0 : scalar::upper(absolute / (1 - g64));
  if (upper_sum > double(std::numeric_limits<float>::max()) / 4)
    throw std::runtime_error("shared LM needs separate F32 overflow analysis");
  const auto count = width / block + std::size_t(width % block != 0);
  const double inner = scalar::gamma(std::min(block, width), 0x1p-53);
  const double outer = scalar::gamma(count, 0x1p-53);
  const double blocked = scalar::upper_add(scalar::upper_add(inner, outer),
      scalar::upper_multiply(inner, outer));
  // Ordinary blocked FP64 summation supplies a bound independent of whether
  // compensation happened to help. Add its actual distance to that center.
  const double center = blocks.value();
  const double oracle = scalar::upper_add(scalar::distance(center, blocks.plain),
      scalar::upper_multiply(blocked, upper_sum));
  const double native = scalar::upper_multiply(scalar::gamma(width, 0x1p-24), upper_sum);
  return {center, absolute == 0 ? 0 : scalar::upper_add(native, oracle), absolute};
}
inline std::vector<std::size_t> candidates(std::span<const float> row) {
  std::vector<std::size_t> result;
  result.reserve(4);
  for (std::size_t token = 0; token < row.size(); ++token) {
    if (!std::isfinite(row[token]))
      throw std::runtime_error("shared head has a nonfinite actual logit");
    auto place = result.begin();
    while (place != result.end() && (row[*place] > row[token] ||
        (row[*place] == row[token] && *place < token))) ++place;
    if (place != result.end() || result.size() < 4) result.insert(place, token);
    if (result.size() > 4) result.pop_back();
  }
  return result;
}
} // namespace detail

inline bool expect_embedding(std::span<const std::uint16_t> actual,
    std::span<const std::byte> original_bank, std::span<const std::int32_t> ids,
    std::size_t vocabulary, std::size_t hidden) {
  if (original_bank.size() != detail::bank_bytes(vocabulary, hidden) || ids.empty() ||
      ids.size() > std::numeric_limits<std::size_t>::max() / hidden || actual.size() != ids.size() * hidden) {
    ADD_FAILURE() << "original shared embedding extent"; return false;
  }
  for (std::size_t row = 0; row < ids.size(); ++row) {
    if (ids[row] < 0 || std::size_t(ids[row]) >= vocabulary) {
      ADD_FAILURE() << "actual shared embedding ID outside full vocabulary"; return false;
    }
    for (std::size_t column = 0; column < hidden; ++column) {
      const auto expected = detail::read_bfloat(original_bank, std::size_t(ids[row]) * hidden + column);
      detail::finite_bfloat(expected);
      if (actual[row * hidden + column] != expected) {
        ADD_FAILURE() << "original embedding row=" << ids[row] << " column=" << column;
        return false;
      }
    }
  }
  return true;
}

inline bool expect_widen(std::span<const float> actual, std::span<const std::uint16_t> source) {
  if (actual.size() != source.size()) { ADD_FAILURE() << "shared BF16 widening extent"; return false; }
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (!std::isfinite(detail::finite_bfloat(source[i])) ||
        std::bit_cast<std::uint32_t>(actual[i]) != (std::uint32_t(source[i]) << 16U)) {
      ADD_FAILURE() << "shared actual BF16 widening[" << i << ']'; return false;
    }
  return true;
}

inline bool expect_sum(std::span<const float> actual, std::span<const float> residual,
                       std::span<const float> ffn) {
  if (actual.size() != residual.size() || actual.size() != ffn.size() || actual.empty()) {
    ADD_FAILURE() << "shared generated residual/FFN addition extent"; return false;
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(residual[i]) || !std::isfinite(ffn[i]))
      throw std::runtime_error("shared generated residual/FFN input is nonfinite");
    const float expected = float(double(residual[i]) + double(ffn[i]));
    if (!std::isfinite(expected) || std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(expected)) {
      ADD_FAILURE() << "actual F32 residual plus FFN[" << i << ']'; return false;
    }
  }
  return true;
}

inline bool expect_norm(std::span<const float> actual_raw,
    std::span<const std::uint16_t> actual_stored, std::span<const float> actual_combined,
    std::span<const std::uint16_t> original_norm, std::size_t hidden, float epsilon = 1.0e-5F) {
  detail::rows(actual_combined.size(), hidden);
  if (actual_raw.size() != actual_combined.size() || actual_stored.size() != actual_raw.size() ||
      original_norm.size() != hidden || !std::isfinite(epsilon) || epsilon <= 0) {
    ADD_FAILURE() << "original shared-head RMS norm extent/epsilon"; return false;
  }
  double maximum_error = 0, maximum_ratio = 0;
  for (std::size_t first = 0; first < actual_combined.size(); first += hidden) {
    detail::scalar::Sum squares;
    for (std::size_t column = 0; column < hidden; ++column) {
      const float value = actual_combined[first + column];
      const double square = double(value) * value;
      if (!std::isfinite(value) || std::fpclassify(value) == FP_SUBNORMAL ||
          (square != 0 && square < 0x1p-103))
        throw std::runtime_error("shared-head RMS needs separate input/underflow analysis");
      squares.add(square);
    }
    if (!std::isfinite(squares.value()) || squares.value() > double(std::numeric_limits<float>::max()) / 4)
      throw std::runtime_error("shared-head RMS needs separate overflow analysis");
    const double inverse = 1 / std::sqrt(squares.value() / double(hidden) + double(epsilon));
    for (std::size_t column = 0; column < hidden; ++column) {
      const auto i = first + column;
      const double center = double(actual_combined[i]) * inverse * detail::finite_bfloat(original_norm[column]);
      // This is the already-established local RMS nonlinear gate, applied
      // only to actual combined F32 values. It is not a full-decoder BF16
      // equivalence claim, nor a tolerance for the later BF16 discontinuity.
      const double local = 8.0e-5 + 4.0e-5 * std::abs(center);
      const double error = std::abs(double(actual_raw[i]) - center);
      if (!std::isfinite(center) || !std::isfinite(actual_raw[i]) || error > local) {
        ADD_FAILURE() << "original shared RMS[" << i << "] actual=" << actual_raw[i]
            << " local F64=" << center << " error=" << error << " established local gate=" << local;
        return false;
      }
      maximum_error = std::max(maximum_error, error);
      maximum_ratio = std::max(maximum_ratio, error / local);
    }
  }
  std::printf("GLM original shared RMS: values=%zu maxAbs=%.17g maxLocalRatio=%.17g\n",
      actual_raw.size(), maximum_error, maximum_ratio);
  return glm_original_indexer_reference::expect_bfloat_store(actual_stored, actual_raw, "original shared RMS actual store");
}

// The first episode checks every vocabulary row. Later episodes check the
// actual stable top4 logits plus supplied selected IDs, with the SAME dot
// contract; greedy() still scans every actual logit on every invocation.
inline bool expect_logits(std::span<const float> actual,
    std::span<const std::uint16_t> actual_normalized, std::span<const std::byte> original_head,
    std::size_t vocabulary, std::size_t hidden, bool full,
    std::span<const std::int32_t> selected_ids = {}) {
  detail::rows(actual_normalized.size(), hidden);
  const auto rows = actual_normalized.size() / hidden;
  if (original_head.size() != detail::bank_bytes(vocabulary, hidden) ||
      rows > std::numeric_limits<std::size_t>::max() / vocabulary || actual.size() != rows * vocabulary ||
      (!selected_ids.empty() && selected_ids.size() != rows)) {
    ADD_FAILURE() << "original shared LM extent"; return false;
  }
  std::vector<float> input(hidden);
  std::size_t checked = 0;
  double maximum_error = 0, maximum_ratio = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < hidden; ++column) {
      const auto bits = actual_normalized[row * hidden + column];
      detail::normal_bfloat(bits); input[column] = detail::finite_bfloat(bits);
    }
    const auto actual_row = actual.subspan(row * vocabulary, vocabulary);
    auto candidates = detail::candidates(actual_row);
    if (!selected_ids.empty()) {
      if (selected_ids[row] < 0 || std::size_t(selected_ids[row]) >= vocabulary) {
        ADD_FAILURE() << "original shared selected LM ID outside vocabulary"; return false;
      }
      const auto selected = std::size_t(selected_ids[row]);
      if (std::find(candidates.begin(), candidates.end(), selected) == candidates.end()) candidates.push_back(selected);
    }
    const auto check = [&](std::size_t token) {
      const auto reference = detail::dot(input, original_head.subspan(token * hidden * 2, hidden * 2));
      const double value = actual_row[token];
      const double error = detail::scalar::distance(value, reference.center);
      if (!std::isfinite(value) || error > reference.bound) {
        ADD_FAILURE() << "original shared LM row=" << row << " token=" << token
            << " actual=" << value << " blocked F64=" << reference.center
            << " sumabs=" << reference.sumabs << " error=" << error << " F32 accumulation bound=" << reference.bound;
        return false;
      }
      ++checked; maximum_error = std::max(maximum_error, error);
      maximum_ratio = std::max(maximum_ratio, reference.bound ? error / reference.bound : 0.0);
      return true;
    };
    if (full) { for (std::size_t token = 0; token < vocabulary; ++token) if (!check(token)) return false; }
    else { for (const auto token : candidates) if (!check(token)) return false; }
  }
  std::printf("GLM original shared LM: full=%d values=%zu K=%zu maxAbs=%.17g maxBoundRatio=%.17g\n",
      int(full), checked, hidden, maximum_error, maximum_ratio);
  return true;
}

inline bool expect_greedy(std::span<const float> actual_logits,
    std::span<const std::int32_t> actual_ids, std::size_t vocabulary,
    std::span<const float> actual_values = {}) {
  if (!vocabulary || actual_ids.empty() || actual_ids.size() > std::numeric_limits<std::size_t>::max() / vocabulary ||
      actual_logits.size() != actual_ids.size() * vocabulary ||
      (!actual_values.empty() && actual_values.size() != actual_ids.size())) {
    ADD_FAILURE() << "shared stable full-vocabulary TopK1 extent"; return false;
  }
  for (std::size_t row = 0; row < actual_ids.size(); ++row) {
    const auto logits = actual_logits.subspan(row * vocabulary, vocabulary);
    const auto candidates = detail::candidates(logits);
    const auto token = candidates.front();
    if (actual_ids[row] < 0 || std::size_t(actual_ids[row]) != token ||
        (!actual_values.empty() && std::bit_cast<std::uint32_t>(actual_values[row]) != std::bit_cast<std::uint32_t>(logits[token]))) {
      ADD_FAILURE() << "actual stable full-vocabulary greedy ID/value row=" << row << " expected ID=" << token;
      return false;
    }
  }
  return true;
}

inline bool expect_feedback(std::span<const std::int32_t> actual_pending,
                           std::span<const std::int32_t> actual_previous_prediction) {
  if (actual_pending.empty() || actual_pending.size() != actual_previous_prediction.size() ||
      !std::equal(actual_pending.begin(), actual_pending.end(), actual_previous_prediction.begin())) {
    ADD_FAILURE() << "next shared embedding must consume the actual preceding prediction ID"; return false;
  }
  return true;
}
} // namespace glm_shared_vocabulary_reference
