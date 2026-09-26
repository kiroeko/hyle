#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "d5_gguf_test_support.h"
#include "d5_qwen38_24t_fixture.h"
#include "d5_qwen_gguf_vocabulary_cases.h"

namespace d5_qwen38_24t_grouped_gguf_fixture {
namespace c = hyle::compute;
namespace q = d5_qwen38_24t_fixture;
namespace profiles = d5_qwen_gguf_vocabulary_cases;

inline constexpr std::int64_t kResidentExperts = q::kRouterResidentExperts;
inline constexpr std::int64_t kSlots = q::kTopK;
inline constexpr std::int64_t kReduction = q::kHidden;
inline constexpr std::int64_t kColumns = q::kExpertHidden;

struct Case final {
  const c::ModelArtifactProfileDescription *profile;
  c::GgufQuantizedType type;
  int seed;
};
inline constexpr std::array kCases{
    Case{&profiles::k24TPrimaryProfile, c::GgufQuantizedType::kIq4Xs,
         163},
    Case{&profiles::k24TFidelityProfile, c::GgufQuantizedType::kQ8_0,
         167}};

inline d5_gguf_test::Packed pack_weight(const Case &test,
                                         std::size_t required_bytes) {
  const auto rows = static_cast<std::size_t>(kResidentExperts * kColumns);
  std::vector<float> physical(rows * static_cast<std::size_t>(kReduction));
  for (std::size_t row = 0; row < rows; ++row) {
    const auto offset = row * static_cast<std::size_t>(kReduction);
    const auto column = (row * 17 + static_cast<std::size_t>(test.seed)) %
                        static_cast<std::size_t>(kReduction);
    physical[offset + column] =
        static_cast<float>(static_cast<int>((row + test.seed) % 11) - 5) /
        32.0F;
  }
  return d5_gguf_test::pack(test.type, physical,
                            static_cast<std::int64_t>(rows), kReduction,
                            required_bytes);
}

inline std::array<std::int32_t, kSlots> ids(int replay) {
  std::array<std::int32_t, kSlots> result{};
  for (std::size_t slot = 0; slot < result.size(); ++slot)
    result[slot] = static_cast<std::int32_t>(
        (slot + (replay == 0 ? 0 : 1)) % kResidentExperts);
  return result;
}

inline std::vector<float> expected(std::span<const float> activation,
                                    std::span<const std::int32_t> selected,
                                    std::span<const float> decoded) {
  std::vector<float> result(selected.size() *
                            static_cast<std::size_t>(kColumns));
  for (std::size_t slot = 0; slot < selected.size(); ++slot) {
    const auto input = activation.subspan(slot * kReduction, kReduction);
    const auto weight = decoded.subspan(
        static_cast<std::size_t>(selected[slot]) * kColumns * kReduction,
        kColumns * kReduction);
    const auto one = d5_gguf_test::matmul(input, weight, kColumns, kReduction);
    std::ranges::copy(one, result.begin() + slot * kColumns);
  }
  return result;
}

} // namespace d5_qwen38_24t_grouped_gguf_fixture
