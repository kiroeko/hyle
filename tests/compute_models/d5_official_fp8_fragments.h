#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "d5_moe_execution_cases.h"
#include "d5_qwen38_24t_semantic_fragments.h"

namespace d5_official_fp8_fragments {
namespace c = hyle::compute;
namespace sem = d5_qwen38_24t_semantic_fragments;
namespace cases = d5_moe_execution_cases;
struct Case final {
  const c::ModelArtifactProfileDescription *profile;
  std::int64_t n, k;
  bool ue8_scale;
  const char *role;
};
inline constexpr std::array kCases{
    Case{cases::kCases[3].profile, 3072, 7168, true, "shared gate/up"},
    Case{cases::kCases[3].profile, 7168, 3072, true, "shared down"},
    Case{cases::kCases[4].profile, 2048, 4096, true, "shared gate/up"},
    Case{cases::kCases[4].profile, 4096, 2048, true, "shared down"},
    Case{cases::kCases[7].profile, 2048, 6144, false, "expert/shared gate/up"},
    Case{cases::kCases[7].profile, 6144, 2048, false, "expert/shared down"},
    Case{cases::kCases[10].profile, 2048, 4096, false, "expert/shared gate/up"},
    Case{cases::kCases[10].profile, 4096, 2048, false, "expert/shared down"}};

inline auto descriptions(const Case &shape, std::int64_t m) {
  return std::array{
      c::float8_activation_description(m, shape.k, c::Float8ScaleMode::kFloat32Block128).value(),
      c::float8_weight_description(shape.k, shape.n, c::Float8ScaleMode::kFloat32Block128).value()};
}

// Metadata-only boundary case: both operands are ordinary external inputs.
// It can expose a cold backend rejection without allocating unusable payloads.
inline sem::Fragment boundary(const Case &shape, std::int64_t m) {
  sem::Fragment f{.profile = shape.profile, .region = 0x0b0603};
  const auto desc = descriptions(shape, m);
  auto a = f.input(desc[0]), w = f.input(desc[1]);
  f.outputs = {f.one(c::Operation(c::MatMul{}), {a, w})};
  return f;
}

inline void record_rejection(const Case &shape, std::int64_t m,
                              const hyle::Status &status) {
  const auto message = status.message();
  std::printf("FP8_REJECT model=%.*s role=%s M=%lld N=%lld K=%lld code=%d message=%.*s\n",
      int(shape.profile->model_name.size()), shape.profile->model_name.data(), shape.role,
      static_cast<long long>(m), static_cast<long long>(shape.n), static_cast<long long>(shape.k),
      int(status.code()), int(message.size()), message.data());
}

// Exact values are independently tied to their E4M3FN codes. Zero weights
// remain encoded zeros; there is no dense decoded weight copy in the fixture.
inline constexpr std::array<std::uint8_t, 8> kCodes{0x30, 0x38, 0x3c, 0x40, 0xb0, 0xb8, 0xbc, 0xc0};
inline constexpr std::array<float, 8> kValues{0.5F, 1, 1.5F, 2, -0.5F, -1, -1.5F, -2};

inline float scale(const Case &shape, std::size_t row, std::size_t group, int seed) {
  if (shape.ue8_scale) {
    const auto source = std::uint8_t(121 + (row + group + seed) % 3);
    return std::ldexp(1.0F, int(source) - 127); // exact UE8M0 -> F32 widening
  }
  return float(17 + (row * 3 + group + seed) % 11) / 1024.0F;
}

inline sem::Fragment projection(const Case &shape, std::int64_t m) {
  sem::Fragment f{.profile = shape.profile, .region = 0x0b0603};
  const auto desc = descriptions(shape, m);
  auto input = f.input(desc[0]);
  std::vector<std::byte> w(desc[1].required_bytes().value());
  const auto blocks = std::size_t(shape.k / 128);
  for (std::size_t row = 0; row < std::size_t(shape.n / 128); ++row)
    for (std::size_t group = 0; group < blocks; ++group) {
      const auto value = scale(shape, row, group, 13);
      std::memcpy(w.data() + shape.n * shape.k + (row * blocks + group) * 4, &value, 4);
    }
  for (std::int64_t row = 0; row < shape.n; ++row)
    for (int term = 0; term < 2; ++term) {
      const auto col = (row * 17 + 31 + term * (shape.k / 2 + 13)) % shape.k;
      w[row * shape.k + col] = std::byte{kCodes[(row + term * 3) % 8]};
    }
  auto weight = f.constant(desc[1], std::move(w));
  f.outputs = {f.one(c::Operation(c::MatMul{}), {input, weight})};
  // Two sparse terms with these dyadic codes/scales have an exact F32 sum.
  // The general semantic-fragment absolute tolerance would mask small outputs.
  f.exact_outputs = {0};
  for (int replay = 0; replay < 2; ++replay) {
    std::vector<std::byte> a(desc[0].required_bytes().value());
    std::vector<float> expected(m * shape.n);
    for (std::int64_t row = 0; row < m; ++row) {
      for (std::size_t group = 0; group < blocks; ++group) {
        const auto value = scale(shape, row, group, 3 + replay);
        std::memcpy(a.data() + m * shape.k + (row * blocks + group) * 4, &value, 4);
      }
      for (std::int64_t col = 0; col < shape.k; ++col)
        a[row * shape.k + col] = std::byte{kCodes[(row * 3 + col + replay) % 8]};
      for (std::int64_t out = 0; out < shape.n; ++out) {
        double value = 0;
        for (int term = 0; term < 2; ++term) {
          const auto col = (out * 17 + 31 + term * (shape.k / 2 + 13)) % shape.k;
          const double x = kValues[(row * 3 + col + replay) % 8] *
                           double(scale(shape, row, col / 128, 3 + replay));
          const double y = kValues[(out + term * 3) % 8] *
                           double(scale(shape, out / 128, col / 128, 13));
          value += x * y;
        }
        expected[row * shape.n + out] = float(value);
      }
    }
    f.replays.push_back({{std::move(a)}, {std::move(expected)}});
  }
  return f;
}
} // namespace d5_official_fp8_fragments
