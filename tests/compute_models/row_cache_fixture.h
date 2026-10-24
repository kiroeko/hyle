#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>
#include "hyle/compute/low_precision.h"

namespace row_cache_test {
namespace c = hyle::compute;
inline double decode(std::uint16_t bits) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}
// Search adjacent BF16 values independently of the product's integer RNE path.
inline std::uint16_t bfloat(double value) {
  const std::uint16_t sign = std::signbit(value) ? 0x8000 : 0;
  if (std::isnan(value)) return static_cast<std::uint16_t>(sign | 0x7fc0);
  const double magnitude = std::abs(value);
  const double largest = decode(0x7f7f), step = largest - decode(0x7f7e);
  if (magnitude >= largest + step * .5)
    return static_cast<std::uint16_t>(sign | 0x7f80);
  std::uint16_t lo = 0, hi = 0x7f7f;
  while (lo < hi) {
    const auto mid = static_cast<std::uint16_t>((unsigned(lo) + hi + 1) / 2);
    if (decode(mid) <= magnitude) lo = mid;
    else hi = static_cast<std::uint16_t>(mid - 1);
  }
  if (lo != 0x7f7f) {
    const auto below = magnitude - decode(lo), above = decode(static_cast<std::uint16_t>(lo + 1)) - magnitude;
    if (above < below || (above == below && (lo & 1) != 0)) ++lo;
  }
  return static_cast<std::uint16_t>(sign | lo);
}
constexpr std::size_t kRows = 5;

inline double value(unsigned code, bool four) {
  if (four) {
    constexpr std::array<double, 8> values{0, .5, 1, 1.5, 2, 3, 4, 6};
    return values[code];
  }
  const unsigned exponent = code >> 3, mantissa = code & 7;
  return exponent == 0 ? std::ldexp(double(mantissa), -9)
                       : std::ldexp(1.0 + double(mantissa) / 8, int(exponent) - 7);
}

inline std::vector<std::uint16_t> input_values(c::RowCacheEncoding encoding, int phase) {
  const bool four = encoding.format() == c::RowCacheFormat::kFloat4Block32;
  const std::size_t depth = encoding.depth(), group = four ? 32 : 64;
  std::vector<std::uint16_t> input(kRows * depth);
  constexpr std::array<double, 12> ties{.25, .75, 1.25, 1.75, 2.5, 3.5,
                                       5, .0009765625, .0029296875, 1.0625, 1.1875, 17};
  for (std::size_t row = 0; row < kRows; ++row)
    for (std::size_t column = 0; column < depth; ++column) {
      double x = (int((column * 7 + row * 11 + phase * 3) % 37) - 18) / 8.0;
      if (row == 0) x = std::ldexp(x, -126);
      if (row == 1) x = ties[(column + phase) % (four ? 7 : ties.size())];
      if ((column & 1) != 0) x = -x;
      input[row * depth + column] = bfloat(x);
    }
  for (std::size_t base = 0; base < (four ? 128u : 448u); base += group)
    input[depth + base] = bfloat(four ? 6.0 : 448.0);
  input[0] = 0; input[1] = 0x8000; input[2] = 1; input[3] = 0x8001;
  input[2 * depth] = 0x7f7f; input[2 * depth + 1] = 0xff7f;
  input[4 * depth] = 0x8000;
  if (!four) {
    constexpr std::array<std::uint16_t, 8> tail{0,0x8000,1,0x8001,0x7f7f,0xff7f,0x3f80,0xbf80};
    for (std::size_t row = 0; row < kRows; ++row)
      for (std::size_t i = 448; i < depth; ++i)
        input[row * depth + i] = tail[(i + row + phase) % tail.size()];
  }
  return input;
}

// Definition-based FP64 table/distance oracle. Only the documented F32
// normalization and BF16 output boundaries are explicitly rounded.
struct Reference { std::vector<std::uint8_t> bytes; std::vector<std::uint16_t> decoded; };
inline Reference reference(c::RowCacheEncoding encoding, std::span<const std::uint16_t> input) {
  const bool four = encoding.format() == c::RowCacheFormat::kFloat4Block32;
  const std::size_t depth = encoding.depth(), width = encoding.row_bytes();
  const std::size_t quantized = four ? 128 : 448, group = four ? 32 : 64;
  Reference result{std::vector<std::uint8_t>(kRows * width),std::vector<std::uint16_t>(input.size())};
  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t base = 0; base < quantized; base += group) {
      double maximum = four ? 6 * std::ldexp(1.0, -126) : double(1.0e-4F);
      for (std::size_t i = 0; i < group; ++i)
        maximum = (std::max)(maximum, std::abs(decode(input[row * depth + base + i])));
      const int exponent = int(std::ceil(std::log2(maximum / (four ? 6 : 448))));
      const double scale = std::ldexp(1.0, exponent);
      result.bytes[row * width + (four ? 64 : 576) + base / group] = std::uint8_t(exponent + 127);
      for (std::size_t i = 0; i < group; ++i) {
        const auto index = row * depth + base + i;
        const float normalized = float(decode(input[index]) / scale);
        const double magnitude = (std::min)(std::abs(double(normalized)), four ? 6.0 : 448.0);
        unsigned chosen = 0; double distance = magnitude;
        for (unsigned code = 1; code < (four ? 8u : 127u); ++code) {
          const auto next = std::abs(magnitude - value(code, four));
          if (next < distance || (next == distance && (code & 1) == 0 && (chosen & 1) != 0)) {
            chosen = code; distance = next;
          }
        }
        const bool negative = (input[index] & 0x8000) != 0;
        const auto code = std::uint8_t(chosen | (negative ? (four ? 8 : 128) : 0));
        const auto column = base + i;
        if (four) result.bytes[row * width + column / 2] |= std::uint8_t(code << ((column & 1) * 4));
        else result.bytes[row * width + column] = code;
        const float reconstructed = float((negative ? -value(chosen, four) : value(chosen, four)) * scale);
        result.decoded[index] = bfloat(reconstructed);
      }
    }
    if (!four) for (std::size_t i = 448; i < depth; ++i) {
      const auto bits = input[row * depth + i];
      result.bytes[row * width + 448 + (i - 448) * 2] = std::uint8_t(bits);
      result.bytes[row * width + 449 + (i - 448) * 2] = std::uint8_t(bits >> 8);
      result.decoded[row * depth + i] = bits;
    }
  }
  return result;
}

} // namespace row_cache_test
