#pragma once

#include <cstdint>
#include <type_traits>

namespace hyle::simd {

struct float16_t {
  std::uint16_t bits;

  static constexpr float16_t from_bits(std::uint16_t value) noexcept {
    return float16_t{value};
  }
};

struct bfloat16_t {
  std::uint16_t bits;

  static constexpr bfloat16_t from_bits(std::uint16_t value) noexcept {
    return bfloat16_t{value};
  }
};

struct float8_e4m3_t {
  std::uint8_t bits;

  static constexpr float8_e4m3_t from_bits(std::uint8_t value) noexcept {
    return float8_e4m3_t{value};
  }
};

struct float8_e5m2_t {
  std::uint8_t bits;

  static constexpr float8_e5m2_t from_bits(std::uint8_t value) noexcept {
    return float8_e5m2_t{value};
  }
};

struct alignas(16) UInt128 {
  std::uint64_t lo;
  std::uint64_t hi;
};

struct alignas(8) KeyValue32 {
  std::uint32_t value;
  std::uint32_t key;
};

struct alignas(16) KeyValue64 {
  std::uint64_t value;
  std::uint64_t key;
};

static_assert(sizeof(float16_t) == 2);
static_assert(sizeof(bfloat16_t) == 2);
static_assert(sizeof(float8_e4m3_t) == 1);
static_assert(sizeof(float8_e5m2_t) == 1);
static_assert(sizeof(UInt128) == 16 && alignof(UInt128) == 16);
static_assert(sizeof(KeyValue32) == 8 && alignof(KeyValue32) == 8);
static_assert(sizeof(KeyValue64) == 16 && alignof(KeyValue64) == 16);
static_assert(std::is_standard_layout_v<UInt128>);
static_assert(std::is_standard_layout_v<KeyValue32>);
static_assert(std::is_standard_layout_v<KeyValue64>);
static_assert(std::is_trivially_copyable_v<UInt128>);
static_assert(std::is_trivially_copyable_v<KeyValue32>);
static_assert(std::is_trivially_copyable_v<KeyValue64>);

}  // namespace hyle::simd
