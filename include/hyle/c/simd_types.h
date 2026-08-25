/* Pre-release language-neutral value types shared by the HYLE SIMD draft FFI
 * schema and its C++ core. This header alone is not a frozen ABI or a native-C
 * product promise; SIMD managed review may revise or remove FFI-facing uses. */
#ifndef HYLE_C_SIMD_TYPES_H_
#define HYLE_C_SIMD_TYPES_H_

#include <stdint.h>

#if defined(__cplusplus)
#  define HYLE_SIMD_ALIGNAS(n) alignas(n)
#elif defined(_MSC_VER)
#  define HYLE_SIMD_ALIGNAS(n) __declspec(align(n))
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_SIMD_ALIGNAS(n) __attribute__((aligned(n)))
#else
#  define HYLE_SIMD_ALIGNAS(n)
#endif

#ifdef __cplusplus

namespace hyle::simd {

struct float16_t {
  uint16_t bits;
  static constexpr float16_t from_bits(uint16_t value) noexcept {
    return float16_t{value};
  }
};
struct bfloat16_t {
  uint16_t bits;
  static constexpr bfloat16_t from_bits(uint16_t value) noexcept {
    return bfloat16_t{value};
  }
};
struct float8_e4m3_t {
  uint8_t bits;
  static constexpr float8_e4m3_t from_bits(uint8_t value) noexcept {
    return float8_e4m3_t{value};
  }
};
struct float8_e5m2_t {
  uint8_t bits;
  static constexpr float8_e5m2_t from_bits(uint8_t value) noexcept {
    return float8_e5m2_t{value};
  }
};
struct HYLE_SIMD_ALIGNAS(16) UInt128 {
  uint64_t lo;
  uint64_t hi;
};
struct HYLE_SIMD_ALIGNAS(8) KeyValue32 {
  uint32_t value;
  uint32_t key;
};
struct HYLE_SIMD_ALIGNAS(16) KeyValue64 {
  uint64_t value;
  uint64_t key;
};

}  // namespace hyle::simd

using hyle_simd_f16 = hyle::simd::float16_t;
using hyle_simd_bf16 = hyle::simd::bfloat16_t;
using hyle_simd_f8e4m3 = hyle::simd::float8_e4m3_t;
using hyle_simd_f8e5m2 = hyle::simd::float8_e5m2_t;
using hyle_simd_u128 = hyle::simd::UInt128;
using hyle_simd_kv32 = hyle::simd::KeyValue32;
using hyle_simd_kv64 = hyle::simd::KeyValue64;

#else

typedef struct hyle_simd_f16 { uint16_t bits; } hyle_simd_f16;
typedef struct hyle_simd_bf16 { uint16_t bits; } hyle_simd_bf16;
typedef struct hyle_simd_f8e4m3 { uint8_t bits; } hyle_simd_f8e4m3;
typedef struct hyle_simd_f8e5m2 { uint8_t bits; } hyle_simd_f8e5m2;
typedef struct HYLE_SIMD_ALIGNAS(16) hyle_simd_u128 {
  uint64_t lo;
  uint64_t hi;
} hyle_simd_u128;
typedef struct HYLE_SIMD_ALIGNAS(8) hyle_simd_kv32 {
  uint32_t value;
  uint32_t key;
} hyle_simd_kv32;
typedef struct HYLE_SIMD_ALIGNAS(16) hyle_simd_kv64 {
  uint64_t value;
  uint64_t key;
} hyle_simd_kv64;

#endif

#undef HYLE_SIMD_ALIGNAS

#endif  /* HYLE_C_SIMD_TYPES_H_ */
