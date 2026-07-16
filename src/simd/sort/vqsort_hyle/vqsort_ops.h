// Copyright 2021 Google LLC
// Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
// SPDX-License-Identifier: BSD-3-Clause
//
// HYLE helpers for the vqsort port of Highway contrib/sort.
// Origin: hwy/base.h + hwy/cache_control.h + hwy/contrib/algo/copy-inl.h
// upstream commit SHA: 83eff39371e5e6494b305b253f8a3c06f831faf4
// This file provides only hyle::simd D-tag primitives / thin local helpers.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <limits>
#include <type_traits>
#if defined(_MSC_VER)
#include <intrin.h>  // _BitScanReverse
#endif

#include "hyle/macros.h"
#include "hyle/simd/vec.h"

// Must be included only from a TU that already defined HYLE_SIMD_TARGET_*.
#if !defined(HYLE_SIMD_TARGET_AVX2) && !defined(HYLE_SIMD_TARGET_AVX512) && \
    !defined(HYLE_SIMD_TARGET_NEON)
#error "vqsort_ops.h requires HYLE_SIMD_TARGET_AVX2 / AVX512 / NEON"
#endif

namespace hyle::simd {
namespace detail {
namespace vqsort {

// ---- Highway-compat type aliases used by the ported headers ---------------------
template <class D>
using Vec = VFromD<D>;
template <class D>
using Mask = MFromD<D>;

// Unqualified size_t / fixed-width names (Highway headers use the C names).
using size_t = std::size_t;
using uint8_t = std::uint8_t;
using uint16_t = std::uint16_t;
using uint32_t = std::uint32_t;
using uint64_t = std::uint64_t;
using int16_t = std::int16_t;
using int32_t = std::int32_t;
using int64_t = std::int64_t;

// ---- Scalar type helpers (Highway base.h) ---------------------------------------
template <class T, class U>
constexpr bool IsSame() noexcept {
  return std::is_same_v<T, U>;
}
template <typename T>
constexpr bool IsFloat() noexcept {
  return std::is_floating_point_v<T>;
}
template <typename T>
constexpr bool IsSigned() noexcept {
  return std::is_signed_v<T>;
}
template <typename T>
constexpr T LimitsMax() noexcept {
  return std::numeric_limits<T>::max();
}
template <typename T>
constexpr T LimitsMin() noexcept {
  return std::numeric_limits<T>::lowest();
}
template <typename T>
constexpr T HighestValue() noexcept {
  return std::numeric_limits<T>::max();
}
template <typename T>
constexpr T LowestValue() noexcept {
  return std::numeric_limits<T>::lowest();
}

// Type-level If (Highway hwy::If).
template <bool B, class Then, class Else>
using If = std::conditional_t<B, Then, Else>;

// 128-bit key (only needed for KeyAdapter completeness; public HYLE API uses KV).
struct alignas(16) uint128_t {
  uint64_t lo;
  uint64_t hi;
};

// Byte-identical to hyle::simd::KeyValue32 / KeyValue64 and detail::KV32Bits / KV64Bits.
struct alignas(8) K32V32 {
  uint32_t value;
  uint32_t key;
};
struct alignas(16) K64V64 {
  uint64_t value;
  uint64_t key;
};

// ---- Sort order tags ------------------------------------------------------------
struct SortAscending {
  static constexpr bool IsAscending() { return true; }
};
struct SortDescending {
  static constexpr bool IsAscending() { return false; }
};

// ---- Memory helpers -------------------------------------------------------------
template <size_t kBytes>
HYLE_FORCE_INLINE void CopyBytes(const void* from, void* to) noexcept {
  std::memcpy(to, from, kBytes);
}
HYLE_FORCE_INLINE void CopyBytes(const void* from, void* to, size_t bytes) noexcept {
  std::memcpy(to, from, bytes);
}

HYLE_FORCE_INLINE void Prefetch(const void* p) noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  _mm_prefetch(static_cast<const char*>(p), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(p, 0, 3);
#else
  (void)p;
#endif
}

// Fill `count` lanes starting at `ptr` with `value` (Highway Fill).
template <class D, typename T>
HYLE_FORCE_INLINE void Fill(D d, T value, size_t count, T* ptr) noexcept {
  const size_t N = Lanes(d);
  const auto v = Set(d, value);
  size_t i = 0;
  for (; i + N <= count; i += N) StoreU(v, d, ptr + i);
  for (; i < count; ++i) ptr[i] = value;
}

// ---- Highway 3-arg Concat* (D, hi, lo) -> HYLE 2-arg (hi, lo) -------------------
template <class D, class V>
HYLE_FORCE_INLINE V ConcatUpperLower(D, V hi, V lo) noexcept {
  return hyle::simd::ConcatUpperLower(hi, lo);
}
template <class D, class V>
HYLE_FORCE_INLINE V ConcatLowerUpper(D, V hi, V lo) noexcept {
  return hyle::simd::ConcatLowerUpper(hi, lo);
}
template <class D, class V>
HYLE_FORCE_INLINE V ConcatLowerLower(D, V hi, V lo) noexcept {
  return hyle::simd::ConcatLowerLower(hi, lo);
}
template <class D, class V>
HYLE_FORCE_INLINE V ConcatUpperUpper(D, V hi, V lo) noexcept {
  return hyle::simd::ConcatUpperUpper(hi, lo);
}

// ---- Entropy for pivot sampling -------------------------------------------------
HYLE_FORCE_INLINE void Fill16BytesStatic(void* bytes) noexcept {
  uint64_t* words = reinterpret_cast<uint64_t*>(bytes);
  uint64_t** seed_stack = &words;
  void (*seed_code)(void*) = &Fill16BytesStatic;
  const uintptr_t bits_stack = reinterpret_cast<uintptr_t>(seed_stack);
  const uintptr_t bits_code = reinterpret_cast<uintptr_t>(seed_code);
  const uint64_t bits_time = static_cast<uint64_t>(clock());
  words[0] = bits_stack ^ bits_time ^ 0xFEDCBA98ull;
  words[1] = bits_code ^ bits_time ^ 0x01234567ull;
}

HYLE_FORCE_INLINE uint64_t* GetGeneratorStateStatic() noexcept {
  thread_local uint64_t state[3] = {0, 0, 0};
  if (HYLE_UNLIKELY(state[2] == 0)) {
    Fill16BytesStatic(state);
    state[2] = 1;
  }
  return state;
}

// ---- Highway base.h bit / cast helpers used by vqsort ---------------------------
HYLE_FORCE_INLINE size_t Num0BitsAboveMS1Bit_Nonzero32(uint32_t x) noexcept {
#if defined(_MSC_VER)
  unsigned long index = 0;
  _BitScanReverse(&index, x);
  return 31u - static_cast<size_t>(index);
#else
  return static_cast<size_t>(__builtin_clz(x));
#endif
}

template <typename TTo, typename TFrom>
HYLE_FORCE_INLINE constexpr TTo ConvertScalarTo(TFrom in) noexcept {
  return static_cast<TTo>(in);
}

template <typename T>
HYLE_FORCE_INLINE void PrintValue(const T&) noexcept {}

// ---- RandomBits / SortConstants (from shared-inl.h target-independent half) -----
HYLE_FORCE_INLINE uint64_t RandomBits(uint64_t* state) noexcept {
  const uint64_t a = state[0];
  const uint64_t b = state[1];
  const uint64_t w = state[2] + 1;
  const uint64_t next = a ^ w;
  state[0] = (b + (b << 3)) ^ (b >> 11);
  const uint64_t rot = (b << 24) | (b >> 40);
  state[1] = rot + next;
  state[2] = w;
  return next;
}

struct SortConstants {
#if defined(_MSC_VER) || !defined(NDEBUG)
  static constexpr size_t kMaxCols = 8;
#else
  static constexpr size_t kMaxCols = 16;
#endif
  static constexpr size_t kMaxRows = 16;

  template <size_t kLPK>
  static constexpr size_t BaseCaseNumLanes(size_t N) {
    return (((N / kLPK) >= 4) ? kMaxRows : 8) * ((N < kMaxCols) ? N : kMaxCols);
  }

  static constexpr size_t kPartitionUnroll = 4;

  static constexpr size_t LanesPerChunk(size_t sizeof_t) { return 64 / sizeof_t; }

  template <typename T>
  static constexpr size_t SampleLanes() {
    return 2 * LanesPerChunk(sizeof(T));
  }

  static constexpr size_t PartitionBufNum(size_t N) {
    return 2 * (3 * kPartitionUnroll + 1) * N;
  }

  template <typename T, size_t kLPK>
  static constexpr size_t BufNum(size_t N) {
    const size_t a = SampleLanes<T>() + BaseCaseNumLanes<kLPK>(N) + N;
    const size_t b = PartitionBufNum(N);
    return a > b ? a : b;
  }

  template <typename T, size_t kLPK>
  static constexpr size_t BufBytes(size_t vector_size) {
    return BufNum<T, kLPK>(vector_size / sizeof(T)) * sizeof(T);
  }

  template <size_t kLPK>
  static constexpr size_t MaxBufBytes(size_t vector_size) {
    return kLPK == 2 ? BufBytes<uint64_t, 2>(vector_size)
                     : (BufBytes<uint16_t, 1>(vector_size) > BufBytes<uint32_t, 1>(vector_size)
                            ? (BufBytes<uint16_t, 1>(vector_size) > BufBytes<uint64_t, 1>(vector_size)
                                   ? BufBytes<uint16_t, 1>(vector_size)
                                   : BufBytes<uint64_t, 1>(vector_size))
                            : (BufBytes<uint32_t, 1>(vector_size) > BufBytes<uint64_t, 1>(vector_size)
                                   ? BufBytes<uint32_t, 1>(vector_size)
                                   : BufBytes<uint64_t, 1>(vector_size)));
  }
};

// SortTag: full native vector (Highway ScalableTag on fixed targets).
template <typename T>
using SortTag = ScalableTag<T>;

// vqsort sometimes packs into a tight buffer region. Public CompressStore is
// allowed to overwrite a full vector's unspecified tail, so this private
// helper narrows the write to the exact active count.
template <class V, class M, class D>
HYLE_FORCE_INLINE size_t SafeCompressStore(V v, M m, D d, TFromD<D>* p) noexcept {
  const size_t count = CountTrue(d, m);
  StoreN(Compress(v, m), d, p, count);
  return count;
}

// Print stubs (VQSORT_PRINT defaults to 0).
template <class D>
HYLE_FORCE_INLINE void MaybePrintVector(D, const char*, Vec<D>, size_t = 0,
                                        size_t = 16) noexcept {}

// IsFull(d): true when d is a full native vector (not a Half/Capped fraction).
template <class D>
constexpr bool IsFull(D d) noexcept {
  return d.MaxLanes() == (kNativeVectorBytes / sizeof(TFromD<D>)) && d.Pow2() == 0;
}

// Highway NaN(d) is defined for any lane type (bit-cast of all-ones). HYLE's public
// NaN is float-only; sort only *executes* it for floats (num_nan==0 for ints) but the
// call site must still compile for integer T.
template <class D>
HYLE_FORCE_INLINE VFromD<D> SortPoisonNaN(D d) noexcept {
  if constexpr (std::is_floating_point_v<TFromD<D>>) {
    return NaN(d);
  } else {
    using TS = MakeSigned<TFromD<D>>;
    const RebindToSigned<D> di;
    return BitCast(d, Set(di, LimitsMax<TS>()));
  }
}

// ---- SFINAE enable macros (Highway HWY_IF_*) ------------------------------------
// Enable when sizeof(TFromD<D>) == bytes.
#define HYLE_VQSORT_IF_T_SIZE_D(D, bytes) \
  typename std::enable_if_t<sizeof(TFromD<D>) == (bytes), int> = 0
#define HYLE_VQSORT_IF_NOT_T_SIZE_D(D, bytes) \
  typename std::enable_if_t<sizeof(TFromD<D>) != (bytes), int> = 0
#define HYLE_VQSORT_IF_U64_D(D) \
  typename std::enable_if_t<std::is_same_v<TFromD<D>, std::uint64_t>, int> = 0
#define HYLE_VQSORT_IF_FLOAT_OR_SPECIAL_D(D) \
  typename std::enable_if_t<std::is_floating_point_v<TFromD<D>>, int> = 0
#define HYLE_VQSORT_IF_NOT_FLOAT_NOR_SPECIAL_D(D) \
  typename std::enable_if_t<!std::is_floating_point_v<TFromD<D>>, int> = 0
#define HYLE_VQSORT_IF_FLOAT(T) \
  typename std::enable_if_t<std::is_floating_point_v<T>, int> = 0
#define HYLE_VQSORT_IF_NOT_FLOAT(T) \
  typename std::enable_if_t<!std::is_floating_point_v<T>, int> = 0
// Lane-count SFINAE for sorting networks (uses compile-time MaxLanes).
#define HYLE_VQSORT_IF_LANES_LE(N, limit) \
  typename std::enable_if_t<(N) <= (limit), int> = 0
#define HYLE_VQSORT_IF_LANES_GT(N, limit) \
  typename std::enable_if_t<((N) > (limit)), int> = 0
// Vector byte size SFINAE (V is a vector type with kPrivateN * sizeof(PrivateT)).
#define HYLE_VQSORT_IF_V_SIZE_V(V, bytes)                                     \
  typename std::enable_if_t<(sizeof(typename V::PrivateT) * V::kPrivateN) ==  \
                                (bytes),                                      \
                            int> = 0

// Map Highway macro names used inside the ported bodies to HYLE equivalents.
// These are local to the vqsort port TUs (not public API).
#ifndef HWY_INLINE
#define HWY_INLINE HYLE_FORCE_INLINE
#endif
#ifndef HWY_API
#define HWY_API HYLE_FORCE_INLINE
#endif
#ifndef HWY_NOINLINE
#if defined(_MSC_VER)
#define HWY_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define HWY_NOINLINE __attribute__((noinline))
#else
#define HWY_NOINLINE
#endif
#endif
#ifndef HWY_RESTRICT
#define HWY_RESTRICT
#endif
#ifndef HWY_DASSERT
#define HWY_DASSERT(cond) ((void)0)
#endif
#ifndef HWY_DASSERT_M
#define HWY_DASSERT_M(cond, msg) ((void)0)
#endif
#ifndef HWY_ASSERT
#define HWY_ASSERT(cond) HYLE_FATAL_ASSERT(cond)
#endif
#ifndef HWY_ASSERT_M
#define HWY_ASSERT_M(cond, msg) HYLE_FATAL_ASSERT(cond)
#endif
#ifndef HWY_LIKELY
#define HWY_LIKELY HYLE_LIKELY
#endif
#ifndef HWY_UNLIKELY
#define HWY_UNLIKELY HYLE_UNLIKELY
#endif
#ifndef HWY_MIN
#define HWY_MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef HWY_MAX
#define HWY_MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef HWY_ALIGN
#define HWY_ALIGN alignas(64)
#endif
#ifndef HWY_MAYBE_UNUSED
#if defined(__GNUC__) || defined(__clang__)
#define HWY_MAYBE_UNUSED __attribute__((unused))
#else
#define HWY_MAYBE_UNUSED
#endif
#endif
#ifndef HWY_ASSUME
#if defined(_MSC_VER)
#define HWY_ASSUME(cond) __assume(cond)
#elif defined(__clang__)
#define HWY_ASSUME(cond) __builtin_assume(!!(cond))
#elif defined(__GNUC__)
#define HWY_ASSUME(cond)          \
  do {                            \
    if (!(cond)) __builtin_unreachable(); \
  } while (0)
#else
#define HWY_ASSUME(cond) ((void)0)
#endif
#endif
#ifndef HWY_UNROLL
#define HWY_UNROLL(n)
#endif
#ifndef HWY_DEFAULT_UNROLL
#define HWY_DEFAULT_UNROLL
#endif
#ifndef HWY_IF_CONSTEXPR
#define HWY_IF_CONSTEXPR(...) if constexpr (__VA_ARGS__)
#endif
#ifndef HWY_WARN
#define HWY_WARN(...) ((void)0)
#endif
#ifndef HWY_DIAGNOSTICS
#define HWY_DIAGNOSTICS(x)
#endif
#ifndef HWY_DIAGNOSTICS_OFF
#define HWY_DIAGNOSTICS_OFF(x, y)
#endif
#ifndef HWY_CONCAT
#define HWY_CONCAT2(a, b) a##b
#define HWY_CONCAT(a, b) HWY_CONCAT2(a, b)
#endif

// Target IDs (Highway bit values; lower = wider). Used only by ported #if paths.
#ifndef HWY_AVX3
#define HWY_AVX3 (1LL << 8)
#define HWY_AVX2 (1LL << 9)
#define HWY_SSSE3 (1LL << 12)
#define HWY_NEON (1LL << 28)
#define HWY_SCALAR (1LL << 62)
#define HWY_SVE_256 (1LL << 22)
#define HWY_ALL_SVE 0
#endif

#if defined(HYLE_SIMD_TARGET_AVX512)
#define HWY_TARGET HWY_AVX3
#define HWY_MAX_BYTES 64
#elif defined(HYLE_SIMD_TARGET_AVX2)
#define HWY_TARGET HWY_AVX2
#define HWY_MAX_BYTES 32
#elif defined(HYLE_SIMD_TARGET_NEON)
#define HWY_TARGET HWY_NEON
#define HWY_MAX_BYTES 16
#else
#define HWY_TARGET HWY_SCALAR
#define HWY_MAX_BYTES 16
#endif

// Prefer fault-safe buf path: HYLE LoadU always touches full native width, so the
// Highway MaskedLoadOr path is riskier for partial rows; buf+SafeCopyN is correct.
#ifndef HWY_MEM_OPS_MIGHT_FAULT
#define HWY_MEM_OPS_MIGHT_FAULT 1
#endif
#ifndef HWY_HAVE_FLOAT64
#define HWY_HAVE_FLOAT64 1
#endif
#ifndef HWY_HAVE_FLOAT16
#define HWY_HAVE_FLOAT16 0
#endif
#ifndef HWY_IS_DEBUG_BUILD
#ifdef NDEBUG
#define HWY_IS_DEBUG_BUILD 0
#else
#define HWY_IS_DEBUG_BUILD 1
#endif
#endif
#if defined(_MSC_VER)
#define HWY_COMPILER_MSVC 1
#else
#define HWY_COMPILER_MSVC 0
#endif
#if defined(__clang__)
#define HWY_COMPILER_CLANG (__clang_major__ * 100 + __clang_minor__)
#else
#define HWY_COMPILER_CLANG 0
#endif
#if defined(__GNUC__) && !defined(__clang__)
#define HWY_COMPILER_GCC_ACTUAL __GNUC__
#else
#define HWY_COMPILER_GCC_ACTUAL 0
#endif
#if defined(__arm__) && !defined(__aarch64__)
#define HWY_ARCH_ARM_V7 1
#else
#define HWY_ARCH_ARM_V7 0
#endif

// SFINAE macro renames used in ported sources (keep HWY_IF_* spelling for minimal diff).
#ifndef HWY_IF_T_SIZE_D
#define HWY_IF_T_SIZE_D(D, bytes) HYLE_VQSORT_IF_T_SIZE_D(D, bytes)
#define HWY_IF_NOT_T_SIZE_D(D, bytes) HYLE_VQSORT_IF_NOT_T_SIZE_D(D, bytes)
#define HWY_IF_U64_D(D) HYLE_VQSORT_IF_U64_D(D)
#define HWY_IF_FLOAT_OR_SPECIAL_D(D) HYLE_VQSORT_IF_FLOAT_OR_SPECIAL_D(D)
#define HWY_IF_NOT_FLOAT_NOR_SPECIAL_D(D) HYLE_VQSORT_IF_NOT_FLOAT_NOR_SPECIAL_D(D)
#define HWY_IF_FLOAT(T) HYLE_VQSORT_IF_FLOAT(T)
#define HWY_IF_NOT_FLOAT(T) HYLE_VQSORT_IF_NOT_FLOAT(T)
#define HWY_IF_LANES_LE(N, limit) HYLE_VQSORT_IF_LANES_LE(N, limit)
#define HWY_IF_LANES_GT(N, limit) HYLE_VQSORT_IF_LANES_GT(N, limit)
#define HWY_IF_V_SIZE_V(V, bytes) HYLE_VQSORT_IF_V_SIZE_V(V, bytes)
#endif

#ifndef VQSORT_PRINT
#define VQSORT_PRINT 0
#endif

// float16 placeholder so KeyString switch compiles (not a supported key type).
struct float16_t {};

}  // namespace vqsort
}  // namespace detail
}  // namespace hyle::simd

// Bring HYLE ops + vqsort helpers into an unqualified lookup surface for the
// ported algorithm headers (included after this file from the same namespace).
using namespace hyle::simd;
using namespace hyle::simd::detail::vqsort;
