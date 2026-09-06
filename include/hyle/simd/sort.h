#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include "hyle/simd/types.h"
#include "hyle/simd/visibility.h"

// In-place whole-buffer sort / partial_sort / select. Accelerated target TUs
// use the licensed, bounded HYLE adaptation of Highway VQSort; HYLE retains
// pointer/count, order, floating-point, public-POD and dispatch contracts. No
// vendor type crosses this public boundary.
//
// TIERS (sort_dispatch.cc, same shape as math_dispatch): x86 avx512 -> avx2 ->
// scalar; arm64 SVE capped-network -> NEON; riscv64 RVV -> scalar. The scalar
// floor is std::sort / std::partial_sort / std::nth_element -- ALWAYS compiled,
// the correctness oracle + the non-x86 / old-CPU fallback (never a SIGILL).
// VQSort is NOT stable (the order of equal keys is unspecified), matching
// std::sort/ std::nth_element. Use radix_sort or argsort below when stability
// is required.
//
// Scalar keys are u16/i16/u32/i32/u64/i64/f32/f64. UInt128 is an unsigned
// 128-bit POD key; KeyValue32/64 provide in-place top-k payload pairs. All
// public data surfaces are pointers, counts, and fixed-layout PODs (no
// standard-library container crosses the library boundary).
//
// SEMANTICS (units of KEYS, mirroring the vqsort static entry points):
//   sort(keys, n, order)            : full sort of keys[0..n).
//   partial_sort(keys, n, k, order) : keys[0..k) become the sorted smallest
//   (asc) / largest
//                                     (desc) k; keys[k..n) unspecified. k <= n
//                                     (contract).
//   select(keys, n, k, order)       : keys[k] lands on the element it would
//   have if fully
//                                     sorted (nth_element); keys[0..k) all
//                                     order-<= keys[k], keys[k+1..n) all
//                                     order->=. k < n (contract).
// For f32/f64, NaNs are placed after all finite values in both directions; NaNs
// are equivalent to each other and their relative order is unspecified. A
// violated size contract (k > n for partial_sort, k >= n for select) is a
// caller bug and aborts via HYLE_FATAL_ASSERT (fail-fast, not a silent clamp)
// -- except the degenerate n==0.

namespace hyle::simd {

// Sort direction. Ascending = smallest first; Descending = largest first.
enum class SortOrder : std::uint8_t { kAscending, kDescending };

// Non-destructive score selection for vocabulary and routing rows.  The caller
// supplies k result slots; those slots are also the only O(k) heap workspace.
// Results are best-first (descending score), NaNs follow every non-NaN, and
// exact score ties -- including signed zero -- use the smaller original index
// first.  `scores` is unchanged.  k must be <= n and a non-empty input must fit
// uint32_t indices; scores and result must not overlap.  k == 0 accepts a null
// result and does not read scores. Complexity is O(n log k + k log k); k == 1
// and the consumer-derived bounded MoE range k <= 8 use an O(n) path.
struct ScoredIndexF32 {
  float score;
  std::uint32_t index;
};

static_assert(sizeof(ScoredIndexF32) == 8 && alignof(ScoredIndexF32) == 4);
static_assert(std::is_standard_layout_v<ScoredIndexF32>);
static_assert(std::is_trivially_copyable_v<ScoredIndexF32>);

HYLE_SIMD_API void top_k_f32(const float* scores, std::size_t n,
                             ScoredIndexF32* result, std::size_t k) noexcept;

// --- full sort (in place) ---
HYLE_SIMD_API void sort(std::uint16_t* keys, std::size_t n,
                        SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void sort(std::int16_t* keys, std::size_t n,
                        SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void sort(std::uint32_t* keys, std::size_t n,
                        SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void sort(std::int32_t* keys, std::size_t n,
                        SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void sort(std::uint64_t* keys, std::size_t n,
                        SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void sort(std::int64_t* keys, std::size_t n,
                        SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void sort(float* keys, std::size_t n,
                        SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void sort(double* keys, std::size_t n,
                        SortOrder order = SortOrder::kAscending) noexcept;

// --- partial sort: sorted smallest/largest k in [0,k) (in place) ---
HYLE_SIMD_API void partial_sort(
    std::uint16_t* keys, std::size_t n, std::size_t k,
    SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void partial_sort(
    std::int16_t* keys, std::size_t n, std::size_t k,
    SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void partial_sort(
    std::uint32_t* keys, std::size_t n, std::size_t k,
    SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void partial_sort(
    std::int32_t* keys, std::size_t n, std::size_t k,
    SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void partial_sort(
    std::uint64_t* keys, std::size_t n, std::size_t k,
    SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void partial_sort(
    std::int64_t* keys, std::size_t n, std::size_t k,
    SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void partial_sort(
    float* keys, std::size_t n, std::size_t k,
    SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void partial_sort(
    double* keys, std::size_t n, std::size_t k,
    SortOrder order = SortOrder::kAscending) noexcept;

// --- select (nth_element): keys[k] placed, partitioned around it (in place)
// ---
HYLE_SIMD_API void select(std::uint16_t* keys, std::size_t n, std::size_t k,
                          SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void select(std::int16_t* keys, std::size_t n, std::size_t k,
                          SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void select(std::uint32_t* keys, std::size_t n, std::size_t k,
                          SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void select(std::int32_t* keys, std::size_t n, std::size_t k,
                          SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void select(std::uint64_t* keys, std::size_t n, std::size_t k,
                          SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void select(std::int64_t* keys, std::size_t n, std::size_t k,
                          SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void select(float* keys, std::size_t n, std::size_t k,
                          SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void select(double* keys, std::size_t n, std::size_t k,
                          SortOrder order = SortOrder::kAscending) noexcept;

// --- unsigned 128-bit keys
// --------------------------------------------------------------- Numeric order
// is lexicographic by (hi, lo). The explicit POD keeps the ABI independent of
// compiler-specific __int128 support.
inline constexpr bool operator==(UInt128 a, UInt128 b) noexcept {
  return a.lo == b.lo && a.hi == b.hi;
}
inline constexpr bool operator<(UInt128 a, UInt128 b) noexcept {
  return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}
inline constexpr bool operator>(UInt128 a, UInt128 b) noexcept { return b < a; }
inline constexpr bool operator<=(UInt128 a, UInt128 b) noexcept {
  return !(b < a);
}
inline constexpr bool operator>=(UInt128 a, UInt128 b) noexcept {
  return !(a < b);
}

HYLE_SIMD_API void sort(UInt128* keys, std::size_t n,
                        SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void partial_sort(
    UInt128* keys, std::size_t n, std::size_t k,
    SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void select(UInt128* keys, std::size_t n, std::size_t k,
                          SortOrder order = SortOrder::kAscending) noexcept;

// --- key-value pairs (top-k): sort by KEY, VALUE rides along
// ------------------------------ The top-k / arg-sort primitive: pair each
// score with its index (or any payload), sort by the key, read the values out
// in key order. vqsort terminates recursion earlier for KV than for a full
// 128-bit key (only the key half is compared), so this is the fast path for
// "sort scores, keep track of which lane each came from" -- attention/logits
// top-k sampling, MoE routing, sparse index building. Layout is {value, key};
// construct with the designated fields and do not use this integer-key shape
// for floating logits. Sorts by `key` only; equal keys' value order is
// unspecified (vqsort is not stable), same as the scalar-T ops.

HYLE_SIMD_API void sort(KeyValue32* kv, std::size_t n,
                        SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void sort(KeyValue64* kv, std::size_t n,
                        SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void partial_sort(
    KeyValue32* kv, std::size_t n, std::size_t k,
    SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void partial_sort(
    KeyValue64* kv, std::size_t n, std::size_t k,
    SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void select(KeyValue32* kv, std::size_t n, std::size_t k,
                          SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void select(KeyValue64* kv, std::size_t n, std::size_t k,
                          SortOrder order = SortOrder::kAscending) noexcept;

// --- Stable LSD radix sort + argsort
// --------------------------------------------------------- Base-256, O(n *
// sizeof(key)) time and O(n) caller-reusable scratch. Equal integer keys, -0/+0
// and all NaNs retain input order. Floating-point finite values use numeric
// order and NaNs are stable-last in BOTH directions, matching the ordinary sort
// family above.
//
// The std::span overloads never allocate and require scratch.size() >= n with
// no overlap between scratch and the live input/output ranges. The convenience
// overloads allocate exactly n scratch elements and therefore may throw
// std::bad_alloc (including array-length failure). This is intentional: a
// stable algorithm never silently falls back to an unstable one on OOM.
// Repeated/hot callers should own one scratch buffer and use the span overload.
//
// argsort writes 32-bit indices as a stable permutation. Therefore n==0 or n-1
// must fit uint32_t; violating that size contract fails fast. keys is borrowed
// and unchanged; indices is output-only.

inline constexpr std::size_t radix_sort_workspace_size(std::size_t n) noexcept {
  return n;
}
inline constexpr std::size_t argsort_workspace_size(std::size_t n) noexcept {
  return n;
}

#define HYLE_SIMD_DECLARE_RADIX(T)                       \
  HYLE_SIMD_API void radix_sort(                         \
      T* keys, std::size_t n, std::span<T> scratch,      \
      SortOrder order = SortOrder::kAscending) noexcept; \
  HYLE_SIMD_API void radix_sort(T* keys, std::size_t n,  \
                                SortOrder order = SortOrder::kAscending);

HYLE_SIMD_DECLARE_RADIX(std::uint16_t)
HYLE_SIMD_DECLARE_RADIX(std::int16_t)
HYLE_SIMD_DECLARE_RADIX(std::uint32_t)
HYLE_SIMD_DECLARE_RADIX(std::int32_t)
HYLE_SIMD_DECLARE_RADIX(std::uint64_t)
HYLE_SIMD_DECLARE_RADIX(std::int64_t)
HYLE_SIMD_DECLARE_RADIX(float)
HYLE_SIMD_DECLARE_RADIX(double)

#undef HYLE_SIMD_DECLARE_RADIX

#define HYLE_SIMD_DECLARE_ARGSORT(T)                                \
  HYLE_SIMD_API void argsort(                                       \
      const T* keys, std::uint32_t* indices, std::size_t n,         \
      std::span<std::uint32_t> scratch,                             \
      SortOrder order = SortOrder::kAscending) noexcept;            \
  HYLE_SIMD_API void argsort(const T* keys, std::uint32_t* indices, \
                             std::size_t n,                         \
                             SortOrder order = SortOrder::kAscending);

HYLE_SIMD_DECLARE_ARGSORT(std::uint16_t)
HYLE_SIMD_DECLARE_ARGSORT(std::int16_t)
HYLE_SIMD_DECLARE_ARGSORT(std::uint32_t)
HYLE_SIMD_DECLARE_ARGSORT(std::int32_t)
HYLE_SIMD_DECLARE_ARGSORT(std::uint64_t)
HYLE_SIMD_DECLARE_ARGSORT(std::int64_t)
HYLE_SIMD_DECLARE_ARGSORT(float)
HYLE_SIMD_DECLARE_ARGSORT(double)

#undef HYLE_SIMD_DECLARE_ARGSORT

}  // namespace hyle::simd
