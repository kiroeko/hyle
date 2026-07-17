#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/simd/dispatch.h"  // Isa, HYLE_SIMD_API

// sort.h -- in-place vectorized sort / partial_sort / select (simd-09 line②, "算法级能力").
// The accelerated path is a HYLE port of Highway's verified vqsort (VQSortStatic /
// VQPartialSortStatic / VQSelectStatic; origin hwy/contrib/sort, SHA 83eff393…). The port lives
// under src/simd/sort/vqsort_hyle/ and uses only hyle::simd D-tag primitives. Per-ISA TUs are
// compiled at their own -march and entered through a baseline dispatcher (sort_dispatch.cc)
// gated on cpu_features().
//
// TIERS (sort_dispatch.cc, same shape as math_dispatch): x86 avx512 -> avx2 -> scalar; arm64
// SVE-256-capped -> NEON; riscv64 RVV -> scalar. The scalar floor is std::sort /
// std::partial_sort / std::nth_element --
// ALWAYS compiled, the correctness oracle + the non-x86 / old-CPU fallback (never a SIGILL).
// vqsort is NOT a stable sort (order of equal keys is unspecified), matching std::sort/
// std::nth_element (use a key-index pair if stability is needed).
//
// Scalar keys are u32/i32/u64/i64/f32/f64; KeyValue32/64 provide zero-copy top-k payload pairs.
// u16/i16 and uint128 remain outside the public surface.
//
// SEMANTICS (units of KEYS, mirroring the vqsort static entry points):
//   sort(keys, n, order)            : full sort of keys[0..n).
//   partial_sort(keys, n, k, order) : keys[0..k) become the sorted smallest (asc) / largest
//                                     (desc) k; keys[k..n) unspecified. k <= n (contract).
//   select(keys, n, k, order)       : keys[k] lands on the element it would have if fully
//                                     sorted (nth_element); keys[0..k) all order-<= keys[k],
//                                     keys[k+1..n) all order->=. k < n (contract).
// For f32/f64, NaNs are placed after all finite values in both directions; NaNs are equivalent
// to each other and their relative order is unspecified. A violated size contract (k > n for
// partial_sort, k >= n for select) is a caller bug and aborts via HYLE_FATAL_ASSERT (fail-fast,
// not a silent clamp) -- except the degenerate n==0.

namespace hyle::simd {

// Sort direction. Ascending = smallest first; Descending = largest first. Maps to the port's
// SortAscending{} / SortDescending{} order tags inside the ISA TUs.
enum class SortOrder : std::uint8_t { kAscending, kDescending };

// --- full sort (in place) ---
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
HYLE_SIMD_API void partial_sort(std::uint32_t* keys, std::size_t n, std::size_t k,
                                SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void partial_sort(std::int32_t* keys, std::size_t n, std::size_t k,
                                SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void partial_sort(std::uint64_t* keys, std::size_t n, std::size_t k,
                                SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void partial_sort(std::int64_t* keys, std::size_t n, std::size_t k,
                                SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void partial_sort(float* keys, std::size_t n, std::size_t k,
                                SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void partial_sort(double* keys, std::size_t n, std::size_t k,
                                SortOrder order = SortOrder::kAscending) noexcept;

// --- select (nth_element): keys[k] placed, partitioned around it (in place) ---
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

// --- key-value pairs (top-k): sort by KEY, VALUE rides along ------------------------------
// The top-k / arg-sort primitive: pair each score with its index (or any payload), sort by the
// key, read the values out in key order. vqsort terminates recursion earlier for KV than for a
// full 128-bit key (only the key half is compared), so this is the fast path for "sort scores,
// keep track of which lane each came from" -- attention/logits top-k sampling, MoE routing,
// sparse index building. Layout is {value, key} (little-endian, value first), byte-identical to
// the port's K32V32/K64V64 for zero-copy reinterpret_cast; construct with the designated fields,
// don't assume field order. Sorts by `key` only; equal keys' value order is unspecified (vqsort
// is not stable), same as the scalar-T ops.
struct alignas(8) KeyValue32 {
  std::uint32_t value;  // payload (e.g. original index)
  std::uint32_t key;    // sort key (e.g. score)
};
struct alignas(16) KeyValue64 {
  std::uint64_t value;
  std::uint64_t key;
};

HYLE_SIMD_API void sort(KeyValue32* kv, std::size_t n, SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void sort(KeyValue64* kv, std::size_t n, SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void partial_sort(KeyValue32* kv, std::size_t n, std::size_t k,
                                SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void partial_sort(KeyValue64* kv, std::size_t n, std::size_t k,
                                SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void select(KeyValue32* kv, std::size_t n, std::size_t k,
                          SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void select(KeyValue64* kv, std::size_t n, std::size_t k,
                          SortOrder order = SortOrder::kAscending) noexcept;

// --- P1-2: LSD radix sort (stable, O(n · passes)) + argsort ---------------------------------
// Integer / float keys via LSD base-256 (8-bit digits). STABLE (equal keys keep relative order),
// unlike vqsort. Needs O(n) temporary storage (malloc); on OOM falls back to sort().
// u32 = 4 passes, u64 = 8, f32 = bit-flip then 4. Intended for large n (auto threshold in
// sort() for integer keys is kRadixSortMinN; explicit radix_sort always uses radix).
//
// argsort: writes indices[0..n) as a permutation of 0..n-1 such that
//   keys[indices[i]] is sorted by `order` (stable). indices is output-only.

inline constexpr std::size_t kRadixSortMinN = 4096;  // sort() auto-picks radix above this

HYLE_SIMD_API void radix_sort(std::uint32_t* keys, std::size_t n,
                              SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void radix_sort(std::int32_t* keys, std::size_t n,
                              SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void radix_sort(std::uint64_t* keys, std::size_t n,
                              SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void radix_sort(std::int64_t* keys, std::size_t n,
                              SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void radix_sort(float* keys, std::size_t n,
                              SortOrder order = SortOrder::kAscending) noexcept;

HYLE_SIMD_API void argsort(const std::uint32_t* keys, std::uint32_t* indices, std::size_t n,
                           SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void argsort(const std::int32_t* keys, std::uint32_t* indices, std::size_t n,
                           SortOrder order = SortOrder::kAscending) noexcept;
HYLE_SIMD_API void argsort(const float* keys, std::uint32_t* indices, std::size_t n,
                           SortOrder order = SortOrder::kAscending) noexcept;

}  // namespace hyle::simd
