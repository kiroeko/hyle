#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/simd/dispatch.h"  // Isa, HYLE_SIMD_API

// sort.h -- in-place vectorized sort / partial_sort / select (simd-09 line②, "算法级能力").
// This is Highway's flagship contrib -- HYLE has no hand-rolled counterpart, so (charter:
// 借核不手搓) the accelerated path BORROWS Highway's verified vqsort core (VQSortStatic /
// VQPartialSortStatic / VQSelectStatic in hwy/contrib/sort/vqsort-inl.h). It is a thin wrapper,
// STRUCTURALLY IDENTICAL to the math borrow (transcendental_highway.cc): the vendor headers are
// static single-target (no dynamic dispatch), compiled per ISA tier at their own -march, and
// entered through a baseline dispatcher (sort_dispatch.cc) gated on cpu_features().
//
// TIERS (sort_dispatch.cc, same shape as math_dispatch): x86 avx512 -> avx2 -> scalar; arm64
// NEON; else scalar. The scalar floor is std::sort / std::partial_sort / std::nth_element --
// ALWAYS compiled, the correctness oracle + the non-x86 / old-CPU fallback (never a SIGILL).
// vqsort is NOT a stable sort (order of equal keys is unspecified), matching std::sort/
// std::nth_element (use a key-index pair if stability is needed).
//
// This header does NOT include Highway. Scalar keys are u32/i32/u64/i64/f32/f64;
// KeyValue32/64 provide zero-copy top-k payload pairs. u16/i16 and uint128 remain outside the
// public surface.
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

// Sort direction. Ascending = smallest first; Descending = largest first. Maps to Highway's
// SortAscending{} / SortDescending{} order tags inside the borrow TUs.
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
// sparse index building. Layout is {value, key} (little-endian, value first) to match Highway's
// K32V32/K64V64 byte-for-byte, so the borrow is a zero-copy reinterpret_cast; construct with the
// designated fields, don't assume field order. Sorts by `key` only; equal keys' value order is
// unspecified (vqsort is not stable), same as the scalar-T ops.
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

// The tier the sort ops selected: kAvx512 if the 512-bit borrow core is live, kAvx2 if the
// 256-bit one is, kNeon on arm64, else kScalar (std::sort floor). For tests/diagnostics.
HYLE_SIMD_API Isa active_sort_variant() noexcept;

}  // namespace hyle::simd
