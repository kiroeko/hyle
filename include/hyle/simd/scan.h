#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/simd/dispatch.h"  // Isa, HYLE_SIMD_API

// scan.h -- inclusive / exclusive prefix sum over contiguous arrays (P1-1).
// Time complexity: scalar chain is O(n) serial dependence (~1 elem/cycle); this path is
// O(n/w · log w) with w = native SIMD width (vector Kogge-Stone + carry across vectors).
// Highway/xsimd do not ship a public scan — this is a capability HYLE owns.
//
// SEMANTICS (in place):
//   inclusive_scan(a, n): a[i] <- sum_{j=0..i} a0[j]   (a0 = values before the call)
//   exclusive_scan(a, n): a[i] <- sum_{j=0..i-1} a0[j]  (a[0] = 0; identity of +)
// Integer types use wrapping arithmetic (same as plain T addition). Floating types use
// the backend's Add (non-associative; result is deterministic for a fixed ISA width).
// n == 0 is a no-op. Pointers must be non-null when n > 0 (caller contract).

namespace hyle::simd {

HYLE_SIMD_API void inclusive_scan(std::uint32_t* data, std::size_t n) noexcept;
HYLE_SIMD_API void inclusive_scan(std::int32_t* data, std::size_t n) noexcept;
HYLE_SIMD_API void inclusive_scan(std::uint64_t* data, std::size_t n) noexcept;
HYLE_SIMD_API void inclusive_scan(std::int64_t* data, std::size_t n) noexcept;
HYLE_SIMD_API void inclusive_scan(float* data, std::size_t n) noexcept;
HYLE_SIMD_API void inclusive_scan(double* data, std::size_t n) noexcept;

HYLE_SIMD_API void exclusive_scan(std::uint32_t* data, std::size_t n) noexcept;
HYLE_SIMD_API void exclusive_scan(std::int32_t* data, std::size_t n) noexcept;
HYLE_SIMD_API void exclusive_scan(std::uint64_t* data, std::size_t n) noexcept;
HYLE_SIMD_API void exclusive_scan(std::int64_t* data, std::size_t n) noexcept;
HYLE_SIMD_API void exclusive_scan(float* data, std::size_t n) noexcept;
HYLE_SIMD_API void exclusive_scan(double* data, std::size_t n) noexcept;

// ISA tier selected for scan kernels (same model as ops active_variant).
HYLE_SIMD_API Isa active_scan_variant() noexcept;

}  // namespace hyle::simd
