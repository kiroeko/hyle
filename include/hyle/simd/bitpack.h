#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/simd/dispatch.h"  // Isa, HYLE_SIMD_API

// bitpack.h -- vectorized fixed-width integer bit packing / unpacking (simd-20 X1). The
// accelerated implementation is HYLE D-tag SIMD: Pack8<1..8>, Pack16<1..16>, and Pack32<1..32>
// compile once for each target ISA and are selected by the baseline dispatcher.
//
// TIERS (bitpack_dispatch.cc): x86 avx512 -> avx2 -> scalar; arm64 NEON; else scalar. The scalar
// floor is a pure contiguous bit-stream codec -- ALWAYS compiled, the non-x86 / old-CPU fallback
// (never a SIGILL). The wide kernels use integer shift/or/and SIMD, and the AVX-512 target is
// compiled on all supported compilers.
//
// This public header declares one overload per lane type: uint8/uint16/uint32.
//
// SEMANTICS: pack takes `count` raw values, each carrying `bits` significant low bits, and writes
// them densely into `packed`. unpack reverses it. Input and output ranges must not overlap.
// The valid range of `bits` is [1, 8*sizeof(T)] (1..8 for uint8, 1..16 for uint16, 1..32 for
// uint32). A `bits` outside that range is a caller contract violation and aborts via
// HYLE_FATAL_ASSERT (fail-fast, not a silent clamp). count==0 is a no-op.
//
// LAYOUT / ROUND-TRIP CONTRACT: the packed byte layout is TIER-DEPENDENT (full blocks are
// interleaved by the live vector width; the sub-block tail uses a scalar bit-stream). The only
// guaranteed invariant is round-trip: unpack(pack(x)) == x when both run on the SAME host (hence
// the same tier). Do NOT pack on one machine and unpack on another with a different ISA tier, and
// do NOT interpret the packed bytes as a portable wire format -- it is an in-memory codec. Use
// packed_size<T>(count, bits) to size the `packed` buffer.
//
// u16 widths 11 and 13 use corrected HYLE SIMD block layouts, including the 13-bit rawF scatter
// mask and inverse vector selection; independent layout-oracle and full-domain tests pin them on
// AVX2/AVX512/NEON.

namespace hyle::simd {

// Number of T-elements pack() writes for `count` values at `bits` bits each. This is tier-
// INDEPENDENT: the block region emits exactly block_vals*bits/(8*sizeof(T)) elements (block_vals
// is a multiple of 8*sizeof(T)) and the tail rounds up to whole elements, so the total is always
// ceil(count*bits / (8*sizeof(T))). A template (not a flat overload) because the element count
// depends on sizeof(T) and there is no runtime argument to select it on. constexpr + header-only,
// so it stays vendor-free.
template <typename T>
constexpr std::size_t packed_size(std::size_t count, unsigned bits) noexcept {
  constexpr std::size_t W = 8 * sizeof(T);
  return (count * bits + (W - 1)) / W;
}

// --- pack: count raw values (each `bits` significant) -> dense `packed` ---
HYLE_SIMD_API void pack(const std::uint8_t* raw, std::size_t count, std::uint8_t* packed,
                        unsigned bits) noexcept;  // bits 1..8
HYLE_SIMD_API void pack(const std::uint16_t* raw, std::size_t count, std::uint16_t* packed,
                        unsigned bits) noexcept;  // bits 1..16
HYLE_SIMD_API void pack(const std::uint32_t* raw, std::size_t count, std::uint32_t* packed,
                        unsigned bits) noexcept;  // bits 1..32

// --- unpack: dense `packed` -> count raw values (each `bits` significant) ---
HYLE_SIMD_API void unpack(const std::uint8_t* packed, std::size_t count, std::uint8_t* raw,
                          unsigned bits) noexcept;  // bits 1..8
HYLE_SIMD_API void unpack(const std::uint16_t* packed, std::size_t count, std::uint16_t* raw,
                          unsigned bits) noexcept;  // bits 1..16
HYLE_SIMD_API void unpack(const std::uint32_t* packed, std::size_t count, std::uint32_t* raw,
                          unsigned bits) noexcept;  // bits 1..32

}  // namespace hyle::simd
