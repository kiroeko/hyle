#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/simd/dispatch.h"  // Isa, HYLE_SIMD_API

// bitpack.h -- vectorized fixed-width integer bit packing / unpacking (simd-09 line②, "算法级
// 能力"). Like sort.h, this is a Highway flagship contrib with no hand-rolled HYLE counterpart, so
// (charter: 借核不手搓) the accelerated path BORROWS Highway's verified bit_pack cores
// (hwy::Pack8<k> / Pack16<k> / Pack32<k> in hwy/contrib/bit_pack/bit_pack-inl.h). It is a thin
// wrapper, STRUCTURALLY IDENTICAL to the vqsort borrow (sort.h): the vendor headers are static
// single-target (no dynamic dispatch), compiled per ISA tier at their own -march, and entered
// through a baseline dispatcher (bitpack_dispatch.cc) gated on cpu_features().
//
// TIERS (bitpack_dispatch.cc, same shape as sort_dispatch): x86 avx512 -> avx2 -> scalar; arm64
// NEON; else scalar. The scalar floor is a pure contiguous bit-stream codec -- ALWAYS compiled,
// the non-x86 / old-CPU fallback (never a SIGILL). bit_pack's math is integer shift/or/and (NOT
// FP transcendentals), so its AVX-512 target is NOT MSVC-broken -- the 512 tier is compiled on
// all compilers (no _MSC_VER exclusion, same as vqsort).
//
// This header does NOT include Highway (a public header stays vendor-free, like sort.h only
// declaring sort). It declares one overload per lane type: uint8/uint16/uint32.
//
// SEMANTICS: pack takes `count` raw values, each carrying `bits` significant low bits, and writes
// them densely into `packed`. unpack reverses it. The valid range of `bits` is [1, 8*sizeof(T)]
// (1..8 for uint8, 1..16 for uint16, 1..32 for uint32). A `bits` outside that range is a caller
// contract violation and aborts via HYLE_FATAL_ASSERT (fail-fast, not a silent clamp). count==0
// is a no-op.
//
// LAYOUT / ROUND-TRIP CONTRACT: the packed byte layout is TIER-DEPENDENT (Highway interleaves the
// block region by the live vector width; the sub-block tail uses a scalar bit-stream). The only
// guaranteed invariant is round-trip: unpack(pack(x)) == x when both run on the SAME host (hence
// the same tier). Do NOT pack on one machine and unpack on another with a different ISA tier, and
// do NOT interpret the packed bytes as a portable wire format -- it is an in-memory codec. Use
// packed_size<T>(count, bits) to size the `packed` buffer.
//
// (Impl note: Highway's Pack16<11> / Pack16<13> specializations do not round-trip full-range
// values -- an upstream bug hidden by their test masking inputs with `& kBits` instead of
// `& ((1<<kBits)-1)`. HYLE routes those two u16 widths through its own verified scalar bit-stream
// codec instead of the borrow; the public contract is unchanged, round-trip holds for all widths.)

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

// The tier the bitpack ops selected: kAvx512 if the 512-bit borrow core is live, kAvx2 if the
// 256-bit one is, kNeon on arm64, else kScalar (the bit-stream floor). For tests/diagnostics.
HYLE_SIMD_API Isa active_bitpack_variant() noexcept;

}  // namespace hyle::simd
