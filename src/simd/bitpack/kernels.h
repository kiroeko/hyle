#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/platform.h"  // HYLE_ARCH_X64 / HYLE_ARCH_ARM64 (SIMD cores are arch-specific)

// kernels.h (internal) -- per-tier HYLE bitpack symbols the dispatcher takes the address of. Same
// shape as sort/kernels.h: the scalar TU (pure contiguous bit-stream codec) is ALWAYS defined + is
// the non-x86 / fallback floor (NOT a stub -- it is a self-consistent codec). The HYLE D-tag
// Pack8/Pack16/Pack32 implementation is compiled per ISA tier on the matching architecture (x86
// avx2/avx512, arm64 neon); bitpack_dispatch enters it only when the CPU supports the tier, so wide
// codegen never runs on an older CPU. Not a public header.
//
// The runtime `bits` is folded to the compile-time kBits template argument inside each HYLE target
// TU (std::index_sequence<1..B>), so one symbol per (type,op,tier) covers every width. Whole N*B
// raw-value blocks use Pack<kBits>; the sub-block tail uses the scalar bit-stream codec (same-tier
// round trip, see bitpack.h layout contract).

namespace hyle::simd::detail {

// X-macro over the three supported lane types this milestone: <C++ type>, <short tag>.
// B (bits-per-lane = max kBits) is 8*sizeof(T): 8 / 16 / 32.
#define HYLE_BITPACK_TYPES(X) \
  X(std::uint8_t, u8)         \
  X(std::uint16_t, u16)       \
  X(std::uint32_t, u32)

// Declares the pack/unpack pair for one (type, tier). `count` is in units of raw values; `bits` is
// the per-value significant width (1..8*sizeof(T), validated fail-fast in the dispatcher). count==0
// is a no-op. The packed buffer must hold packed_size<T>(count, bits) elements (see bitpack.h).
#define HYLE_BITPACK_DECL_PAIR(T, tag, tier)                                                  \
  void pack_##tag##_##tier(const T* raw, std::size_t count, T* packed, unsigned bits) noexcept;\
  void unpack_##tag##_##tier(const T* packed, std::size_t count, T* raw, unsigned bits) noexcept;

// scalar floor (bit-stream codec) -- always available, every arch.
#define HYLE_BITPACK_DECL_SCALAR(T, tag) HYLE_BITPACK_DECL_PAIR(T, tag, scalar)
HYLE_BITPACK_TYPES(HYLE_BITPACK_DECL_SCALAR)
#undef HYLE_BITPACK_DECL_SCALAR

#if defined(HYLE_ARCH_X64)
// AVX2 (256-bit) + AVX-512 (512-bit) HYLE bitpack cores. Entered by bitpack_dispatch when
// cpu_features().avx2 / the avx512 group. The kernels use integer shift/or/and SIMD (not FP
// transcendentals), so the 512 core exists on MSVC too.
#define HYLE_BITPACK_DECL_AVX2(T, tag) HYLE_BITPACK_DECL_PAIR(T, tag, avx2)
HYLE_BITPACK_TYPES(HYLE_BITPACK_DECL_AVX2)
#undef HYLE_BITPACK_DECL_AVX2
#define HYLE_BITPACK_DECL_AVX512(T, tag) HYLE_BITPACK_DECL_PAIR(T, tag, avx512)
HYLE_BITPACK_TYPES(HYLE_BITPACK_DECL_AVX512)
#undef HYLE_BITPACK_DECL_AVX512
#endif  // HYLE_ARCH_X64

#if defined(HYLE_ARCH_ARM64)
// aarch64 NEON HYLE bitpack core (ARMv8 baseline; bitpack_dispatch enters it unconditionally,
// no runtime gate -- NEON is guaranteed present).
#define HYLE_BITPACK_DECL_NEON(T, tag) HYLE_BITPACK_DECL_PAIR(T, tag, neon)
HYLE_BITPACK_TYPES(HYLE_BITPACK_DECL_NEON)
#undef HYLE_BITPACK_DECL_NEON
#endif  // HYLE_ARCH_ARM64

#undef HYLE_BITPACK_DECL_PAIR

}  // namespace hyle::simd::detail
