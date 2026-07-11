#include "hyle/simd/bitpack.h"

#include <cstddef>
#include <cstdint>

#include "hyle/macros.h"  // HYLE_FATAL_ASSERT (fail-fast on bits-contract violation)
#include "hyle/platform.h"
#include "hyle/simd/dispatch.h"
#include "src/simd/bitpack/bitstream.h"
#include "src/simd/bitpack/kernels.h"

// bitpack_dispatch.cc -- baseline dispatcher for the Highway bit_pack borrow ops. Mirrors
// sort_dispatch.cc: per-type it routes to the widest LIVE tier -> x86 avx512 -> avx2 -> scalar;
// arm64 NEON (unconditional -- ARMv8 baseline); everything else the scalar bit-stream floor. The
// borrow cores are vendored header-only (can't be missing) + compiled per tier; this dispatcher
// only enters a wide tier when the CPU supports it (cpu_features()), so an old CPU never SIGILLs.
//
// bit_pack borrows ONE lib (Highway -- only it ships a vectorized fixed-width packer), so there is
// no per-op library pick, just the tier ladder. bit_pack's AVX-512 target is NOT MSVC-broken
// (integer shift/or/and SIMD, not FP transcendentals), so the 512 tier is used on all compilers --
// no _MSC_VER exclusion (same as vqsort).

namespace hyle::simd {
namespace {

bool have_avx2() noexcept {
#if defined(HYLE_ARCH_X64)
  return cpu_features().avx2;
#else
  return false;
#endif
}

// AVX-512 group (f+bw+vl+dq+cd) -- same condition as sort_dispatch's have_avx512.
bool have_avx512() noexcept {
#if defined(HYLE_ARCH_X64)
  const auto& f = cpu_features();
  return f.avx512f && f.avx512bw && f.avx512vl && f.avx512dq && f.avx512cd;
#else
  return false;
#endif
}

}  // namespace

// Generates the two public overloads (pack/unpack) for one lane type. The tier ladder is identical
// for all types; only the detail:: symbol suffix (tag) differs. `bits` contract (fail-fast):
// 1 <= bits <= 8*sizeof(T). count==0 is a no-op.
#if defined(HYLE_ARCH_X64)
#define HYLE_BITPACK_TIER_PACK(tag) \
  do { \
    static const bool hw512 = have_avx512(); \
    if (hw512) { detail::pack_##tag##_avx512(raw, count, packed, bits); return; } \
    static const bool hw2 = have_avx2(); \
    if (hw2) { detail::pack_##tag##_avx2(raw, count, packed, bits); return; } \
    detail::pack_##tag##_scalar(raw, count, packed, bits); \
  } while (0)
#define HYLE_BITPACK_TIER_UNPACK(tag) \
  do { \
    static const bool hw512 = have_avx512(); \
    if (hw512) { detail::unpack_##tag##_avx512(packed, count, raw, bits); return; } \
    static const bool hw2 = have_avx2(); \
    if (hw2) { detail::unpack_##tag##_avx2(packed, count, raw, bits); return; } \
    detail::unpack_##tag##_scalar(packed, count, raw, bits); \
  } while (0)
#elif defined(HYLE_ARCH_ARM64)
#define HYLE_BITPACK_TIER_PACK(tag)   detail::pack_##tag##_neon(raw, count, packed, bits)
#define HYLE_BITPACK_TIER_UNPACK(tag) detail::unpack_##tag##_neon(packed, count, raw, bits)
#else
#define HYLE_BITPACK_TIER_PACK(tag)   detail::pack_##tag##_scalar(raw, count, packed, bits)
#define HYLE_BITPACK_TIER_UNPACK(tag) detail::unpack_##tag##_scalar(packed, count, raw, bits)
#endif

// Every SIMD tier's smallest block is 128 raw values (NEON); below that all wide TUs execute
// only the identical scalar tail. Do it here to avoid a useless dispatch + cross-TU round trip.
#define HYLE_BITPACK_DISPATCH(T, tag)                                                      \
  void pack(const T* raw, std::size_t count, T* packed, unsigned bits) noexcept {          \
    HYLE_FATAL_ASSERT(bits >= 1 && bits <= 8 * sizeof(T));                                 \
    if (count == 0) return;                                                                \
    if (count < 128) {                                                                     \
      detail::bitstream_pack(raw, count, packed, bits);                                    \
      return;                                                                              \
    }                                                                                      \
    HYLE_BITPACK_TIER_PACK(tag);                                                           \
  }                                                                                        \
  void unpack(const T* packed, std::size_t count, T* raw, unsigned bits) noexcept {        \
    HYLE_FATAL_ASSERT(bits >= 1 && bits <= 8 * sizeof(T));                                 \
    if (count == 0) return;                                                                \
    if (count < 128) {                                                                     \
      detail::bitstream_unpack(packed, count, raw, bits);                                  \
      return;                                                                              \
    }                                                                                      \
    HYLE_BITPACK_TIER_UNPACK(tag);                                                         \
  }

HYLE_BITPACK_TYPES(HYLE_BITPACK_DISPATCH)
#undef HYLE_BITPACK_DISPATCH

Isa active_bitpack_variant() noexcept {
#if defined(HYLE_ARCH_X64)
  static const Isa tier =
      have_avx512() ? Isa::kAvx512 : (have_avx2() ? Isa::kAvx2 : Isa::kScalar);
  return tier;
#elif defined(HYLE_ARCH_ARM64)
  return Isa::kNeon;  // Highway NEON bit_pack core always compiled (ARMv8 baseline)
#else
  return Isa::kScalar;
#endif
}

}  // namespace hyle::simd
