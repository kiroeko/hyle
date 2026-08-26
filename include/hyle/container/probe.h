#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "hyle/core/compiler.h"
#include "hyle/platform.h"

// SwissTable control-byte group probing. Self-contained: hand-written baseline-ISA SIMD, NOT
// routed through the simd/vec.h abstraction -- ON PURPOSE, and the reason is INLINEABILITY, not
// a dependency rule. (container is topologically after simd in the DAG and CAN depend on it --
// sibling str_search.cc does, for count/rfind.) But `match()` is inlined into the innermost
// find/insert/erase probe loop, so it must use the COMPILE-TIME baseline ISA (SSE2 everywhere
// on x86-64, NEON baseline on aarch64) and inline with zero dispatch. vec.h reaches wide ISAs
// (AVX2/AVX-512) via per-TU -march compilation + a runtime best_isa() indirect call -- fine to
// pay ONCE per whole-buffer scan (str_search), fatal to pay PER PROBE. A fixed 16-wide (SSE2) /
// 8-wide (NEON/SWAR) group is exactly abseil's choice, and the right one here.
//
// GroupPortable (SWAR, 8-wide) is ALWAYS compiled, so it is testable on every platform
// (including x64). GroupSse2 (16-wide) is compiled + aliased as `Group` on x86-64; GroupNeon
// (8-wide) on aarch64 (real-hardware validated on Apple Silicon -- swiss_table_test green).
// `Group`/`GroupMask`/`kGroupWidth` alias the best available pick.
//
// Control byte values are the abseil-chosen ones (do NOT change -- picked for
// SIMD): empty=0x80, deleted=0xFE, sentinel=0xFF, full=0hhhhhhh (top bit 0,
// low 7 bits = H2 = hash's high 7 bits).

#if defined(_MSC_VER)
#  include <intrin.h>
#endif

namespace hyle::container::detail {

// 32-bit bit-scan helpers (count trailing / leading zeros). MSVC lacks the GCC
// builtins, so bridge to _BitScan*. Used by BitMask below (never exported).
#if defined(_MSC_VER)
HYLE_FORCE_INLINE unsigned ctz32(std::uint32_t x) noexcept {
  unsigned long i;
  _BitScanForward(&i, x);
  return static_cast<unsigned>(i);
}
HYLE_FORCE_INLINE unsigned clz32(std::uint32_t x) noexcept {
  unsigned long i;
  _BitScanReverse(&i, x);
  return static_cast<unsigned>(31 - i);
}
#else
HYLE_FORCE_INLINE unsigned ctz32(std::uint32_t x) noexcept {
  return static_cast<unsigned>(__builtin_ctz(x));
}
HYLE_FORCE_INLINE unsigned clz32(std::uint32_t x) noexcept {
  return static_cast<unsigned>(__builtin_clz(x));
}
#endif

using ctrl_t = std::int8_t;
inline constexpr ctrl_t kEmpty = -128;     // 0x80
inline constexpr ctrl_t kDeleted = -2;     // 0xFE
inline constexpr ctrl_t kSentinel = -1;    // 0xFF
// full slots hold 0..127 (H2). A control byte is "full" iff its top bit is 0.
inline constexpr bool ctrl_is_full(ctrl_t c) noexcept { return c >= 0; }

// abseil's split: H1 (index) = the hash itself (low bits land in the table after
// `& mask`), H2 = the TOP 7 bits. H1 is thus identity -- NO shift, saving one insn
// per probe vs a `hash >> 7` (the find/insert/erase inner loops all hit this).
// Correctness rests on mix64's avalanche: low bits (index) and top 7 bits (H2) are
// independent, so they stay disjoint -- same property our old (>>7 / low-7) split
// had, just without the h1 shift. H2 = hash>>57 yields 0..127 (top bit clear =>
// "full" marker), exactly what the control byte needs.
inline constexpr std::size_t h1(std::uint64_t hash) noexcept {
  return static_cast<std::size_t>(hash);
}
inline constexpr ctrl_t h2(std::uint64_t hash) noexcept {
  return static_cast<ctrl_t>(hash >> 57);  // top 7 bits; top bit 0 => "full"
}

// A bitmask over a group's lanes (one bit per lane, low bit = lane 0);
// range-for yields set bit positions (low first).
class BitMask {
 public:
  explicit BitMask(std::uint32_t mask) noexcept : mask_(mask) {}
  explicit operator bool() const noexcept { return mask_ != 0; }
  std::uint32_t lowest() const noexcept { return ctz32(mask_); }
  // Lane index of the lowest set bit (alias of lowest(), abseil naming). UB if 0.
  std::uint32_t trailing_zeros() const noexcept { return ctz32(mask_); }
  // Number of clear high lanes within a `width`-lane group (UB if mask_ == 0).
  // The mask occupies the low `width` bits, so strip the 32-width high zeros.
  std::uint32_t leading_zeros(unsigned width) const noexcept {
    return clz32(mask_) - (32 - width);
  }
  BitMask begin() const noexcept { return *this; }
  BitMask end() const noexcept { return BitMask(0); }
  std::uint32_t operator*() const noexcept { return lowest(); }
  void operator++() noexcept { mask_ &= (mask_ - 1); }
  bool operator!=(const BitMask& o) const noexcept { return mask_ != o.mask_; }

 private:
  std::uint32_t mask_;
};

// ---- portable SWAR group (8-wide), ALWAYS compiled --------------------------
// abseil GroupPortableImpl: operate on a uint64 of 8 ctrl bytes.
class GroupPortable {
 public:
  using Mask = BitMask;
  static constexpr std::size_t kWidth = 8;
  explicit GroupPortable(const ctrl_t* pos) noexcept {
    std::memcpy(&ctrl_, pos, sizeof(ctrl_));
  }
  BitMask match(ctrl_t h) const noexcept {
    constexpr std::uint64_t lsb = 0x0101010101010101ull;
    constexpr std::uint64_t msb = 0x8080808080808080ull;
    std::uint64_t x = ctrl_ ^ (lsb * static_cast<std::uint8_t>(h));
    std::uint64_t z = (x - lsb) & ~x & msb;  // top bit set per zero (==) byte
    return BitMask(pack(z));
  }
  BitMask match_empty() const noexcept {
    constexpr std::uint64_t msb = 0x8080808080808080ull;
    // kEmpty=0x80 is the only ctrl with top bit set AND next bit clear.
    return BitMask(pack(ctrl_ & ~(ctrl_ << 1) & msb));
  }
  BitMask match_empty_or_deleted() const noexcept {
    constexpr std::uint64_t msb = 0x8080808080808080ull;
    return BitMask(pack(ctrl_ & msb));  // empty/deleted/sentinel: top bit set
  }
  // Full (occupied) lanes = complement of empty/deleted within the 8-lane group.
  BitMask match_full() const noexcept {
    constexpr std::uint64_t msb = 0x8080808080808080ull;
    return BitMask((pack(ctrl_ & msb) ^ 0xFFu) & 0xFFu);
  }

 private:
  static std::uint32_t pack(std::uint64_t z) noexcept {
    std::uint32_t m = 0;
    for (int i = 0; i < 8; ++i)
      if (z & (std::uint64_t{0x80} << (8 * i))) m |= (1u << i);
    return m;
  }
  std::uint64_t ctrl_;
};

}  // namespace hyle::container::detail

#if defined(HYLE_ARCH_X64)
#  include <emmintrin.h>  // SSE2
#  if defined(HYLE_HAVE_SSSE3)
#    include <tmmintrin.h>  // SSSE3: _mm_sign_epi8 (1-insn match_empty)
#  endif
namespace hyle::container::detail {
// ---- x86-64 SSE2 group (16-wide) --------------------------------------------
class GroupSse2 {
 public:
  using Mask = BitMask;
  static constexpr std::size_t kWidth = 16;
  explicit GroupSse2(const ctrl_t* pos) noexcept
      : ctrl_(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pos))) {}
  BitMask match(ctrl_t h) const noexcept {
    return BitMask(static_cast<std::uint32_t>(
        _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_set1_epi8(h), ctrl_))));
  }
  BitMask match_empty() const noexcept {
#if defined(HYLE_HAVE_SSSE3)
    // abseil's trick: _mm_sign_epi8(c,c) negates per byte -> c<0:-c, 0:0, c>0:c.
    // Only kEmpty=0x80(-128) negates to itself (overflow), keeping its top bit
    // set; kDeleted/kSentinel(-2/-1) become +2/+1 (top bit clear) and full(0..127)
    // stays positive. So movemask flags exactly the empty lanes -- ONE insn, no
    // broadcast constant to rebuild each probe (vs the SSE2 cmpeq path below).
    return BitMask(
        static_cast<std::uint32_t>(_mm_movemask_epi8(_mm_sign_epi8(ctrl_, ctrl_))));
#else
    return BitMask(static_cast<std::uint32_t>(
        _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_set1_epi8(kEmpty), ctrl_))));
#endif
  }
  BitMask match_empty_or_deleted() const noexcept {
    // empty/deleted/sentinel all have the top bit set; full does not.
    return BitMask(static_cast<std::uint32_t>(_mm_movemask_epi8(ctrl_)));
  }
  // Full (occupied) lanes = complement of empty/deleted within the 16-lane group.
  // for_each scans groups and iterates only these bits -- 16 ctrl bytes per SIMD op
  // + branchless bit iteration, vs a per-slot byte load + branch (abseil's iterator
  // shape; the naive per-byte scan lost absl ~1.2x on iterate).
  BitMask match_full() const noexcept {
    return BitMask(static_cast<std::uint32_t>(_mm_movemask_epi8(ctrl_)) ^ 0xFFFFu);
  }

 private:
  __m128i ctrl_;
};
using Group = GroupSse2;
}  // namespace hyle::container::detail
#elif defined(HYLE_ARCH_ARM64)
#  include <arm_neon.h>
namespace hyle::container::detail {
// ---- ARM64 NEON group (8-wide) ----------------------------------------------
// Real-hardware validated on Apple Silicon (swiss_table_test + probe_test green).
// AArch64 has no movemask. Keep the comparison result's high bit in each byte and
// iterate it with a shift-3 mask, matching mature Swiss-table implementations.
// This avoids a horizontal NEON reduction on every match/empty query; the public
// group contract still reports ordinary lane indices.
class NeonBitMask {
 public:
  explicit NeonBitMask(std::uint64_t mask) noexcept : mask_(mask) {}
  explicit operator bool() const noexcept { return mask_ != 0; }
  std::uint32_t lowest() const noexcept {
    return static_cast<std::uint32_t>(std::countr_zero(mask_) >> 3);
  }
  std::uint32_t trailing_zeros() const noexcept { return lowest(); }
  std::uint32_t leading_zeros(unsigned width) const noexcept {
    (void)width;
    return static_cast<std::uint32_t>(std::countl_zero(mask_) >> 3);
  }
  NeonBitMask begin() const noexcept { return *this; }
  NeonBitMask end() const noexcept { return NeonBitMask(0); }
  std::uint32_t operator*() const noexcept { return lowest(); }
  void operator++() noexcept { mask_ &= mask_ - 1; }
  bool operator!=(const NeonBitMask &other) const noexcept {
    return mask_ != other.mask_;
  }

 private:
  std::uint64_t mask_;
};

class GroupNeon {
 public:
  using Mask = NeonBitMask;
  static constexpr std::size_t kWidth = 8;
  explicit GroupNeon(const ctrl_t* pos) noexcept
      : ctrl_(vld1_u8(reinterpret_cast<const std::uint8_t*>(pos))) {}
  Mask match(ctrl_t h) const noexcept {
    uint8x8_t eq = vceq_u8(ctrl_, vdup_n_u8(static_cast<std::uint8_t>(h)));
    return Mask(high_bits(eq));
  }
  Mask match_empty() const noexcept {
    // kEmpty=0x80 is the only ctrl == 0x80.
    uint8x8_t eq = vceq_u8(ctrl_, vdup_n_u8(0x80));
    return Mask(high_bits(eq));
  }
  Mask match_empty_or_deleted() const noexcept {
    // empty/deleted/sentinel all have the top bit set; full does not.
    uint8x8_t hi = vtst_u8(ctrl_, vdup_n_u8(0x80));  // 0xFF where top bit set
    return Mask(high_bits(hi));
  }
  // Full (occupied) lanes = complement of empty/deleted within the 8-lane group.
  Mask match_full() const noexcept {
    uint8x8_t hi = vtst_u8(ctrl_, vdup_n_u8(0x80));
    return Mask((~high_bits(hi)) & 0x8080808080808080ull);
  }

 private:
  static std::uint64_t high_bits(uint8x8_t mask) noexcept {
    return vget_lane_u64(vreinterpret_u64_u8(mask), 0) &
           0x8080808080808080ull;
  }
  uint8x8_t ctrl_;
};
using Group = GroupNeon;
}  // namespace hyle::container::detail
#else
namespace hyle::container::detail {
using Group = GroupPortable;
}  // namespace hyle::container::detail
#endif

namespace hyle::container::detail {
using GroupMask = typename Group::Mask;
inline constexpr std::size_t kGroupWidth = Group::kWidth;
}
