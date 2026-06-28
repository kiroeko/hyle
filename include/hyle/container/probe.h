#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "hyle/macros.h"
#include "hyle/platform.h"

// SwissTable control-byte group probing, self-contained (no dependency on the
// full simd/ module -- keeps container at Tier 0).
//
// GroupPortable (SWAR, 8-wide) is ALWAYS compiled, so it is testable on every
// platform (including x64). GroupSse2 (16-wide) is additionally compiled on
// x86-64. `Group`/`kGroupWidth` alias the best available pick. NEON (ARM, 8-wide)
// is deferred (no ARM machine to verify).
//
// Control byte values are the abseil-chosen ones (do NOT change -- picked for
// SIMD): empty=0x80, deleted=0xFE, sentinel=0xFF, full=0hhhhhhh (top bit 0,
// low 7 bits = H2 = hash's high 7 bits).

#if defined(_MSC_VER)
#  include <intrin.h>
HYLE_FORCE_INLINE unsigned hyle_ctz32(std::uint32_t x) noexcept {
  unsigned long i;
  _BitScanForward(&i, x);
  return static_cast<unsigned>(i);
}
HYLE_FORCE_INLINE unsigned hyle_clz32(std::uint32_t x) noexcept {
  unsigned long i;
  _BitScanReverse(&i, x);
  return static_cast<unsigned>(31 - i);
}
#else
HYLE_FORCE_INLINE unsigned hyle_ctz32(std::uint32_t x) noexcept {
  return static_cast<unsigned>(__builtin_ctz(x));
}
HYLE_FORCE_INLINE unsigned hyle_clz32(std::uint32_t x) noexcept {
  return static_cast<unsigned>(__builtin_clz(x));
}
#endif

namespace hyle::container::detail {

using ctrl_t = std::int8_t;
inline constexpr ctrl_t kEmpty = -128;     // 0x80
inline constexpr ctrl_t kDeleted = -2;     // 0xFE
inline constexpr ctrl_t kSentinel = -1;    // 0xFF
// full slots hold 0..127 (H2). A control byte is "full" iff its top bit is 0.
inline constexpr bool ctrl_is_full(ctrl_t c) noexcept { return c >= 0; }

inline constexpr std::size_t h1(std::uint64_t hash) noexcept {
  return static_cast<std::size_t>(hash >> 7);
}
inline constexpr ctrl_t h2(std::uint64_t hash) noexcept {
  return static_cast<ctrl_t>(hash & 0x7F);  // low 7 bits, top bit 0 => "full"
}

// A bitmask over a group's lanes (one bit per lane, low bit = lane 0);
// range-for yields set bit positions (low first).
class BitMask {
 public:
  explicit BitMask(std::uint32_t mask) noexcept : mask_(mask) {}
  explicit operator bool() const noexcept { return mask_ != 0; }
  std::uint32_t lowest() const noexcept { return hyle_ctz32(mask_); }
  // Lane index of the lowest set bit (alias of lowest(), abseil naming). UB if 0.
  std::uint32_t trailing_zeros() const noexcept { return hyle_ctz32(mask_); }
  // Number of clear high lanes within a `width`-lane group (UB if mask_ == 0).
  // The mask occupies the low `width` bits, so strip the 32-width high zeros.
  std::uint32_t leading_zeros(unsigned width) const noexcept {
    return hyle_clz32(mask_) - (32 - width);
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
namespace hyle::container::detail {
// ---- x86-64 SSE2 group (16-wide) --------------------------------------------
class GroupSse2 {
 public:
  static constexpr std::size_t kWidth = 16;
  explicit GroupSse2(const ctrl_t* pos) noexcept
      : ctrl_(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pos))) {}
  BitMask match(ctrl_t h) const noexcept {
    return BitMask(static_cast<std::uint32_t>(
        _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_set1_epi8(h), ctrl_))));
  }
  BitMask match_empty() const noexcept {
    return BitMask(static_cast<std::uint32_t>(
        _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_set1_epi8(kEmpty), ctrl_))));
  }
  BitMask match_empty_or_deleted() const noexcept {
    // empty/deleted/sentinel all have the top bit set; full does not.
    return BitMask(static_cast<std::uint32_t>(_mm_movemask_epi8(ctrl_)));
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
// WRITTEN BUT NOT RUN: cross-compiles clean with aarch64 GCC14, but there is no
// ARM machine to execute the test suite on. Produces a 1-bit-per-lane mask (a
// matched lane's compare result 0xFF has its top bit set), so it satisfies the
// SAME shared BitMask contract as GroupSse2/GroupPortable -- iteration, lowest(),
// leading/trailing_zeros all work unchanged. (abseil's aarch64 group uses a
// byte-mask with a /8 shift; HYLE packs to bits for one BitMask across all ISAs,
// trading a few ops for not forking the mask type. Correctness identical.)
class GroupNeon {
 public:
  static constexpr std::size_t kWidth = 8;
  explicit GroupNeon(const ctrl_t* pos) noexcept
      : ctrl_(vld1_u8(reinterpret_cast<const std::uint8_t*>(pos))) {}
  BitMask match(ctrl_t h) const noexcept {
    uint8x8_t eq = vceq_u8(ctrl_, vdup_n_u8(static_cast<std::uint8_t>(h)));
    return BitMask(pack(eq));
  }
  BitMask match_empty() const noexcept {
    // kEmpty=0x80 is the only ctrl == 0x80.
    uint8x8_t eq = vceq_u8(ctrl_, vdup_n_u8(0x80));
    return BitMask(pack(eq));
  }
  BitMask match_empty_or_deleted() const noexcept {
    // empty/deleted/sentinel all have the top bit set; full does not.
    uint8x8_t hi = vtst_u8(ctrl_, vdup_n_u8(0x80));  // 0xFF where top bit set
    return BitMask(pack(hi));
  }

 private:
  // Each lane of `m` is 0xFF (match) or 0x00; fold to bit i <- (lane i top bit).
  static std::uint32_t pack(uint8x8_t m) noexcept {
    std::uint64_t x = vget_lane_u64(vreinterpret_u64_u8(m), 0);
    std::uint32_t r = 0;
    for (int i = 0; i < 8; ++i)
      if (x & (std::uint64_t{0x80} << (8 * i))) r |= (1u << i);
    return r;
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
inline constexpr std::size_t kGroupWidth = Group::kWidth;
}
