#pragma once

// hash_inline.h -- OPT-IN header-inline variant of hash_bytes(), for HOT call sites
// hashing SHORT byte ranges (roughly 17..128B) that want to shave the out-of-line
// seam. NOT included by the default hash.h.
//
// WHY THIS EXISTS (the seam, and who pays it): hash.h's hash_bytes() sends >16B
// inputs to an OUT-OF-LINE core (detail::hash_bytes_xxh3 / _wyhash, compiled in
// hash.cc) so the 7490-line xxhash.h + wyhash.h never reach a consumer TU. That
// out-of-line call costs ~1ns (function call + no cross-TU inlining + DLL/.so
// indirection). ~1ns is a FIXED cost, so its VISIBILITY depends on the length:
//   - 64B  : the hash itself is ~1.9ns, so +1ns is ~37% -> visible (0.78x vs raw).
//   - 256B : the hash itself is ~10ns, so +1ns is ~8%  -> lost in noise (~tie).
// So the seam only matters for SHORT byte ranges. hash_bytes_inline() below inlines
// the hasher into YOUR TU (no call, no seam) -- but the PRICE is that including this
// header pulls xxhash.h + wyhash.h into your TU (compile-time cost + those headers'
// symbols). You opt in per-TU only where a profiler shows short-byte-range hashing
// is hot. The DEFAULT (hash.h hash_bytes) stays seam-isolated for everyone else.
//
// BIT-IDENTICAL to hash_bytes(): same length dispatch (<=16 inline mix / <=128 XXH3
// / >128 wyhash), same seed/secret, so a key hashed via hash_bytes() and looked up
// via hash_bytes_inline() (or vice versa) agree. Verified by HashInlineMatchesOutOfLine.
//
// USE IT FOR: a profiled hot loop hashing many SHORT (17..~128B) byte ranges where
// the ~1ns seam is a measurable fraction. For <=16B keys the default already inlines
// (no seam -- don't bother). For long strings the seam is already amortized (don't
// bother). When unsure, use hash.h's hash_bytes().

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "hyle/container/hash.h"  // mix64 / mix64_2 / hash_small / kWyhashThreshold

// The vendored single-header hashers, pulled DIRECTLY into this TU (this is the
// opt-in cost). XXH_INLINE_ALL makes XXH3 fully inline here.
#define XXH_INLINE_ALL
#include "xxhash.h"
#include "wyhash.h"

namespace hyle::container {

// Header-inline twin of hash_bytes(): identical result, no out-of-line seam.
HYLE_FORCE_INLINE std::uint64_t hash_bytes_inline(const void* data, std::size_t len,
                                                  std::uint64_t seed = 0) noexcept {
  if (len <= 16) {
    // Identical to hash.h hash_bytes()'s <=16 path (already inline there too).
    std::uint64_t lo = 0, hi = 0;
    if (len != 0) {
      std::memcpy(&lo, data, len < 8 ? len : 8);
      if (len > 8) std::memcpy(&hi, static_cast<const char*>(data) + 8, len - 8);
    }
    const std::uint64_t s = lo ^ seed ^ (len * 0x9e3779b97f4a7c15ull);
    if (len > 8) return detail::mix64_2(s, hi ^ 0xff51afd7ed558ccdull);
    return detail::mix64(s);
  }
  // >16B: SAME best-of-breed split as hash_bytes(), but the hasher is inlined here
  // (no call into hash.cc's out-of-line core -> no seam).
  if (len <= detail::kWyhashThreshold) return XXH3_64bits_withSeed(data, len, seed);
  return wyhash(data, len, seed, _wyp);
}

}  // namespace hyle::container
