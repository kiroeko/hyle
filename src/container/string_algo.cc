#include "hyle/container/string_algo.h"

#if defined(HYLE_SIMD_SIMDUTF)
#  include "simdutf.h"
#endif

// string_algo.cc -- the runtime-dispatched entry for utf8_length. The scalar
// Utf8Decoder loop in string_algo.h stays the constexpr/compile-time path and the
// always-available fallback; at RUNTIME utf8_length routes here, which BORROWS
// simdutf's verified SIMD cores when wired (-DHYLE_SIMD_SIMDUTF), exactly as
// simd/math borrows Highway/xsimd ("借核不手搓"). Without simdutf this is
// the same scalar scan (no behavior change, just not vectorized).
//
// CONTRACT MATCH: utf8_length returns the code-point count, or -1 on malformed
// input -- i.e. VALIDATE + COUNT. simdutf splits these (validate_utf8 +
// count_utf8), so the borrow does both: validate first, count only if valid.
// That is the honest same-contract path (a bench against count_utf8 ALONE would
// be apples-to-oranges -- count_utf8 skips validation).

namespace hyle::container::detail {

std::int64_t utf8_length_runtime(string_ref s) noexcept {
#if defined(HYLE_SIMD_SIMDUTF)
  // Borrowed SIMD core: validate then count (matches utf8_length's validate+count
  // contract). simdutf runs its own internal runtime ISA dispatch.
  const char* data = s.data == nullptr ? "" : s.data;
  if (!simdutf::validate_utf8(data, s.size)) return -1;
  return static_cast<std::int64_t>(simdutf::count_utf8(data, s.size));
#else
  // Scalar fallback: identical to the header's compile-time path.
  Utf8Decoder d(s);
  std::int64_t n = 0;
  while (!d.is_end()) {
    d.next();
    if (!d.valid()) return -1;
    ++n;
  }
  return n;
#endif
}

}  // namespace hyle::container::detail
