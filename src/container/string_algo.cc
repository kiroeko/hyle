#include "hyle/container/string_algo.h"

#if defined(HYLE_SIMD_SIMDUTF)
#  include "simdutf.h"

static_assert(simdutf::SIMDUTF_VERSION_MAJOR == 9 &&
                  simdutf::SIMDUTF_VERSION_MINOR == 1 &&
                  simdutf::SIMDUTF_VERSION_REVISION == 0,
              "HYLE's reviewed simdutf contract requires v9.1.0");
#endif

// string_algo.cc -- runtime-dispatched UTF-8 validate/count entries. The shipping
// container target directly embeds simdutf's verified SIMD cores ("借核不手搓").
// The conditional keeps this TU usable in stripped/standalone builds; the scalar
// Utf8Decoder is also the constexpr/reference fallback.
//
// CONTRACT MATCH: utf8_length returns the code-point count, or -1 on malformed
// input -- i.e. VALIDATE + COUNT. simdutf splits these (validate_utf8 +
// count_utf8), so the borrow does both: validate first, count only if valid.
// That is the honest same-contract path (a bench against count_utf8 ALONE would
// be apples-to-oranges -- count_utf8 skips validation).

namespace hyle::container::detail {

bool utf8_validate_runtime(string_ref s) noexcept {
  if (s.data == nullptr) return s.size == 0;
#if defined(HYLE_SIMD_SIMDUTF)
  return simdutf::validate_utf8(s.data, s.size);
#else
  Utf8Decoder decoder(s);
  while (!decoder.is_end()) {
    decoder.next();
    if (!decoder.valid()) return false;
  }
  return true;
#endif
}

std::int64_t utf8_length_runtime(string_ref s) noexcept {
  if (s.data == nullptr) return s.size == 0 ? 0 : -1;
#if defined(HYLE_SIMD_SIMDUTF)
  // Borrowed SIMD core: validate then count (matches utf8_length's validate+count
  // contract). simdutf runs its own internal runtime ISA dispatch.
  if (!simdutf::validate_utf8(s.data, s.size)) return -1;
  return static_cast<std::int64_t>(simdutf::count_utf8(s.data, s.size));
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
