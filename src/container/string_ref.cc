#include "hyle/container/string_ref.h"

#include <string_view>  // membership/substring family forwards to std's SOTA (see below)

#include "hyle/container/string_algo.h"
#include "hyle/macros.h"
#include "hyle/platform.h"
#include "hyle/platform/cpu.h"  // best_isa() -> pick the widest compiled count/rfind kernel
#include "src/container/str_search.h"  // per-ISA str_count/str_rfind_char variant symbols

// Out-of-line SIMD cores for string_ref's search family. Compiled ONCE here and
// exported (HYLE_CONTAINER_API) so the public header stays free of <string_view>
// (string_ref.h pulls in no std string headers, this .cc does).
//
// count/rfind DISPATCH (2026-07, unified onto simd/vec.h): str_count / str_rfind_char are
// now function-pointer dispatchers picking the widest compiled kernel by best_isa(), exactly
// like simd/ops.h's dot/saxpy. The kernels live in str_search_{scalar,sse4,avx2,avx512,neon}
// .cc (shared body str_search_impl.h, written against the vec.h D-tag abstraction). This
// REPLACED the old hand-rolled string_ref_avx2.cc (256b) + inline SSE2 core here: one source
// for every ISA, and aarch64/NEON + AVX-512 count/rfind for free (previously x86-only ->
// arm64 fell to scalar). The SAD count keeps its 4-accumulator psadbw structure (via SumsOf8).
//
// SPLIT STRATEGY (measured, 2026-07): HYLE self-implements the ops where it matches or beats
// std (find(char)/rfind(char)/count via the dispatched kernels, starts_with/split via memcmp),
// but FORWARDS to std::string_view for the ops where MSVC STL's hand-written vectorized asm
// (two-way substring search, PCMPESTRI char-set scans) is genuinely ahead. The forward is
// header-inline-free (stays out of the public header + DLL symbol table).

namespace hyle::container::detail {

// --- count/rfind: pick the widest compiled kernel by best_isa(), cached once ------------
namespace {
using CountFn = std::uint32_t (*)(const char*, std::uint32_t, char) noexcept;
using RfindFn = std::uint32_t (*)(const char*, std::uint32_t, char) noexcept;
using TransformFn = std::uint32_t (*)(const char*, std::uint32_t, char*,
                                      std::uint32_t) noexcept;

CountFn resolve_count() noexcept {
#if defined(HYLE_ARCH_X64)
  const auto v = static_cast<std::uint32_t>(::hyle::platform::best_isa());
  if (v >= static_cast<std::uint32_t>(::hyle::platform::Isa::kAvx512)) return &str_count_avx512;
  if (v >= static_cast<std::uint32_t>(::hyle::platform::Isa::kAvx2)) return &str_count_avx2;
  if (v >= static_cast<std::uint32_t>(::hyle::platform::Isa::kSse42)) return &str_count_sse4;
  return &str_count_scalar;
#elif defined(HYLE_ARCH_ARM64)
  if (static_cast<std::uint32_t>(::hyle::platform::best_isa()) >=
      static_cast<std::uint32_t>(::hyle::platform::Isa::kNeon))
    return &str_count_neon;
  return &str_count_scalar;
#else
  return &str_count_scalar;
#endif
}
RfindFn resolve_rfind() noexcept {
#if defined(HYLE_ARCH_X64)
  const auto v = static_cast<std::uint32_t>(::hyle::platform::best_isa());
  if (v >= static_cast<std::uint32_t>(::hyle::platform::Isa::kAvx512)) return &str_rfind_char_avx512;
  if (v >= static_cast<std::uint32_t>(::hyle::platform::Isa::kAvx2)) return &str_rfind_char_avx2;
  if (v >= static_cast<std::uint32_t>(::hyle::platform::Isa::kSse42)) return &str_rfind_char_sse4;
  return &str_rfind_char_scalar;
#elif defined(HYLE_ARCH_ARM64)
  if (static_cast<std::uint32_t>(::hyle::platform::best_isa()) >=
      static_cast<std::uint32_t>(::hyle::platform::Isa::kNeon))
    return &str_rfind_char_neon;
  return &str_rfind_char_scalar;
#else
  return &str_rfind_char_scalar;
#endif
}
TransformFn resolve_transform(bool lower) noexcept {
#if defined(HYLE_ARCH_X64)
  const auto v = static_cast<std::uint32_t>(::hyle::platform::best_isa());
  if (v >= static_cast<std::uint32_t>(::hyle::platform::Isa::kAvx512))
    return lower ? &str_ascii_lower_avx512 : &str_ascii_upper_avx512;
  if (v >= static_cast<std::uint32_t>(::hyle::platform::Isa::kAvx2))
    return lower ? &str_ascii_lower_avx2 : &str_ascii_upper_avx2;
  if (v >= static_cast<std::uint32_t>(::hyle::platform::Isa::kSse42))
    return lower ? &str_ascii_lower_sse4 : &str_ascii_upper_sse4;
  return lower ? &str_ascii_lower_scalar : &str_ascii_upper_scalar;
#elif defined(HYLE_ARCH_ARM64)
  if (static_cast<std::uint32_t>(::hyle::platform::best_isa()) >=
      static_cast<std::uint32_t>(::hyle::platform::Isa::kNeon))
    return lower ? &str_ascii_lower_neon : &str_ascii_upper_neon;
  return lower ? &str_ascii_lower_scalar : &str_ascii_upper_scalar;
#else
  return lower ? &str_ascii_lower_scalar : &str_ascii_upper_scalar;
#endif
}
}  // namespace

// Meyers-cached fn ptr (thread-safe init-once); after the first call it's a predictable load
// + one indirect call, dwarfed by the scan. A short input still enters the kernel, whose own
// scalar tail handles a sub-vector length -- no separate small-size branch needed.
std::uint32_t str_count(const char* data, std::uint32_t size, char c) noexcept {
  static const CountFn fn = resolve_count();
  return fn(data, size, c);
}
std::uint32_t str_rfind_char(const char* data, std::uint32_t size, char c) noexcept {
  static const RfindFn fn = resolve_rfind();
  return fn(data, size, c);
}
std::uint32_t str_ascii_lower(const char* data, std::uint32_t size, char* out,
                              std::uint32_t cap) noexcept {
  static const TransformFn fn = resolve_transform(true);
  return fn(data, size, out, cap);
}
std::uint32_t str_ascii_upper(const char* data, std::uint32_t size, char* out,
                              std::uint32_t cap) noexcept {
  static const TransformFn fn = resolve_transform(false);
  return fn(data, size, out, cap);
}

// Forward char-set scan. `want`=true: first index IN set (find_first_of); false:
// first NOT in set (find_first_not_of). Forwards to std::string_view, whose MSVC
// vectorized find_first_of/not_of is the measured SOTA here (PCMPESTRI-class asm we
// couldn't out-engineer; our own PCMPESTRI attempt only reached ~0.78x). npos maps to
// our 0xFFFFFFFF.
std::uint32_t str_find_first_of(const char* data, std::uint32_t size,
                                const char* set, std::uint32_t set_size,
                                std::uint32_t pos, bool want) noexcept {
  constexpr std::uint32_t kNpos = 0xFFFFFFFFu;
  std::string_view hay(data, size);
  std::string_view s(set, set_size);
  std::size_t r = want ? hay.find_first_of(s, pos)
                       : hay.find_first_not_of(s, pos);
  return r == std::string_view::npos ? kNpos : static_cast<std::uint32_t>(r);
}

// Backward char-set scan. `pos` is the highest index to consider (inclusive).
// want=true => find_last_of; false => find_last_not_of. Forwards to std::string_view
// (same SOTA rationale as the forward variant).
std::uint32_t str_find_last_of(const char* data, std::uint32_t size,
                               const char* set, std::uint32_t set_size,
                               std::uint32_t pos, bool want) noexcept {
  constexpr std::uint32_t kNpos = 0xFFFFFFFFu;
  std::string_view hay(data, size);
  std::string_view s(set, set_size);
  std::size_t r = want ? hay.find_last_of(s, pos)
                       : hay.find_last_not_of(s, pos);
  return r == std::string_view::npos ? kNpos : static_cast<std::uint32_t>(r);
}

// Substring search, forwards to std::string_view (two-way vectorized SOTA). rev=false
// => find (first at/after pos); rev=true => rfind (last at/before pos).
std::uint32_t str_find_substr(const char* data, std::uint32_t size,
                              const char* needle, std::uint32_t needle_size,
                              std::uint32_t pos, bool rev) noexcept {
  constexpr std::uint32_t kNpos = 0xFFFFFFFFu;
  std::string_view hay(data, size);
  std::string_view n(needle, needle_size);
  std::size_t r = rev ? hay.rfind(n, pos) : hay.find(n, pos);
  return r == std::string_view::npos ? kNpos : static_cast<std::uint32_t>(r);
}

}  // namespace hyle::container::detail
