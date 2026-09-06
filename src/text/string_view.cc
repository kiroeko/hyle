#include "hyle/container/string_view.h"

#include <string_view>

#include "hyle/container/string_algo.h"
#include "hyle/core/compiler.h"
#include "hyle/core/environment.h"
#include "hyle/platform/cpu.h"
#include "cpu_dispatch_internal.h"
#include "str_search.h"  // per-ISA str_count/str_rfind_char variant symbols

// Out-of-line SIMD cores for StringView's search family. Compiled once here and
// exported so the public header stays free of ISA-specific headers.
//
// count/rfind dispatch resolves the widest safe compiled whole-buffer kernel.
// Generated variant TUs share str_search_impl.h and use locked Highway register
// types directly; Container retains the public pointer/count contract.
//
// HYLE owns dispatched count/reverse-char/case kernels. Substring and
// character-set searches delegate to the standard library instead of
// duplicating its mature search implementation.

namespace hyle::container::detail {

// Pick the widest compiled kernel whose complete requirements are present;
// each public operation caches the resulting pointer once.
namespace {
using CountFn = std::uint32_t (*)(const char*, std::uint32_t, char) noexcept;
using RfindFn = std::uint32_t (*)(const char*, std::uint32_t, char) noexcept;
using TransformFn = std::uint32_t (*)(const char*, std::uint32_t, char*,
                                      std::uint32_t) noexcept;

CountFn resolve_count() noexcept {
#if HYLE_ARCH_X64
  const auto& features = ::hyle::platform::cpu_features();
#if defined(HYLE_TEXT_HAS_AVX512)
  if (supports_highway_avx512(features)) return &str_count_avx512;
#endif
  if (supports_highway_avx2(features)) return &str_count_avx2;
  if (supports_highway_sse4(features)) return &str_count_sse4;
  return &str_count_scalar;
#elif HYLE_ARCH_ARM64
  if (supports_highway_neon(::hyle::platform::cpu_features()))
    return &str_count_neon;
  return &str_count_scalar;
#else
  return &str_count_scalar;
#endif
}
RfindFn resolve_rfind() noexcept {
#if HYLE_ARCH_X64
  const auto& features = ::hyle::platform::cpu_features();
#if defined(HYLE_TEXT_HAS_AVX512)
  if (supports_highway_avx512(features)) return &str_rfind_char_avx512;
#endif
  if (supports_highway_avx2(features)) return &str_rfind_char_avx2;
  if (supports_highway_sse4(features)) return &str_rfind_char_sse4;
  return &str_rfind_char_scalar;
#elif HYLE_ARCH_ARM64
  if (supports_highway_neon(::hyle::platform::cpu_features()))
    return &str_rfind_char_neon;
  return &str_rfind_char_scalar;
#else
  return &str_rfind_char_scalar;
#endif
}
TransformFn resolve_transform(bool lower) noexcept {
#if HYLE_ARCH_X64
  const auto& features = ::hyle::platform::cpu_features();
#if defined(HYLE_TEXT_HAS_AVX512)
  if (supports_highway_avx512(features))
    return lower ? &str_ascii_lower_avx512 : &str_ascii_upper_avx512;
#endif
  if (supports_highway_avx2(features))
    return lower ? &str_ascii_lower_avx2 : &str_ascii_upper_avx2;
  if (supports_highway_sse4(features))
    return lower ? &str_ascii_lower_sse4 : &str_ascii_upper_sse4;
  return lower ? &str_ascii_lower_scalar : &str_ascii_upper_scalar;
#elif HYLE_ARCH_ARM64
  if (supports_highway_neon(::hyle::platform::cpu_features()))
    return lower ? &str_ascii_lower_neon : &str_ascii_upper_neon;
  return lower ? &str_ascii_lower_scalar : &str_ascii_upper_scalar;
#else
  return lower ? &str_ascii_lower_scalar : &str_ascii_upper_scalar;
#endif
}
}  // namespace

// Function selection is thread-safe and cached once. A short input is handled
// by each kernel's scalar tail without a separate public API branch.
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

// Forward char-set scan. `want`=true: first index IN set (find_first_of);
// false: first NOT in set (find_first_not_of). npos maps to 0xFFFFFFFF.
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
// want=true => find_last_of; false => find_last_not_of.
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

// Substring search delegates to std::string_view. rev=false => find (first
// at/after pos); rev=true => rfind (last at/before pos).
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
