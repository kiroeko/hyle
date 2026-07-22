#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

// Compile-time type name + stable type ID without RTTI (clay type_name). The
// name is sliced out of the compiler's pretty-function string. clay hardcoded
// the FULL per-compiler signature as the prefix/suffix to strip -- brittle:
// renaming the function or namespace silently breaks the slice. HYLE instead
// DERIVES the prefix/suffix by probing a sentinel type (the magic_enum/ctti
// technique), so it survives refactors. conventions §8: keep, but centralize +
// unit-test (the pretty-function spelling is compiler-fragile).
//
// Returns std::string_view (a borrowed view into the compiler's static string)
// -- the conventions-approved view use, and it keeps util independent of
// container/string_ref (whose placement is a deferred architecture decision).
// The exact spelling differs per compiler (MSVC prefixes "class"/"struct"); use
// it for stable IDs and diagnostics, not for cross-compiler string equality.

namespace hyle::util {

namespace detail {

// The compiler's decorated signature of THIS function, with T substituted.
template <class T>
constexpr std::string_view wrapped_type_name() noexcept {
#if defined(__clang__) || defined(__GNUC__)
  return __PRETTY_FUNCTION__;
#elif defined(_MSC_VER)
  return __FUNCSIG__;
#else
  return std::string_view{};  // unknown compiler: degrade to empty (safe)
#endif
}

// Probe with a known type to learn how many chars wrap the substituted name.
// `double` is the sentinel: it has no occurrence in this namespace/function
// spelling, so it appears exactly once -- where T was substituted.
struct ProbeTag {};
inline constexpr std::string_view kProbe = wrapped_type_name<double>();
inline constexpr std::size_t kPrefixLen = kProbe.find("double");
inline constexpr std::size_t kSuffixLen =
    kProbe.size() - kPrefixLen - std::string_view("double").size();

// Whether the slicing markers were found (false on an unknown compiler whose
// wrapped name is empty -> kPrefixLen == npos).
inline constexpr bool kSupported = kPrefixLen != std::string_view::npos;

}  // namespace detail

// Fully-qualified compile-time name of T. Empty on an unsupported compiler.
template <class T>
constexpr std::string_view type_name() noexcept {
  if constexpr (!detail::kSupported) {
    return std::string_view{};
  } else {
    std::string_view w = detail::wrapped_type_name<T>();
    w.remove_prefix(detail::kPrefixLen);
    w.remove_suffix(detail::kSuffixLen);
    return w;
  }
}

// Stable 64-bit type ID = FNV-1a of type_name<T>(). constexpr (XXH3, HYLE's
// runtime hash, is not constexpr; FNV-1a is trivially so and fine for type IDs).
// Two distinct types almost surely differ; the same type always agrees, within
// one compiler build. NOT stable across compilers (the name spelling differs).
template <class T>
constexpr std::uint64_t type_hash() noexcept {
  std::string_view n = type_name<T>();
  std::uint64_t h = 14695981039346656037ull;  // FNV offset basis
  for (char c : n) {
    h ^= static_cast<std::uint8_t>(c);
    h *= 1099511628211ull;  // FNV prime
  }
  return h;
}

}  // namespace hyle::util
