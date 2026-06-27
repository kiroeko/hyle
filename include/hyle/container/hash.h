#pragma once

#include <cstddef>
#include <cstdint>
#include <concepts>
#include <string_view>
#include <type_traits>

#include "hyle/macros.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_CONTAINER_API
#elif defined(HYLE_CONTAINER_BUILD)
#  define HYLE_CONTAINER_API HYLE_EXPORT
#else
#  define HYLE_CONTAINER_API HYLE_IMPORT
#endif

namespace hyle::container {

// Hash an arbitrary byte range with XXH3 (xxHash). Defined in hash.cc so the
// 7490-line xxhash.h stays out of this header (ABI-clean + no compile-time
// bloat in every consumer). This is HYLE's default RUNTIME hasher (clay's murmur
// is retained only for compile-time type IDs).
HYLE_CONTAINER_API std::uint64_t hash_bytes(const void* data, std::size_t len,
                                            std::uint64_t seed = 0) noexcept;

namespace detail {
// 64-bit finalizer (splitmix64) for small integral/pointer keys: a function
// call into XXH3 would cost more than the key is worth, so mix inline. Spreads
// low bits to high so consecutive integers don't collide in a power-of-two mask.
HYLE_FORCE_INLINE constexpr std::uint64_t mix64(std::uint64_t x) noexcept {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebull;
  x ^= x >> 31;
  return x;
}
}  // namespace detail

// Default hasher. Specialized for the common cheap cases; falls back to hashing
// the object's bytes via XXH3 for trivially-copyable types. Users may specialize
// Hash<T> for their own types.
template <class T, class = void>
struct Hash {
  static_assert(std::is_trivially_copyable_v<T>,
                "Hash<T>: specialize Hash for non-trivially-copyable T");
  std::uint64_t operator()(const T& v) const noexcept {
    return hash_bytes(&v, sizeof(T));
  }
};

// Integral / enum: inline mix, no XXH3 call.
template <class T>
struct Hash<T, std::enable_if_t<std::is_integral_v<T> || std::is_enum_v<T>>> {
  std::uint64_t operator()(T v) const noexcept {
    return detail::mix64(static_cast<std::uint64_t>(v));
  }
};

// Pointers: mix the address (drop the always-zero low alignment bits first).
template <class T>
struct Hash<T*> {
  std::uint64_t operator()(T* p) const noexcept {
    return detail::mix64(reinterpret_cast<std::uintptr_t>(p));
  }
};

// Strings / byte views: XXH3 over the bytes.
template <>
struct Hash<std::string_view> {
  std::uint64_t operator()(std::string_view s) const noexcept {
    return hash_bytes(s.data(), s.size());
  }
};

// A type usable as a hash key: Hash<T> yields a uint64 for it.
template <class T>
concept Hashable = requires(const T& v) {
  { Hash<T>{}(v) } -> std::convertible_to<std::uint64_t>;
};

}  // namespace hyle::container
