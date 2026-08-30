#include "hyle/simd/sort.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <type_traits>

#include "hyle/core/assert.h"

// Stable LSD radix ordering. Digits are derived from keys while complete key/index objects are
// scattered, so no implementation path aliases a signed or floating object through an unsigned
// pointer. The scratch overload is the hot/repeated route and performs no allocation.

namespace hyle::simd {
namespace {

constexpr unsigned kBits = 8;
constexpr unsigned kBins = 1u << kBits;
constexpr unsigned kMask = kBins - 1;
constexpr unsigned kHistWays = 4;

template <typename GetDigit>
void multi_histogram_prefix(std::size_t n, std::size_t count[kBins], GetDigit digit) noexcept {
  alignas(64) std::size_t c0[kBins] = {};
  alignas(64) std::size_t c1[kBins] = {};
  alignas(64) std::size_t c2[kBins] = {};
  alignas(64) std::size_t c3[kBins] = {};
  std::size_t i = 0;
  for (; i + kHistWays <= n; i += kHistWays) {
    ++c0[digit(i)];
    ++c1[digit(i + 1)];
    ++c2[digit(i + 2)];
    ++c3[digit(i + 3)];
  }
  for (; i < n; ++i) ++c0[digit(i)];
  std::size_t sum = 0;
  for (unsigned b = 0; b < kBins; ++b) {
    const std::size_t total = c0[b] + c1[b] + c2[b] + c3[b];
    count[b] = sum;
    sum += total;
  }
}

struct ByteRange {
  std::uintptr_t begin;
  std::uintptr_t end;
};

ByteRange byte_range(const void* data, std::size_t count,
                     std::size_t item_size) noexcept {
  HYLE_FATAL_ASSERT(item_size != 0);
  HYLE_FATAL_ASSERT(count <= std::numeric_limits<std::size_t>::max() / item_size);
  const std::size_t byte_count = count * item_size;
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(data);
  HYLE_FATAL_ASSERT(byte_count <= std::numeric_limits<std::uintptr_t>::max());
  HYLE_FATAL_ASSERT(begin <=
                    std::numeric_limits<std::uintptr_t>::max() - byte_count);
  return {begin, begin + static_cast<std::uintptr_t>(byte_count)};
}

bool disjoint(ByteRange a, ByteRange b) noexcept {
  return a.end <= b.begin || b.end <= a.begin;
}

template <typename T>
using UnsignedOf = std::make_unsigned_t<T>;

template <typename T>
UnsignedOf<T> sortable_integral(T value, bool ascending) noexcept {
  using U = UnsignedOf<T>;
  U bits = static_cast<U>(value);
  if constexpr (std::is_signed_v<T>) bits ^= U{1} << (sizeof(T) * 8 - 1);
  return ascending ? bits : static_cast<U>(~bits);
}

template <typename U>
U float_flip(U bits) noexcept {
  constexpr U kSign = U{1} << (sizeof(U) * 8 - 1);
  return bits ^ ((bits & kSign) != 0 ? std::numeric_limits<U>::max() : kSign);
}

template <typename F>
auto sortable_float(F value, bool ascending) noexcept {
  static_assert(std::is_same_v<F, float> || std::is_same_v<F, double>);
  using U = std::conditional_t<sizeof(F) == 4, std::uint32_t, std::uint64_t>;
  constexpr unsigned kMantissa = sizeof(F) == 4 ? 23u : 52u;
  constexpr unsigned kExponent = sizeof(F) == 4 ? 8u : 11u;
  constexpr U kMantissaMask = (U{1} << kMantissa) - 1;
  constexpr U kExponentMask = ((U{1} << kExponent) - 1) << kMantissa;
  constexpr U kSign = U{1} << (sizeof(U) * 8 - 1);
  const U bits = std::bit_cast<U>(value);
  const U magnitude = bits & ~kSign;
  const bool is_nan = (magnitude & kExponentMask) == kExponentMask &&
                      (magnitude & kMantissaMask) != 0;
  if (is_nan) return std::numeric_limits<U>::max();

  // Stable sorting treats signed zeros as equivalent, just like operator<. Use the -0 rank for
  // both encodings; the unused adjacent rank is harmless.
  U rank = magnitude == 0 ? float_flip(kSign) : float_flip(bits);
  return ascending ? rank : static_cast<U>(~rank);
}

template <typename K>
auto sortable_key(K value, bool ascending) noexcept {
  if constexpr (std::is_integral_v<K>) return sortable_integral(value, ascending);
  else return sortable_float(value, ascending);
}

template <typename Item, typename U, typename KeyAt>
void lsd_radix(Item* items, Item* scratch, std::size_t n, KeyAt key_at) noexcept {
  if (n < 2) return;
  Item* in = items;
  Item* out = scratch;
  constexpr unsigned kPasses = sizeof(U);
  for (unsigned pass = 0; pass < kPasses; ++pass) {
    const unsigned shift = pass * kBits;
    std::size_t count[kBins];
    multi_histogram_prefix(n, count, [&](std::size_t i) noexcept {
      return static_cast<std::size_t>((key_at(in[i]) >> shift) & static_cast<U>(kMask));
    });
    for (std::size_t i = 0; i < n; ++i) {
      const Item item = in[i];
      const U key = key_at(item);
      out[count[static_cast<std::size_t>((key >> shift) & static_cast<U>(kMask))]++] = item;
    }
    Item* swap = in;
    in = out;
    out = swap;
  }
  if (in != items) {
    for (std::size_t i = 0; i < n; ++i) items[i] = in[i];
  }
}

template <typename K>
void radix_sort_scratch(K* keys, std::size_t n, std::span<K> scratch,
                        SortOrder order) noexcept {
  if (n == 0) return;
  HYLE_FATAL_ASSERT(keys != nullptr);
  HYLE_FATAL_ASSERT(scratch.size() >= n);
  HYLE_FATAL_ASSERT(scratch.data() != nullptr);
  HYLE_FATAL_ASSERT(disjoint(byte_range(keys, n, sizeof(K)),
                             byte_range(scratch.data(), n, sizeof(K))));
  const bool ascending = order == SortOrder::kAscending;
  using U = decltype(sortable_key(keys[0], ascending));
  lsd_radix<K, U>(keys, scratch.data(), n,
                  [ascending](K key) noexcept { return sortable_key(key, ascending); });
}

template <typename K>
void radix_sort_allocating(K* keys, std::size_t n, SortOrder order) {
  if (n == 0) return;
  HYLE_FATAL_ASSERT(keys != nullptr);
  auto scratch = std::make_unique_for_overwrite<K[]>(n);
  radix_sort_scratch(keys, n, std::span<K>(scratch.get(), n), order);
}

template <typename K>
void argsort_scratch(const K* keys, std::uint32_t* indices, std::size_t n,
                     std::span<std::uint32_t> scratch, SortOrder order) noexcept {
  if (n == 0) return;
  HYLE_FATAL_ASSERT(keys != nullptr);
  HYLE_FATAL_ASSERT(indices != nullptr);
  HYLE_FATAL_ASSERT(n - 1 <= std::numeric_limits<std::uint32_t>::max());
  HYLE_FATAL_ASSERT(scratch.size() >= n);
  HYLE_FATAL_ASSERT(scratch.data() != nullptr);
  const ByteRange key_range = byte_range(keys, n, sizeof(K));
  const ByteRange index_range = byte_range(indices, n, sizeof(std::uint32_t));
  const ByteRange scratch_range =
      byte_range(scratch.data(), n, sizeof(std::uint32_t));
  HYLE_FATAL_ASSERT(disjoint(key_range, index_range));
  HYLE_FATAL_ASSERT(disjoint(key_range, scratch_range));
  HYLE_FATAL_ASSERT(disjoint(index_range, scratch_range));
  for (std::size_t i = 0; i < n; ++i) indices[i] = static_cast<std::uint32_t>(i);
  const bool ascending = order == SortOrder::kAscending;
  using U = decltype(sortable_key(keys[0], ascending));
  lsd_radix<std::uint32_t, U>(
      indices, scratch.data(), n,
      [keys, ascending](std::uint32_t index) noexcept {
        return sortable_key(keys[index], ascending);
      });
}

template <typename K>
void argsort_allocating(const K* keys, std::uint32_t* indices, std::size_t n,
                        SortOrder order) {
  if (n == 0) return;
  HYLE_FATAL_ASSERT(keys != nullptr);
  HYLE_FATAL_ASSERT(indices != nullptr);
  HYLE_FATAL_ASSERT(n - 1 <= std::numeric_limits<std::uint32_t>::max());
  auto scratch = std::make_unique_for_overwrite<std::uint32_t[]>(n);
  argsort_scratch(keys, indices, n, std::span<std::uint32_t>(scratch.get(), n), order);
}

}  // namespace

#define HYLE_SIMD_DEFINE_RADIX(T)                                                   \
  void radix_sort(T* keys, std::size_t n, std::span<T> scratch,                   \
                  SortOrder order) noexcept {                                      \
    radix_sort_scratch(keys, n, scratch, order);                                   \
  }                                                                                 \
  void radix_sort(T* keys, std::size_t n, SortOrder order) {                       \
    radix_sort_allocating(keys, n, order);                                         \
  }

HYLE_SIMD_DEFINE_RADIX(std::uint16_t)
HYLE_SIMD_DEFINE_RADIX(std::int16_t)
HYLE_SIMD_DEFINE_RADIX(std::uint32_t)
HYLE_SIMD_DEFINE_RADIX(std::int32_t)
HYLE_SIMD_DEFINE_RADIX(std::uint64_t)
HYLE_SIMD_DEFINE_RADIX(std::int64_t)
HYLE_SIMD_DEFINE_RADIX(float)
HYLE_SIMD_DEFINE_RADIX(double)

#undef HYLE_SIMD_DEFINE_RADIX

#define HYLE_SIMD_DEFINE_ARGSORT(T)                                                \
  void argsort(const T* keys, std::uint32_t* indices, std::size_t n,              \
               std::span<std::uint32_t> scratch, SortOrder order) noexcept {       \
    argsort_scratch(keys, indices, n, scratch, order);                             \
  }                                                                                 \
  void argsort(const T* keys, std::uint32_t* indices, std::size_t n,              \
               SortOrder order) {                                                  \
    argsort_allocating(keys, indices, n, order);                                   \
  }

HYLE_SIMD_DEFINE_ARGSORT(std::uint16_t)
HYLE_SIMD_DEFINE_ARGSORT(std::int16_t)
HYLE_SIMD_DEFINE_ARGSORT(std::uint32_t)
HYLE_SIMD_DEFINE_ARGSORT(std::int32_t)
HYLE_SIMD_DEFINE_ARGSORT(std::uint64_t)
HYLE_SIMD_DEFINE_ARGSORT(std::int64_t)
HYLE_SIMD_DEFINE_ARGSORT(float)
HYLE_SIMD_DEFINE_ARGSORT(double)

#undef HYLE_SIMD_DEFINE_ARGSORT

}  // namespace hyle::simd
