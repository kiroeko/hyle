#include "hyle/simd/sort.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <new>
#include <type_traits>

// LSD radix sort (base 256) + argsort. Stable by construction (counting sort per digit).
// Histogram uses kHistWays private 256-bin counters (unrolled) then reduces — cuts store-port
// conflicts on ++count[digit] without needing SIMD conflict detection.

namespace hyle::simd {
namespace {

constexpr unsigned kBits = 8;
constexpr unsigned kBins = 1u << kBits;  // 256
constexpr unsigned kMask = kBins - 1;
constexpr unsigned kHistWays = 4;

// 4-way private histograms → one count[256], then in-place prefix into the same buffer.
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

template <typename U>
bool lsd_radix_unsigned(U* keys, std::size_t n) noexcept {
  if (n < 2) return true;
  U* tmp = static_cast<U*>(::operator new(n * sizeof(U), std::nothrow));
  if (tmp == nullptr) return false;
  U* in = keys;
  U* out = tmp;
  constexpr unsigned kPasses = static_cast<unsigned>(sizeof(U) * 8 / kBits);
  for (unsigned pass = 0; pass < kPasses; ++pass) {
    const unsigned shift = pass * kBits;
    std::size_t count[kBins];
    multi_histogram_prefix(n, count, [&](std::size_t i) noexcept {
      return static_cast<std::size_t>(in[i] >> shift) & kMask;
    });
    // Direct scatter wins e2e vs per-bin buffered (kScatterBuf=16): Win 1M u32
    // 6.28ms direct vs 7.56ms buffered (~20% tax from double write + flush branch).
    for (std::size_t i = 0; i < n; ++i) {
      const U k = in[i];
      out[count[(static_cast<std::size_t>(k >> shift) & kMask)]++] = k;
    }
    U* swap = in;
    in = out;
    out = swap;
  }
  if (in != keys) std::memcpy(keys, in, n * sizeof(U));
  ::operator delete(tmp);
  return true;
}

inline std::uint32_t to_sortable_u32(std::uint32_t x, bool ascending) noexcept {
  return ascending ? x : ~x;
}
inline std::uint32_t to_sortable_i32(std::int32_t x, bool ascending) noexcept {
  const std::uint32_t u = static_cast<std::uint32_t>(x) ^ 0x80000000u;
  return ascending ? u : ~u;
}
inline std::uint64_t to_sortable_i64(std::int64_t x, bool ascending) noexcept {
  const std::uint64_t u = static_cast<std::uint64_t>(x) ^ 0x8000000000000000ull;
  return ascending ? u : ~u;
}
inline std::uint32_t float_flip(std::uint32_t f) noexcept {
  const std::uint32_t mask =
      static_cast<std::uint32_t>(-static_cast<std::int32_t>(f >> 31)) | 0x80000000u;
  return f ^ mask;
}
inline std::uint32_t float_unflip(std::uint32_t f) noexcept {
  const std::uint32_t mask = ((f >> 31) - 1u) | 0x80000000u;
  return f ^ mask;
}
inline std::uint32_t to_sortable_f32(std::uint32_t bits, bool ascending) noexcept {
  const std::uint32_t s = float_flip(bits);
  return ascending ? s : ~s;
}
inline std::uint32_t from_sortable_f32(std::uint32_t s, bool ascending) noexcept {
  const std::uint32_t t = ascending ? s : ~s;
  return float_unflip(t);
}

template <typename T>
void std_sort_order(T* keys, std::size_t n, SortOrder order) noexcept {
  if (order == SortOrder::kAscending) std::sort(keys, keys + n);
  else std::sort(keys, keys + n, std::greater<T>());
}

// argsort: LSD on indices, digit from keys[idx] after sortable map.
template <typename K, typename ToSortableU>
void argsort_impl(const K* keys, std::uint32_t* indices, std::size_t n, bool ascending,
                  ToSortableU map) noexcept {
  if (n == 0) return;
  for (std::size_t i = 0; i < n; ++i) indices[i] = static_cast<std::uint32_t>(i);
  if (n < 2) return;

  using U = decltype(map(keys[0], true));
  std::uint32_t* tmp =
      static_cast<std::uint32_t*>(::operator new(n * sizeof(std::uint32_t), std::nothrow));
  if (tmp == nullptr) {
    std::stable_sort(indices, indices + n, [&](std::uint32_t a, std::uint32_t b) {
      const U ka = map(keys[a], ascending);
      const U kb = map(keys[b], ascending);
      return ka < kb;
    });
    return;
  }

  std::uint32_t* in = indices;
  std::uint32_t* out = tmp;
  constexpr unsigned kPasses = static_cast<unsigned>(sizeof(U) * 8 / kBits);
  for (unsigned pass = 0; pass < kPasses; ++pass) {
    const unsigned shift = pass * kBits;
    std::size_t count[kBins];
    multi_histogram_prefix(n, count, [&](std::size_t i) noexcept {
      const U sk = map(keys[in[i]], ascending);
      return static_cast<std::size_t>(sk >> shift) & kMask;
    });
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint32_t idx = in[i];
      const U sk = map(keys[idx], ascending);
      out[count[(static_cast<std::size_t>(sk >> shift) & kMask)]++] = idx;
    }
    std::uint32_t* swap = in;
    in = out;
    out = swap;
  }
  if (in != indices) std::memcpy(indices, in, n * sizeof(std::uint32_t));
  ::operator delete(tmp);
}

}  // namespace

void radix_sort(std::uint32_t* keys, std::size_t n, SortOrder order) noexcept {
  if (n < 2) return;
  const bool asc = order == SortOrder::kAscending;
  if (!asc) {
    for (std::size_t i = 0; i < n; ++i) keys[i] = ~keys[i];
  }
  if (!lsd_radix_unsigned(keys, n)) {
    if (!asc) {
      for (std::size_t i = 0; i < n; ++i) keys[i] = ~keys[i];
    }
    std_sort_order(keys, n, order);
    return;
  }
  if (!asc) {
    for (std::size_t i = 0; i < n; ++i) keys[i] = ~keys[i];
  }
}

void radix_sort(std::int32_t* keys, std::size_t n, SortOrder order) noexcept {
  if (n < 2) return;
  const bool asc = order == SortOrder::kAscending;
  auto* u = reinterpret_cast<std::uint32_t*>(keys);
  for (std::size_t i = 0; i < n; ++i) u[i] = to_sortable_i32(keys[i], asc);
  if (!lsd_radix_unsigned(u, n)) {
    // Restore original bits then comparison-sort.
    for (std::size_t i = 0; i < n; ++i) {
      std::uint32_t x = u[i];
      if (!asc) x = ~x;
      keys[i] = static_cast<std::int32_t>(x ^ 0x80000000u);
    }
    std_sort_order(keys, n, order);
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    std::uint32_t x = u[i];
    if (!asc) x = ~x;
    keys[i] = static_cast<std::int32_t>(x ^ 0x80000000u);
  }
}

void radix_sort(std::uint64_t* keys, std::size_t n, SortOrder order) noexcept {
  if (n < 2) return;
  const bool asc = order == SortOrder::kAscending;
  if (!asc) {
    for (std::size_t i = 0; i < n; ++i) keys[i] = ~keys[i];
  }
  if (!lsd_radix_unsigned(keys, n)) {
    if (!asc) {
      for (std::size_t i = 0; i < n; ++i) keys[i] = ~keys[i];
    }
    std_sort_order(keys, n, order);
    return;
  }
  if (!asc) {
    for (std::size_t i = 0; i < n; ++i) keys[i] = ~keys[i];
  }
}

void radix_sort(std::int64_t* keys, std::size_t n, SortOrder order) noexcept {
  if (n < 2) return;
  const bool asc = order == SortOrder::kAscending;
  auto* u = reinterpret_cast<std::uint64_t*>(keys);
  for (std::size_t i = 0; i < n; ++i) u[i] = to_sortable_i64(keys[i], asc);
  if (!lsd_radix_unsigned(u, n)) {
    for (std::size_t i = 0; i < n; ++i) {
      std::uint64_t x = u[i];
      if (!asc) x = ~x;
      keys[i] = static_cast<std::int64_t>(x ^ 0x8000000000000000ull);
    }
    std_sort_order(keys, n, order);
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    std::uint64_t x = u[i];
    if (!asc) x = ~x;
    keys[i] = static_cast<std::int64_t>(x ^ 0x8000000000000000ull);
  }
}

void radix_sort(float* keys, std::size_t n, SortOrder order) noexcept {
  if (n < 2) return;
  const bool asc = order == SortOrder::kAscending;
  auto* u = reinterpret_cast<std::uint32_t*>(keys);
  for (std::size_t i = 0; i < n; ++i) u[i] = to_sortable_f32(u[i], asc);
  if (!lsd_radix_unsigned(u, n)) {
    for (std::size_t i = 0; i < n; ++i) u[i] = from_sortable_f32(u[i], asc);
    std_sort_order(keys, n, order);
    return;
  }
  for (std::size_t i = 0; i < n; ++i) u[i] = from_sortable_f32(u[i], asc);
}

void argsort(const std::uint32_t* keys, std::uint32_t* indices, std::size_t n,
             SortOrder order) noexcept {
  const bool asc = order == SortOrder::kAscending;
  argsort_impl(keys, indices, n, asc,
               [](std::uint32_t k, bool a) { return to_sortable_u32(k, a); });
}

void argsort(const std::int32_t* keys, std::uint32_t* indices, std::size_t n,
             SortOrder order) noexcept {
  const bool asc = order == SortOrder::kAscending;
  argsort_impl(keys, indices, n, asc,
               [](std::int32_t k, bool a) { return to_sortable_i32(k, a); });
}

void argsort(const float* keys, std::uint32_t* indices, std::size_t n, SortOrder order) noexcept {
  const bool asc = order == SortOrder::kAscending;
  argsort_impl(keys, indices, n, asc, [](float k, bool a) {
    std::uint32_t bits;
    std::memcpy(&bits, &k, sizeof(bits));
    return to_sortable_f32(bits, a);
  });
}

}  // namespace hyle::simd
