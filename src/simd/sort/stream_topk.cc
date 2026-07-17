#include "hyle/simd/sort.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "src/simd/sort/kernels.h"

// P1-3: streaming partial_sort for k ≪ n.
// Keep keys[0..k) sorted best-first; threshold = keys[k-1]. Scan the rest with a single
// compare against the threshold; rare hits insert into the sorted prefix (O(k) move).
// Random data: after the top-k settles, almost every element is one compare + skip — read-only
// on the main stream, no full-array partition/write like VQPartialSort.
//
// Measured vs VQPartialSort on small-k shapes (dispatch threshold kStreamPartialMaxK).
// Float uses the same NaN-after-finite order as the scalar/vqsort contract.

namespace hyle::simd {
namespace {

// Max k for the streaming path. Above this, heap/VQ paths win (insert cost O(k) per hit).
constexpr std::size_t kStreamPartialMaxK = 64;
// Only when n is large enough that full partition is wasteful.
constexpr std::size_t kStreamPartialMinN = 256;

template <typename T, bool kAscending>
struct StreamOrder {
  bool operator()(T a, T b) const noexcept {
    if constexpr (std::is_floating_point_v<T>) {
      const bool a_nan = std::isnan(a);
      const bool b_nan = std::isnan(b);
      if (a_nan || b_nan) return !a_nan && b_nan;  // finite before NaN
    }
    if constexpr (kAscending) return a < b;
    else return a > b;
  }
};

template <typename T, typename Better>
void stream_partial_sorted(T* keys, std::size_t n, std::size_t k, Better better) noexcept {
  // Sort first k best-first; keys[k-1] is the reject threshold.
  std::sort(keys, keys + k, better);
  for (std::size_t i = k; i < n; ++i) {
    if (!better(keys[i], keys[k - 1])) continue;
    // Preserve multiset/permutation: park the displaced threshold at i, then insert v.
    const T v = keys[i];
    keys[i] = keys[k - 1];
    std::size_t pos = k - 1;
    while (pos > 0 && better(v, keys[pos - 1])) {
      keys[pos] = keys[pos - 1];
      --pos;
    }
    keys[pos] = v;
  }
}

template <typename T>
bool try_stream_partial(T* keys, std::size_t n, std::size_t k, bool ascending) noexcept {
  if (k == 0 || n == 0) return true;
  if (k > kStreamPartialMaxK || n < kStreamPartialMinN || k >= n) return false;
  if (ascending)
    stream_partial_sorted(keys, n, k, StreamOrder<T, true>{});
  else
    stream_partial_sorted(keys, n, k, StreamOrder<T, false>{});
  return true;
}

template <typename KVT>
bool try_stream_partial_kv(KVT* kv, std::size_t n, std::size_t k, bool ascending) noexcept {
  if (k == 0 || n == 0) return true;
  if (k > kStreamPartialMaxK || n < kStreamPartialMinN || k >= n) return false;
  auto lt = [](const KVT& a, const KVT& b) noexcept { return a.key < b.key; };
  auto gt = [](const KVT& a, const KVT& b) noexcept { return a.key > b.key; };
  if (ascending)
    stream_partial_sorted(kv, n, k, lt);
  else
    stream_partial_sorted(kv, n, k, gt);
  return true;
}

}  // namespace

// Called from sort_dispatch; returns true if handled.
namespace detail {

bool stream_partial_u16(std::uint16_t* keys, std::size_t n, std::size_t k, bool asc) noexcept {
  return try_stream_partial(keys, n, k, asc);
}
bool stream_partial_i16(std::int16_t* keys, std::size_t n, std::size_t k, bool asc) noexcept {
  return try_stream_partial(keys, n, k, asc);
}
bool stream_partial_u32(std::uint32_t* keys, std::size_t n, std::size_t k, bool asc) noexcept {
  return try_stream_partial(keys, n, k, asc);
}
bool stream_partial_i32(std::int32_t* keys, std::size_t n, std::size_t k, bool asc) noexcept {
  return try_stream_partial(keys, n, k, asc);
}
bool stream_partial_u64(std::uint64_t* keys, std::size_t n, std::size_t k, bool asc) noexcept {
  return try_stream_partial(keys, n, k, asc);
}
bool stream_partial_i64(std::int64_t* keys, std::size_t n, std::size_t k, bool asc) noexcept {
  return try_stream_partial(keys, n, k, asc);
}
bool stream_partial_f32(float* keys, std::size_t n, std::size_t k, bool asc) noexcept {
  return try_stream_partial(keys, n, k, asc);
}
bool stream_partial_f64(double* keys, std::size_t n, std::size_t k, bool asc) noexcept {
  return try_stream_partial(keys, n, k, asc);
}
bool stream_partial_kv32(KV32Bits* kv, std::size_t n, std::size_t k, bool asc) noexcept {
  return try_stream_partial_kv(kv, n, k, asc);
}
bool stream_partial_kv64(KV64Bits* kv, std::size_t n, std::size_t k, bool asc) noexcept {
  return try_stream_partial_kv(kv, n, k, asc);
}

}  // namespace detail
}  // namespace hyle::simd
