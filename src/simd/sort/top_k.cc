#include "hyle/simd/sort.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "hyle/core/compiler.h"

namespace hyle::simd {
namespace {

struct BetterScore {
  bool operator()(const ScoredIndexF32& lhs,
                  const ScoredIndexF32& rhs) const noexcept {
    const bool lhs_nan = std::isnan(lhs.score);
    const bool rhs_nan = std::isnan(rhs.score);
    if (lhs_nan != rhs_nan) return !lhs_nan;
    if (!lhs_nan) {
      if (lhs.score > rhs.score) return true;
      if (lhs.score < rhs.score) return false;
    }
    return lhs.index < rhs.index;
  }
};

struct ByteRange {
  std::uintptr_t begin;
  std::uintptr_t end;
};

ByteRange byte_range(const void* data, std::size_t count,
                     std::size_t item_size) noexcept {
  HYLE_FATAL_ASSERT(item_size != 0);
  HYLE_FATAL_ASSERT(count <=
                    std::numeric_limits<std::size_t>::max() / item_size);
  const std::size_t bytes = count * item_size;
  const auto begin = reinterpret_cast<std::uintptr_t>(data);
  HYLE_FATAL_ASSERT(bytes <= std::numeric_limits<std::uintptr_t>::max());
  HYLE_FATAL_ASSERT(begin <=
                    std::numeric_limits<std::uintptr_t>::max() - bytes);
  return {begin, begin + bytes};
}

bool disjoint(ByteRange lhs, ByteRange rhs) noexcept {
  return lhs.end <= rhs.begin || rhs.end <= lhs.begin;
}

}  // namespace

void top_k_f32(const float* scores, std::size_t n, ScoredIndexF32* result,
               std::size_t k) noexcept {
  HYLE_FATAL_ASSERT(k <= n);
  if (k == 0) return;

  HYLE_FATAL_ASSERT(scores != nullptr && result != nullptr);
  HYLE_FATAL_ASSERT(n - 1 <= std::numeric_limits<std::uint32_t>::max());
  HYLE_FATAL_ASSERT(disjoint(byte_range(scores, n, sizeof(float)),
                             byte_range(result, k, sizeof(ScoredIndexF32))));

  const BetterScore better;
  if (k == 1) {
    ScoredIndexF32 best{scores[0], 0};
    for (std::size_t i = 1; i < n; ++i) {
      const ScoredIndexF32 candidate{scores[i], static_cast<std::uint32_t>(i)};
      if (better(candidate, best)) best = candidate;
    }
    result[0] = best;
    return;
  }

  // Current MoE consumers select at most eight experts. For that bounded
  // shape, maintaining the already-sorted result is O(n) with smaller state
  // and fewer branches than a general heap. This is a fixed consumer-derived
  // algorithm boundary, not a size-dependent public route or metadata field.
  if (k <= 8) {
    std::size_t retained = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const ScoredIndexF32 candidate{scores[i], static_cast<std::uint32_t>(i)};
      if (retained == k && !better(candidate, result[k - 1])) continue;
      std::size_t position = retained < k ? retained++ : k - 1;
      while (position > 0 && better(candidate, result[position - 1])) {
        result[position] = result[position - 1];
        --position;
      }
      result[position] = candidate;
    }
    return;
  }

  for (std::size_t i = 0; i < k; ++i) {
    result[i] = {scores[i], static_cast<std::uint32_t>(i)};
  }
  if (k == n) {
    std::sort(result, result + k, better);
    return;
  }
  // BetterScore acts like std::greater: the least desirable retained item is
  // at the root, so each remaining candidate needs one comparison before the
  // O(log k) replacement path.
  std::make_heap(result, result + k, better);
  for (std::size_t i = k; i < n; ++i) {
    const ScoredIndexF32 candidate{scores[i], static_cast<std::uint32_t>(i)};
    if (!better(candidate, result[0])) continue;
    std::pop_heap(result, result + k, better);
    result[k - 1] = candidate;
    std::push_heap(result, result + k, better);
  }
  std::sort_heap(result, result + k, better);
}

}  // namespace hyle::simd
