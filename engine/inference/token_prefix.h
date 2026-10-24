#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace hyle::inference {
namespace detail {
inline std::size_t token_plan_add(std::size_t first, std::size_t second) {
  if (second > (std::numeric_limits<std::size_t>::max)() - first)
    throw std::runtime_error("prefix token plan size overflows");
  return first + second;
}
} // namespace detail

struct TokenPlan final {
  std::vector<std::int32_t> ids;
  std::vector<std::uint64_t> hashes;
  std::vector<std::size_t> checkpoints;
};
using TokenPlanRef = std::shared_ptr<const TokenPlan>;
inline TokenPlanRef token_plan(std::span<const std::span<const std::int32_t>> chunks, std::size_t capacity) {
  auto result = std::make_shared<TokenPlan>();
  std::size_t total = 0;
  for (const auto chunk : chunks) total = detail::token_plan_add(total, chunk.size());
  if (!total || total > capacity) throw std::runtime_error("prefix token plan exceeds context capacity");
  result->ids.reserve(total); result->hashes.reserve(detail::token_plan_add(total, 1));
  result->checkpoints.reserve(chunks.size()); result->hashes.push_back(1469598103934665603ULL);
  for (const auto chunk : chunks) {
    for (const auto id : chunk) {
      result->ids.push_back(id);
      result->hashes.push_back((result->hashes.back() ^ static_cast<std::uint32_t>(id)) * 1099511628211ULL);
    }
    result->checkpoints.push_back(result->ids.size()); // Caller-provided complete input boundaries.
  }
  return result;
}
inline TokenPlanRef token_plan(std::span<const std::int32_t> ids, std::size_t capacity) {
  const std::array chunks{ids};
  return token_plan(chunks, capacity);
}
inline bool same_prefix(const TokenPlan &a, const TokenPlan &b, std::size_t count) noexcept {
  return count <= a.ids.size() && count <= b.ids.size() && a.hashes[count] == b.hashes[count] &&
      std::equal(a.ids.begin(), a.ids.begin() + static_cast<std::ptrdiff_t>(count), b.ids.begin());
}

} // namespace hyle::inference
