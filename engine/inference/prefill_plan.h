#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace hyle::inference {

struct PrefillPart final {
  std::size_t begin = 0;
  std::size_t rows = 0;
  bool complete = false;
};

// Shared partition facts for cold capacity/shape planning and actual input
// slicing. The caller owns the input and advances its cursor only on commit.
class PrefillPolicy final {
public:
  explicit constexpr PrefillPolicy(
      std::size_t max_rows = (std::numeric_limits<std::size_t>::max)())
      : max_rows_(max_rows) {
    if (!max_rows_) throw std::invalid_argument("prefill quantum must be positive");
  }

  constexpr PrefillPart next(std::size_t total, std::size_t consumed = 0) const {
    const auto remaining = remaining_rows(total, consumed);
    // Physical bucket capacity is independent of real consumption. A final
    // short chunk is one episode, not a decomposition into exact powers of two.
    const auto rows = remaining < max_rows_ ? remaining : max_rows_;
    return {consumed, rows, rows == remaining};
  }

  constexpr std::size_t parts(std::size_t total, std::size_t consumed = 0) const {
    const auto remaining = remaining_rows(total, consumed);
    // Division before rounding avoids overflow even at SIZE_MAX.
    return remaining / max_rows_ + static_cast<std::size_t>(remaining % max_rows_ != 0);
  }

  constexpr bool bounded() const noexcept {
    return max_rows_ != (std::numeric_limits<std::size_t>::max)();
  }

  // Cold carrier capacities, not exact lengths of every possible suffix.
  // One bucket covers all shorter real chunks when the prepared math supports
  // effective lengths. Exact-only implementations still admit their actual
  // lengths explicitly. No provider, payload or shape inventory is retained.
  template<class Visit>
  constexpr void for_each_capacity(std::size_t maximum_remaining, Visit &&visit) const {
    if (!maximum_remaining) return;
    const auto rows = next(maximum_remaining).rows;
    visit(rows, true);
    if (rows < maximum_remaining) visit(rows, false);
  }

private:
  static constexpr std::size_t remaining_rows(std::size_t total, std::size_t consumed) {
    if (consumed > total) throw std::out_of_range("prefill cursor exceeds its input");
    return total - consumed;
  }

  std::size_t max_rows_;
};

} // namespace hyle::inference
