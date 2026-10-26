#pragma once

#include <bit>
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
    // Consume only real input. A finite quantum has one full block and a
    // logarithmic set of tail blocks; an unbounded input retains its one call.
    const auto rows = !bounded() || remaining >= max_rows_
        ? (remaining < max_rows_ ? remaining : max_rows_)
        : std::bit_floor(remaining);
    return {consumed, rows, rows == remaining};
  }

  constexpr std::size_t parts(std::size_t total, std::size_t consumed = 0) const {
    const auto remaining = remaining_rows(total, consumed);
    if (!bounded()) return remaining != 0;
    return remaining / max_rows_ + static_cast<std::size_t>(std::popcount(remaining % max_rows_));
  }

  constexpr bool bounded() const noexcept {
    return max_rows_ != (std::numeric_limits<std::size_t>::max)();
  }

  // Exact parts is not monotonic: Q8 needs one call for 8 rows, three for 7.
  // Cold owners that may adopt an unknown prefix reserve for every remaining
  // length up to this bound, rather than using parts of the complete input.
  constexpr std::size_t parts_bound(std::size_t maximum_remaining) const noexcept {
    if (!bounded()) return maximum_remaining != 0;
    const auto tail = maximum_remaining < max_rows_ ? maximum_remaining : max_rows_ - 1;
    return maximum_remaining / max_rows_ + static_cast<std::size_t>(std::bit_width(tail));
  }

  // Possible exact row/completion keys of any suffix up to this input length.
  // Borrow the visitor only; no shape inventory, payload, provider or execution
  // is retained. Unknown unbounded suffixes continue to use exact admission.
  template<class Visit>
  constexpr void for_each_shape(std::size_t maximum_remaining, Visit &&visit) const {
    if (!maximum_remaining) return;
    if (!bounded()) { visit(maximum_remaining, true); return; }
    if (maximum_remaining >= max_rows_) {
      visit(max_rows_, true);
      if (maximum_remaining > max_rows_) visit(max_rows_, false);
    }
    const auto tail = maximum_remaining < max_rows_ ? maximum_remaining : max_rows_ - 1;
    for (auto rows = std::bit_floor(tail); rows; rows >>= 1) {
      visit(rows, true);
      if (rows > 1 && rows < max_rows_ - 1 && maximum_remaining > rows) visit(rows, false);
    }
  }

private:
  static constexpr std::size_t remaining_rows(std::size_t total, std::size_t consumed) {
    if (consumed > total) throw std::out_of_range("prefill cursor exceeds its input");
    return total - consumed;
  }

  std::size_t max_rows_;
};

} // namespace hyle::inference
