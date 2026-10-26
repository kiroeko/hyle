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
    const auto rows = remaining < max_rows_ ? remaining : max_rows_;
    return {consumed, rows, rows == remaining};
  }

  constexpr std::size_t parts(std::size_t total, std::size_t consumed = 0) const {
    const auto remaining = remaining_rows(total, consumed);
    // Avoid remaining + max_rows - 1: both inputs may be SIZE_MAX.
    return remaining / max_rows_ + (remaining % max_rows_ != 0);
  }

private:
  static constexpr std::size_t remaining_rows(std::size_t total, std::size_t consumed) {
    if (consumed > total) throw std::out_of_range("prefill cursor exceeds its input");
    return total - consumed;
  }

  std::size_t max_rows_;
};

} // namespace hyle::inference
