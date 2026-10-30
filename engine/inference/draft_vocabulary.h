#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hyle/error/result.h"

namespace hyle::inference {

// Cold target-token IDs only. This value owns no execution or weight resource.
// Canonical target-ID order preserves the original greedy tie order when the
// corresponding head rows are selected; input frequency rank is not row order.
class DraftVocabulary final {
 public:
  static Result<DraftVocabulary> create(std::int64_t full_vocab,
      std::span<const std::int32_t> ids) noexcept {
    if (full_vocab <= 0 || full_vocab > (std::numeric_limits<std::int32_t>::max)())
      return std::unexpected(Status::InvalidArgument("draft vocabulary requires a positive I32 full vocabulary"));
    if (ids.empty())
      return std::unexpected(Status::InvalidArgument("draft vocabulary must contain a target token ID"));
    for (const auto id : ids)
      if (id < 0 || static_cast<std::int64_t>(id) >= full_vocab)
        return std::unexpected(Status::OutOfRange("draft token ID is outside the full vocabulary"));
    try {
      std::vector<std::int32_t> ordered(ids.begin(), ids.end());
      std::sort(ordered.begin(), ordered.end());
      if (std::adjacent_find(ordered.begin(), ordered.end()) != ordered.end())
        return std::unexpected(Status::InvalidArgument("draft vocabulary repeats a target token ID"));
      return DraftVocabulary(full_vocab, std::move(ordered));
    } catch (const std::bad_alloc &) {
      return std::unexpected(Status::ResourceExhausted("draft vocabulary cold ID allocation failed"));
    } catch (const std::length_error &) {
      return std::unexpected(Status::ResourceExhausted("draft vocabulary cold ID count exceeds allocation limits"));
    }
  }

  std::int64_t full_size() const noexcept { return full_vocab_; }
  std::size_t size() const noexcept { return ids_.size(); }
  std::span<const std::int32_t> ids() const noexcept { return ids_; }

 private:
  DraftVocabulary(std::int64_t full_vocab, std::vector<std::int32_t> ids) noexcept
      : full_vocab_(full_vocab), ids_(std::move(ids)) {}
  std::int64_t full_vocab_;
  std::vector<std::int32_t> ids_;
};

} // namespace hyle::inference
