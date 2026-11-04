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

// A request-boundary extension of a frozen base vocabulary. All storage is
// reserved in the cold phase; prepare only consumes already known tokens.
// Padding rows have valid IDs but are masked out before draft selection.
class DraftVocabularyContext final {
 public:
  static Result<DraftVocabularyContext> create(const DraftVocabulary &base,
      std::size_t extra_rows) noexcept {
    if (!extra_rows || extra_rows > static_cast<std::size_t>(base.full_size()) - base.size())
      return std::unexpected(Status::InvalidArgument("draft context capacity exceeds the remaining vocabulary"));
    try {
      DraftVocabularyContext result(base.full_size(), base.size(), extra_rows);
      const auto status = result.prepare(base, {});
      if (!status.ok()) return std::unexpected(status);
      return result;
    } catch (const std::bad_alloc &) {
      return std::unexpected(Status::ResourceExhausted("draft context cold allocation failed"));
    } catch (const std::length_error &) {
      return std::unexpected(Status::ResourceExhausted("draft context capacity exceeds allocation limits"));
    }
  }

  Status prepare(const DraftVocabulary &base, std::span<const std::int32_t> recent) noexcept {
    if (base.full_size() != full_size_ || base.size() != base_rows_ || recent.size() > scratch_.size())
      return Status::InvalidArgument("draft context differs from its cold capacity");
    for (const auto id : recent)
      if (id < 0 || static_cast<std::int64_t>(id) >= base.full_size())
        return Status::OutOfRange("draft context token is outside the full vocabulary");
    // Validate first: a refusal leaves the published host ID/mask pair intact.
    std::size_t count = 0;
    for (const auto id : recent)
      if (!std::binary_search(base.ids().begin(), base.ids().end(), id)) scratch_[count++] = id;
    std::sort(scratch_.begin(), scratch_.begin() + count);
    count = static_cast<std::size_t>(std::unique(scratch_.begin(), scratch_.begin() + count) - scratch_.begin());
    std::merge(base.ids().begin(), base.ids().end(), scratch_.begin(), scratch_.begin() + count, ids_.begin());
    active_rows_ = base_rows_ + count;
    std::fill(ids_.begin() + active_rows_, ids_.end(), base.ids().front());
    std::fill(allowed_.begin(), allowed_.begin() + active_rows_, std::uint8_t{1});
    std::fill(allowed_.begin() + active_rows_, allowed_.end(), std::uint8_t{0});
    return Status::Ok();
  }
  std::span<const std::int32_t> ids() const noexcept { return ids_; }
  std::span<const std::uint8_t> allowed() const noexcept { return allowed_; }
  std::span<std::int32_t> recent_storage() noexcept { return recent_; }
  std::size_t active_rows() const noexcept { return active_rows_; }

 private:
  DraftVocabularyContext(std::int64_t full_size, std::size_t base_rows, std::size_t extra_rows)
      : full_size_(full_size), base_rows_(base_rows), ids_(base_rows + extra_rows), allowed_(base_rows + extra_rows),
        scratch_(extra_rows), recent_(extra_rows) {}
  std::int64_t full_size_;
  std::size_t base_rows_, active_rows_ = 0;
  std::vector<std::int32_t> ids_;
  std::vector<std::uint8_t> allowed_;
  std::vector<std::int32_t> scratch_, recent_;
};

} // namespace hyle::inference
