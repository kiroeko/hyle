#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "hyle/compute/batch_layout.h"

namespace hyle::inference {

// Reused host integers only. Layout versions synchronously copy these ranges;
// page references, source footprints and all native permissions stay outside.
class PagedLayoutScratch final {
 public:
  PagedLayoutScratch() = default;
  PagedLayoutScratch(const PagedLayoutScratch &) = delete;
  PagedLayoutScratch &operator=(const PagedLayoutScratch &) = delete;
  PagedLayoutScratch(PagedLayoutScratch &&) noexcept = default;
  PagedLayoutScratch &operator=(PagedLayoutScratch &&) noexcept = default;

  // Cold preparation can grow but never lower an already prepared bound.
  // Reserve all storage before resetting the current integer ranges.
  void prepare(std::size_t max_members, std::size_t max_page_refs) {
    if (max_members > integer_max || max_page_refs > integer_max)
      throw std::out_of_range("paged layout scratch capacity exceeds I32");
    const auto members = (std::max)(max_members_, max_members);
    const auto refs = (std::max)(max_page_refs_, max_page_refs);
    query_.reserve(members + 1); kv_.reserve(members + 1);
    pages_.reserve(refs); tails_.reserve(members);
    max_members_ = members; max_page_refs_ = refs;
    reset();
  }

  void reset() noexcept {
    query_.clear(); kv_.clear(); pages_.clear(); tails_.clear();
    // A default or moved-from scratch must not allocate in reset().
    if (query_.capacity() && kv_.capacity()) {
      query_.push_back(0); kv_.push_back(0);
    }
  }

  std::size_t max_page_refs() const noexcept { return max_page_refs_; }

  // Every fallible check precedes mutation. Logical query/KV legality and
  // physical page-ID range remain the original BatchSchema's responsibility.
  // Shared/repeated pages retain their original order; no permission is copied.
  template<class PageRef>
  Status append(std::size_t query_rows, std::size_t visible_rows, std::size_t page_size,
      std::span<const PageRef> pages) noexcept {
    using PageId = std::remove_cvref_t<decltype(std::declval<const PageRef &>().page())>;
    static_assert(std::is_integral_v<PageId> && noexcept(std::declval<const PageRef &>().page()),
        "page references must expose their integer ID without throwing");
    if (query_.empty() || kv_.empty())
      return Status::FailedPrecondition("paged layout scratch has not been cold prepared");
    if (!page_size) return Status::InvalidArgument("paged layout requires a positive page size");
    if (query_rows > integer_max - static_cast<std::size_t>(query_.back()) ||
        visible_rows > integer_max || page_size > integer_max)
      return Status::OutOfRange("paged layout row extent exceeds I32");
    const auto expected_pages = visible_rows / page_size + (visible_rows % page_size != 0);
    if (pages.size() != expected_pages)
      return Status::InvalidArgument("paged layout page count differs from its visible rows");
    if (tails_.size() >= max_members_ || pages.size() > max_page_refs_ - pages_.size())
      return Status::OutOfRange("paged layout exceeds its cold prepared capacity");
    if (query_.size() == query_.capacity() || kv_.size() == kv_.capacity() ||
        tails_.size() == tails_.capacity() || pages.size() > pages_.capacity() - pages_.size())
      return Status::FailedPrecondition("paged layout scratch storage is not prepared for this extent");
    for (const auto &page : pages)
      if (!std::in_range<std::int32_t>(page.page()))
        return Status::OutOfRange("paged layout page ID exceeds I32");
    const auto last = visible_rows ? (visible_rows - 1) % page_size + 1 : 0;
    query_.push_back(query_.back() + static_cast<std::int32_t>(query_rows));
    kv_.push_back(kv_.back() + static_cast<std::int32_t>(pages.size()));
    for (const auto &page : pages) pages_.push_back(static_cast<std::int32_t>(page.page()));
    tails_.push_back(static_cast<std::int32_t>(last));
    return Status::Ok();
  }

  compute::HostBatchLayout layout() const noexcept { return {query_, kv_, pages_, tails_}; }

  // R1 retains its original active-sized schema version. Only the offsets
  // are rebased on the stack; this scratch owns no immutable version.
  Result<compute::BatchLayoutVersion> member_version(const compute::BatchSchema &schema,
      std::size_t index) const noexcept {
    if (index >= tails_.size())
      return std::unexpected(Status::OutOfRange("paged layout member is outside its active range"));
    const auto first = static_cast<std::size_t>(kv_[index]);
    const auto count = static_cast<std::size_t>(kv_[index + 1] - kv_[index]);
    const std::array<std::int32_t, 2> query{0, query_[index + 1] - query_[index]},
        kv{0, static_cast<std::int32_t>(count)};
    return schema.version({query, kv, std::span<const std::int32_t>(pages_).subspan(first, count),
        std::span<const std::int32_t>(tails_).subspan(index, 1)});
  }

 private:
  static constexpr auto integer_max = static_cast<std::size_t>((std::numeric_limits<std::int32_t>::max)());
  std::vector<std::int32_t> query_, kv_, pages_, tails_;
  std::size_t max_members_ = 0, max_page_refs_ = 0;
};

} // namespace hyle::inference
