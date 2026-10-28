#pragma once

#include <cstddef>
#include <cstdint>
#include <climits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hyle/compute/resource_budget.h"
#include "hyle/mem/std_allocator.h"

namespace hyle::inference {

// Derived metadata only. The original ledger remains the sole token history.
// Allocations are charged to its model's original metadata budget; this cache
// owns no Session, payload permission, Context or execution controller.
class HistoryCounts final {
  compute::ResourceBudget budget_;
  std::vector<std::int32_t, mem::StdAllocator<std::int32_t>> counts_;
  std::size_t consumed_length_ = 0, ordinal_ = 0;

  static std::size_t checked_vocabulary(std::size_t vocabulary) {
    if (!vocabulary || vocabulary > static_cast<std::size_t>(INT32_MAX))
      throw std::invalid_argument("history counts require a finite I32 vocabulary");
    return vocabulary;
  }
  void validate(std::span<const std::int32_t> ids) const {
    if (ids.size() > static_cast<std::size_t>(INT32_MAX) - consumed_length_)
      throw std::out_of_range("history count length exceeds I32");
    for (const auto id : ids)
      if (id < 0 || static_cast<std::size_t>(id) >= counts_.size())
        throw std::out_of_range("history token is outside vocabulary");
  }
public:
  HistoryCounts(std::size_t vocabulary, compute::ResourceBudget budget, mem::AllocatorRef allocator)
      : budget_(std::move(budget)), counts_(checked_vocabulary(vocabulary), 0, mem::StdAllocator<std::int32_t>(allocator)) {}
  std::size_t vocabulary() const noexcept { return counts_.size(); }
  std::size_t consumed_length() const noexcept { return consumed_length_; }
  std::size_t ordinal() const noexcept { return ordinal_; }
  std::span<const std::int32_t> values() const noexcept { return counts_; }
  void append_committed(std::span<const std::int32_t> ids) {
    validate(ids);
    for (const auto id : ids) ++counts_[static_cast<std::size_t>(id)];
    consumed_length_ += ids.size(); ++ordinal_;
  }
  // A synchronous C0 upload temporarily includes this episode's real input.
  // Failure rolls back that local contribution; no ledger ordinal is changed.
  class Conditional final {
    HistoryCounts &cache_;
    std::span<const std::int32_t> ids_;
  public:
    Conditional(HistoryCounts &cache, std::span<const std::int32_t> ids) : cache_(cache), ids_(ids) {
      cache_.validate(ids_);
      for (const auto id : ids_) ++cache_.counts_[static_cast<std::size_t>(id)];
    }
    Conditional(const Conditional &) = delete;
    Conditional &operator=(const Conditional &) = delete;
    ~Conditional() { for (const auto id : ids_) --cache_.counts_[static_cast<std::size_t>(id)]; }
  };
  Conditional condition(std::span<const std::int32_t> ids) { return Conditional(*this, ids); }
};

} // namespace hyle::inference
