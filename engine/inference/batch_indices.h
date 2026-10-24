#pragma once

#include <algorithm>
#include <cstddef>
#include <new>
#include <span>
#include <stdexcept>

#include "hyle/container/small_vector.h"

namespace hyle::inference {

// Reused host indices only. Callers own compatibility, admission, execution
// and all payloads; completion may mark only a resource-admitted prefix.
class BatchIndices final {
public:
  BatchIndices() = default;
  BatchIndices(const BatchIndices &) = delete;
  BatchIndices &operator=(const BatchIndices &) = delete;
  void prepare(std::size_t max_candidates, std::size_t max_batch) {
    if (!max_candidates || !max_batch || max_batch > max_candidates)
      throw std::invalid_argument("batch indices require positive candidate/batch bounds with batch <= candidates");
    if (!done_.try_reserve(max_candidates) || !selected_.try_reserve(max_batch)) throw std::bad_alloc{};
    max_candidates_ = max_candidates;
    max_batch_ = max_batch;
    done_.clear(); selected_.clear();
  }
  void reset(std::size_t candidates) {
    if (!max_batch_ || candidates > max_candidates_)
      throw std::out_of_range("batch candidates exceed the cold-prepared index bound");
    done_.resize(candidates);
    std::fill(done_.begin(), done_.end(), char{});
    selected_.clear();
  }
  bool completed(std::size_t index) const {
    if (index >= done_.size()) throw std::out_of_range("batch completion index is outside its candidates");
    return done_[index] != 0;
  }
  // The borrowed selection remains valid until the next prepare/reset/select.
  template<class Compatible>
  std::span<const std::size_t> select(std::size_t first, std::size_t limit, Compatible compatible) {
    if (!limit || limit > max_batch_)
      throw std::out_of_range("batch selection exceeds the cold-prepared batch bound");
    if (completed(first)) throw std::logic_error("batch selection starts at an already completed candidate");
    selected_.clear();
    for (auto index = first; index < done_.size() && selected_.size() < limit; ++index)
      if (!done_[index] && compatible(index)) selected_.push_back(index);
    return {selected_.data(), selected_.size()};
  }
  void mark_selected(std::size_t completed_count) {
    if (completed_count > selected_.size()) throw std::out_of_range("batch completion exceeds its selected prefix");
    for (std::size_t i = 0; i < completed_count; ++i) done_[selected_[i]] = 1;
  }

private:
  container::SmallVector<char, 4> done_;
  container::SmallVector<std::size_t, 4> selected_;
  std::size_t max_candidates_ = 0, max_batch_ = 0;
};

} // namespace hyle::inference
