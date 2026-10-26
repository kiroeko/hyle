#pragma once

#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>

#include "batch_indices.h"

namespace hyle::inference {

// Traverse one caller-prepared wave using existing cold index storage. The
// predicates describe stable eligibility/compatibility facts; batch_limit is
// bounded by BatchIndices::prepare(). Every eligible first must match itself.
// execute borrows selected only for this call and returns its actually completed
// prefix, in [1, selected.size()]. An unadmitted tail stays pending until its
// original candidate order is visited; it does not jump ahead of other classes.
//
// Callbacks must not mutate indices or retain its borrowed selection. They own
// admission, payloads, execution, publication, observation and retirement. An
// exception exits immediately: this function neither retries nor rolls back
// work that the caller may already have committed. Valid traversal allocates
// nothing and uses statically resolved calls.
template<class Eligible, class BatchLimit, class Compatible, class Execute>
  requires requires(Eligible &eligible, BatchLimit &batch_limit,
                    Compatible &compatible, Execute &execute, std::size_t index,
                    std::span<const std::size_t> selected) {
    { eligible(index) } -> std::convertible_to<bool>;
    { batch_limit(index) } -> std::same_as<std::size_t>;
    { compatible(index, index) } -> std::convertible_to<bool>;
    { execute(selected) } -> std::same_as<std::size_t>;
  }
void dispatch_batches(BatchIndices &indices, std::size_t candidates,
                      Eligible &&eligible, BatchLimit &&batch_limit,
                      Compatible &&compatible, Execute &&execute) {
  indices.reset(candidates);
  for (std::size_t first = 0; first < candidates; ++first) {
    if (indices.completed(first) || !eligible(first)) continue;
    const auto selected = indices.select(first, batch_limit(first), [&](std::size_t index) {
      return eligible(index) && compatible(first, index);
    });
    if (selected.empty() || selected.front() != first)
      throw std::logic_error("batch compatibility must include its eligible first candidate");
    const auto completed = execute(selected);
    if (!completed)
      throw std::logic_error("batch execution must complete a nonempty selected prefix");
    indices.mark_selected(completed);
  }
}

} // namespace hyle::inference
