#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>

#include "hyle/compute/completion_retention.h"
#include "hyle/container/small_vector.h"
#include "hyle/graph/executable.h"

namespace hyle::inference {

// One cold node per serial execution carrier. Only an accepted original
// Completion owns the temporary scope reference. It inherits Graph's actual
// retirement attachment, so observation failure/abandonment cannot release it.
template<class Scope> struct RetainedResources final {
  std::atomic<bool> active{false};
  // Host-owner guard only. Native callbacks publish idleness through active;
  // the next reserve resets this after acquiring that publication.
  bool attached = false;
  std::shared_ptr<Scope> owner;
  hyle::compute::CompletionRetention retention;

  bool pending() const noexcept { return active.load(std::memory_order_acquire); }
  hyle::Status reserve(const std::shared_ptr<Scope> &scope) noexcept {
    if (!scope || pending())
      return hyle::Status::Unavailable("original carrier resource scope is still retained");
    attached = false;
    owner = scope;
    active.store(true, std::memory_order_release);
    return hyle::Status::Ok();
  }
  void reject() noexcept {
    auto held = std::move(owner);
    active.store(false, std::memory_order_release);
  }
  hyle::Status attach(hyle::compute::Completion &completion) noexcept {
    if (!pending() || attached || !owner)
      return hyle::Status::FailedPrecondition("carrier resource scope was not reserved once");
    attached = true;
    const auto result = retention.attach(completion, this, +[](void *opaque) noexcept {
      auto &node = *static_cast<RetainedResources *>(opaque);
      // Move before publication: a host may reuse an idle serial carrier.
      // The local can destroy this node's whole scope; no access follows.
      auto held = std::move(node.owner);
      node.active.store(false, std::memory_order_release);
    });
    if (!result.ok()) { attached = false; reject(); }
    return result;
  }
};

using QueueBindings = hyle::container::SmallVector<hyle::graph::QueueBinding, 2>;

// Cold preparation binds only queues required by the actual compiled lanes.
// Delegated native lanes retain their original endpoint without a Queue shell.
inline hyle::Result<QueueBindings> prepare_queue_bindings(
    std::span<const hyle::graph::QueueBinding> available,
    const hyle::graph::CompileReport &report) noexcept {
  QueueBindings selected;
  for (std::size_t i = 0; i < report.lane_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto lane, report.lane(i));
    if (!lane.queue_required) continue;
    const auto found = std::ranges::find(available, lane.lane,
        &hyle::graph::QueueBinding::lane);
    if (found == available.end())
      return std::unexpected(hyle::Status::FailedPrecondition(
          "compiled graph requires an unavailable execution queue"));
    if (!selected.try_emplace_back(*found))
      return std::unexpected(hyle::Status::ResourceExhausted(
          "compiled graph execution queue binding metadata"));
  }
  return selected;
}

inline std::span<const hyle::graph::QueueBinding> queue_span(
    const QueueBindings &queues) noexcept {
  return {queues.data(), queues.size()};
}

} // namespace hyle::inference
