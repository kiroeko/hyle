#pragma once

#include "hyle/compute/resource_leases.h"
#include "hyle/compute/detail/queue_ref.h"
#include "hyle/container/small_vector.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/workspace.h"
#include "hyle/mem/lifetime_plan.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/spin_lock.h"

namespace hyle::graph::detail {
struct InvocationWorkspaceState final : util::RefCounted<InvocationWorkspaceState> {
  explicit InvocationWorkspaceState(mem::AllocatorRef owner) noexcept : allocator(owner), routes(owner) {}
  HYLE_GRAPH_API void destroy() noexcept;
  mem::AllocatorRef allocator;
  container::SmallVector<Executable, 0, mem::AllocatorRef> routes;
  compute::LeasePool pool;
  compute::WriteLease writer;
  // Coordinates only. The private writer/use admission remains authoritative.
  mem::BufferView coordinates;
  compute::LeaseUsePool preparation_uses;
  // Identity of the Queue shared by all currently admitted native writers.
  // The lease's original use counters/completions remain the only retirement
  // authority. Exclusive preparation clears this optional ordering fact.
  mutable util::SpinLock admission_lock;
  compute::detail::QueueRef ordered_queue;
};
struct WorkspaceAccess final {
  static InvocationWorkspace adopt(InvocationWorkspaceState *state) noexcept { return InvocationWorkspace(state); }
  static InvocationWorkspaceState *state(const InvocationWorkspace &owner) noexcept { return owner.state_; }
};

// Original per-Invocation storage, or validated views borrowing a shared root.
// Provider invocations are destroyed before this backing/owner is released.
class WorkspaceInstance final {
 public:
  WorkspaceInstance() noexcept = default;
  explicit WorkspaceInstance(mem::PlannedBuffers owned) noexcept : owned_(std::move(owned)) {}
  WorkspaceInstance(WorkspaceInstance &&) noexcept = default;
  WorkspaceInstance &operator=(WorkspaceInstance &&) noexcept = default;
  WorkspaceInstance(const WorkspaceInstance &) = delete;
  WorkspaceInstance &operator=(const WorkspaceInstance &) = delete;
  static Result<WorkspaceInstance> borrow(const InvocationWorkspace &, const PlanState &,
                                         mem::AllocatorRef) noexcept;
  Result<mem::BufferView> view(std::size_t request) noexcept {
    if (!shared_) return owned_.view(request);
    if (request >= views_.size()) return std::unexpected(Status::OutOfRange("shared workspace request"));
    return views_[request];
  }
  Result<mem::ConstBufferView> view(std::size_t request) const noexcept {
    if (!shared_) return owned_.view(request);
    if (request >= views_.size()) return std::unexpected(Status::OutOfRange("shared workspace request"));
    return mem::ConstBufferView(views_[request]);
  }
  bool shared() const noexcept { return static_cast<bool>(shared_); }
  Status admit(compute::LeaseUseBatch &use,
      const compute::detail::QueueRef *ordered_queue = nullptr,
      bool has_ordered_source = false) const noexcept;
  Result<compute::LeaseUseBatch> begin_preparation() const noexcept;
 private:
  mem::PlannedBuffers owned_;
  util::Ref<InvocationWorkspaceState> shared_;
  container::SmallVector<mem::Buffer, 0, mem::AllocatorRef> empty_backings_;
  container::SmallVector<mem::BufferView, 0, mem::AllocatorRef> views_;
};
} // namespace hyle::graph::detail
