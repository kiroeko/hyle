#pragma once

#include "hyle/compute/resource_leases.h"
#include "hyle/graph/executable.h"

namespace hyle::graph {
namespace detail { struct LeasedInvocationState; }

struct LeasedInvocationOptions final {
  mem::AllocatorRef allocator = mem::system_allocator();
  // Wrapper metadata only. Native workspace/program limits still belong to
  // the original Executable's CompileOptions resource/storage budgets.
  compute::ResourceBudget resource_budget{};
};

// A ready readonly view plus its original Invocation/backing owner. Every
// copy prevents replay/reset/rebind of that Invocation. A borrowed tensor
// remains valid only while a result handle or its exported storage lease lives.
class LeasedGraphOutput final {
public:
  LeasedGraphOutput() noexcept = default;
  HYLE_GRAPH_API LeasedGraphOutput(const LeasedGraphOutput &) noexcept;
  HYLE_GRAPH_API LeasedGraphOutput &operator=(const LeasedGraphOutput &) noexcept;
  HYLE_GRAPH_API LeasedGraphOutput(LeasedGraphOutput &&) noexcept;
  HYLE_GRAPH_API LeasedGraphOutput &operator=(LeasedGraphOutput &&) noexcept;
  HYLE_GRAPH_API ~LeasedGraphOutput();
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_GRAPH_API Result<compute::ConstTensorView> tensor() const noexcept;
  // Caller/source-alias backing can flow to lease-aware Direct Compute.
  // Private workspace/Executable constants keep their original Graph owner;
  // they cannot be represented as an independently owned external ReadLease.
  HYLE_GRAPH_API Result<compute::ReadLease> storage_lease() const noexcept;
  HYLE_GRAPH_API void reset() noexcept;
private:
  friend class LeasedInvocation;
  LeasedGraphOutput(detail::LeasedInvocationState *state,
                    compute::ConstTensorView tensor) noexcept;
  detail::LeasedInvocationState *state_ = nullptr;
  compute::ConstTensorView tensor_{};
};

// Opt-in fixed-capacity bindings over an existing compiled plan. Context,
// resources, Queues, compile-time borrowed constants/effect footprints and
// injected allocator outlive this owner, all submitted
// native work and results. Control is thread-compatible; distinct readonly
// result/lease handles may be copied/dropped under their native lifetime rules.
class LeasedInvocation final {
public:
  LeasedInvocation() noexcept = default;
  LeasedInvocation(const LeasedInvocation &) = delete;
  LeasedInvocation &operator=(const LeasedInvocation &) = delete;
  HYLE_GRAPH_API LeasedInvocation(LeasedInvocation &&) noexcept;
  HYLE_GRAPH_API LeasedInvocation &operator=(LeasedInvocation &&) noexcept;
  HYLE_GRAPH_API ~LeasedInvocation();
  HYLE_GRAPH_API static Result<LeasedInvocation>
  create(const Executable &executable, LeasedInvocationOptions options = {},
         std::span<const ShapeBinding> shape = {}) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_GRAPH_API bool in_flight() const noexcept;
  HYLE_GRAPH_API Status bind(ValueId value, compute::TensorDesc description,
                             compute::ReadLease lease) noexcept;
  // Takes exclusive ownership on success only. Inputs or caller outputs are
  // allowed; original Graph alias/description/resource checks still apply.
  HYLE_GRAPH_API Status bind(ValueId value, compute::TensorDesc description,
                             compute::WriteLease &&lease) noexcept;
  HYLE_GRAPH_API Status prepare_replay(std::span<const QueueBinding> queues,
                                       ReplayOptions options = {}) noexcept;
  HYLE_GRAPH_API ReplayRoute replay_route() const noexcept;
  HYLE_GRAPH_API Result<ReplayReport> replay_report() const noexcept;
  HYLE_GRAPH_API Status reset() noexcept;
  HYLE_GRAPH_API Result<compute::Completion> submit() noexcept;
  HYLE_GRAPH_API Result<compute::Completion> submit(compute::Queue &queue) noexcept;
  HYLE_GRAPH_API Result<compute::Completion> submit(std::span<const QueueBinding> queues) noexcept;
  HYLE_GRAPH_API Result<LeasedGraphOutput> output(ValueId value) noexcept;
  // Control and public Completion observations are externally serialized.
  // Observe/drop the old public Completion before reusing its one cold slot.
  // Stops new admission without a device-wide wait. Existing results survive.
  HYLE_GRAPH_API Status close() noexcept;
  HYLE_GRAPH_API Status drain() noexcept;
private:
  explicit LeasedInvocation(detail::LeasedInvocationState *state) noexcept : state_(state) {}
  detail::LeasedInvocationState *state_ = nullptr;
};
} // namespace hyle::graph
