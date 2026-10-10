#pragma once

#include <cstddef>
#include <cstdint>
#include <array>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/result.h"
#include "hyle/compute/cuda_weights.h"
#include "hyle/graph/executable.h"
#include "hyle/sched/scheduler.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_GRAPH_ROUTE_CPU_CUDA_API
#elif defined(HYLE_GRAPH_ROUTE_CPU_CUDA_BUILD)
#define HYLE_GRAPH_ROUTE_CPU_CUDA_API HYLE_EXPORT
#else
#define HYLE_GRAPH_ROUTE_CPU_CUDA_API HYLE_IMPORT
#endif

namespace hyle::graph::cpu_cuda {

inline constexpr ExecutionTargetId kCudaTarget{0};
inline constexpr ExecutionTargetId kCpuTarget{1};
inline constexpr StorageDomainId kDeviceStorage{0};
inline constexpr StorageDomainId kHostStorage{1};
inline constexpr StorageDomainId kStagingStorage{2};

// Explicit execution endpoints. Resources and their CUDA Context are borrowed
// through every compiled artifact, invocation and completion. CPU math uses
// ordinary host-accessible storage. Only crossing values use the exact-context
// CUDA pinned staging domain; its workspace limit is per Invocation.
struct Environment final {
  const compute::Context *cpu = nullptr;
  const compute::Context *cuda = nullptr;
  mem::MemoryResourceRef host = mem::system_resource();
  const compute::cuda::MemoryResource *staging = nullptr;
  const compute::cuda::MemoryResource *device = nullptr;
  std::size_t host_workspace_budget_bytes = SIZE_MAX;
  std::size_t device_workspace_budget_bytes = SIZE_MAX;
  std::size_t staging_workspace_budget_bytes = SIZE_MAX;
};

// Value snapshot of the exact successful cold preparation. Native ordering
// permits overlap with independent work; neither these counts nor a pending
// handle certify that particular hardware executed copy and math concurrently.
struct ReplayReport final {
  graph::ReplayReport graph{};
  std::size_t staging_copy_count = 0;
  std::size_t staging_copy_bytes = 0;
  std::size_t weight_device_wait_count = 0;
  std::size_t weight_host_wait_count = 0;
};

// Cold shared-scope snapshot, indexed by StorageDomainId, including every
// plan/slot/pool using that budget and old leases. Boundary ranges may already
// be charged: never blindly add them to managed bytes. Context/Queue/Scheduler
// and driver storage are excluded; opaque bounds are not domain totals.
struct ResourceReport final {
  std::array<compute::ResourceBudgetUsage, 3> storage{};
  std::array<std::size_t, 3> boundary_view_bytes{};
  compute::UnattributedStorageUsage unattributed{};
  std::size_t other_resource_count = 0, other_live_bytes = 0;
  bool has_unknown_graph_storage_history = false;
};

// CUDA is the ordinary target; required CPU nodes use kCpuTarget in existing
// NodePlacement. Providers retain their original programs/regions and fixed
// cold rules. This never enables cross-device execution in ordinary compile().
HYLE_GRAPH_ROUTE_CPU_CUDA_API Result<CompiledGraph>
compile(const Graph &graph, Environment environment,
        CompileOptions options = {}) noexcept;

namespace detail {
struct RunState;
struct InvocationState;
} // namespace detail

class Invocation;

// Retains the graph Invocation and its private workspace. Caller-bound inputs,
// outputs, Contexts and resources remain borrowed; a lease cannot make an
// externally mutable allocation immutable. Obtain views after successful
// OwnedResult completion. Keeping any result prevents this slot's next bind.
class Results final {
public:
  Results() noexcept = default;
  Results(const Results &) = delete;
  Results &operator=(const Results &) = delete;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Results(Results &&other) noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Results &operator=(Results &&other) noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API ~Results();
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Result<compute::ConstTensorView>
  output(ValueId value) const noexcept;

private:
  friend class Invocation;
  explicit Results(detail::RunState *state) noexcept;
  detail::RunState *state_ = nullptr;
};

// One fixed mutable slot. Create independent slots from the Executable for
// overlapping runs; neither a pending run nor an old result may be overwritten.
// This handle is thread-compatible. Cancellation is cooperative at coarse
// segment boundaries and never preempts an accepted native kernel.
//
// The Scheduler, Queues, Contexts, resources, allocator and external tensor
// backing are borrowed through pending work; resources/allocator additionally
// outlive result leases. Destruction does not wait: accepted work retains its
// own state. close() is the explicit drain of this slot, not of caller Queues.
class Invocation final {
public:
  Invocation() noexcept = default;
  Invocation(const Invocation &) = delete;
  Invocation &operator=(const Invocation &) = delete;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Invocation(Invocation &&other) noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Invocation &operator=(Invocation &&other) noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API ~Invocation();
  HYLE_GRAPH_ROUTE_CPU_CUDA_API static Result<Invocation>
  create(graph::Invocation &&invocation, sched::Scheduler &scheduler,
         std::span<const QueueBinding> queues,
         mem::AllocatorRef allocator = mem::system_allocator()) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  explicit operator bool() const noexcept { return valid(); }
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Status
  bind(ValueId value, compute::ConstTensorView tensor) noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Status
  bind(ValueId value, compute::TensorView tensor) noexcept;
  // Keeps this selection alive through replay/results. A native CUDA lane
  // records its DMA dependency; a delegated lane preserves an explicit host
  // barrier. Changing the selection requires normal bind/prepare admission.
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Status
  bind(ValueId value, const compute::cuda::WeightBatch &batch, std::uint64_t id) noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Status reset() noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Status
  prepare_replay(ReplayOptions options = {}) noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Result<ReplayReport> replay_report() const noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Result<ResourceReport> resource_report() const noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Result<ReplayDependencyRoute>
  dependency_route(std::size_t segment, std::size_t dependency) const noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Result<compute::OwnedResult<Results>>
  submit() noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Result<compute::ResultLease<Results>>
  run() noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Status close() noexcept;

private:
  explicit Invocation(detail::InvocationState *state) noexcept : state_(state) {}
  detail::InvocationState *state_ = nullptr;
};

} // namespace hyle::graph::cpu_cuda
