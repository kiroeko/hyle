#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/result.h"
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

// Explicit execution endpoints. Resources and their CUDA Context are borrowed
// through every compiled artifact, invocation and completion. Host storage is
// exact-context CUDA pinned memory; pageable inputs need explicit staging.
struct Environment final {
  const compute::Context *cpu = nullptr;
  const compute::Context *cuda = nullptr;
  const compute::cuda::MemoryResource *host = nullptr;
  const compute::cuda::MemoryResource *device = nullptr;
  std::size_t host_workspace_budget_bytes = SIZE_MAX;
  std::size_t device_workspace_budget_bytes = SIZE_MAX;
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
struct WeightState;
struct WeightAccess;
} // namespace detail

class Invocation;

struct WeightSpec final {
  std::uint64_t id = 0;
  compute::TensorDesc description{};
};

// Each source is host-accessible, immutable for upload()'s synchronous staging
// copy, and has the exact registered physical description. No quantization or
// conversion is performed here. Repeated IDs in one selection are rejected.
struct WeightFragment final {
  std::uint64_t id = 0;
  compute::ConstTensorView source{};
};

// A selected device batch and its real upload prefix. binding() may describe
// pending bytes: use order() or wait() before consuming them. Graph Invocation's
// typed bind overload retains this batch and records that dependency for you.
// Direct consumers retain the batch through their final compute Completion.
class WeightBatch final {
public:
  WeightBatch() noexcept = default;
  WeightBatch(const WeightBatch &) = delete;
  WeightBatch &operator=(const WeightBatch &) = delete;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API WeightBatch(WeightBatch &&other) noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API WeightBatch &operator=(WeightBatch &&other) noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API ~WeightBatch();
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Result<compute::ConstTensorView>
  binding(std::uint64_t id) const noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Result<compute::CompletionPoll> poll() const noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Status wait() const noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Result<compute::Completion>
  order(compute::Queue &destination) const noexcept;

private:
  friend class WeightSlots;
  friend struct detail::WeightAccess;
  explicit WeightBatch(detail::WeightState *state) noexcept : state_(state) {}
  detail::WeightState *state_ = nullptr;
};

// Two preallocated pinned/device slots, each with capacity_bytes and a fixed
// catalog. IDs select fragments; the engine owns placement/cache/eviction.
// Slot reuse is nonblocking and rejected while an upload, batch or bound
// consumer is still live. upload() never grows metadata or allocation capacity.
// All handles are thread-compatible. Context/resources/allocator are borrowed
// through every batch and in-flight native prefix. close() drains uploads only.
class WeightSlots final {
public:
  WeightSlots() noexcept = default;
  WeightSlots(const WeightSlots &) = delete;
  WeightSlots &operator=(const WeightSlots &) = delete;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API WeightSlots(WeightSlots &&other) noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API WeightSlots &operator=(WeightSlots &&other) noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API ~WeightSlots();
  HYLE_GRAPH_ROUTE_CPU_CUDA_API static Result<WeightSlots>
  create(const compute::Context &context,
         const compute::cuda::MemoryResource &host,
         const compute::cuda::MemoryResource &device,
         std::span<const WeightSpec> catalog, std::size_t capacity_bytes,
         mem::AllocatorRef allocator = mem::system_allocator()) noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Result<WeightBatch>
  upload(std::size_t slot, std::span<const WeightFragment> selection,
         compute::Queue &transfer_queue) noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Status close() noexcept;

private:
  detail::WeightState *states_[2]{};
  bool closed_ = false;
};

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
  bind(ValueId value, const WeightBatch &batch, std::uint64_t id) noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Status reset() noexcept;
  HYLE_GRAPH_ROUTE_CPU_CUDA_API Status
  prepare_replay(ReplayOptions options = {}) noexcept;
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
