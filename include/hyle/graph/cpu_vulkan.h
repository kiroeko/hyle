#pragma once

#include "hyle/compute/vulkan_weights.h"
#include "hyle/graph/vulkan.h"
#include "hyle/graph/executable.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#define HYLE_GRAPH_ROUTE_CPU_VULKAN_API
#elif defined(HYLE_GRAPH_ROUTE_CPU_VULKAN_BUILD)
#define HYLE_GRAPH_ROUTE_CPU_VULKAN_API HYLE_EXPORT
#else
#define HYLE_GRAPH_ROUTE_CPU_VULKAN_API HYLE_IMPORT
#endif

#if defined(HYLE_COMPUTE_VULKAN)
#include <array>
#include <memory>
#include <span>
#include <vector>

namespace hyle::sched { class Scheduler; }

namespace hyle::graph::cpu_vulkan {
inline constexpr ExecutionTargetId kVulkanTarget{0};
inline constexpr ExecutionTargetId kCpuTarget{1};

struct Environment final {
  const compute::Context *cpu = nullptr;
  const compute::vulkan::Context *vulkan = nullptr;
  const compute::vulkan::MemoryResource *device = nullptr;
  const compute::vulkan::MemoryResource *staging = nullptr;
  mem::MemoryResourceRef host = mem::system_resource();
  std::size_t host_workspace_budget_bytes = SIZE_MAX;
  std::size_t native_boundary_budget_bytes = SIZE_MAX;
  std::size_t staging_budget_bytes = SIZE_MAX;
};

struct CompileOptions final {
  std::span<const NodePlacement> node_placements{};
  std::span<const ConstantBinding> constants{};
  compute::WorkloadContext workload{};
  compute::ResourceBudget resource_budget{};
  mem::AllocatorRef allocator = mem::system_allocator();
  vulkan::CustomLowering custom{};
};

struct FragmentReport final {
  ExecutionTargetId target{};
  std::size_t node_count = 0;
  graph::CompileReport cpu{};
  vulkan::CompileReport vulkan{};
};
// Independent cold diagnostics. Provider-private bytes remain unknown; neither
// declared view bytes nor observed VkBuffer lengths prove a total memory cap.
struct CompileReport final {
  std::vector<FragmentReport> fragments;
  std::size_t host_workspace_bytes = 0;
  bool has_unknown_native_storage = false;
};
struct ReplayReport final {
  std::size_t cpu_fragments = 0, vulkan_fragments = 0;
  std::size_t host_copy_count = 0, host_copy_bytes = 0;
  std::size_t host_repack_count = 0;
  std::size_t native_completion_wait_count = 0;
  std::size_t weight_preparation_wait_count = 0;
};
struct ResourceReport final {
  // Requested managed bytes in the same scope, including other invocations,
  // migration pools and old leases. A native backing is never charged per view.
  std::array<compute::ResourceBudgetUsage, 3> storage{}; // host, device, staging
  bool has_shared_budget = false;
  bool has_unknown_native_storage = false;
  std::size_t managed_host_workspace_bytes = 0;
  std::size_t managed_native_boundary_bytes = 0;
  std::size_t managed_staging_capacity_bytes = 0;
  compute::UnattributedStorageUsage unattributed{};
  std::size_t other_resource_count = 0, other_live_bytes = 0;
  std::size_t observed_native_backing_bytes = 0;
  std::size_t observed_native_backing_count = 0;
};

namespace detail { struct Plan; struct RunState; struct InvocationState; }
class Invocation;

// Immutable coarse plan. Vulkan is the default; kCpuTarget is an explicit
// placement, never a fallback for a GPU preparation/execution failure. CPU
// fragments keep original CPU Graph rules and Vulkan fragments keep whole native
// regions. Fixed pure DAGs only; effects/symbolic/control edges are rejected.
class Executable final {
public:
  Executable() noexcept = default;
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_GRAPH_ROUTE_CPU_VULKAN_API Result<Invocation>
  create_invocation(sched::Scheduler &scheduler, compute::Queue &cpu_queue) const noexcept;
private:
  friend struct detail::Plan;
  friend HYLE_GRAPH_ROUTE_CPU_VULKAN_API Result<Executable>
  compile(const Graph &, Environment, CompileOptions, CompileReport *) noexcept;
  explicit Executable(std::shared_ptr<const detail::Plan> state) noexcept
      : state_(std::move(state)) {}
  std::shared_ptr<const detail::Plan> state_;
};

HYLE_GRAPH_ROUTE_CPU_VULKAN_API Result<Executable>
compile(const Graph &graph, Environment environment, CompileOptions options = {},
        CompileReport *report = nullptr) noexcept;

// Output references are borrowed through the successful ResultLease. Retain
// that lease through CPU reads or GPU use; copying a native Tensor does not authorize
// overwriting a shared boundary. Caller inputs/resources stay borrowed.
class Results final {
public:
  Results() noexcept = default;
  Results(const Results &) = delete;
  Results &operator=(const Results &) = delete;
  HYLE_GRAPH_ROUTE_CPU_VULKAN_API Results(Results &&) noexcept;
  HYLE_GRAPH_ROUTE_CPU_VULKAN_API Results &operator=(Results &&) noexcept;
  HYLE_GRAPH_ROUTE_CPU_VULKAN_API ~Results();
  HYLE_GRAPH_ROUTE_CPU_VULKAN_API Result<compute::ConstTensorView>
  output(ValueId value) const noexcept;
  HYLE_GRAPH_ROUTE_CPU_VULKAN_API Result<const compute::vulkan::Tensor &>
  tensor(ValueId value) const noexcept;
private:
  friend class Invocation;
  explicit Results(detail::RunState *state) noexcept;
  detail::RunState *state_ = nullptr;
};

// One mutable slot. Independent slots share only the immutable plan. The
// borrowed Scheduler drives CPU coarse progress; GPU keeps its original native
// queue, fences and retirement.
// Bind/reset invalidate preparation. Pending work and old leases prevent slot
// reuse. Destruction never waits; close cancels successors and drains this slot.
// Context/resources/allocator/Scheduler/CPU Queue outlive pending work; backing
// resources/allocator additionally outlive retained result leases.
class Invocation final {
public:
  Invocation() noexcept = default;
  Invocation(const Invocation &) = delete;
  Invocation &operator=(const Invocation &) = delete;
  HYLE_GRAPH_ROUTE_CPU_VULKAN_API Invocation(Invocation &&) noexcept;
  HYLE_GRAPH_ROUTE_CPU_VULKAN_API Invocation &operator=(Invocation &&) noexcept;
  HYLE_GRAPH_ROUTE_CPU_VULKAN_API ~Invocation();
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_GRAPH_ROUTE_CPU_VULKAN_API Status bind(ValueId, compute::ConstTensorView) noexcept;
  HYLE_GRAPH_ROUTE_CPU_VULKAN_API Status bind(ValueId, const compute::vulkan::Tensor &) noexcept;
  HYLE_GRAPH_ROUTE_CPU_VULKAN_API Status
  bind(ValueId, const compute::vulkan::WeightBatch &, std::uint64_t id) noexcept;
  HYLE_GRAPH_ROUTE_CPU_VULKAN_API Status reset() noexcept;
  HYLE_GRAPH_ROUTE_CPU_VULKAN_API Status prepare_replay(ReplayOptions options = {}) noexcept;
  HYLE_GRAPH_ROUTE_CPU_VULKAN_API Result<ReplayReport> replay_report() const noexcept;
  HYLE_GRAPH_ROUTE_CPU_VULKAN_API Result<ResourceReport> resource_report() const noexcept;
  HYLE_GRAPH_ROUTE_CPU_VULKAN_API Result<compute::OwnedResult<Results>> submit() noexcept;
  HYLE_GRAPH_ROUTE_CPU_VULKAN_API Result<compute::ResultLease<Results>> run() noexcept;
  HYLE_GRAPH_ROUTE_CPU_VULKAN_API Status close() noexcept;
private:
  friend class Executable;
  explicit Invocation(detail::InvocationState *state) noexcept : state_(state) {}
  detail::InvocationState *state_ = nullptr;
};
} // namespace hyle::graph::cpu_vulkan
#endif
