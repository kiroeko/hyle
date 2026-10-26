#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <utility>
#include <variant>

#include "host_storage.h"
#include "hyle/compute/file_stream.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/graph/batch.h"
#include "hyle/mem/buffer.h"

#if defined(HYLE_COMPUTE_CPU)
#include "hyle/compute/cpu_file_stream.h"
#include "hyle/graph/cpu.h"
#endif
#if defined(HYLE_COMPUTE_CUDA)
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/cuda_file_stream.h"
#include "hyle/graph/cuda.h"
#endif
#if defined(HYLE_COMPUTE_APPLE) && defined(__APPLE__)
#include "hyle/compute/apple_file_stream.h"
#include "hyle/compute/apple_memory.h"
#include "hyle/graph/apple.h"
#endif
#if defined(HYLE_COMPUTE_VULKAN)
#include "hyle/compute/vulkan_file_stream.h"
#include "hyle/compute/vulkan_memory.h"
#include "hyle/graph/vulkan.h"
#endif
#if !defined(HYLE_COMPUTE_CPU) && !defined(HYLE_COMPUTE_CUDA) && !defined(HYLE_COMPUTE_VULKAN) && !(defined(HYLE_COMPUTE_APPLE) && defined(__APPLE__))
#error "Hyle inference requires a CPU, Apple, CUDA or Vulkan backend"
#endif

namespace hyle::compute::cpu { struct DefaultContextOptions; }

namespace hyle::inference {

enum class BackendKind : std::uint8_t { kDefault, kCpu, kCuda, kApple, kVulkan };

struct BackendSelection final {
  BackendKind primary = BackendKind::kDefault;
  bool cpu_assist = false;
};

struct WorkspaceBudgets final {
  std::size_t device = SIZE_MAX, host = SIZE_MAX, staging = SIZE_MAX;
};

// Cold endpoint selection only. The selected original Contexts/queues own all
// native work; Graph/Compute still perform planning, execution and retirement.
class Backend final {
private:
#if defined(HYLE_COMPUTE_CPU)
  struct CpuEndpoint final {
    compute::Context context;
    mem::MemoryResourceRef memory;
    compute::Queue queue;
    graph::ExecutionSource source;
  };
#endif
#if defined(HYLE_COMPUTE_CUDA)
  struct CudaEndpoint final {
    compute::Context context;
    compute::StorageOwner memory, mixed_staging;
    compute::cuda::MemoryResource staging;
    compute::Queue queue;
    graph::ExecutionSource source;
  };
#endif
#if defined(HYLE_COMPUTE_APPLE) && defined(__APPLE__)
  struct AppleEndpoint final {
    compute::apple::Context context;
    compute::apple::MemoryResource memory, staging;
    graph::ExecutionSource source;
  };
#endif
#if defined(HYLE_COMPUTE_VULKAN)
  struct VulkanEndpoint final {
    compute::vulkan::Context context;
    compute::vulkan::MemoryResource memory, staging;
    graph::ExecutionSource source;
  };
#endif
  struct State final {
    // Independent identity only. Escaped typed host storage may retain this
    // empty token without keeping endpoints, model state or weights alive.
    std::shared_ptr<const void> host_storage_identity;
    // Endpoints are constructed in place once. Their Contexts are declared
    // first and destroyed last; cached sources/owners below die before them.
    std::variant<std::monostate
#if defined(HYLE_COMPUTE_CPU)
        , CpuEndpoint
#endif
#if defined(HYLE_COMPUTE_CUDA)
        , CudaEndpoint
#endif
#if defined(HYLE_COMPUTE_APPLE) && defined(__APPLE__)
        , AppleEndpoint
#endif
#if defined(HYLE_COMPUTE_VULKAN)
        , VulkanEndpoint
#endif
        > primary;
#if defined(HYLE_COMPUTE_CPU)
    std::optional<CpuEndpoint> cpu;
#endif
    BackendSelection selection;
    mem::MemoryResourceRef device, host, staging;
    std::array<graph::StorageDomain, 3> domains{};
    std::size_t domain_count = 0;
    std::array<graph::ExecutionTarget, 2> targets{};
    std::size_t target_count = 0;
    std::array<graph::ExecutionLane, 2> lanes{};
    std::size_t lane_count = 0;
    std::array<graph::QueueBinding, 2> queues{};
    std::size_t queue_count = 0;
    // Only Vulkan requires explicit native collection/drain. These point to
    // its original stable Context, never a temporary visitor or Backend.
    void *retirement_context = nullptr;
    Status (*collect)(void *) noexcept = [](void *) noexcept { return Status::Ok(); };
    Status (*drain)(void *, graph::Invocation &) noexcept =
        [](void *, graph::Invocation &run) noexcept { return run.drain(); };
  };

public:
  Backend() noexcept = default;
  Backend(const Backend &) = delete;
  Backend &operator=(const Backend &) = delete;
  Backend(Backend &&) noexcept = default;
  Backend &operator=(Backend &&) noexcept = default;

  static std::span<const BackendKind> compiled_backends() noexcept {
    static constexpr BackendKind kinds[]{
#if defined(HYLE_COMPUTE_CPU)
        BackendKind::kCpu,
#endif
#if defined(HYLE_COMPUTE_CUDA)
        BackendKind::kCuda,
#endif
#if defined(HYLE_COMPUTE_APPLE) && defined(__APPLE__)
        BackendKind::kApple,
#endif
#if defined(HYLE_COMPUTE_VULKAN)
        BackendKind::kVulkan,
#endif
    };
    return kinds;
  }

  static Result<BackendSelection> default_selection() noexcept {
    BackendKind gpu = BackendKind::kDefault;
    std::size_t gpu_count = 0;
    bool cpu = false;
    for (const auto kind : compiled_backends()) {
      if (kind == BackendKind::kCpu) cpu = true;
      else { gpu = kind; ++gpu_count; }
    }
    if (gpu_count > 1)
      return std::unexpected(Status::InvalidArgument(
          "multiple GPU backends are compiled; select the primary backend explicitly"));
    if (gpu_count == 1) return BackendSelection{gpu, false};
    if (cpu) return BackendSelection{BackendKind::kCpu, false};
    return std::unexpected(Status::Unimplemented("no inference backend is compiled"));
  }

  static Result<Backend> create(BackendSelection selection = {},
                                WorkspaceBudgets budgets = {}) noexcept {
    return create_impl(selection, budgets, nullptr);
  }

  // Cold CPU endpoint configuration. The original allocator/executor backings
  // remain borrowed through every derived artifact and native use. Options
  // require a CPU primary or an explicitly selected CPU assistance endpoint;
  // they never create one implicitly or reconfigure a shared runtime.
  static Result<Backend> create(BackendSelection selection,
                                WorkspaceBudgets budgets,
                                const compute::cpu::DefaultContextOptions &cpu_options) noexcept {
    return create_impl(selection, budgets, &cpu_options);
  }

private:
  static Result<Backend> create_impl(BackendSelection selection,
                                     WorkspaceBudgets budgets,
                                     const compute::cpu::DefaultContextOptions *cpu_options) noexcept {
    if (selection.primary == BackendKind::kDefault) {
      HYLE_ASSIGN_OR_RETURN(auto chosen, default_selection());
      selection.primary = chosen.primary;
    }
    switch (selection.primary) {
    case BackendKind::kCpu: case BackendKind::kCuda:
    case BackendKind::kApple: case BackendKind::kVulkan: break;
    default: return std::unexpected(Status::InvalidArgument("backend selection kind is invalid"));
    }
    if (cpu_options != nullptr) {
#if defined(HYLE_COMPUTE_CPU)
      if (selection.primary != BackendKind::kCpu && !selection.cpu_assist)
        return std::unexpected(Status::InvalidArgument("CPU options require a selected CPU endpoint"));
#else
      return std::unexpected(Status::Unimplemented("CPU options require the compiled CPU backend"));
#endif
    }
    if (!is_compiled(selection.primary))
      return std::unexpected(Status::Unimplemented("selected backend is not compiled into this image"));
    if (selection.cpu_assist && selection.primary == BackendKind::kCpu)
      return std::unexpected(Status::InvalidArgument("CPU assistance requires a native GPU primary"));
    if (selection.cpu_assist && !is_compiled(BackendKind::kCpu))
      return std::unexpected(Status::Unimplemented("CPU assistance requires the compiled CPU backend"));

    Backend result;
    result.state_.reset(new (std::nothrow) State);
    if (!result.state_)
      return std::unexpected(Status::ResourceExhausted("cannot allocate the cold backend state"));
    auto &state = *result.state_;
    state.selection = selection;
    if (selection.primary == BackendKind::kApple || selection.primary == BackendKind::kVulkan) {
      try { state.host_storage_identity = std::make_shared<const unsigned char>(0); }
      catch (const std::bad_alloc &) {
        return std::unexpected(Status::ResourceExhausted("native Backend identity metadata"));
      }
    }
    compute::StorageOwner staging_owner;
    switch (selection.primary) {
#if defined(HYLE_COMPUTE_CPU)
    case BackendKind::kCpu: {
      auto &endpoint = state.primary.emplace<CpuEndpoint>();
      HYLE_RETURN_IF_ERROR(initialize_cpu(endpoint, cpu_options));
      state.device = state.host = state.staging = endpoint.memory;
      state.domains[0] = {.resource = endpoint.memory, .workspace_budget_bytes = budgets.host};
      state.targets[0] = {endpoint.source, graph::StorageDomainId{0}};
      state.queues[state.queue_count++] = {.lane = graph::ExecutionLaneId{0}, .queue = &endpoint.queue};
      break;
    }
#endif
#if defined(HYLE_COMPUTE_CUDA)
    case BackendKind::kCuda: {
      auto &endpoint = state.primary.emplace<CudaEndpoint>();
      HYLE_ASSIGN_OR_RETURN(endpoint.context, graph::cuda::create_context());
      HYLE_ASSIGN_OR_RETURN(auto memory, compute::cuda::MemoryResource::create(
          endpoint.context, compute::cuda::MemoryKind::kDevice));
      HYLE_ASSIGN_OR_RETURN(endpoint.memory, compute::StorageOwner::adopt(endpoint.context, std::move(memory)));
      HYLE_ASSIGN_OR_RETURN(endpoint.staging, compute::cuda::MemoryResource::create(
          endpoint.context, compute::cuda::MemoryKind::kPinnedHost));
      if (selection.cpu_assist) {
        HYLE_ASSIGN_OR_RETURN(endpoint.mixed_staging,
            compute::StorageOwner::adopt(endpoint.context, std::move(endpoint.staging)));
        staging_owner = endpoint.mixed_staging;
      }
      HYLE_ASSIGN_OR_RETURN(endpoint.queue, endpoint.context.create_queue());
      HYLE_ASSIGN_OR_RETURN(endpoint.source, graph::ExecutionSource::acquire(endpoint.context));
      state.device = endpoint.memory.ref();
      state.staging = endpoint.mixed_staging ? endpoint.mixed_staging.ref() : endpoint.staging.ref();
      state.domains[0] = {.resource = state.device, .workspace_budget_bytes = budgets.device, .owner = endpoint.memory};
      state.targets[0] = {endpoint.source, graph::StorageDomainId{0}};
      state.queues[state.queue_count++] = {.lane = graph::ExecutionLaneId{0}, .queue = &endpoint.queue};
      break;
    }
#endif
#if defined(HYLE_COMPUTE_APPLE) && defined(__APPLE__)
    case BackendKind::kApple: {
      auto &endpoint = state.primary.emplace<AppleEndpoint>();
      HYLE_ASSIGN_OR_RETURN(endpoint.context, compute::apple::Context::create());
      HYLE_ASSIGN_OR_RETURN(endpoint.memory, compute::apple::MemoryResource::create(endpoint.context));
      HYLE_ASSIGN_OR_RETURN(endpoint.staging, compute::apple::MemoryResource::create(endpoint.context));
      HYLE_ASSIGN_OR_RETURN(endpoint.source, graph::apple::execution_source(endpoint.context));
      state.device = endpoint.memory.ref(); state.staging = endpoint.staging.ref();
      state.domains[0] = {.resource = state.device, .workspace_budget_bytes = budgets.device};
      state.targets[0] = {endpoint.source, graph::StorageDomainId{0}};
      break;
    }
#endif
#if defined(HYLE_COMPUTE_VULKAN)
    case BackendKind::kVulkan: {
      auto &endpoint = state.primary.emplace<VulkanEndpoint>();
      HYLE_ASSIGN_OR_RETURN(endpoint.context, compute::vulkan::Context::create());
      HYLE_ASSIGN_OR_RETURN(endpoint.memory, compute::vulkan::MemoryResource::create(
          endpoint.context, compute::vulkan::MemoryKind::kDevice));
      HYLE_ASSIGN_OR_RETURN(endpoint.staging, compute::vulkan::MemoryResource::create(
          endpoint.context, compute::vulkan::MemoryKind::kStaging));
      HYLE_ASSIGN_OR_RETURN(endpoint.source, graph::vulkan::execution_source(endpoint.context));
      state.device = endpoint.memory.ref(); state.staging = endpoint.staging.ref();
      state.domains[0] = {.resource = state.device, .workspace_budget_bytes = budgets.device};
      state.targets[0] = {endpoint.source, graph::StorageDomainId{0}};
      state.retirement_context = &endpoint.context;
      state.collect = [](void *context) noexcept {
        return static_cast<compute::vulkan::Context *>(context)->collect();
      };
      state.drain = [](void *context, graph::Invocation &) noexcept {
        return static_cast<compute::vulkan::Context *>(context)->drain();
      };
      break;
    }
#endif
    default:
      return std::unexpected(Status::Unimplemented("selected backend has no compiled endpoint"));
    }
    state.domain_count = state.target_count = 1;
#if defined(HYLE_COMPUTE_CPU)
    if (selection.cpu_assist) {
      auto &endpoint = state.cpu.emplace();
      HYLE_RETURN_IF_ERROR(initialize_cpu(endpoint, cpu_options));
      state.host = endpoint.memory;
      state.domains[state.domain_count++] = {.resource = state.host, .workspace_budget_bytes = budgets.host};
      state.targets[state.target_count++] = {endpoint.source, graph::StorageDomainId{1}};
      state.lanes[0] = {graph::ExecutionTargetId{0}};
      state.lanes[1] = {graph::ExecutionTargetId{1}};
      state.lane_count = 2;
      state.queues[state.queue_count++] = {.lane = graph::ExecutionLaneId{1}, .queue = &endpoint.queue};
    }
#endif
    if (result.native_generated_storage() || selection.cpu_assist)
      state.domains[state.domain_count++] = {
          .resource = state.staging, .workspace_budget_bytes = budgets.staging, .owner = std::move(staging_owner)};
    return result;
  }

public:
  BackendKind kind() const noexcept { return state_ ? state_->selection.primary : BackendKind::kDefault; }
  BackendSelection selection() const noexcept { return state_ ? state_->selection : BackendSelection{}; }
  bool host_only() const noexcept { return kind() == BackendKind::kCpu; }
  bool native_generated_storage() const noexcept {
    return kind() == BackendKind::kApple || kind() == BackendKind::kVulkan;
  }
  // A secondary CPU endpoint on a GPU primary, distinct from a CPU primary.
  bool has_cpu_assist() const noexcept { return state_ && state_->selection.cpu_assist; }
  const char *name() const noexcept {
    switch (kind()) {
    case BackendKind::kCpu: return "cpu";
    case BackendKind::kCuda: return "cuda";
    case BackendKind::kApple: return "apple";
    case BackendKind::kVulkan: return "vulkan";
    default: return "unselected";
    }
  }
  mem::MemoryResourceRef device() const noexcept { return state_ ? state_->device : mem::MemoryResourceRef{}; }
  mem::MemoryResourceRef host_device() const noexcept { return state_ ? state_->host : mem::MemoryResourceRef{}; }
  mem::MemoryResourceRef staging() const noexcept { return state_ ? state_->staging : mem::MemoryResourceRef{}; }

  // Independent cold host-storage scope for complete parked state. Every
  // native route uses its own original host resource, never CPU assistance
  // or the Backend's borrowed staging object. No model/device bank is retained.
  Result<HostStorage> create_host_storage() const noexcept {
    if (!state_) return std::unexpected(Status::FailedPrecondition("Backend is not initialized"));
#if defined(HYLE_COMPUTE_CPU)
    if (const auto *endpoint = std::get_if<CpuEndpoint>(&state_->primary)) {
      // cpu::memory_resource is the process-lifetime host resource. The
      // original Context/metadata allocator contract stays with StorageOwner.
      struct HostResource final {
        mem::MemoryResourceRef resource;
        mem::MemoryResourceRef ref() const noexcept { return resource; }
      };
      HYLE_ASSIGN_OR_RETURN(auto owner,
          compute::StorageOwner::adopt(endpoint->context, HostResource{endpoint->memory}));
      return HostStorage::retain(std::move(owner));
    }
#endif
#if defined(HYLE_COMPUTE_CUDA)
    if (const auto *endpoint = std::get_if<CudaEndpoint>(&state_->primary)) {
      HYLE_ASSIGN_OR_RETURN(auto memory, compute::cuda::MemoryResource::create(
          endpoint->context, compute::cuda::MemoryKind::kPinnedHost));
      HYLE_ASSIGN_OR_RETURN(auto owner, compute::StorageOwner::adopt(endpoint->context, std::move(memory)));
      return HostStorage::retain(std::move(owner));
    }
#endif
#if defined(HYLE_COMPUTE_APPLE) && defined(__APPLE__)
    if (const auto *endpoint = std::get_if<AppleEndpoint>(&state_->primary)) {
      // Shared Metal storage has its own allocation controller. Host reads do
      // not require retaining MLX jobs, the execution Context or any model bank.
      HYLE_ASSIGN_OR_RETURN(auto memory, compute::apple::MemoryResource::create(endpoint->context));
      return HostStorage::adopt(std::move(memory), state_->host_storage_identity);
    }
#endif
#if defined(HYLE_COMPUTE_VULKAN)
    if (const auto *endpoint = std::get_if<VulkanEndpoint>(&state_->primary)) {
      // This resource retains the original Runtime/device and uses its own
      // process-lifetime system allocator. It owns no Backend execution state.
      HYLE_ASSIGN_OR_RETURN(auto memory, compute::vulkan::MemoryResource::create(
          endpoint->context, compute::vulkan::MemoryKind::kStaging));
      return HostStorage::adopt(std::move(memory), state_->host_storage_identity);
    }
#endif
    return std::unexpected(Status::Unimplemented(
        "selected backend has no independent host storage"));
  }

  Result<const void *> host_storage_identity() const noexcept {
    if (!state_) return std::unexpected(Status::FailedPrecondition("Backend is not initialized"));
#if defined(HYLE_COMPUTE_CPU)
    if (const auto *endpoint = std::get_if<CpuEndpoint>(&state_->primary)) {
      if (!endpoint->context.valid())
        return std::unexpected(Status::FailedPrecondition("context is invalid"));
      if (!endpoint->context.is_open())
        return std::unexpected(Status::FailedPrecondition("context is closed"));
      return endpoint->context.identity_key();
    }
#endif
#if defined(HYLE_COMPUTE_CUDA)
    if (const auto *endpoint = std::get_if<CudaEndpoint>(&state_->primary))
      return endpoint->memory.context_identity();
#endif
    if (state_->host_storage_identity) return state_->host_storage_identity.get();
    return std::unexpected(Status::FailedPrecondition("Backend has no selected host-storage identity"));
  }

  Result<mem::MemoryResourceRef> resource_for(graph::StorageDomainId domain) const noexcept {
    if (!domain.valid()) return std::unexpected(Status::InvalidArgument("Backend storage domain is invalid"));
    if (!state_) return std::unexpected(Status::FailedPrecondition("Backend is not initialized"));
    if (domain.ordinal() >= state_->domain_count)
      return std::unexpected(Status::OutOfRange("Backend storage domain is not configured"));
    return state_->domains[domain.ordinal()].resource;
  }
  Result<compute::StorageOwner> storage_owner_for(mem::MemoryResourceRef resource) const noexcept {
    if (!resource.valid()) return std::unexpected(Status::InvalidArgument("Backend resource is empty"));
    if (!state_) return std::unexpected(Status::FailedPrecondition("Backend is not initialized"));
    for (const auto &domain : domains()) if (resource == domain.resource) return domain.owner;
    return std::unexpected(Status::InvalidArgument("Backend resource is outside its configured allocation domains"));
  }
  const compute::StorageOwner &storage_owner() const noexcept {
    if (state_) return state_->domains[0].owner;
    static const compute::StorageOwner empty;
    return empty;
  }
  Result<compute::ReadLease> adopt_buffer(mem::Buffer &&buffer,
      compute::ResourceBudget budget = {}) const noexcept {
    if (buffer.valid() && has_cpu_assist() && buffer.resource() == host_device())
      return compute::ReadLease::adopt(std::move(buffer), {.resource_budget = std::move(budget)});
    if (!buffer.valid() || buffer.resource() != device())
      return std::unexpected(Status::InvalidArgument("Backend buffer belongs to another backend resource"));
    if (storage_owner()) return storage_owner().adopt_buffer(std::move(buffer), std::move(budget));
    return compute::ReadLease::adopt(std::move(buffer), {.resource_budget = std::move(budget)});
  }
  Status freeze(mem::Buffer &buffer, compute::ReadLease &ready) const noexcept {
    if (ready.valid()) return Status::Ok();
    HYLE_ASSIGN_OR_RETURN(ready, adopt_buffer(std::move(buffer)));
    return Status::Ok();
  }
  Status write(mem::BufferView target, std::span<const std::byte> bytes) const noexcept { return target.write(bytes); }
  Status read(mem::ConstBufferView source, std::span<std::byte> bytes) const noexcept { return source.read(bytes); }
  Status fill(mem::BufferView target, std::byte value = {}) const noexcept { return target.fill(value); }

  Result<graph::CompiledGraph> compile(const graph::Graph &source_graph,
      std::span<const graph::ConstantBinding> constants, graph::CompileOptions options = {}) const noexcept {
    if (!state_) return std::unexpected(Status::FailedPrecondition("Backend is not initialized"));
    options.constants = constants;
    if (has_cpu_assist() && options.resource_budget) for (const auto &domain : domains()) {
      auto usage = options.resource_budget.usage(domain.resource);
      if (!usage) return std::unexpected(std::move(usage).error());
    }
    return graph::batch::compile(source_graph, {
        .storage_domains = domains(),
        .targets = std::span(state_->targets).first(state_->target_count),
        .lanes = std::span(state_->lanes).first(state_->lane_count)}, options);
  }
  std::span<const graph::QueueBinding> queue_bindings() const noexcept {
    return state_ ? std::span<const graph::QueueBinding>(state_->queues).first(state_->queue_count)
                  : std::span<const graph::QueueBinding>{};
  }
  Status collect_retired() noexcept {
    if (!state_) return Status::FailedPrecondition("Backend is not initialized");
    return state_->collect(state_->retirement_context);
  }
  Status drain_retired(graph::Invocation &run) noexcept {
    if (!state_) return Status::FailedPrecondition("Backend is not initialized");
    return state_->drain(state_->retirement_context, run);
  }

  Result<compute::FileResourcePipeline> create_file_pipeline(io::IoContext &io,
      std::span<const compute::CopyResourceSpec> targets,
      compute::FileStreamOptions options = {}) noexcept {
    if (!state_) return std::unexpected(Status::FailedPrecondition("Backend is not initialized"));
    switch (kind()) {
#if defined(HYLE_COMPUTE_CPU)
    case BackendKind::kCpu: {
      auto &endpoint = std::get<CpuEndpoint>(state_->primary);
      return compute::cpu::create_file_pipeline(io, endpoint.context, endpoint.queue, staging(), targets, options);
    }
#endif
#if defined(HYLE_COMPUTE_CUDA)
    case BackendKind::kCuda: {
      auto &endpoint = std::get<CudaEndpoint>(state_->primary);
      return compute::cuda::create_file_pipeline(io, endpoint.context, endpoint.queue, staging(), targets, options);
    }
#endif
#if defined(HYLE_COMPUTE_APPLE) && defined(__APPLE__)
    case BackendKind::kApple:
      return compute::apple::create_file_pipeline(io,
          std::get<AppleEndpoint>(state_->primary).context, staging(), targets, options);
#endif
#if defined(HYLE_COMPUTE_VULKAN)
    case BackendKind::kVulkan:
      return compute::vulkan::create_file_pipeline(io,
          std::get<VulkanEndpoint>(state_->primary).context, staging(), targets, options);
#endif
    default: return std::unexpected(Status::FailedPrecondition("Backend has no selected input endpoint"));
    }
  }
  Result<compute::FileResourcePipeline> create_host_file_pipeline(io::IoContext &io,
      std::span<const compute::CopyResourceSpec> targets,
      compute::FileStreamOptions options = {}) noexcept {
#if defined(HYLE_COMPUTE_CPU)
    if (!has_cpu_assist())
      return std::unexpected(Status::FailedPrecondition("CPU file input requires CPU assistance on a GPU primary"));
    for (const auto &target : targets)
      if (target.resource != host_device())
        return std::unexpected(Status::InvalidArgument("CPU file target must use this Backend's host resource"));
    auto &endpoint = *state_->cpu;
    return compute::cpu::create_file_pipeline(io, endpoint.context, endpoint.queue, host_device(), targets, options);
#else
    (void)io; (void)targets; (void)options;
    return std::unexpected(Status::Unimplemented("CPU file input requires the CPU route in this SDK"));
#endif
  }

  // Cold composition/diagnostics borrow only original selected endpoints. No getter
  // constructs an otherwise unselected Context or substitutes a native queue.
  Result<compute::Context *> common_context() noexcept {
    if (!state_) return std::unexpected(Status::FailedPrecondition("Backend is not initialized"));
#if defined(HYLE_COMPUTE_CPU)
    if (auto *endpoint = std::get_if<CpuEndpoint>(&state_->primary)) return &endpoint->context;
#endif
#if defined(HYLE_COMPUTE_CUDA)
    if (auto *endpoint = std::get_if<CudaEndpoint>(&state_->primary)) return &endpoint->context;
#endif
    return std::unexpected(Status::FailedPrecondition("selected primary has no Compute Context"));
  }
  Result<compute::Queue *> common_queue() noexcept {
    if (!state_) return std::unexpected(Status::FailedPrecondition("Backend is not initialized"));
#if defined(HYLE_COMPUTE_CPU)
    if (auto *endpoint = std::get_if<CpuEndpoint>(&state_->primary)) return &endpoint->queue;
#endif
#if defined(HYLE_COMPUTE_CUDA)
    if (auto *endpoint = std::get_if<CudaEndpoint>(&state_->primary)) return &endpoint->queue;
#endif
    return std::unexpected(Status::FailedPrecondition("selected primary has no Compute Queue"));
  }
  Result<compute::Context *> cpu_context() noexcept {
#if defined(HYLE_COMPUTE_CPU)
    if (!state_) return std::unexpected(Status::FailedPrecondition("Backend is not initialized"));
    if (auto *endpoint = std::get_if<CpuEndpoint>(&state_->primary)) return &endpoint->context;
    if (state_->cpu) return &state_->cpu->context;
    return std::unexpected(Status::FailedPrecondition("Backend has no selected CPU endpoint"));
#else
    return std::unexpected(Status::Unimplemented("CPU backend is not compiled"));
#endif
  }
  Result<compute::Queue *> cpu_queue() noexcept {
#if defined(HYLE_COMPUTE_CPU)
    if (!state_) return std::unexpected(Status::FailedPrecondition("Backend is not initialized"));
    if (auto *endpoint = std::get_if<CpuEndpoint>(&state_->primary)) return &endpoint->queue;
    if (state_->cpu) return &state_->cpu->queue;
    return std::unexpected(Status::FailedPrecondition("Backend has no selected CPU endpoint"));
#else
    return std::unexpected(Status::Unimplemented("CPU backend is not compiled"));
#endif
  }
#if defined(HYLE_COMPUTE_APPLE) && defined(__APPLE__)
  Result<compute::apple::Context *> apple_context() noexcept {
    if (state_) if (auto *endpoint = std::get_if<AppleEndpoint>(&state_->primary)) return &endpoint->context;
    return std::unexpected(Status::FailedPrecondition("Apple primary is not selected"));
  }
#endif
#if defined(HYLE_COMPUTE_VULKAN)
  Result<compute::vulkan::Context *> vulkan_context() noexcept {
    if (state_) if (auto *endpoint = std::get_if<VulkanEndpoint>(&state_->primary)) return &endpoint->context;
    return std::unexpected(Status::FailedPrecondition("Vulkan primary is not selected"));
  }
#endif

private:
  static bool is_compiled(BackendKind kind) noexcept {
    for (const auto compiled : compiled_backends()) if (compiled == kind) return true;
    return false;
  }
#if defined(HYLE_COMPUTE_CPU)
  static Status initialize_cpu(CpuEndpoint &endpoint,
                               const compute::cpu::DefaultContextOptions *cpu_options) noexcept {
    if (cpu_options != nullptr) {
      HYLE_ASSIGN_OR_RETURN(endpoint.context, graph::cpu::create_context(*cpu_options));
    } else {
      HYLE_ASSIGN_OR_RETURN(endpoint.context, graph::cpu::create_context());
    }
    HYLE_ASSIGN_OR_RETURN(endpoint.memory, compute::cpu::memory_resource(compute::cpu::device()));
    HYLE_ASSIGN_OR_RETURN(endpoint.queue, endpoint.context.create_queue());
    HYLE_ASSIGN_OR_RETURN(endpoint.source, graph::ExecutionSource::acquire(endpoint.context));
    return Status::Ok();
  }
#endif
  std::span<const graph::StorageDomain> domains() const noexcept {
    return std::span<const graph::StorageDomain>(state_->domains).first(state_->domain_count);
  }
  // Moving Backend moves this pointer only. Native Contexts, cached Queue
  // bindings and exceptional-retirement callbacks keep their exact addresses.
  std::unique_ptr<State> state_;
};

} // namespace hyle::inference
