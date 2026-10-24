#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <utility>

#include "hyle/compute/file_stream.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/graph/batch.h"
#include "hyle/mem/buffer.h"

#if defined(HYLE_COMPUTE_APPLE) && defined(__APPLE__)
#include "hyle/compute/apple_file_stream.h"
#include "hyle/compute/apple_memory.h"
#include "hyle/graph/apple.h"
#elif defined(HYLE_COMPUTE_CUDA)
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/cuda_file_stream.h"
#include "hyle/graph/cuda.h"
#elif defined(HYLE_COMPUTE_VULKAN)
#include "hyle/compute/vulkan_file_stream.h"
#include "hyle/compute/vulkan_memory.h"
#include "hyle/graph/vulkan.h"
#else
#error "Qwen text consumer requires an Apple, CUDA or Vulkan backend"
#endif

namespace qwen35 {

// One cold environment for this consumer. It owns the real allocation domains
// and endpoint; Graph/Compute retain their existing executors and retirement.
// In particular, native Apple resources remain borrowed by Buffer/LeasePool:
// this owner must outlive every tensor, executable, invocation and native use.
class Backend final {
public:
#if defined(HYLE_COMPUTE_APPLE) && defined(__APPLE__)
  static constexpr bool native_generated_storage = true;
  static constexpr const char *name() noexcept { return "apple"; }
  // Cold test interop only: the retirement probe uses this original Context,
  // never a replacement device, stream or execution source.
  hyle::compute::apple::Context &native_probe_context() noexcept { return context_; }
#elif defined(HYLE_COMPUTE_CUDA)
  static constexpr bool native_generated_storage = false;
  static constexpr const char *name() noexcept { return "cuda"; }
  // Existing CUDA-only native queue diagnostics borrow these original owners.
  hyle::compute::Context &common_context() noexcept { return context_; }
  hyle::compute::Queue &common_queue() noexcept { return queue_; }
#else
  static constexpr bool native_generated_storage = true;
  static constexpr const char *name() noexcept { return "vulkan"; }
  hyle::compute::vulkan::Context &native_probe_context() noexcept { return context_; }
#endif
  Backend() noexcept = default;
  Backend(const Backend &) = delete;
  Backend &operator=(const Backend &) = delete;
  Backend(Backend &&) noexcept = default;
  Backend &operator=(Backend &&) noexcept = default;

  static hyle::Result<Backend> create() noexcept {
    Backend result;
#if defined(HYLE_COMPUTE_APPLE) && defined(__APPLE__)
    HYLE_ASSIGN_OR_RETURN(result.context_, hyle::compute::apple::Context::create());
    HYLE_ASSIGN_OR_RETURN(result.memory_, hyle::compute::apple::MemoryResource::create(result.context_));
    HYLE_ASSIGN_OR_RETURN(result.staging_, hyle::compute::apple::MemoryResource::create(result.context_));
    HYLE_ASSIGN_OR_RETURN(result.source_, hyle::graph::apple::execution_source(result.context_));
#elif defined(HYLE_COMPUTE_CUDA)
    HYLE_ASSIGN_OR_RETURN(result.context_, hyle::graph::cuda::create_context());
    HYLE_ASSIGN_OR_RETURN(auto resource, hyle::compute::cuda::MemoryResource::create(
        result.context_, hyle::compute::cuda::MemoryKind::kDevice));
    HYLE_ASSIGN_OR_RETURN(result.memory_, hyle::compute::StorageOwner::adopt(result.context_, std::move(resource)));
    HYLE_ASSIGN_OR_RETURN(result.staging_, hyle::compute::cuda::MemoryResource::create(
        result.context_, hyle::compute::cuda::MemoryKind::kPinnedHost));
    HYLE_ASSIGN_OR_RETURN(result.queue_, result.context_.create_queue());
    HYLE_ASSIGN_OR_RETURN(result.source_, hyle::graph::ExecutionSource::acquire(result.context_));
#else
    HYLE_ASSIGN_OR_RETURN(result.context_, hyle::compute::vulkan::Context::create());
    HYLE_ASSIGN_OR_RETURN(result.memory_, hyle::compute::vulkan::MemoryResource::create(
        result.context_, hyle::compute::vulkan::MemoryKind::kDevice));
    HYLE_ASSIGN_OR_RETURN(result.staging_, hyle::compute::vulkan::MemoryResource::create(
        result.context_, hyle::compute::vulkan::MemoryKind::kStaging));
    HYLE_ASSIGN_OR_RETURN(result.source_, hyle::graph::vulkan::execution_source(result.context_));
#endif
    return result;
  }

  hyle::mem::MemoryResourceRef device() const noexcept { return memory_.ref(); }
  hyle::mem::MemoryResourceRef staging() const noexcept { return staging_.ref(); }
  const hyle::compute::StorageOwner &storage_owner() const noexcept {
#if defined(HYLE_COMPUTE_APPLE) && defined(__APPLE__)
    // ConstantBinding/StorageDomain accept caller-owned native resources.
    return borrowed_storage_;
#elif defined(HYLE_COMPUTE_CUDA)
    return memory_;
#else
    return borrowed_storage_;
#endif
  }

  hyle::Result<hyle::compute::ReadLease> adopt_buffer(hyle::mem::Buffer &&buffer,
      hyle::compute::ResourceBudget budget = {}) const noexcept {
    if (!buffer.valid() || buffer.resource() != device())
      return std::unexpected(hyle::Status::InvalidArgument("Qwen buffer belongs to another backend resource"));
#if defined(HYLE_COMPUTE_APPLE) && defined(__APPLE__)
    return hyle::compute::ReadLease::adopt(std::move(buffer), {.resource_budget = std::move(budget)});
#elif defined(HYLE_COMPUTE_CUDA)
    return memory_.adopt_buffer(std::move(buffer), std::move(budget));
#else
    return hyle::compute::ReadLease::adopt(std::move(buffer), {.resource_budget = std::move(budget)});
#endif
  }
  hyle::Status freeze(hyle::mem::Buffer &buffer, hyle::compute::ReadLease &ready) const noexcept {
    if (ready.valid()) return hyle::Status::Ok();
    HYLE_ASSIGN_OR_RETURN(ready, adopt_buffer(std::move(buffer)));
    return hyle::Status::Ok();
  }

  // External I/O only. Callers must already have exclusive/retired access;
  // shared Apple storage does not manufacture a GPU completion or permission.
  hyle::Status write(hyle::mem::BufferView target, std::span<const std::byte> bytes) const noexcept {
    return target.write(bytes);
  }
  hyle::Status read(hyle::mem::ConstBufferView source, std::span<std::byte> bytes) const noexcept {
    return source.read(bytes);
  }
  hyle::Status fill(hyle::mem::BufferView target, std::byte value = {}) const noexcept {
    return target.fill(value);
  }

  hyle::Result<hyle::graph::CompiledGraph> compile(const hyle::graph::Graph &graph,
      std::span<const hyle::graph::ConstantBinding> constants,
      hyle::graph::CompileOptions options = {}) const noexcept {
    options.constants = constants;
#if defined(HYLE_COMPUTE_APPLE) && defined(__APPLE__)
    const std::array domains{
        hyle::graph::StorageDomain{.resource = device(), .owner = storage_owner()},
        hyle::graph::StorageDomain{.resource = staging()}};
#elif defined(HYLE_COMPUTE_CUDA)
    // Preserve the original CUDA placement domain. Pinned staging is supplied
    // only to the loader and explicit host-workspace preparation.
    const std::array domains{
        hyle::graph::StorageDomain{.resource = device(), .owner = storage_owner()}};
#else
    const std::array domains{
        hyle::graph::StorageDomain{.resource = device(), .owner = storage_owner()},
        hyle::graph::StorageDomain{.resource = staging()}};
#endif
    const std::array targets{hyle::graph::ExecutionTarget{source_, hyle::graph::StorageDomainId{0}}};
    return hyle::graph::batch::compile(graph, {.storage_domains = domains, .targets = targets}, options);
  }

  std::span<const hyle::graph::QueueBinding> queue_bindings() noexcept {
#if defined(HYLE_COMPUTE_APPLE) && defined(__APPLE__)
    return {};
#elif defined(HYLE_COMPUTE_CUDA)
    // Refresh the borrow after a Backend move. No Queue shell is constructed.
    bindings_[0] = {.lane = hyle::graph::ExecutionLaneId{0}, .queue = &queue_};
    return bindings_;
#else
    return {};
#endif
  }

  // Nonblocking progress for abandoned native work. Vulkan deliberately uses
  // explicit collection; Apple/CUDA already retire through their original
  // workers. This keeps native collection out of session/state logic.
  hyle::Status collect_retired() noexcept {
#if defined(HYLE_COMPUTE_APPLE) && defined(__APPLE__)
    return hyle::Status::Ok();
#elif defined(HYLE_COMPUTE_CUDA)
    return hyle::Status::Ok();
#else
    return context_.collect();
#endif
  }

  hyle::Result<hyle::compute::FileResourcePipeline> create_file_pipeline(
      hyle::io::IoContext &io, std::span<const hyle::compute::CopyResourceSpec> targets,
      hyle::compute::FileStreamOptions options = {}) noexcept {
#if defined(HYLE_COMPUTE_APPLE) && defined(__APPLE__)
    return hyle::compute::apple::create_file_pipeline(io, context_, staging(), targets, options);
#elif defined(HYLE_COMPUTE_CUDA)
    return hyle::compute::cuda::create_file_pipeline(io, context_, queue_, staging(), targets, options);
#else
    return hyle::compute::vulkan::create_file_pipeline(io, context_, staging(), targets, options);
#endif
  }

private:
  // Destruction reverses this order: endpoints disappear before their native
  // Context/resource owners. The Model drains its borrowed work first.
#if defined(HYLE_COMPUTE_APPLE) && defined(__APPLE__)
  hyle::compute::apple::Context context_;
  hyle::compute::apple::MemoryResource memory_, staging_;
  hyle::compute::StorageOwner borrowed_storage_;
#elif defined(HYLE_COMPUTE_CUDA)
  hyle::compute::Context context_;
  hyle::compute::StorageOwner memory_;
  hyle::compute::cuda::MemoryResource staging_;
  hyle::compute::Queue queue_;
  std::array<hyle::graph::QueueBinding, 1> bindings_{};
#else
  hyle::compute::vulkan::Context context_;
  hyle::compute::vulkan::MemoryResource memory_, staging_;
  hyle::compute::StorageOwner borrowed_storage_;
#endif
  hyle::graph::ExecutionSource source_;
};
} // namespace qwen35
