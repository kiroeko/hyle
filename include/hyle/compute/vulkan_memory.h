#pragma once

#include "hyle/compute/vulkan.h"

#if defined(HYLE_COMPUTE_VULKAN)
#include <optional>

#include "hyle/compute/resource_budget.h"
#include "hyle/compute/resource_leases.h"
#include "hyle/compute/batch_transfer.h"
#include "hyle/compute/tensor.h"
#include "hyle/mem/buffer.h"

namespace hyle::compute::vulkan {
namespace detail { struct MemoryState; struct LeasedInvocationState; struct MemoryAccess; struct TransferState; struct TransferAccess; }
namespace detail { struct CopyProgramState; struct CopyInvocationState; }

class CopyBatchInvocation;
class CopyBatchSubmission;
// Original Vulkan Runtime and native resources, with immutable cold limits.
class CopyBatchProgram final {
public:
  CopyBatchProgram() noexcept = default;
  HYLE_COMPUTE_ROUTE_VULKAN_API CopyBatchProgram(const CopyBatchProgram &) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API CopyBatchProgram &operator=(const CopyBatchProgram &) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API CopyBatchProgram(CopyBatchProgram &&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API CopyBatchProgram &operator=(CopyBatchProgram &&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API ~CopyBatchProgram();
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<CopyBatchProgram> prepare(const Context &,
      std::span<const CopyResourceSpec>, CopyBatchOptions options = {}) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_ROUTE_VULKAN_API void reset() noexcept;
private:
  friend class CopyBatchInvocation;
  explicit CopyBatchProgram(detail::CopyProgramState *state) noexcept : state_(state) {}
  detail::CopyProgramState *state_ = nullptr;
};
// Independent phase Run/fence slots; no common Compute Queue or second runtime.
class CopyBatchInvocation final {
public:
  CopyBatchInvocation() noexcept = default;
  CopyBatchInvocation(const CopyBatchInvocation &) = delete;
  CopyBatchInvocation &operator=(const CopyBatchInvocation &) = delete;
  HYLE_COMPUTE_ROUTE_VULKAN_API CopyBatchInvocation(CopyBatchInvocation &&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API CopyBatchInvocation &operator=(CopyBatchInvocation &&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API ~CopyBatchInvocation();
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<CopyBatchInvocation> create(const CopyBatchProgram &) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_ROUTE_VULKAN_API Status bind_source(CopyResourceId, const ReadLease &) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status bind_mutable(CopyResourceId, WriteLease &&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status bind_producer(CopyResourceId, Completion &&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status prepare_replay(std::span<const CopyPhase>, std::span<const CopyRange>) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<CopyBatchSubmission> submit() noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status reset() noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status close() noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status drain() noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API bool in_flight() const noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API CopyBatchReport report() const noexcept;
private:
  explicit CopyBatchInvocation(detail::CopyInvocationState *state) noexcept : state_(state) {}
  detail::CopyInvocationState *state_ = nullptr;
};
class CopyBatchSubmission final {
public:
  CopyBatchSubmission() noexcept = default;
  CopyBatchSubmission(const CopyBatchSubmission &) = delete;
  CopyBatchSubmission &operator=(const CopyBatchSubmission &) = delete;
  HYLE_COMPUTE_ROUTE_VULKAN_API CopyBatchSubmission(CopyBatchSubmission &&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API CopyBatchSubmission &operator=(CopyBatchSubmission &&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API ~CopyBatchSubmission();
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<ReadLease> output(CopyPhaseId, CopyResourceId,
      std::size_t offset, std::size_t bytes) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status wait() noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status request_cancel() noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API bool in_flight() const noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API CopyBatchReport report() const noexcept;
private:
  friend class CopyBatchInvocation;
  explicit CopyBatchSubmission(detail::CopyInvocationState *state) noexcept : state_(state) {}
  detail::CopyInvocationState *state_ = nullptr;
};

enum class MemoryKind : unsigned char { kDevice, kStaging };

struct AllocationOptions final {
  ResourceBudget resource_budget{};
  ResourceBudgetCategory category = ResourceBudgetCategory::kWorkspace;
};

struct NativeStorage final {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceSize offset_bytes = 0, capacity_bytes = 0;
  // Requested capacity differs from VkDeviceMemory requirements. Unknown for
  // external imports; never add this observation to managed scope usage.
  std::optional<std::size_t> allocation_bytes;
  bool transfer_eligible = false; // Original compute and transfer families.
};

// Original route VkBuffer/VkDeviceMemory in the Context's exact Device/family.
// Mem handles are opaque allocation tokens, never CPU or VkBuffer pointers.
// Staging is explicitly host mapped; device storage is not advertised as host
// accessible even on a shared-memory GPU. Both kinds use original native owners.
// Resources/allocator remain borrowed through Buffer/quotas. Tensor allocations
// retain device and budget until their last original native owner disappears.
class MemoryResource final {
public:
  MemoryResource() noexcept = default;
  MemoryResource(const MemoryResource &) = delete;
  MemoryResource &operator=(const MemoryResource &) = delete;
  HYLE_COMPUTE_ROUTE_VULKAN_API MemoryResource(MemoryResource &&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API MemoryResource &operator=(MemoryResource &&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API ~MemoryResource();
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<MemoryResource>
  create(const Context &context, MemoryKind kind,
         mem::AllocatorRef allocator = mem::system_allocator()) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_ROUTE_VULKAN_API mem::MemoryResourceRef ref() const noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<Tensor>
  allocate(const TensorDesc &description, AllocationOptions options = {}) const noexcept;
  // Cold readonly native view over a lease from this exact device resource.
  // Four-byte native tail capacity must be inside the owned source range.
  // Tensor/Program/result owners retain the lease; no copy/new Vk allocation.
  // Resource/allocator outlive aliases. Native writable aliases are forbidden.
  // Options account wrapper metadata; original backing remains charged once.
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<Tensor>
  binding(ReadLease source, const TensorDesc &description,
          LeaseImportOptions options = {}) const noexcept;
  // The producer must have retired successfully. Retains the actual native
  // Tensor owner, permanently readonly; options cover new metadata only.
  // Original native accounting/resource lifetime remains in force.
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<ReadLease>
  import_readonly(const Tensor &source, LeaseImportOptions options = {}) const noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<NativeStorage>
  storage_info(const Tensor &tensor) const noexcept;
private:
  friend struct detail::MemoryAccess;
  explicit MemoryResource(std::unique_ptr<detail::MemoryState> state) noexcept;
  std::unique_ptr<detail::MemoryState> state_;
};

// One cold staging allocation, command pool/buffer and fence. Host sources are
// borrowed only for upload's synchronous staging preparation. Returned native
// Completions protect accepted work; download data is exposed by read() only
// after successful terminal observation. All operations preserve exact bytes.
// Copy origins must be four-byte aligned; use an existing typed Copy/repack for
// other views. Reuse never grows capacity or reselects a Provider.
// Handles are thread-compatible. Resource/allocator outlive late work; Context
// collect/close retire abandoned prefixes using the original native runtime.
// Upload of eligible MemoryResource storage uses the original transfer queue
// and a semaphore handoff to compute. Ready remains compute ordered-prefix
// completion. Legacy/imported storage and downloads use the original compute
// queue; queue availability alone does not prove physical overlap.
class TransferSlot final {
public:
  TransferSlot() noexcept = default;
  TransferSlot(const TransferSlot &) = delete;
  TransferSlot &operator=(const TransferSlot &) = delete;
  HYLE_COMPUTE_ROUTE_VULKAN_API TransferSlot(TransferSlot &&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API TransferSlot &operator=(TransferSlot &&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API ~TransferSlot();
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<TransferSlot>
  create(const Context &context, const MemoryResource &staging,
         std::size_t capacity_bytes, AllocationOptions options = {}) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<Completion>
  upload(const Tensor &destination, ConstTensorView source) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<Completion> download(const Tensor &source) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status read(TensorView destination) const noexcept;
private:
  friend struct detail::TransferAccess;
  std::shared_ptr<detail::TransferState> state_;
};
struct LeasedInput final {
  TensorDesc description{};
  ReadLease source{};
};

// Opt-in native submission owner for both Direct and the route's Graph
// Programs. bind constructs native views cold; submit automatically admits
// sources and attaches their guards to the original native Completion.
class LeasedInvocation final {
public:
  LeasedInvocation() noexcept = default;
  LeasedInvocation(const LeasedInvocation &) = delete;
  LeasedInvocation &operator=(const LeasedInvocation &) = delete;
  HYLE_COMPUTE_ROUTE_VULKAN_API LeasedInvocation(LeasedInvocation &&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API LeasedInvocation &operator=(LeasedInvocation &&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API ~LeasedInvocation();
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<LeasedInvocation>
  create(const Program &program, const MemoryResource &memory,
         LeaseImportOptions options = {}, InvocationAllocation allocation = {}) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_ROUTE_VULKAN_API bool in_flight() const noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status bind(std::span<const LeasedInput> inputs) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<OwnedResult<Results>> submit() noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status close() noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status drain() noexcept;
private:
  explicit LeasedInvocation(detail::LeasedInvocationState *state) noexcept : state_(state) {}
  detail::LeasedInvocationState *state_ = nullptr;
};

} // namespace hyle::compute::vulkan
#endif
