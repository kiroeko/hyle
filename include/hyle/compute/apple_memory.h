#pragma once

#include "hyle/compute/apple.h"

#if defined(HYLE_COMPUTE_APPLE)
#include "hyle/compute/resource_budget.h"
#include "hyle/compute/resource_leases.h"
#include "hyle/compute/batch_transfer.h"
#include "hyle/compute/tensor.h"
#include "hyle/mem/buffer.h"

namespace hyle::compute::apple {
namespace detail { struct MemoryState; struct LeasedInvocationState; }
namespace detail { struct CopyProgramState; struct CopyInvocationState; struct CopyStreamAccess; }

class CopyBatchInvocation;
class CopyBatchSubmission;
// Original Apple Runtime and native resources, with immutable cold limits.
class CopyBatchProgram final {
public:
  CopyBatchProgram() noexcept = default;
  HYLE_COMPUTE_ROUTE_APPLE_API CopyBatchProgram(const CopyBatchProgram &) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API CopyBatchProgram &operator=(const CopyBatchProgram &) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API CopyBatchProgram(CopyBatchProgram &&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API CopyBatchProgram &operator=(CopyBatchProgram &&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API ~CopyBatchProgram();
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<CopyBatchProgram> prepare(const Context &,
      std::span<const CopyResourceSpec>, CopyBatchOptions options = {}) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_ROUTE_APPLE_API void reset() noexcept;
private:
  friend class CopyBatchInvocation;
  explicit CopyBatchProgram(detail::CopyProgramState *state) noexcept : state_(state) {}
  detail::CopyProgramState *state_ = nullptr;
};
// Independent FIFO native Job slots; no common Compute Queue or second runtime.
class CopyBatchInvocation final {
public:
  CopyBatchInvocation() noexcept = default;
  CopyBatchInvocation(const CopyBatchInvocation &) = delete;
  CopyBatchInvocation &operator=(const CopyBatchInvocation &) = delete;
  HYLE_COMPUTE_ROUTE_APPLE_API CopyBatchInvocation(CopyBatchInvocation &&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API CopyBatchInvocation &operator=(CopyBatchInvocation &&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API ~CopyBatchInvocation();
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<CopyBatchInvocation> create(const CopyBatchProgram &) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  // Accept the exact nominated resource or a trusted readonly MLX export
  // from its native family. Retains that export's actual owner and byte range;
  // replay preparation still verifies the original Metal Device. Mutable
  // endpoints require the exact resource. No implicit evaluation or wait.
  HYLE_COMPUTE_ROUTE_APPLE_API Status bind_source(CopyResourceId, const ReadLease &) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status bind_mutable(CopyResourceId, WriteLease &&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status bind_producer(CopyResourceId, Completion &&) noexcept;
  // Return the original mutable binding after all accepted phases actually
  // succeed and retire, or withdraw it before submission. Retained submissions,
  // pending/failed/unknown work and published outputs reject without consuming.
  // Nonblocking retirement observation, no wait/publication; untouched bytes
  // remain caller-owned content. Abandonment alone does not retire a writer.
  HYLE_COMPUTE_ROUTE_APPLE_API Result<WriteLease> unbind_mutable(CopyResourceId) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status prepare_replay(std::span<const CopyPhase>, std::span<const CopyRange>) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Result<CopyBatchSubmission> submit() noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status reset() noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status close() noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status drain() noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API bool in_flight() const noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API CopyBatchReport report() const noexcept;
private:
  friend struct detail::CopyStreamAccess;
  explicit CopyBatchInvocation(detail::CopyInvocationState *state) noexcept : state_(state) {}
  detail::CopyInvocationState *state_ = nullptr;
};
class CopyBatchSubmission final {
public:
  CopyBatchSubmission() noexcept = default;
  CopyBatchSubmission(const CopyBatchSubmission &) = delete;
  CopyBatchSubmission &operator=(const CopyBatchSubmission &) = delete;
  HYLE_COMPUTE_ROUTE_APPLE_API CopyBatchSubmission(CopyBatchSubmission &&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API CopyBatchSubmission &operator=(CopyBatchSubmission &&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API ~CopyBatchSubmission();
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_ROUTE_APPLE_API Result<ReadLease> output(CopyPhaseId, CopyResourceId,
      std::size_t offset, std::size_t bytes) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status wait() noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status request_cancel() noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API bool in_flight() const noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API CopyBatchReport report() const noexcept;
private:
  friend class CopyBatchInvocation;
  explicit CopyBatchSubmission(detail::CopyInvocationState *state) noexcept : state_(state) {}
  detail::CopyInvocationState *state_ = nullptr;
};


struct NativeStorage final {
  const void *buffer = nullptr; // Original opaque MTLBuffer identity.
  std::size_t offset_bytes = 0;
  std::size_t view_bytes = 0;
  std::size_t backing_bytes = 0;
};

// The original MLX allocator's shared Metal buffers. Host accessibility does
// not imply execution ordering: wait for GPU use before mapping or mutating.
// This resource and its native device outlive all Buffer/array views and quotas.
class MemoryResource final {
public:
  HYLE_COMPUTE_ROUTE_APPLE_API MemoryResource() noexcept;
  MemoryResource(const MemoryResource &) = delete;
  MemoryResource &operator=(const MemoryResource &) = delete;
  HYLE_COMPUTE_ROUTE_APPLE_API MemoryResource(MemoryResource &&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API MemoryResource &operator=(MemoryResource &&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API ~MemoryResource();
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<MemoryResource>
  create(const Context &context) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_ROUTE_APPLE_API mem::MemoryResourceRef ref() const noexcept;
  // Cold allocation. The MLX array retains the native Buffer and quota until
  // its last alias disappears. Native allocator caches/rounding are excluded.
  // MLX descriptor storage is opaque; a supplied budget must explicitly allow
  // unattributed owners. A strict total scope rejects before native allocation.
  HYLE_COMPUTE_ROUTE_APPLE_API Result<mlx::core::array>
  allocate(const TensorDesc &description, ResourceBudget budget = {},
           ResourceBudgetCategory category = ResourceBudgetCategory::kWorkspace) const noexcept;
  // Cold readonly native view, no payload copy/eval/wait. Original native
  // handles retain the source lease through bind/submit/results. Source must
  // belong to this exact resource and use its dense physical carrier; native
  // aliases must not mutate it. Resource outlives every returned array alias.
  // Budget applies to wrapper/opaque descriptor metadata, not backing twice.
  HYLE_COMPUTE_ROUTE_APPLE_API Result<mlx::core::array>
  binding(ReadLease source, const TensorDesc &description,
          LeaseImportOptions options = {}) const noexcept;
  // Successfully ready dense native carrier. Retains the real MLX array,
  // exports permanently readonly bytes; no inferred producer fence/copy.
  // Options cover new metadata and acknowledge opaque native descriptors;
  // existing native backing accounting is kept, never charged a second time.
  HYLE_COMPUTE_ROUTE_APPLE_API Result<ReadLease>
  import_readonly(const mlx::core::array &source, const TensorDesc &description,
                  LeaseImportOptions options = {}) const noexcept;
  // Ready native Metal arrays only. Byte copies preserve exact physical
  // encoding and require dense carriers; strided host inputs use the existing
  // CPU Copy operation before this boundary. No implicit GPU wait/evaluation.
  // The destination must be exclusive through all of its aliases.
  HYLE_COMPUTE_ROUTE_APPLE_API Status
  copy_from_host(const mlx::core::array &destination, ConstTensorView source) const noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status
  copy_to_host(const mlx::core::array &source, TensorView destination) const noexcept;
  // Cold observation, not a quota or private allocation bound. Aliases with
  // the same buffer identify one backing and must not be counted repeatedly.
  HYLE_COMPUTE_ROUTE_APPLE_API Result<NativeStorage>
  storage_info(const mlx::core::array &array) const noexcept;
private:
  explicit MemoryResource(std::unique_ptr<detail::MemoryState> state) noexcept;
  std::unique_ptr<detail::MemoryState> state_;
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
  HYLE_COMPUTE_ROUTE_APPLE_API LeasedInvocation(LeasedInvocation &&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API LeasedInvocation &operator=(LeasedInvocation &&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API ~LeasedInvocation();
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<LeasedInvocation>
  create(const Program &program, const MemoryResource &memory,
         LeaseImportOptions options = {}) noexcept;
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_ROUTE_APPLE_API bool in_flight() const noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status bind(std::span<const LeasedInput> inputs) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Result<OwnedResult<Results>> submit() noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status close() noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status drain() noexcept;
private:
  explicit LeasedInvocation(detail::LeasedInvocationState *state) noexcept : state_(state) {}
  detail::LeasedInvocationState *state_ = nullptr;
};

} // namespace hyle::compute::apple
#endif
