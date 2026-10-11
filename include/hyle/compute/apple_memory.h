#pragma once

#include "hyle/compute/apple.h"

#if defined(HYLE_COMPUTE_APPLE)
#include "hyle/compute/resource_budget.h"
#include "hyle/compute/resource_leases.h"
#include "hyle/compute/tensor.h"
#include "hyle/mem/buffer.h"

namespace hyle::compute::apple {
namespace detail { struct MemoryState; }

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
  MemoryResource() noexcept = default;
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

} // namespace hyle::compute::apple
#endif
