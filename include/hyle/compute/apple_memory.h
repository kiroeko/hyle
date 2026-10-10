#pragma once

#include "hyle/compute/apple.h"

#if defined(HYLE_COMPUTE_APPLE)
#include "hyle/compute/resource_budget.h"
#include "hyle/mem/buffer.h"

namespace hyle::compute::apple {
namespace detail { struct MemoryState; }

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
private:
  explicit MemoryResource(std::unique_ptr<detail::MemoryState> state) noexcept;
  std::unique_ptr<detail::MemoryState> state_;
};

} // namespace hyle::compute::apple
#endif
