#pragma once

#include "hyle/compute/detail/vulkan_plan.h"

namespace hyle::compute::vulkan::detail {

// Optional expert lowering, borrowed only for one synchronous cold prepare.
// The real typed handlers get first refusal. No capability list, Graph type,
// resolver, or caller span survives into Program/Invocation/replay.
struct ColdCustomDispatch final {
  Kernel kernel;
  std::array<std::uint32_t, 3> groups{1, 1, 1};
  std::vector<std::byte> push_constants;
};
struct ColdCustomResolver final {
  void *state = nullptr;
  Result<ColdCustomDispatch> (*resolve)(
      void *, const Operation &, std::span<const TensorDesc>,
      std::span<const TensorDesc>) noexcept = nullptr;
};

HYLE_COMPUTE_ROUTE_VULKAN_API Result<Program>
prepare_plan(const Context &, const Plan &, PlanFacts *, ColdCustomResolver) noexcept;

// Same cold preparation body for callers that already retain the native owner.
HYLE_COMPUTE_ROUTE_VULKAN_API Result<Program>
prepare_plan(const std::shared_ptr<Runtime> &, const Plan &,
             PlanFacts * = nullptr, ColdCustomResolver = {}) noexcept;
// Cold execution constraint for a composition that relies on this Runtime's
// original queue ordering. Never silently substitute an asynchronous carrier.
enum class PlanQueuePolicy : std::uint8_t { kMayDelegate, kRuntimeQueueOnly };
HYLE_COMPUTE_ROUTE_VULKAN_API
Result<Program> prepare_plan(const std::shared_ptr<Runtime> &, const Plan &,
    PlanQueuePolicy, PlanFacts * = nullptr, ColdCustomResolver = {}) noexcept;
HYLE_COMPUTE_ROUTE_VULKAN_API Result<Program>
prepare_operation(const std::shared_ptr<Runtime> &, const Operation &,
                  std::span<const Argument>, const WorkloadContext & = {}) noexcept;

struct ProgramState;
Status lower_operation(ProgramState &, const PlanStep &, ColdCustomResolver,
                       RowCacheAccessFacts * = nullptr) noexcept;

} // namespace hyle::compute::vulkan::detail
