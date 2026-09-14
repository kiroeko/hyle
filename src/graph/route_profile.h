#pragma once

#include "hyle/container/small_vector.h"
#include "hyle/graph/executable.h"

namespace hyle::graph::detail {

struct GraphState;
struct PlanState;
struct ReportState;

// Cold report storage. No owner/borrow of a Context, artifact or payload.
struct GraphProfileState final {
  explicit GraphProfileState(mem::AllocatorRef allocator) noexcept
      : targets(allocator), providers(allocator), inputs(allocator),
        outputs(allocator) {}

  compute::RouteProfileIdentity view() const noexcept {
    auto result = identity;
    result.targets = targets;
    result.providers = providers;
    result.inputs = inputs;
    result.outputs = outputs;
    return result;
  }

  compute::RouteProfileIdentity
      identity{}; // Spans stay empty; view binds them.
  container::SmallVector<compute::ProfileTargetIdentity, 0, mem::AllocatorRef>
      targets;
  container::SmallVector<compute::ProfileProviderIdentity, 0, mem::AllocatorRef>
      providers;
  container::SmallVector<compute::TensorDesc, 0, mem::AllocatorRef> inputs;
  container::SmallVector<compute::TensorDesc, 0, mem::AllocatorRef> outputs;
};

Status validate_profile_context(CompileEnvironment environment,
                                CompileOptions options) noexcept;
Status build_profile_identity(ReportState &report, const PlanState &plan,
                              const GraphState &logical,
                              const CompileProfileContext &context) noexcept;

} // namespace hyle::graph::detail
