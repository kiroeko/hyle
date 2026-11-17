#pragma once

#include "hyle/compute/vulkan.h"
#include "hyle/compute/batch_layout.h"

namespace hyle::compute::vulkan::detail {
HYLE_COMPUTE_ROUTE_VULKAN_API Status qualify_device_lengths_paged_schema(
    const BatchSchema &, float scale, bool causal, std::int32_t window_left) noexcept;

// Cold native recipe, with Q/K/V, actual I32 Q/KV lengths and the original
// readonly capacity metadata. Lengths only shorten the owned page envelope.
// The typed Graph caller still owns page admission, dependencies and publication.
HYLE_COMPUTE_ROUTE_VULKAN_API Result<Program> prepare_device_lengths_paged_graph(
    std::shared_ptr<Runtime>, BatchSchema, std::uint32_t physical_query_rows,
    float scale, bool causal, std::int32_t window_left,
    const WorkloadContext & = {}, bool guard_writes = false) noexcept;

// Private preparation descriptor, never a second executor or public operation.
struct DeviceLengthsPagedRead final {
  BatchSchema schema;
  std::uint32_t rows = 0;
  float scale = 1.0F;
  bool causal = true;
  std::int32_t window_left = -1;
  bool guard_writes = false;
  static constexpr OperationId operation_id{0x48594c45564b444cULL, 0x5041474544520001ULL};
  static constexpr std::string_view operation_name = "vulkan-device-lengths-paged-read";
  constexpr OperationTraits operation_traits() const noexcept {
    const auto count = std::uint16_t(guard_writes ? 8 : 6);
    return {{count, count}, 1, AliasPolicy::kNone, EffectKind::kPure,
        PrecisionPolicy::kExact, DataType::kFloat32};
  }
  Status infer_output_descriptions(std::span<const TensorDesc>, std::span<TensorDesc>) const noexcept;
};
static_assert(compute::detail::InlineCustomOperationDescriptor<DeviceLengthsPagedRead>);
struct ProgramState;
struct PlanStep;
Status lower_device_lengths_paged(ProgramState &, const PlanStep &) noexcept;
} // namespace hyle::compute::vulkan::detail
