#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "../compute/native_route.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/compute/prepared_operation.h"

namespace hyle::compute::native::internal {

struct NativePlan;
enum class ReductionRoute : std::uint8_t {
  kGeneral = 0,
  kAllDense,
  kLastAxisDense,
};
using SubmitRoute = detail::EnqueueResult (*)(
    const NativePlan &plan, std::span<const ConstTensorView> inputs,
    std::span<const TensorView> outputs) noexcept;
using WorkspaceSubmitRoute = detail::EnqueueResult (*)(
    const NativePlan &plan, std::span<const ConstTensorView> inputs,
    std::span<const TensorView> outputs, mem::BufferView workspace) noexcept;
using LinearElementwiseRoute = void (*)(const void *first, const void *second,
                                        const void *third, void *output,
                                        std::size_t count) noexcept;
using ScalarElementwiseRoute = void (*)(const std::byte *first,
                                        const std::byte *second,
                                        const std::byte *third,
                                        std::byte *output) noexcept;
using RopeKernelRoute = void (*)(const float *input, const float *frequencies,
                                 float *output, std::size_t pairs,
                                 float position) noexcept;
using ParallelRangeRoute = void (*)(void *context, std::size_t task_index,
                                    std::size_t begin,
                                    std::size_t end) noexcept;

// Cold prepared metadata shared by the native operation families. A family
// selects one submit route during prepare; submit never searches by kind or
// grows metadata. The compact scalar/parameter slots hold only route-specific
// immutable values, not benchmark-driven per-tensor state.
struct NativePlan final {
  ParallelExecutorRef executor{};
  detail::NativeParameters parameters{};
  float scalar = 0.0F;
  std::uint64_t scalar_bits = 0;
  ReductionKind reduction_kind = ReductionKind::kSum;
  ReductionRoute reduction_route = ReductionRoute::kGeneral;
  std::uint8_t reduction_axes_mask = 0;
  std::size_t axis = 0;
  std::size_t count = 0;
  std::size_t rotary_dimensions = 0;
  std::size_t sequence_axis = 0;
  IndexBoundsPolicy index_bounds = IndexBoundsPolicy::kChecked;
  ScatterKind scatter_kind = ScatterKind::kAssign;
  RopeLayout rope_layout = RopeLayout::kInterleaved;
  SoftmaxKind softmax_kind = SoftmaxKind::kSoftmax;
  MaskKind mask_kind = MaskKind::kNone;
  AttentionLayout attention_layout = AttentionLayout::kBatchSequenceHeadDepth;
  bool largest = true;
  bool sorted = true;
  bool stable = true;
  bool causal = false;
  std::size_t attention_batches = 0;
  std::size_t attention_heads = 0;
  std::size_t attention_queries = 0;
  std::size_t attention_keys = 0;
  std::size_t attention_query_depth = 0;
  std::size_t attention_value_depth = 0;
  std::size_t attention_workspace_stride = 0;
  WorkspaceRequirement workspace{};
  detail::NativeRoute route = nullptr;
  LinearElementwiseRoute linear_elementwise_route = nullptr;
  ScalarElementwiseRoute scalar_elementwise_route = nullptr;
  RopeKernelRoute rope_kernel_route = nullptr;
  SubmitRoute submit_route = nullptr;
  WorkspaceSubmitRoute workspace_submit_route = nullptr;
  std::size_t parallel_tasks = 1;
  std::size_t row_first_stride = 0;
  std::size_t row_second_stride = 0;
  std::size_t row_output_stride = 0;
  std::size_t alias_input0_offset = 0;
  std::uint8_t alias_input_mask = 0;
};

struct ParallelRangeEpisode final {
  std::size_t units = 0;
  std::size_t tasks = 1;
  ParallelRangeRoute route = nullptr;
  void *context = nullptr;
};

inline void execute_parallel_range(void *episode_value, std::size_t index,
                                   std::size_t count) noexcept {
  auto &episode = *static_cast<ParallelRangeEpisode *>(episode_value);
  if (episode.route == nullptr || count != episode.tasks || count == 0 ||
      index >= count)
    return;
  const std::size_t base = episode.units / count;
  const std::size_t remainder = episode.units % count;
  const std::size_t begin = index * base + (std::min)(index, remainder);
  const std::size_t end =
      begin + base + static_cast<std::size_t>(index < remainder);
  if (begin != end)
    episode.route(episode.context, index, begin, end);
}

// Executes one synchronous fan-out over independent work units. The task count
// is selected during prepare; submit performs no allocation or policy search.
// A detectable nested episode deliberately stays on the caller thread. A
// backing without in_parallel() remains responsible for its documented nested
// progress guarantee.
inline void parallel_for_ranges(const NativePlan &plan, std::size_t units,
                                ParallelRangeRoute route,
                                void *context) noexcept {
  if (units == 0 || route == nullptr)
    return;
  const std::size_t tasks = (std::min)(plan.parallel_tasks, units);
  if (tasks <= 1 || plan.executor.in_parallel()) {
    route(context, 0, 0, units);
    return;
  }
  ParallelRangeEpisode episode{
      .units = units,
      .tasks = tasks,
      .route = route,
      .context = context,
  };
  plan.executor.parallel_for(tasks, &execute_parallel_range, &episode);
}

inline void preserve_first_error(Status &terminal, Status candidate) noexcept {
  if (terminal.ok() && !candidate.ok())
    terminal = std::move(candidate);
}

inline detail::EnqueueResult completed(Status status) noexcept {
  // Native routes execute synchronously on the admitted CPU Queue. Treat the
  // provider episode as accepted even when a mapping close/flush reports its
  // terminal error, so callers never infer that an output remained untouched.
  return detail::EnqueueResult::accepted_command(std::move(status));
}

inline bool aligned_for_storage(const void *address,
                                std::size_t alignment) noexcept {
  return address == nullptr ||
         (alignment != 0 &&
          (reinterpret_cast<std::uintptr_t>(address) & (alignment - 1)) == 0);
}

// TensorDesc construction already proved that every non-negative stride and
// the complete byte extent fit size_t. This bounded-rank helper therefore
// needs no checked arithmetic in the steady reference loop.
inline std::size_t logical_byte_offset(const TensorDesc &description,
                                       std::size_t linear_index) noexcept {
  std::size_t offset = 0;
  const auto dimensions = description.dimensions();
  const auto strides = description.byte_strides();
  for (std::size_t reverse = 0; reverse < dimensions.size(); ++reverse) {
    const std::size_t axis = dimensions.size() - 1 - reverse;
    const std::size_t extent = static_cast<std::size_t>(dimensions[axis]);
    const std::size_t coordinate = linear_index % extent;
    linear_index /= extent;
    offset += coordinate * static_cast<std::size_t>(strides[axis]);
  }
  return offset;
}

// Map one dense output coordinate back into a broadcast-compatible input.
// The schema has already proved rank and dimension compatibility.
inline std::size_t
logical_broadcast_byte_offset(const TensorDesc &input,
                              std::span<const std::int64_t> output_dimensions,
                              std::size_t linear_index) noexcept {
  std::size_t offset = 0;
  const auto input_dimensions = input.dimensions();
  const auto input_strides = input.byte_strides();
  for (std::size_t reverse = 0; reverse < output_dimensions.size(); ++reverse) {
    const std::size_t output_axis = output_dimensions.size() - 1 - reverse;
    const std::size_t output_extent =
        static_cast<std::size_t>(output_dimensions[output_axis]);
    const std::size_t coordinate = linear_index % output_extent;
    linear_index /= output_extent;
    if (reverse >= input_dimensions.size())
      continue;
    const std::size_t input_axis = input_dimensions.size() - 1 - reverse;
    if (input_dimensions[input_axis] != 1) {
      offset +=
          coordinate * static_cast<std::size_t>(input_strides[input_axis]);
    }
  }
  return offset;
}

Status prepare_transform(Operation operation,
                         std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs,
                         NativePlan &plan) noexcept;
Status prepare_cast(Operation operation, std::span<const TensorDesc> inputs,
                    std::span<const TensorDesc> outputs,
                    NativePlan &plan) noexcept;
Status prepare_elementwise(Operation operation,
                           std::span<const TensorDesc> inputs,
                           std::span<const TensorDesc> outputs,
                           NativePlan &plan) noexcept;
Status prepare_reduction(Operation operation,
                         std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs,
                         NativePlan &plan) noexcept;
Status prepare_layer_norm(Operation operation,
                          std::span<const TensorDesc> inputs,
                          std::span<const TensorDesc> outputs,
                          NativePlan &plan) noexcept;
Status prepare_indexing(Operation operation, std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs,
                        NativePlan &plan) noexcept;
Status prepare_transformer(Operation operation,
                           std::span<const TensorDesc> inputs,
                           std::span<const TensorDesc> outputs,
                           NativePlan &plan) noexcept;
Status prepare_attention(Operation operation,
                         std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs,
                         NativePlan &plan) noexcept;

} // namespace hyle::compute::native::internal
