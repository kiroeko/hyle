#include "vulkan_tensor.h"

#include <algorithm>
#include <array>
#include <limits>

#include "hyle/compute/model_ops.h"
#include "vulkan_shaders.h"
#include "../compute_batch/row_cache_map.h"

namespace hyle::compute::vulkan::detail {
Status lower_row_cache_indexer(ProgramState &program, const PlanStep &step,
                              RowCacheAccessFacts *access) noexcept try {
  if (access) *access = {};
  const auto paged = step.operation.custom_as<compute::detail::RowCachePagedIndexerScores>();
  const auto input_count = paged ? 5u : 4u;
  if (step.inputs.size() != input_count || step.outputs.size() != 1)
    return Status::InvalidArgument("Vulkan row indexer ports differ");
  std::array<TensorDesc, 5> inputs;
  for (std::size_t i = 0; i < input_count; ++i)
    inputs[i] = program.values[step.inputs[i]].description;
  std::array<TensorDesc, 1> output;
  HYLE_RETURN_IF_ERROR(infer_output_descriptions(step.operation,
      std::span(inputs).first(input_count), output));
  if (output[0] != program.values[step.outputs[0]].description)
    return Status::InvalidArgument("Vulkan row indexer output differs");
  const auto tokens = std::uint64_t(inputs[0].dimensions()[0]);
  const auto columns = paged ? std::uint64_t(paged->score_columns)
                            : std::uint64_t(inputs[1].dimensions()[0]);
  if (tokens > UINT32_MAX || columns > UINT32_MAX)
    return Status::Unimplemented("Vulkan row indexer native index extent");
  RowCacheAccessFacts fact;
  if (paged)
    fact = {.kind = RowCacheAccessKind::kPagedIndexerScores,
        .bank_value = step.inputs[1], .bank_input = 1, .bank_binding = 1,
        .query_rows = std::uint32_t(tokens),
        .max_requests = paged->mapping.max_requests(),
        .max_refs = paged->mapping.max_refs(), .page_size = paged->mapping.page_size,
        .score_columns = paged->score_columns};
  const auto first_dispatch = program.dispatches.size();
  const auto finish = [&] {
    fact.dispatch_begin = first_dispatch;
    fact.dispatch_end = program.dispatches.size();
    if (access) *access = fact;
    return Status::Ok();
  };
  if (!tokens) return finish();
  const auto &limits = program.runtime->device->properties.limits;
  const auto bindings = paged ? 7u : 6u, push_bytes = paged ? 24u : 20u;
  if (limits.maxComputeWorkGroupInvocations < 128 || limits.maxComputeWorkGroupSize[0] < 128 ||
      limits.maxComputeSharedMemorySize < 192 * 4 ||
      limits.maxPerStageDescriptorStorageBuffers < bindings ||
      limits.maxDescriptorSetStorageBuffers < bindings ||
      limits.maxPushConstantsSize < push_bytes)
    return Status::Unimplemented("Vulkan row indexer workgroup/descriptor limits");
  for (const auto &description : std::span(inputs).first(input_count)) {
    HYLE_ASSIGN_OR_RETURN(auto bytes, description.required_bytes());
    if (bytes > limits.maxStorageBufferRange || bytes > UINT32_MAX - 3u)
      return Status::Unimplemented("Vulkan row indexer byte-address capacity");
  }
  HYLE_ASSIGN_OR_RETURN(auto output_bytes, output[0].required_bytes());
  if (output_bytes > limits.maxStorageBufferRange || output_bytes > UINT32_MAX - 3u)
    return Status::Unimplemented("Vulkan row indexer output byte-address capacity");
  while (program.aliases.size() < program.values.size()) {
    const auto id = std::uint32_t(program.aliases.size());
    program.aliases.push_back({id, 0});
  }
  if (!program.status_value) {
    HYLE_ASSIGN_OR_RETURN(auto description, TensorDesc::dense(DataType::kInt32, {}));
    const auto id = std::uint32_t(program.values.size());
    program.values.push_back({description, {}});
    program.aliases.push_back({id, 0});
    program.status_value = id;
  }
  const auto append = [&](std::uint32_t phase, std::uint64_t groups) -> Status {
    const auto x = std::min(groups, std::uint64_t(limits.maxComputeWorkGroupCount[0]));
    if (!x || (groups + x - 1) / x > limits.maxComputeWorkGroupCount[1])
      return Status::Unimplemented("Vulkan row indexer native dispatch grid");
    const auto m = paged ? paged->mapping : compute::detail::RowCacheMap{};
    const std::array<SpecializationConstant, 11> constants{{
        {0, phase}, {1, std::uint32_t(tokens)}, {2, std::uint32_t(columns)},
        {3, std::uint32_t(m.query_offset / 4)}, {4, std::uint32_t(m.kv_offset / 4)},
        {5, std::uint32_t(m.page_offset / 4)}, {6, std::uint32_t(m.tail_offset / 4)},
        {7, m.page_size}, {8, m.max_pages},
        {9, paged ? m.max_requests() : 0u}, {10, paged ? m.max_refs() : 0u}}};
    Dispatch dispatch;
    dispatch.values = {step.inputs[0], step.inputs[1], *program.status_value,
                       step.inputs[2], step.inputs[3]};
    if (paged) dispatch.values.push_back(step.inputs[4]);
    dispatch.values.push_back(step.outputs[0]);
    HYLE_ASSIGN_OR_RETURN(dispatch.pipeline, create_pipeline(program.runtime->device,
        {paged ? shaders::row_cache_paged_indexer() : shaders::row_cache_indexer(),
         bindings, push_bytes, std::span(constants).first(paged ? 11u : 3u)}));
    dispatch.groups = {std::uint32_t(x), std::uint32_t((groups + x - 1) / x), 1};
    dispatch.tensor_byte_origins = true;
    dispatch.checked_status = true;
    dispatch.push.resize(push_bytes);
    program.dispatches.push_back(std::move(dispatch));
    return Status::Ok();
  };
  // All metadata and ranges retire through the existing checked-status barrier
  // before any invocation of the original packed-key mathematics can read bank.
  // The same token-sized grid strides over actual GPU request/ref counts;
  // reserving a larger metadata capacity never launches capacity-sized work.
  HYLE_RETURN_IF_ERROR(append(0, (tokens + 127) / 128));
  if (columns) HYLE_RETURN_IF_ERROR(append(1, tokens * columns));
  return finish();
} catch (const std::bad_alloc &) {
  return Status::ResourceExhausted("Vulkan row indexer metadata");
}
} // namespace hyle::compute::vulkan::detail
