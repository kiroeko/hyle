#include "vulkan_tensor.h"

#include <algorithm>
#include <array>
#include <bit>
#include <new>

namespace hyle::compute::vulkan::detail {
Status lower_scatter(ProgramState &program, const PlanStep &step) noexcept {
  try {
    const auto base = program.values[step.inputs[0]].description;
    const auto indices = program.values[step.inputs[1]].description;
    const auto updates = program.values[step.inputs[2]].description;
    const auto operation = *step.operation.get_if<Scatter>();
    const bool adding = operation.kind == ScatterKind::kAdd;
    const bool checked = operation.bounds == IndexBoundsPolicy::kChecked;
    if (adding && *tensor_storage_type(base.data_type()) > 2)
      return Status::Unimplemented("Vulkan Scatter add requires F32/F16/BF16");
    const auto count = *indices.element_count();
    if (count == 0)
      return lower_tensor(program, {Operation(Copy{}),
                                    {step.inputs[0]},
                                    step.outputs,
                                    {},
                                    {1, 1, 1},
                                    {}});
    const auto axis = static_cast<std::size_t>(
        operation.axis < 0
            ? operation.axis + static_cast<std::int32_t>(base.rank())
            : operation.axis);
    const auto extent = static_cast<std::uint64_t>(base.dimensions()[axis]);
    const auto &limits = program.runtime->device->properties.limits;
    if (count > limits.maxStorageBufferRange / (adding ? 8 : 4) ||
        (adding && count > INT32_MAX / 2) ||
        limits.maxComputeWorkGroupInvocations < 128 ||
        limits.maxComputeWorkGroupSize[0] < 128)
      return Status::Unimplemented(
          "Vulkan Scatter metadata exceeds native limits");
    while (program.aliases.size() < program.values.size()) {
      const auto id = static_cast<std::uint32_t>(program.aliases.size());
      program.aliases.push_back({id, 0});
    }
    const auto append = [&](const TensorDesc &description) {
      const auto id = static_cast<std::uint32_t>(program.values.size());
      program.values.push_back({description, {}});
      program.aliases.push_back({id, 0});
      return id;
    };
    const auto alias = [&](const TensorDesc &description,
                           std::uint32_t source) {
      const auto root = program.aliases[source];
      const auto id = append(description);
      program.aliases[id] = root;
      return id;
    };
    const auto pack = [&](std::uint32_t source) -> Result<std::uint32_t> {
      const auto description = program.values[source].description;
      if (description.is_dense())
        return source;
      HYLE_ASSIGN_OR_RETURN(
          auto dense,
          TensorDesc::dense(description.data_type(), description.dimensions()));
      const auto output = append(dense);
      HYLE_RETURN_IF_ERROR(lower_tensor(
          program, {Operation(Copy{}), {source}, {output}, {}, {1, 1, 1}, {}}));
      return output;
    };
    if (checked && !program.status_value) {
      HYLE_ASSIGN_OR_RETURN(auto status,
                            TensorDesc::dense(DataType::kInt32, {}));
      program.status_value = append(status);
    }
    HYLE_ASSIGN_OR_RETURN(
        auto safe_desc,
        TensorDesc::dense(DataType::kInt32, indices.dimensions()));
    const auto safe = append(safe_desc);
    HYLE_RETURN_IF_ERROR(
        lower_index_check(program, step.inputs[1], safe, extent, checked));
    // Check nonempty indices even when a different dimension has no payload.
    if (*base.element_count() == 0)
      return Status::Ok();
    // The vendor computes log2 in F32. Exact powers of two avoid rounding a
    // large non-power width down; NaN sentinels sort after every encoded key.
    const auto padded = std::bit_ceil(static_cast<std::uint32_t>(count));
    const auto max_group =
        std::bit_floor(limits.maxComputeWorkGroupInvocations);
    const auto pipeline = (std::min)(std::bit_width(padded) - 1u, 10u);
    const auto group = (std::min)(1u << pipeline, max_group);
    const bool small =
        padded <= max_group &&
        std::uint64_t{padded} * 8 <= limits.maxComputeSharedMemorySize;
    const auto smallest_group =
        small ? padded
              : (std::min)(1u << (pipeline > 2 ? pipeline - 2 : 0), max_group);
    if (padded > limits.maxStorageBufferRange / 8 ||
        group > limits.maxComputeWorkGroupSize[0] ||
        std::uint64_t{group} * 8 > limits.maxComputeSharedMemorySize ||
        (padded + smallest_group - 1) / smallest_group >
            limits.maxComputeWorkGroupCount[0])
      return Status::Unimplemented(
          "Vulkan Scatter native sort padding exceeds device limits");
    const std::array<std::int64_t, 1> index_shape{
        static_cast<std::int64_t>(count)};
    HYLE_ASSIGN_OR_RETURN(
        auto key_desc, TensorDesc::dense(DataType::kFloat32,
                                         std::array<std::int64_t, 1>{padded}));
    HYLE_ASSIGN_OR_RETURN(auto unused_desc,
                          TensorDesc::dense(DataType::kFloat32, index_shape));
    HYLE_ASSIGN_OR_RETURN(auto order_desc,
                          TensorDesc::dense(DataType::kInt32, index_shape));
    const auto safe_vector = alias(order_desc, safe);
    const auto keys = append(key_desc), unused_values = append(unused_desc),
               order = append(order_desc);
    HYLE_RETURN_IF_ERROR(
        lower_scatter_dispatch(program, 0, std::array{safe_vector}, keys,
                               static_cast<std::uint32_t>(count), 0, 0));
    Plan sorting;
    sorting.outputs = {
        order}; // Do not execute the unused TopK values/GetRows branch.
    sorting.steps.push_back(
        {Operation(
             TopK{.count = static_cast<std::size_t>(count), .largest = false}),
         {keys},
         {unused_values, order},
         {},
         {1, 1, 1},
         {}});
    std::vector<std::size_t> last_uses(program.values.size(), 0);
    HYLE_RETURN_IF_ERROR(lower_region(program, sorting, 0, 1, last_uses));
    std::uint64_t inner = 1;
    for (std::size_t i = axis + 1; i < base.rank(); ++i)
      inner *= static_cast<std::uint64_t>(base.dimensions()[i]);
    const auto outer = *base.element_count() / extent / inner;
    HYLE_ASSIGN_OR_RETURN(auto packed_updates, pack(step.inputs[2]));
    HYLE_ASSIGN_OR_RETURN(
        auto updates_matrix,
        TensorDesc::dense(
            base.data_type(),
            std::array<std::int64_t, 3>{static_cast<std::int64_t>(outer),
                                        static_cast<std::int64_t>(count),
                                        static_cast<std::int64_t>(inner)}));
    const auto update_rows = alias(updates_matrix, packed_updates);
    if (!adding) {
      const auto combined_elements =
          *base.element_count() + *updates.element_count();
      if (combined_elements > INT32_MAX ||
          combined_elements >
              limits.maxStorageBufferRange / data_type_size(base.data_type()))
        return Status::Unimplemented(
            "Vulkan Scatter combined row table exceeds device range");
      HYLE_ASSIGN_OR_RETURN(auto packed_base, pack(step.inputs[0]));
      HYLE_ASSIGN_OR_RETURN(
          auto base_matrix,
          TensorDesc::dense(
              base.data_type(),
              std::array<std::int64_t, 3>{static_cast<std::int64_t>(outer),
                                          static_cast<std::int64_t>(extent),
                                          static_cast<std::int64_t>(inner)}));
      const auto base_rows = alias(base_matrix, packed_base);
      HYLE_ASSIGN_OR_RETURN(
          auto combined_desc,
          TensorDesc::dense(base.data_type(),
                            std::array<std::int64_t, 3>{
                                static_cast<std::int64_t>(outer),
                                static_cast<std::int64_t>(extent + count),
                                static_cast<std::int64_t>(inner)}));
      const auto combined = append(combined_desc);
      HYLE_RETURN_IF_ERROR(lower_tensor(program, {Operation(Concat{1}),
                                                  {base_rows, update_rows},
                                                  {combined},
                                                  {},
                                                  {1, 1, 1},
                                                  {}}));
      HYLE_ASSIGN_OR_RETURN(
          auto map_desc,
          TensorDesc::dense(
              DataType::kInt32,
              std::array<std::int64_t, 1>{static_cast<std::int64_t>(extent)}));
      const auto mapped = append(map_desc);
      HYLE_RETURN_IF_ERROR(
          lower_scatter_dispatch(program, 1, std::array{safe_vector, order},
                                 mapped, static_cast<std::uint32_t>(count),
                                 static_cast<std::uint32_t>(extent), 0));
      const auto output_rows = alias(base_matrix, step.outputs[0]);
      // Reuse the admitted raw-word GetRows adapter and its final checked
      // publication; a prior bad input index remains set in the shared status.
      return lower_gather(program, {Operation(Gather{1, operation.bounds}),
                                    {combined, mapped},
                                    {output_rows},
                                    {},
                                    {1, 1, 1},
                                    {}});
    }
    HYLE_ASSIGN_OR_RETURN(
        auto pairs_desc,
        TensorDesc::dense(
            DataType::kInt32,
            std::array<std::int64_t, 1>{static_cast<std::int64_t>(count * 2)}));
    const auto pairs = append(pairs_desc);
    HYLE_RETURN_IF_ERROR(lower_tensor(program, {Operation(Concat{0}),
                                                {safe_vector, order},
                                                {pairs},
                                                {},
                                                {1, 1, 1},
                                                {}}));
    const auto output =
        checked ? append(program.values[step.outputs[0]].description)
                : step.outputs[0];
    HYLE_RETURN_IF_ERROR(lower_scatter_dispatch(
        program, 2, std::array{step.inputs[0], update_rows, pairs}, output,
        static_cast<std::uint32_t>(count), static_cast<std::uint32_t>(extent),
        static_cast<std::uint32_t>(inner)));
    return checked ? lower_checked_copy(program, output, step.outputs[0])
                   : Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan Scatter adapter metadata");
  }
}
} // namespace hyle::compute::vulkan::detail
