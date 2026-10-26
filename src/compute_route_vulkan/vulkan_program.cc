#include "../compute_ops/view_alias.h"
#include "vulkan_iree.h"
#include "vulkan_program_internal.h"
#include "vulkan_carrier.h"
#include "vulkan_iq1.h"
#include "vulkan_policy.h"
#include "vulkan_tensor.h"
#include "../compute_batch/row_cache_map.h"
#include "../compute_packed_shared/static_fp8_linear.h"
#include "vulkan_memory_internal.h"
#include "vulkan_composition_access.h"
#include "hyle/compute/detail/vulkan_batch_access.h"

#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/sampling_ops.h"
#include "hyle/obs/trace.h"

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <new>
#include <numeric>
#include <unordered_map>
#include <utility>

namespace hyle::compute::vulkan::detail {

Status plan_storage(ProgramState &program) noexcept {
  try {
    const auto none = (std::numeric_limits<std::uint32_t>::max)();
    program.block_bytes.clear();
    const auto count = program.values.size();
    std::vector<std::uint32_t> first(count, none), last(count, 0);
    std::vector<bool> external(count, false);
    for (const auto input : program.inputs)
      external[input] = true;
    if (program.status_value)
      external[*program.status_value] = true;
    for (std::size_t value = 0; value < count; ++value)
      external[value] =
          external[value] || program.values[value].constant.valid();
    for (std::uint32_t index = 0; index < program.dispatches.size(); ++index)
      for (const auto alias : program.dispatches[index].values) {
        const auto value = program.aliases[alias].root;
        first[value] = (std::min)(first[value], index);
        last[value] = index;
      }
    for (const auto alias : program.outputs) {
      const auto output = program.aliases[alias].root;
      last[output] = static_cast<std::uint32_t>(program.dispatches.size());
      // An empty computed value still has a valid result owner, even when no
      // payload dispatch needs it. Keep separate published outputs distinct.
      if (first[output] == none)
        first[output] = 0;
    }
    std::vector<std::uint32_t> order(count);
    std::iota(order.begin(), order.end(), 0u);
    std::ranges::stable_sort(order, [&](auto left, auto right) {
      return first[left] < first[right];
    });
    program.storage_blocks.assign(count, none);
    std::vector<std::uint32_t> block_last;
    for (const auto value : order) {
      if (external[value] || first[value] == none)
        continue;
      HYLE_ASSIGN_OR_RETURN(
          auto bytes,
          buffer_capacity(*program.values[value].description.required_bytes()));
      std::uint32_t selected = none;
      VkDeviceSize smallest_growth = (std::numeric_limits<VkDeviceSize>::max)();
      for (std::uint32_t block = 0; block < block_last.size(); ++block) {
        // Strict inequality is essential: a current dispatch still reads an
        // input whose last use is this index. Its output cannot alias it.
        if (block_last[block] >= first[value])
          continue;
        const auto growth = bytes > program.block_bytes[block]
                                ? bytes - program.block_bytes[block]
                                : 0;
        if (growth < smallest_growth) {
          selected = block;
          smallest_growth = growth;
        }
      }
      if (selected == none) {
        selected = static_cast<std::uint32_t>(program.block_bytes.size());
        program.block_bytes.push_back(bytes);
        block_last.push_back(last[value]);
      } else {
        program.block_bytes[selected] =
            (std::max)(program.block_bytes[selected],
                       static_cast<VkDeviceSize>(bytes));
        block_last[selected] = last[value];
      }
      program.storage_blocks[value] = selected;
    }
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan liveness plan metadata");
  }
}

Status allocate_values(const ProgramState &program, Run &run) noexcept {
  try {
    std::vector<std::shared_ptr<Buffer>> blocks;
    blocks.reserve(program.block_bytes.size());
    for (const auto bytes : program.block_bytes) {
      auto buffer = allocate_buffer(program.runtime->device, bytes, false, false,
          {.budget = run.allocation.resource_budget,
           .resource = run.allocation.device});
      if (!buffer)
        return std::move(buffer).error();
      blocks.push_back(std::move(*buffer));
    }
    run.retained_buffers.resize(program.values.size());
    for (std::size_t value = 0; value < program.values.size(); ++value) {
      if (program.status_value == value) {
        HYLE_ASSIGN_OR_RETURN(
            run.retained_buffers[value],
            allocate_buffer(program.runtime->device, 4, true, true,
                {.budget = run.allocation.resource_budget,
                 .resource = run.allocation.staging}));
      } else if (program.values[value].constant)
        run.retained_buffers[value] =
            Access::buffer(program.values[value].constant);
      else if (program.storage_blocks[value] !=
               (std::numeric_limits<std::uint32_t>::max)())
        run.retained_buffers[value] = blocks[program.storage_blocks[value]];
    }
    for (std::size_t value = 0; value < program.values.size(); ++value) {
      const auto alias = program.aliases[value];
      if (alias.root == value || program.values[value].constant)
        continue;
      auto buffer = std::make_shared<Buffer>();
      buffer->device = program.runtime->device;
      bind_view_buffer(*buffer, run.retained_buffers[alias.root],
                       alias.byte_offset);
      run.retained_buffers[value] = std::move(buffer);
    }
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan invocation resources");
  }
}

namespace {
Result<std::uint32_t> owned_output_block(const ProgramState &program,
    const Run &run, std::uint32_t output) noexcept {
  const auto root = program.aliases[output].root;
  const auto none = (std::numeric_limits<std::uint32_t>::max)();
  const auto block = program.storage_blocks[root];
  const auto &buffer = run.retained_buffers[root];
  if (block == none || block >= program.block_bytes.size() ||
      program.values[root].constant ||
      std::ranges::find(program.inputs, root) != program.inputs.end() ||
      !buffer || buffer->device != program.runtime->device || buffer->is_view ||
      buffer->external_owner || buffer->memory == VK_NULL_HANDLE || buffer->offset ||
      buffer->bytes != program.block_bytes[block])
    return std::unexpected(Status::Unimplemented(
        "Vulkan retained output has no original owned storage block"));
  return block;
}

bool official_word_storage(const TensorDesc &description) noexcept {
  if (description.rank() != 3)
    return false;
  const auto shape = description.dimensions();
  const auto fp8 = float8_grouped_weight_description(shape[0], shape[1], shape[2]);
  const auto fp4 = float4_grouped_weight_description(shape[0], shape[1], shape[2]);
  const auto lanes = fp8 && *fp8 == description ? 4
                     : fp4 && *fp4 == description ? 8 : 0;
  if (lanes == 0 || shape[2] / lanes == 0 ||
      shape[0] > INT32_MAX / shape[1])
    return false;
  return shape[0] * shape[1] <= INT32_MAX / (shape[2] / lanes);
}
Status prepare_constant_casts(const std::shared_ptr<Runtime> &runtime,
                              ProgramState &program,
                              const Plan &plan,
                              std::vector<bool> &prepared) {
  std::vector<bool> published(program.values.size(), false);
  for (const auto output : plan.outputs) published[output] = true;
  for (auto node = plan.steps.rbegin(); node != plan.steps.rend(); ++node)
    if (node->operation.get_if<View>() && published[node->outputs[0]])
      published[node->inputs[0]] = true;
  std::vector<bool> derived;
  std::vector<Tensor> converted;
  for (std::size_t index = 0; index < plan.steps.size(); ++index) {
    const auto &step = plan.steps[index];
    if (const auto *view = step.operation.get_if<View>();
        view && !derived.empty() && derived[step.inputs[0]]) {
      HYLE_ASSIGN_OR_RETURN(auto value,
          program.values[step.inputs[0]].constant.view(*view));
      program.values[step.outputs[0]].constant = std::move(value);
      derived[step.outputs[0]] = true;
      continue;
    }
    const auto *cast = step.operation.get_if<Cast>();
    if (!cast || cast->output_type != DataType::kFloat32 ||
        step.inputs.size() != 1 || step.outputs.size() != 1)
      continue;
    const auto source = step.inputs[0], output = step.outputs[0];
    const auto &from = program.values[source].description;
    const auto &to = program.values[output].description;
    const auto constant = program.values[source].constant;
    if (!constant || published[output] || !from.is_plain() || !from.is_dense() ||
        from.rank() != 2 || from.data_type() != DataType::kBFloat16 ||
        !to.is_plain() || !to.is_dense())
      continue;
    if (converted.empty()) {
      converted.resize(program.values.size());
      derived.resize(program.values.size(), false);
      prepared.resize(plan.steps.size(), false);
    }
    if (!converted[source]) {
      const std::array arguments{Argument(constant)};
      HYLE_ASSIGN_OR_RETURN(auto cast_program,
          prepare_operation(runtime, step.operation, arguments));
      HYLE_ASSIGN_OR_RETURN(auto invocation, cast_program.create_invocation());
      HYLE_ASSIGN_OR_RETURN(auto work, invocation.submit());
      HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
      converted[source] = result.get()->tensors()[0];
    }
    program.values[output].constant = converted[source];
    derived[output] = prepared[index] = true;
  }
  return Status::Ok();
}

Status pack_matrix_constants(const std::shared_ptr<Runtime> &runtime,
                             ProgramState &program,
                             const Plan &source_plan, Plan &packed_plan) {
  // The same immutable source can have native and promoted consumers. Their
  // physical packs have different dtypes and cannot share a replacement.
  std::array<std::vector<std::uint32_t>, 2> replacements;
  for (std::size_t index = 0; index < source_plan.steps.size(); ++index) {
    const auto &step = source_plan.steps[index];
    if (!step.operation.get_if<MatMul>() || step.inputs.size() != 2 ||
        !matches_mature_default(program, step, DefaultCarrier::kGgml) ||
        (!managed_operation(program, step) && !bfloat_region_operation(program, step)))
      continue;
    const auto source = step.inputs[1];
    const auto description = program.values[source].description;
    const auto constant = program.values[source].constant;
    const bool bfloat = description.data_type() == DataType::kBFloat16;
    const bool promote = bfloat && bfloat_region_operation(program, step);
    if (!constant || !description.is_plain() || !description.is_dense() ||
        (description.data_type() != DataType::kFloat32 &&
         description.data_type() != DataType::kFloat16 && !bfloat))
      continue;
    HYLE_ASSIGN_OR_RETURN(auto transposed, description.transposed());
    if (transposed.is_dense() && !promote) continue; // No hot conversion to remove.
    if (replacements[0].empty()) {
      for (auto &cache : replacements)
        cache.resize(program.values.size(), UINT32_MAX);
      packed_plan.steps = source_plan.steps;
      packed_plan.outputs = source_plan.outputs;
    }
    auto &replacement = replacements[promote ? 1 : 0][source];
    if (replacement == UINT32_MAX) {
      std::array<std::int32_t, kMaxTensorRank> axes{};
      std::iota(axes.begin(), axes.end(), 0);
      std::swap(axes[description.rank() - 1], axes[description.rank() - 2]);
      HYLE_ASSIGN_OR_RETURN(auto transpose,
          View::permute(std::span(axes).first(description.rank())));
      auto source_tensor = constant;
      if (promote) {
        // Reuse the original widening primitive before the physical pack.
        // Keep its owned result alive until the cold Copy has retired.
        const std::array arguments{Argument(constant)};
        HYLE_ASSIGN_OR_RETURN(auto cast,
            prepare_operation(runtime, Operation(Cast{DataType::kFloat32}),
                              arguments));
        HYLE_ASSIGN_OR_RETURN(auto slot, cast.create_invocation());
        HYLE_ASSIGN_OR_RETURN(auto work, slot.submit());
        HYLE_ASSIGN_OR_RETURN(auto ready, work.wait());
        source_tensor = ready.get()->tensors()[0];
      }
      HYLE_ASSIGN_OR_RETURN(auto view, source_tensor.view(transpose));
      // Existing byte-addressed Copy owns the cold materialization and its
      // accepted-prefix retirement. Only immutable captures enter this path.
      const std::array arguments{Argument(view)};
      HYLE_ASSIGN_OR_RETURN(auto copy,
          prepare_operation(runtime, Operation(Copy{}), arguments));
      HYLE_ASSIGN_OR_RETURN(auto slot, copy.create_invocation());
      HYLE_ASSIGN_OR_RETURN(auto work, slot.submit());
      HYLE_ASSIGN_OR_RETURN(auto ready, work.wait());
      HYLE_ASSIGN_OR_RETURN(auto packed,
          ready.get()->tensors()[0].view(transpose));
      const auto id = static_cast<std::uint32_t>(program.values.size());
      program.values.push_back({packed.description(), std::move(packed)});
      program.aliases.push_back({id, 0});
      replacement = id;
    }
    packed_plan.steps[index].inputs[1] = replacement;
  }
  return Status::Ok();
}
} // namespace

Result<Program> prepare_plan(const Context &context, const Plan &plan,
                             PlanFacts *facts) noexcept {
  return prepare_plan(Access::runtime(context), plan, facts, {});
}

Result<Program> prepare_plan(const Context &context, const Plan &plan,
                             PlanFacts *facts, ColdCustomResolver resolver) noexcept {
  return prepare_plan(Access::runtime(context), plan, facts, resolver);
}

Result<Program> prepare_plan(const std::shared_ptr<Runtime> &runtime,
                             const Plan &plan, PlanFacts *facts,
                             ColdCustomResolver resolver) noexcept {
  return prepare_plan(runtime, plan, PlanQueuePolicy::kMayDelegate, facts, resolver);
}

Result<Program> prepare_plan(const std::shared_ptr<Runtime> &runtime,
                             const Plan &plan, PlanQueuePolicy queue_policy,
                             PlanFacts *facts, ColdCustomResolver resolver) noexcept {
  HYLE_TRACE_ZONE_NAMED("hyle.vulkan.prepare");
  if (facts != nullptr)
    *facts = {};
  if (!plan.workload.valid())
    return std::unexpected(Status::InvalidArgument("invalid Vulkan workload"));
  if (!runtime || plan.values.empty() || plan.steps.empty() ||
      plan.outputs.empty() ||
      plan.values.size() > (std::numeric_limits<std::uint32_t>::max)() / 2 ||
      plan.steps.size() > (std::numeric_limits<std::uint32_t>::max)() / 2)
    return std::unexpected(
        Status::InvalidArgument("invalid Vulkan compute plan"));
  {
    std::lock_guard lock(runtime->mutex);
    if (runtime->closed)
      return std::unexpected(
          Status::FailedPrecondition("Vulkan context is closed"));
    if (!runtime->failure.ok())
      return std::unexpected(runtime->failure);
  }
  try {
    std::vector<bool> available(plan.values.size(), false);
    std::vector<bool> produced(plan.values.size(), false);
    for (std::size_t index = 0; index < plan.values.size(); ++index) {
      const auto &value = plan.values[index];
      const auto &description = value.description;
      const auto bytes = description.required_bytes();
      const auto elements = description.element_count();
      if (!description.valid() ||
          !tensor_storage_type(description.data_type()) ||
          (!description.is_plain() && !named_compressed_storage(description)) ||
          !bytes ||
          *bytes > runtime->device->properties.limits.maxStorageBufferRange ||
          !elements || (*elements > (std::numeric_limits<std::int32_t>::max)() &&
                        !iq1_storage_description(description) &&
                        !official_word_storage(description)))
        return std::unexpected(Status::Unimplemented(
            "Vulkan plan requires supported plain or named compressed "
            "storage within device limits"));
      HYLE_ASSIGN_OR_RETURN(auto capacity, buffer_capacity(*bytes));
      if (capacity > runtime->device->properties.limits.maxStorageBufferRange)
        return std::unexpected(Status::Unimplemented(
            "Vulkan padded storage exceeds device limits"));
      if (value.constant) {
        if (value.constant.description() != description ||
            Access::buffer(value.constant)->device != runtime->device)
          return std::unexpected(
              Status::InvalidArgument("Vulkan constant description or device"));
        available[index] = true;
      }
    }
    for (const auto input : plan.inputs) {
      if (input >= available.size() || available[input])
        return std::unexpected(
            Status::InvalidArgument("duplicate or constant Vulkan input"));
      available[input] = true;
    }
    for (const auto &step : plan.steps) {
      const auto traits = operation_traits(step.operation);
      if (traits.effect != EffectKind::kPure ||
          (traits.alias == AliasPolicy::kMustAliasInput0 &&
           step.operation.kind() != OperationKind::kView) ||
          traits.outputs == 0 || traits.outputs != step.outputs.size() ||
          !traits.inputs.accepts(step.inputs.size()))
        return std::unexpected(Status::Unimplemented(
            "Vulkan requires pure SSA operations with supported arity"));
      std::vector<TensorDesc> descriptions;
      descriptions.reserve(step.inputs.size());
      for (std::size_t port = 0; port < step.inputs.size(); ++port) {
        const auto input = step.inputs[port];
        if (input >= available.size() || !available[input])
          return std::unexpected(Status::InvalidArgument(
              "Vulkan plan is not topologically ordered"));
        const auto &description = plan.values[input].description;
        if (description.data_type() == DataType::kFloat8E4M3FN) {
          const bool scale = port == 2 &&
              (step.operation.custom_as<model_ops::Nvfp4Linear>() ||
               step.operation.custom_as<model_ops::GroupedNvfp4Linear>() ||
               step.operation.custom_as<model_ops::LocalNvfp4Linear>() ||
               step.operation.custom_as<model_ops::GroupedLocalNvfp4Linear>());
          const bool static_weight = port == 1 &&
              step.operation.custom_as<packed::internal::StaticFloat8Linear>().has_value();
          const bool storage = step.operation.kind() == OperationKind::kCopy ||
                               step.operation.kind() == OperationKind::kView ||
                               step.operation.kind() == OperationKind::kCast;
          // Gather's checked row adapter copies raw words. Only its data
          // port carries E4 bytes; it neither decodes nor computes in FP8.
          const bool gathered_storage = port == 0 &&
              step.operation.kind() == OperationKind::kGather;
          const auto paged = step.operation.custom_as<ggml_vulkan::detail::PagedAttentionRead>();
          const bool cache = paged && (paged->paged || paged->compact_mla) && paged->float8_cache &&
                             (port == 1 || (!paged->compact_mla && port == 2));
          if (!scale && !storage && !gathered_storage && !static_weight && !cache)
            return std::unexpected(Status::Unimplemented(
                "Vulkan E4M3 requires a typed storage or scale consumer"));
        }
        // This carrier exception does not widen ordinary tensor/shader
        // indexing. Large banks enter only the checked word-based consumer.
        if (*description.element_count() > INT32_MAX &&
            official_word_storage(description) &&
            (port != 0 ||
             (!step.operation.custom_as<model_ops::GroupedFloat8Linear>() &&
              !step.operation.custom_as<model_ops::GroupedFloat4Linear>())))
          return std::unexpected(Status::Unimplemented(
              "large official word banks require a grouped weight port"));
        descriptions.push_back(description);
      }
      std::vector<TensorDesc> inferred(step.outputs.size());
      HYLE_RETURN_IF_ERROR(
          infer_output_descriptions(step.operation, descriptions, inferred));
      for (std::size_t port = 0; port < step.outputs.size(); ++port) {
        const auto output = step.outputs[port];
        if (output >= available.size() || available[output] ||
            inferred[port] != plan.values[output].description)
          return std::unexpected(Status::InvalidArgument(
              "Vulkan plan output violates SSA or operation schema"));
        if (inferred[port].data_type() == DataType::kFloat8E4M3FN &&
            step.operation.kind() != OperationKind::kCopy &&
            step.operation.kind() != OperationKind::kView &&
            step.operation.kind() != OperationKind::kCast &&
            step.operation.kind() != OperationKind::kGather &&
            !step.operation.custom_as<model_ops::CacheEncode>() &&
            !step.operation.custom_as<model_ops::Float8BlockQuantize>())
          return std::unexpected(Status::Unimplemented(
              "Vulkan E4M3 outputs require Cast or a storage copy/gather"));
        if (*inferred[port].element_count() > INT32_MAX &&
            official_word_storage(inferred[port]))
          return std::unexpected(Status::Unimplemented(
              "large official word banks must be external readonly values"));
        available[output] = true;
        produced[output] = true;
      }
    }
    std::vector<bool> published(plan.values.size(), false);
    for (const auto output : plan.outputs) {
      if (output >= available.size() || !produced[output] || published[output])
        return std::unexpected(Status::InvalidArgument(
            "Vulkan output must name a unique computed value"));
      published[output] = true;
    }
    auto program = std::make_shared<ProgramState>();
    program->runtime = runtime;
    program->workload = plan.workload;
    program->values = plan.values;
    program->inputs = plan.inputs;
    program->outputs = plan.outputs;
    for (std::uint32_t index = 0; index < plan.values.size(); ++index)
      program->aliases.push_back({index, 0});
    for (const auto input : plan.inputs)
      program->input_descriptions.push_back(plan.values[input].description);
    for (const auto output : plan.outputs)
      program->output_descriptions.push_back(plan.values[output].description);
    std::vector<bool> prepared_casts;
    HYLE_RETURN_IF_ERROR(prepare_constant_casts(runtime, *program, plan, prepared_casts));
    Plan packed_plan;
    HYLE_RETURN_IF_ERROR(pack_matrix_constants(runtime, *program, plan, packed_plan));
    const auto &lowered = packed_plan.steps.empty() ? plan : packed_plan;
    std::vector<RowCacheAccessFacts> row_cache_accesses;
    std::vector<std::size_t> last_uses(program->values.size(), 0);
    for (std::size_t index = 0; index < lowered.steps.size(); ++index)
      for (const auto input : lowered.steps[index].inputs)
        last_uses[input] = index;
    for (const auto &step : lowered.steps)
      if (step.preserve_outputs)
        for (const auto output : step.outputs)
          last_uses[output] = lowered.steps.size();
    for (std::size_t index = 0; index < lowered.steps.size();) {
      const auto &step = lowered.steps[index];
      if (!prepared_casts.empty() && prepared_casts[index]) {
        ++index; // Keep the original logical step as a managed-region boundary.
        continue;
      }
      if(step.operation.custom_as<model_ops::CacheEncode>()){
        HYLE_RETURN_IF_ERROR(lower_cache_encoder(*program,step,last_uses));++index;continue;
      }
      if (queue_policy == PlanQueuePolicy::kRuntimeQueueOnly &&
          step.operation.get_if<Cast>()) {
        // The existing typed shader preserves Cast's rounding and byte-origin
        // contract on the original queue. Unconstrained plans keep defaults.
        HYLE_RETURN_IF_ERROR(lower_tensor(*program, step));
        ++index;
        continue;
      }
      if (matches_mature_default(*program, step, DefaultCarrier::kIree) &&
          iree_operation(*program, step)) {
        if (queue_policy == PlanQueuePolicy::kRuntimeQueueOnly)
          return std::unexpected(Status::Unimplemented("Vulkan ordered plan needs an original-queue implementation"));
        HYLE_RETURN_IF_ERROR(lower_iree(*program, step));
        ++index;
        continue;
      }
      if (matches_mature_default(*program, step, DefaultCarrier::kGgml) &&
          bfloat_region_operation(*program, step)) {
        HYLE_RETURN_IF_ERROR(lower_bfloat_region(*program, step));
        ++index;
        continue;
      }
      if (matches_mature_default(*program, step, DefaultCarrier::kGgml) &&
          !step.operation.get_if<View>() &&
          managed_operation(*program, step)) {
        auto end = index + 1;
        const auto internal_reshape = [&](const PlanStep &next) {
          const auto *view = next.operation.get_if<View>();
          if (!view || view->kind() != ViewKind::kReshape ||
              next.inputs.size() != 1 || next.outputs.size() != 1)
            return false;
          // Only reshape an output already produced inside this region.
          // External capacity aliases retain the ordinary Plan view path.
          const auto &producer = lowered.steps[end - 1];
          return std::ranges::find(producer.outputs, next.inputs[0]) !=
                     producer.outputs.end() &&
                 managed_operation(*program, next);
        };
        while (!step.control_boundary && end < lowered.steps.size() &&
               (prepared_casts.empty() || !prepared_casts[end]) &&
               !lowered.steps[end].control_boundary &&
               ((!lowered.steps[end].operation.get_if<View>() &&
                 matches_mature_default(*program, lowered.steps[end], DefaultCarrier::kGgml) &&
                 managed_operation(*program, lowered.steps[end])) ||
                internal_reshape(lowered.steps[end])))
          ++end;
        HYLE_RETURN_IF_ERROR(
            lower_region(*program, lowered, index, end, last_uses));
        index = end;
        continue;
      }
      if (matches_mature_default(*program, step, DefaultCarrier::kGgml) &&
          checked_grouped_operation(*program, step)) {
        HYLE_RETURN_IF_ERROR(lower_checked_grouped(*program, step));
        ++index;
        continue;
      }
      if (matches_mature_default(*program, step, DefaultCarrier::kGgml) &&
          quantized_gather_operation(*program, step)) {
        HYLE_RETURN_IF_ERROR(lower_gather(*program, step));
        ++index;
        continue;
      }
      ++index;
      for (const auto input : step.inputs)
        if (!program->values[input].description.is_plain())
          return std::unexpected(Status::Unimplemented(
              "Vulkan compressed storage requires a matching mature region"));
      if (step.operation.get_if<GatedDeltaNet>())
        return std::unexpected(Status::Unimplemented(
            "Vulkan GatedDeltaNet requires dense F32 equal heads, "
            "D16/32/64/128 and legal native resource limits"));
      if (const auto *view = step.operation.get_if<View>()) {
        HYLE_ASSIGN_OR_RETURN(
            auto offset, compute::detail::view_alias_offset(
                             *view, program->values[step.inputs[0]].description,
                             program->values[step.outputs[0]].description));
        auto alias = program->aliases[step.inputs[0]];
        alias.byte_offset +=
            offset; // Every derived span is contained in its root.
        program->aliases[step.outputs[0]] = alias;
        if (const auto &source = program->values[step.inputs[0]].constant;
            source) {
          // Preserve immutable weights for downstream cold packing and dedup.
          HYLE_ASSIGN_OR_RETURN(auto constant, source.view(*view));
          program->values[step.outputs[0]].constant = std::move(constant);
        }
        continue;
      }
      RowCacheAccessFacts access;
      auto status = lower_operation(*program, step, resolver,
                                    facts ? &access : nullptr);
      if (!status.ok())
        return std::unexpected(status);
      if (facts && access.kind != RowCacheAccessKind::kNone) {
        access.step = index - 1;
        row_cache_accesses.push_back(access);
      }
    }
    if (queue_policy == PlanQueuePolicy::kRuntimeQueueOnly &&
        std::ranges::any_of(program->dispatches, [](const auto &dispatch) { return dispatch.iree.has_value(); }))
      return std::unexpected(Status::Unimplemented("Vulkan ordered plan contains an asynchronous carrier"));
    while (program->aliases.size() < program->values.size()) {
      const auto index = static_cast<std::uint32_t>(program->aliases.size());
      program->aliases.push_back({index, 0});
    }
    // Packing may replace a dense captured buffer with an immutable native
    // layout. Keep originals through all lowering for deduplication, then drop
    // owners no final dispatch uses instead of retaining two complete weights.
    std::vector<bool> referenced(program->values.size(), false);
    for (const auto &dispatch : program->dispatches)
      for (const auto value : dispatch.values)
        referenced[program->aliases[value].root] = true;
    for (const auto value : program->outputs)
      referenced[program->aliases[value].root] = true;
    for (std::size_t value = 0; value < program->values.size(); ++value)
      if (!referenced[value])
        program->values[value].constant = {};
    auto status = plan_storage(*program);
    if (!status.ok())
      return std::unexpected(status);
    if (facts != nullptr) {
      PlanFacts snapshot;
      if (!row_cache_accesses.empty()) {
        // Qualify the final native bank path, including hidden dispatches and
        // aliases introduced during lowering. Binding a native descriptor is
        // permitted; importing/copying/converting the whole bank is not.
        // The index contains only sparse access-bearing banks, not another DAG.
        std::unordered_map<std::uint32_t, std::size_t> bank_accesses;
        std::vector<bool> qualified(row_cache_accesses.size(), true);
        std::vector<bool> boundary(row_cache_accesses.size(), false);
        for (std::size_t i = 0; i < row_cache_accesses.size(); ++i) {
          const auto bank = row_cache_accesses[i].bank_value;
          if (bank == UINT32_MAX) continue;
          const auto [position, inserted] = bank_accesses.emplace(bank, i);
          if (!inserted) {
            qualified[i] = false;
            if (position->second != SIZE_MAX)
              qualified[position->second] = false;
            position->second = SIZE_MAX;
          }
        }
        const auto find_bank = [&](std::uint32_t value) {
          const auto found = bank_accesses.find(program->aliases[value].root);
          return found == bank_accesses.end() ? SIZE_MAX : found->second;
        };
        for (const auto input : program->inputs) {
          const auto i = find_bank(input);
          if (i != SIZE_MAX) boundary[i] = true;
        }
        for (std::size_t value = 0; value < program->aliases.size(); ++value) {
          const auto i = find_bank(static_cast<std::uint32_t>(value));
          if (i != SIZE_MAX && value != row_cache_accesses[i].bank_value)
            qualified[i] = false;
        }
        for (std::size_t step_index = 0; step_index < lowered.steps.size(); ++step_index) {
          const auto &native_step = lowered.steps[step_index];
          for (std::size_t port = 0; port < native_step.inputs.size(); ++port) {
            const auto i = find_bank(native_step.inputs[port]);
            if (i != SIZE_MAX &&
                (step_index != row_cache_accesses[i].step ||
                 port != row_cache_accesses[i].bank_input))
              qualified[i] = false;
          }
        }
        for (std::size_t dispatch_index = 0;
             dispatch_index < program->dispatches.size(); ++dispatch_index) {
          const auto &dispatch = program->dispatches[dispatch_index];
          for (const auto value : dispatch.values) {
            const auto i = find_bank(value);
            if (i == SIZE_MAX) continue;
            const auto &access = row_cache_accesses[i];
            if (dispatch_index < access.dispatch_begin ||
                dispatch_index >= access.dispatch_end ||
                !dispatch.pipeline || dispatch.region || dispatch.iree ||
                !dispatch.checked_status ||
                access.bank_binding >= dispatch.values.size() ||
                dispatch.values[access.bank_binding] != access.bank_value)
              qualified[i] = false;
          }
        }
        for (const auto output : program->outputs) {
          const auto i = find_bank(output);
          if (i != SIZE_MAX) qualified[i] = false;
        }
        for (std::size_t i = 0; i < row_cache_accesses.size(); ++i)
          if (qualified[i] &&
              (row_cache_accesses[i].bank_value == UINT32_MAX || boundary[i]))
            snapshot.row_cache_accesses.push_back(row_cache_accesses[i]);
      }
      for (const auto &dispatch : program->dispatches)
        if (dispatch.region || dispatch.iree)
          ++snapshot.managed_region_count;
        else
          ++snapshot.native_dispatch_count;
      snapshot.invocation_storage_block_bytes.reserve(
          program->block_bytes.size());
      for (const auto bytes : program->block_bytes) {
        if (!std::in_range<std::size_t>(bytes))
          return std::unexpected(Status::ResourceExhausted(
              "Vulkan report block capacity overflows"));
        snapshot.invocation_storage_block_bytes.push_back(
            static_cast<std::size_t>(bytes));
      }
      if (program->status_value)
        snapshot.invocation_storage_block_bytes.push_back(4);
      std::vector<const Buffer *> constants;
      for (const auto &value : program->values)
        if (value.constant) {
          const auto *root = Access::buffer(value.constant).get();
          while (root->is_view)
            root = static_cast<const Buffer *>(root->external_owner.get());
          constants.push_back(root);
        }
      std::ranges::sort(constants, std::less<>{});
      const auto duplicate = std::ranges::unique(constants);
      constants.erase(duplicate.begin(), duplicate.end());
      snapshot.retained_constant_owner_count = constants.size();
      for (const auto *buffer : constants) {
        if (buffer->bytes >
            SIZE_MAX - snapshot.retained_constant_capacity_bytes)
          return std::unexpected(Status::ResourceExhausted(
              "Vulkan report constant capacity overflows"));
        snapshot.retained_constant_capacity_bytes +=
            static_cast<std::size_t>(buffer->bytes);
        if (buffer->memory == VK_NULL_HANDLE) {
          snapshot.has_borrowed_constants = true;
        } else {
          if (buffer->allocation_bytes >
              SIZE_MAX - snapshot.owned_constant_allocation_bytes)
            return std::unexpected(Status::ResourceExhausted(
                "Vulkan report constant allocation overflows"));
          snapshot.owned_constant_allocation_bytes +=
              static_cast<std::size_t>(buffer->allocation_bytes);
        }
      }
      *facts = std::move(snapshot);
    }
    return Access::program(std::move(program));
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("Vulkan program preparation"));
  }
}

} // namespace hyle::compute::vulkan::detail

namespace hyle::compute::vulkan {

Result<Program> prepare(const Context &context, const Operation &operation,
                        std::span<const Argument> arguments,
                        const WorkloadContext &workload) noexcept {
  return detail::prepare_operation(detail::Access::runtime(context), operation,
                                   arguments, workload);
}

Result<Program> detail::prepare_operation(
    const std::shared_ptr<Runtime> &runtime, const Operation &operation,
    std::span<const Argument> arguments,
    const WorkloadContext &workload) noexcept {
  if (!workload.valid())
    return std::unexpected(Status::InvalidArgument("invalid Vulkan workload"));
  const auto output_count = operation_traits(operation).outputs;
  if (arguments.size() >
      (std::numeric_limits<std::uint32_t>::max)() - output_count)
    return std::unexpected(
        Status::InvalidArgument("too many Vulkan arguments"));
  try {
    detail::Plan plan;
    plan.workload = workload;
    detail::PlanStep step;
    step.operation = operation;
    std::vector<TensorDesc> descriptions;
    for (const auto &argument : arguments) {
      const auto index = static_cast<std::uint32_t>(plan.values.size());
      plan.values.push_back({argument.description(), argument.constant()});
      if (!argument.constant())
        plan.inputs.push_back(index);
      step.inputs.push_back(index);
      descriptions.push_back(argument.description());
    }
    std::vector<TensorDesc> outputs(output_count);
    HYLE_RETURN_IF_ERROR(
        infer_output_descriptions(operation, descriptions, outputs));
    for (const auto &output : outputs) {
      const auto id = static_cast<std::uint32_t>(plan.values.size());
      plan.values.push_back({output, {}});
      step.outputs.push_back(id);
      plan.outputs.push_back(id);
    }
    plan.steps.push_back(std::move(step));
    return detail::prepare_plan(runtime, plan);
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("Vulkan argument preparation"));
  }
}

std::span<const Tensor> Results::tensors() const noexcept {
  return values_ ? std::span<const Tensor>(values_->tensors)
                 : std::span<const Tensor>{};
}
WorkloadContext Program::workload() const noexcept {
  return state_ ? state_->workload : WorkloadContext{};
}
std::span<const TensorDesc> Program::input_descriptions() const noexcept {
  return state_ ? std::span<const TensorDesc>(state_->input_descriptions)
                : std::span<const TensorDesc>{};
}
std::span<const TensorDesc> Program::output_descriptions() const noexcept {
  return state_ ? std::span<const TensorDesc>(state_->output_descriptions)
                : std::span<const TensorDesc>{};
}

Result<Invocation>
Program::create_invocation(mem::AllocatorRef allocator, InvocationAllocation allocation) const noexcept {
  if (!state_)
    return std::unexpected(Status::InvalidArgument("invalid Vulkan program"));
  compute::ResourceBudget::Reservation private_storage;
  if (allocation.resource_budget) {
    HYLE_RETURN_IF_ERROR(detail::MemoryAccess::validate_resource(state_->runtime, allocation.device));
    if (allocation.device.properties().has(mem::MemoryProperty::kHostAccessible))
      return std::unexpected(Status::InvalidArgument("Vulkan invocation budget needs device storage"));
    if (allocation.staging.valid()) {
      HYLE_RETURN_IF_ERROR(detail::MemoryAccess::validate_resource(state_->runtime, allocation.staging));
      if (!allocation.staging.properties().has(mem::MemoryProperty::kHostAccessible))
        return std::unexpected(Status::InvalidArgument("Vulkan invocation budget needs mapped staging storage"));
    } else if (detail::requires_staging(*state_)) {
      return std::unexpected(Status::InvalidArgument("checked Vulkan invocation budget needs staging storage"));
    }
    HYLE_ASSIGN_OR_RETURN(allocator, allocation.resource_budget.metadata_allocator());
    // Native region/control vectors retain their original owner allocations.
    // Their unknown private cost is explicit, never silently counted as zero.
    HYLE_ASSIGN_OR_RETURN(private_storage, allocation.resource_budget.reserve_unattributed(std::nullopt));
  }
  {
    std::lock_guard lock(state_->runtime->mutex);
    if (state_->runtime->closed)
      return std::unexpected(
          Status::FailedPrecondition("Vulkan context is closed"));
    if (!state_->runtime->failure.ok())
      return std::unexpected(state_->runtime->failure);
  }
  auto slot = detail::cd::ResultSlot<Results>::create(allocator);
  if (!slot)
    return std::unexpected(std::move(slot).error());
  try {
    auto invocation = std::allocate_shared<detail::InvocationState>(mem::StdAllocator<detail::InvocationState>(allocator));
    invocation->program = state_;
    invocation->slot = std::move(*slot);
    auto run = detail::create_run(state_->runtime, allocator);
    if (!run)
      return std::unexpected(std::move(run).error());
    invocation->run = std::move(*run);
    invocation->run->allocation = allocation;
    invocation->run->private_storage = std::move(private_storage);
    invocation->run->recording_owner = state_;
    auto status = detail::allocate_values(*state_, *invocation->run);
    if (!status.ok())
      return std::unexpected(status);
    invocation->results = std::make_shared<detail::ResultValues>();
    for (const auto output : state_->outputs)
      invocation->results->tensors.push_back(
          detail::Access::tensor(state_->values[output].description,
                                 invocation->run->retained_buffers[output]));
    invocation->old_inputs.resize(state_->inputs.size());
    status = detail::record(invocation->run, state_->values, state_->dispatches,
                            &invocation->dynamic_bindings);
    if (!status.ok())
      return std::unexpected(status);
    for (auto &binding : invocation->dynamic_bindings) {
      const auto root = state_->aliases[binding.value].root;
      const auto position = std::ranges::find(state_->inputs, root);
      if (position == state_->inputs.end())
        return std::unexpected(
            Status::Internal("Vulkan dynamic alias has no input root"));
      binding.input_port =
          static_cast<std::uint32_t>(position - state_->inputs.begin());
    }
    for (std::size_t index = 0; index < invocation->run->stages.size();
         ++index) {
      const auto &stage = invocation->run->stages[index];
      if (!stage.managed)
        continue;
      for (std::size_t port = 0; port < stage.values.size(); ++port) {
        const auto value = stage.values[port];
        const auto position =
            std::ranges::find(state_->inputs, state_->aliases[value].root);
        if (position != state_->inputs.end())
          invocation->region_inputs.push_back(
              {index, port, value,
               static_cast<std::uint32_t>(position - state_->inputs.begin())});
      }
    }
    // Native children and imported managed ports also retain boundary owners.
    // Capture the idle baseline only after the entire execution slot exists.
    for (const auto output : state_->outputs)
      invocation->output_owner_counts.push_back(
          invocation->run->retained_buffers[output].use_count());
    invocation->bound = state_->inputs.empty();
    return detail::Access::invocation(std::move(invocation));
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("Vulkan invocation metadata"));
  }
}

Status detail::CompositionAccess::prepare_bindings(Invocation &invocation) noexcept {
  if (!invocation.state_)
    return Status::FailedPrecondition("empty Vulkan cold composition slot");
  auto &state = *invocation.state_;
  // Only Graph output preparation opts in. Original Direct/materialized
  // callers retain their exact exported-owner admission gate.
  if (!state.retained_output_reuse)
    return detail::PagedBatchAccess::reuse_gate(invocation);
  auto &run = *state.run;
  const auto &program = *state.program;
  std::lock_guard lock(program.runtime->mutex);
  if (program.runtime->closed || !program.runtime->failure.ok() ||
      !run.submission_failure.ok())
    return Status::FailedPrecondition("Vulkan cold composition slot is closed/failed");
  if (run.pending || run.observers || run.retiring || run.completion_owner)
    return Status::Unavailable("Vulkan cold composition slot is still in use");
  HYLE_RETURN_IF_ERROR(detail::cd::can_detach_retired(state.slot));
  for (std::size_t i = 0; i < program.outputs.size(); ++i) {
    const auto output = program.outputs[i];
    const auto root = program.aliases[output].root;
    if (program.values[root].constant) continue;
    if (std::ranges::find(program.inputs, root) != program.inputs.end()) {
      // A true input alias has no output writer to rearm. Keep its original
      // exact-owner gate before allowing the stable view to bind a new input.
      if (state.composition_sealed &&
          run.retained_buffers[output].use_count() != state.output_owner_counts[i])
        return Status::Unavailable("Vulkan input-alias output retained");
      continue;
    }
    HYLE_ASSIGN_OR_RETURN(auto block, detail::owned_output_block(program, run, output));
    (void)block;
  }
  // Reverse unbind can already have removed private downstream imports.
  // Do not infer escape deltas from that count or establish a new baseline.
  // An unpublished preparation/retry keeps its already selected epoch.
  if (state.composition_sealed && !state.composition_preparing)
    state.output_rearm_required = true;
  state.bound = false;
  state.composition_preparing = true;
  return Status::Ok();
}

Status detail::CompositionAccess::prepare_outputs(Invocation &invocation,
    bool *refresh) noexcept {
  if (!invocation.state_ || !refresh)
    return Status::InvalidArgument("Vulkan retained output preparation is empty");
  *refresh = false;
  auto &state = *invocation.state_;
  auto &run = *state.run;
  const auto &program = *state.program;
  std::lock_guard lock(program.runtime->mutex);
  if (program.runtime->closed || !program.runtime->failure.ok() ||
      !run.submission_failure.ok())
    return Status::FailedPrecondition("Vulkan retained output slot is closed/failed");
  if (run.pending || run.observers || run.retiring || run.completion_owner)
    return Status::Unavailable("Vulkan retained output slot is still in use");
  HYLE_RETURN_IF_ERROR(detail::cd::can_detach_retired(state.slot));
  const bool prepared = state.retained_output_reuse &&
      state.composition_preparing && !state.output_rearm_required;
  state.bound = false;
  state.composition_preparing = true;
  // A failed caller export/bind may retry this same unpublished epoch. Its
  // changing private references must not become an old-reader baseline.
  if (state.output_epoch_pending || prepared) {
    state.bound = false;
    state.composition_preparing = true;
    *refresh = state.output_epoch_pending;
    return Status::Ok();
  }
  const bool rearm = state.output_rearm_required;
  // Until selection has succeeded, a failed allocation/layout check must not
  // look like an already prepared no-rotation epoch on retry.
  state.output_rearm_required = true;
  try {
    if (!state.retained_output_reuse) {
      state.output_spares.resize(program.block_bytes.size());
      state.retained_output_reuse = true;
    }
    std::vector<bool> rotate(program.block_bytes.size(), false);
    bool changed = false;
    for (std::size_t i = 0; i < program.outputs.size(); ++i) {
      const auto output = program.outputs[i];
      if (!rearm) {
        const auto count = run.retained_buffers[output].use_count();
        if (count < state.output_owner_counts[i])
          return Status::Unavailable("Vulkan retained output private owners changed");
        if (count == state.output_owner_counts[i]) continue;
      } else if (program.values[program.aliases[output].root].constant) {
        continue; // Immutable constants are never a new output writer.
      } else if (std::ranges::find(program.inputs, program.aliases[output].root) != program.inputs.end()) {
        // Rearm only owned writers. An input-alias view keeps the original
        // exact sealed-owner rule and is rebound by Invocation::bind below.
        if (run.retained_buffers[output].use_count() != state.output_owner_counts[i])
          return Status::Unavailable("Vulkan input-alias output retained");
        continue;
      }
      HYLE_ASSIGN_OR_RETURN(auto block, detail::owned_output_block(program, run, output));
      rotate[block] = true;
      changed = true;
    }
    if (changed) {
      // All fallible backing/view preparation precedes selection. Keeping one
      // spare is bounded; an escaped spare is never overwritten or made free.
      auto prospective = run.retained_buffers;
      std::vector<std::shared_ptr<detail::Buffer>> blocks(program.block_bytes.size());
      std::vector<std::shared_ptr<detail::Buffer>> old_blocks(program.block_bytes.size());
      for (std::size_t value = 0; value < program.values.size(); ++value) {
        const auto block = program.storage_blocks[value];
        if (program.aliases[value].root == value && block < rotate.size() && rotate[block])
          old_blocks[block] = run.retained_buffers[value];
      }
      for (std::size_t block = 0; block < rotate.size(); ++block) {
        if (!rotate[block]) continue;
        const auto &spare = state.output_spares[block];
        if (spare && spare.use_count() == 1) {
          blocks[block] = spare;
        } else {
          HYLE_ASSIGN_OR_RETURN(blocks[block], detail::allocate_buffer(
              program.runtime->device, program.block_bytes[block], false, false,
              {.budget = run.allocation.resource_budget, .resource = run.allocation.device}));
        }
      }
      for (std::size_t value = 0; value < program.values.size(); ++value) {
        const auto alias = program.aliases[value];
        const auto block = program.storage_blocks[alias.root];
        if (block >= rotate.size() || !rotate[block]) continue;
        if (alias.root == value) {
          prospective[value] = blocks[block];
        } else {
          if (alias.byte_offset > blocks[block]->bytes)
            return Status::InvalidArgument("Vulkan retained output alias exceeds its block");
          auto view = std::make_shared<detail::Buffer>();
          view->device = program.runtime->device;
          detail::bind_view_buffer(*view, blocks[block], alias.byte_offset);
          prospective[value] = std::move(view);
        }
      }
      state.bound = false;
      state.composition_preparing = true;
      state.output_epoch_pending = true;
      run.retained_buffers.swap(prospective);
      for (std::size_t block = 0; block < rotate.size(); ++block)
        if (rotate[block]) state.output_spares[block] = std::move(old_blocks[block]);
      for (std::size_t i = 0; i < program.outputs.size(); ++i)
        state.results->tensors[i] = detail::Access::tensor(
            program.output_descriptions[i], run.retained_buffers[program.outputs[i]]);
      *refresh = true;
    }
    // Explicit preparation revokes direct submit even if no physical block
    // needed changing. Only the subsequent full bind and seal may restore it.
    state.bound = false;
    state.composition_preparing = true;
    state.output_rearm_required = changed;
    detail::cd::discard_retired_payload(state.slot);
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan retained output epoch metadata");
  }
}

Result<Tensor> detail::CompositionAccess::output(Invocation &invocation, std::size_t index) noexcept {
  if (!invocation.state_ ||
      (invocation.state_->composition_sealed && !invocation.state_->composition_preparing) ||
      index >= invocation.state_->results->tensors.size())
    return std::unexpected(Status::InvalidArgument("invalid cold Vulkan composition output"));
  std::lock_guard lock(invocation.state_->program->runtime->mutex);
  if (invocation.state_->run->pending || invocation.state_->run->completion_owner)
    return std::unexpected(Status::FailedPrecondition("Vulkan composition must bind before execution"));
  invocation.state_->composition_preparing = true;
  return invocation.state_->results->tensors[index];
}
Status detail::CompositionAccess::seal(Invocation &invocation) noexcept {
  if (!invocation.state_ || invocation.state_->composition_sealed)
    return Status::FailedPrecondition("Vulkan composition output references already sealed");
  auto &state = *invocation.state_;
  std::lock_guard lock(state.program->runtime->mutex);
  if (state.run->pending || state.run->completion_owner || state.run->observers)
    return Status::FailedPrecondition("Vulkan composition references cannot change after admission");
  if (state.retained_output_reuse && !state.bound)
    return Status::FailedPrecondition("Vulkan retained output epoch is not rebound");
  for (std::size_t i = 0; i < state.program->outputs.size(); ++i)
    state.output_owner_counts[i] = state.run->retained_buffers[state.program->outputs[i]].use_count();
  state.composition_sealed = true;
  state.composition_preparing = false;
  state.output_epoch_pending = false;
  state.output_rearm_required = false;
  return Status::Ok();
}

Status detail::PagedBatchAccess::reuse_gate(Invocation&invocation)noexcept{
  auto*state=invocation.state_.get();if(!state)return Status::FailedPrecondition("empty Vulkan batch native slot");
  auto&runtime=*state->program->runtime;auto&run=*state->run;
  std::lock_guard lock(runtime.mutex);
  if(runtime.closed || !runtime.failure.ok() || !run.submission_failure.ok())return Status::FailedPrecondition("Vulkan batch native slot closed/failed");
  if(run.pending || run.observers || run.retiring || run.completion_owner)return Status::Unavailable("Vulkan batch native slot in use");
  if(state->retained_output_reuse && state->composition_preparing)
    return Status::FailedPrecondition("Vulkan retained output epoch needs complete cold binding/seal");
  for(std::size_t i=0;i<state->program->outputs.size();++i){const auto output=state->program->outputs[i];if(run.retained_buffers[output].use_count()!=state->output_owner_counts[i])return Status::Unavailable("Vulkan batch native output retained");}
  HYLE_ASSIGN_OR_RETURN(auto writer,state->slot.begin());return Status::Ok();
}
Status detail::PagedBatchAccess::reseal_composition(Invocation& invocation) noexcept {
  auto* state=invocation.state_.get();
  if(!state || !state->composition_sealed)
    return Status::FailedPrecondition("Vulkan batch composition is not sealed");
  auto& run=*state->run;
  std::lock_guard lock(state->program->runtime->mutex);
  if(state->program->runtime->closed || !state->program->runtime->failure.ok() ||
     !run.submission_failure.ok() || run.pending || run.observers || run.retiring || run.completion_owner)
    return Status::FailedPrecondition("Vulkan batch composition is not idle");
  if (state->retained_output_reuse && !state->bound)
    return Status::FailedPrecondition("Vulkan retained output epoch is not rebound");
  // The enclosing Graph owns its result writer and checked the prior baseline
  // before adding these internal references. Public result admission stays exact.
  for(std::size_t i=0;i<state->program->outputs.size();++i)
    state->output_owner_counts[i]=run.retained_buffers[state->program->outputs[i]].use_count();
  state->composition_preparing = false;
  state->output_epoch_pending = false;
  state->output_rearm_required = false;
  return Status::Ok();
}
Status detail::PagedBatchAccess::release_inputs(Invocation&invocation)noexcept{
  HYLE_RETURN_IF_ERROR(detail::CompositionAccess::prepare_bindings(invocation));auto&state=*invocation.state_;auto&run=*state.run;
  Status status;
  {
    std::lock_guard lock(state.program->runtime->mutex);state.bound=false;
    const auto is_input=[&](std::uint32_t value){return std::ranges::find(state.program->inputs,state.program->aliases[value].root)!=state.program->inputs.end();};
    // Keep original roots alive until after the queue lock. Idle alias and
    // managed ports become unbound without replacing their prepared owners.
    for(std::size_t i=0;i<state.program->inputs.size();++i)
      state.old_inputs[i]=std::move(run.retained_buffers[state.program->inputs[i]]);
    for(std::size_t value=0;value<state.program->aliases.size();++value){const auto alias=state.program->aliases[value];
      if(alias.root!=value && is_input(std::uint32_t(value)))detail::bind_view_buffer(*run.retained_buffers[value],{},alias.byte_offset);
    }
    detail::refresh_region_bindings(run);
    for(auto&stage:run.stages){
      if(stage.managed){const auto cleared=stage.managed->bind(stage.bindings);if(!cleared.ok() && status.ok())status=cleared;}
      else if(stage.native){for(std::size_t port=0;port<stage.values.size();++port)
        if(is_input(stage.values[port]))stage.native->retained_buffers[port].reset();}
    }
  }
  for(auto&input:state.old_inputs)input.reset();return status;
}
Status detail::PagedBatchAccess::can_detach_inputs(const Invocation &invocation,
    const cd::ResultStorageOwner &result) noexcept {
  auto *state = invocation.state_.get();
  if (!state) return Status::FailedPrecondition("empty Vulkan native slot");
  auto &run = *state->run;
  std::lock_guard lock(state->program->runtime->mutex);
  if (run.pending || run.observers || run.retiring || run.completion_owner)
    return Status::Unavailable("Vulkan native inputs are still in use");
  for (std::size_t i = 0; i < state->program->outputs.size(); ++i) {
    const auto output = state->program->outputs[i];
    if (!state->output_epoch_pending && !state->output_rearm_required) {
      const auto count = run.retained_buffers[output].use_count();
      if (count < state->output_owner_counts[i])
        return Status::Unavailable("Vulkan native private output owners changed");
      if (count != state->output_owner_counts[i]) {
        if (!state->retained_output_reuse)
          return Status::Unavailable("Vulkan native output retained");
        // This only permits retracting retired imports. The old readonly
        // backing remains retained and cannot be submitted as a new writer.
        HYLE_ASSIGN_OR_RETURN(auto block,
            detail::owned_output_block(*state->program, run, output));
        (void)block;
      }
    }
    if (std::ranges::find(state->program->inputs, state->program->aliases[output].root) !=
        state->program->inputs.end())
      return Status::Unimplemented("Vulkan input-alias output cannot detach its fixed carrier");
    if (state->retained_output_reuse &&
        !state->program->values[state->program->aliases[output].root].constant) {
      HYLE_ASSIGN_OR_RETURN(auto block,
          detail::owned_output_block(*state->program, run, output));
      (void)block;
    }
  }
  for (const auto &stage : run.stages)
    if (stage.managed) HYLE_RETURN_IF_ERROR(stage.managed->can_detach_bindings());
  // Native retirement, not Source admission: a closed/failed runtime is legal.
  return cd::can_detach_retired(state->slot, result);
}
void detail::PagedBatchAccess::detach_inputs(Invocation &invocation) noexcept {
  auto &state = *invocation.state_;
  auto &run = *state.run;
  {
    std::lock_guard lock(state.program->runtime->mutex);
    cd::discard_retired_payload(state.slot);
    state.bound = false;
    if (state.retained_output_reuse) {
      state.composition_preparing = true;
      state.output_rearm_required = true;
    }
    const auto is_input = [&](std::uint32_t value) {
      return std::ranges::find(state.program->inputs, state.program->aliases[value].root) !=
          state.program->inputs.end();
    };
    for (std::size_t i = 0; i < state.program->inputs.size(); ++i)
      state.old_inputs[i] = std::move(run.retained_buffers[state.program->inputs[i]]);
    for (std::size_t value = 0; value < state.program->aliases.size(); ++value) {
      const auto alias = state.program->aliases[value];
      if (alias.root != value && is_input(std::uint32_t(value)))
        detail::bind_view_buffer(*run.retained_buffers[value], {}, alias.byte_offset);
    }
    for (auto &stage : run.stages) {
      if (stage.managed) {
        stage.managed->detach_bindings();
        for (auto &binding : stage.bindings) binding = {};
      } else if (stage.native) {
        for (std::size_t port = 0; port < stage.values.size(); ++port)
          if (is_input(stage.values[port])) stage.native->retained_buffers[port].reset();
      }
    }
    // Other Graph slots can already have dropped internal references during
    // reverse release. Never infer an escaped delta from that changed count.
    // Opt-in output blocks must be replaced/reclaimed before the next seal.
    if (!state.retained_output_reuse)
      for (std::size_t i = 0; i < state.program->outputs.size(); ++i)
        state.output_owner_counts[i] =
            run.retained_buffers[state.program->outputs[i]].use_count();
  }
  // External tokens may destroy resource controls; release outside queue lock.
  for (auto &input : state.old_inputs) input.reset();
}
Status detail::PagedBatchAccess::update_layout(Invocation& invocation,
    const BatchLayoutVersion& layout) noexcept {
  HYLE_RETURN_IF_ERROR(detail::CompositionAccess::prepare_bindings(invocation));
  auto& state=*invocation.state_;const auto host=layout.host();
  std::lock_guard lock(state.program->runtime->mutex);
  for(auto& stage:state.run->stages)
    if(stage.managed)HYLE_RETURN_IF_ERROR(stage.managed->update_batch_layout(host.kv_indptr,host.last_page_lengths));
  return Status::Ok();
}
Result<Program> detail::PagedBatchAccess::prepare_ragged_program(const std::shared_ptr<Runtime>& runtime,
    const Plan& plan,BatchLayoutVersion initial,bool causal,std::int32_t window_left) noexcept {
  try {
    HYLE_ASSIGN_OR_RETURN(auto program,prepare_plan(runtime,plan));
    program.state_->ragged=std::make_shared<const RaggedContract>(
        RaggedContract{std::move(initial),causal,window_left});
    return program;
  } catch(const std::bad_alloc&) {
    return std::unexpected(Status::ResourceExhausted("Vulkan ragged update contract"));
  }
}
Status detail::PagedBatchAccess::update_ragged(Invocation& invocation,
    const BatchLayoutVersion& layout) noexcept {
  auto* state=invocation.state_.get();
  if(!state || !state->program->ragged)
    return Status::InvalidArgument("Vulkan ragged update requires a ragged Program");
  const auto& contract=*state->program->ragged;
  const auto a=contract.initial.host(),b=layout.host();
  if(!contract.initial.origin_schema().owns(layout) || a.query_indptr.size()!=b.query_indptr.size() ||
      !std::equal(a.query_indptr.begin(),a.query_indptr.end(),b.query_indptr.begin()))
    return Status::InvalidArgument("Vulkan ragged update schema/Q topology differs");
  HYLE_RETURN_IF_ERROR(validate_batch_attention_layout(layout,contract.causal,contract.window_left));
  HYLE_RETURN_IF_ERROR(detail::CompositionAccess::prepare_bindings(invocation));
  auto& runtime=*state->program->runtime;auto& run=*state->run;
  {
    std::lock_guard lock(runtime.mutex);
    for(const auto& stage:run.stages)
      if(stage.managed)HYLE_RETURN_IF_ERROR(stage.managed->validate_batch_layout(b.kv_indptr,{}));
  }
  const auto cleared=release_inputs(invocation);
  std::lock_guard lock(runtime.mutex);
  // Once old imports have been released, no failed suffix may publish a
  // partially updated slot as usable. The original bind/submit gates retain it.
  if(!cleared.ok()){run.submission_failure=cleared;return cleared;}
  for(auto& stage:run.stages)if(stage.managed){
    const auto status=stage.managed->update_batch_layout(b.kv_indptr,{});
    if(!status.ok()){run.submission_failure=status;return status;}
  }
  return Status::Ok();
}
Status update_ragged_batch(Invocation& invocation,const BatchLayoutVersion& layout) noexcept {
  return detail::PagedBatchAccess::update_ragged(invocation,layout);
}
Status Invocation::bind(std::span<const Tensor> inputs) noexcept {
  if (!state_ || inputs.size() != state_->program->inputs.size())
    return Status::InvalidArgument("Vulkan input binding arity");
  auto &runtime = *state_->program->runtime;
  auto &run = *state_->run;
  for (std::size_t index = 0; index < inputs.size(); ++index)
    if (!inputs[index] ||
        inputs[index].description() !=
            state_->program->input_descriptions[index] ||
        detail::Access::buffer(inputs[index])->device != runtime.device)
      return Status::InvalidArgument(
          "Vulkan input description or device mismatch");
  auto status = Status::Ok();
  {
    std::lock_guard lock(runtime.mutex);
    if (runtime.closed)
      return Status::FailedPrecondition("Vulkan context is closed");
    if (!runtime.failure.ok())
      return runtime.failure;
    if (!run.submission_failure.ok())
      return run.submission_failure;
    if (run.pending || run.observers != 0 || run.retiring ||
        run.completion_owner)
      return Status::FailedPrecondition("Vulkan input slot is still in use");
    for (std::size_t index = 0; index < state_->program->outputs.size();
         ++index) {
      const auto output = state_->program->outputs[index];
      if (!state_->composition_preparing && state_->program->aliases[output].root != output &&
          run.retained_buffers[output].use_count() !=
              state_->output_owner_counts[index])
        return Status::FailedPrecondition(
            "Vulkan view result is still retained");
    }
    // Validate every prospective range before replacing any owner or native
    // descriptor. Failed metadata binding keeps the previous input set usable.
    for (auto &binding : state_->dynamic_bindings) {
      const auto alias = state_->program->aliases[binding.value];
      HYLE_ASSIGN_OR_RETURN(
          binding.buffer,
          detail::descriptor_buffer(
              detail::Access::buffer(inputs[binding.input_port]).get(),
              state_->program->values[binding.value].description,
              binding.byte_origin, alias.byte_offset, binding.storage_granularity));
    }
    for (const auto &binding : state_->region_inputs) {
      const auto &buffer = detail::Access::buffer(inputs[binding.input_port]);
      const auto offset = state_->program->aliases[binding.value].byte_offset;
      if (offset > buffer->bytes || buffer->offset > SIZE_MAX - offset)
        return Status::InvalidArgument("Vulkan managed input view range");
    }
    for (const auto &stage : run.stages) {
      if (!stage.iree)
        continue;
      for (std::size_t port = 0; port < stage.iree->input_count; ++port) {
        const auto &input = stage.iree->inputs[port];
        const auto alias = state_->program->aliases[input.value];
        const auto position =
            std::ranges::find(state_->program->inputs, alias.root);
        if (position == state_->program->inputs.end())
          continue;
        const auto &buffer =
            detail::Access::buffer(inputs[static_cast<std::size_t>(
                position - state_->program->inputs.begin())]);
        if (alias.byte_offset > buffer->bytes ||
            input.offset > buffer->bytes - alias.byte_offset ||
            input.destination.bytes >
                buffer->bytes - alias.byte_offset - input.offset)
          return Status::InvalidArgument("Vulkan IREE input view range");
      }
    }
    for (const auto &binding : state_->region_inputs) {
      const auto &buffer = detail::Access::buffer(inputs[binding.input_port]);
      const auto offset = state_->program->aliases[binding.value].byte_offset;
      run.stages[binding.stage].bindings[binding.port] = {
          buffer->handle, static_cast<std::size_t>(buffer->offset + offset),
          static_cast<std::size_t>(buffer->bytes - offset), buffer};
    }
    Status region_status;
    for (const auto &stage : run.stages) {
      if (stage.managed) {
        region_status = stage.managed->validate(stage.bindings);
        if (!region_status.ok())
          break;
      }
    }
    // Prospective metadata never retains a rejected input. Native imports
    // still point at the previous input set until the commit below succeeds.
    detail::refresh_region_bindings(run);
    if (!region_status.ok())
      return region_status;
    state_->bound = false;
    for (std::size_t index = 0; index < inputs.size(); ++index) {
      auto &buffer = run.retained_buffers[state_->program->inputs[index]];
      state_->old_inputs[index] = std::move(buffer);
      buffer = detail::Access::buffer(inputs[index]);
    }
    for (std::size_t value = 0; value < state_->program->aliases.size();
         ++value) {
      const auto alias = state_->program->aliases[value];
      if (alias.root != value && !state_->program->values[value].constant)
        detail::bind_view_buffer(*run.retained_buffers[value],
                                 run.retained_buffers[alias.root],
                                 alias.byte_offset);
    }
    for (auto &binding : state_->dynamic_bindings) {
      binding.write.pBufferInfo = &binding.buffer;
      runtime.device->fn.UpdateDescriptorSets(runtime.device->native.device, 1,
                                              &binding.write, 0, nullptr);
    }
    if (!run.stages.empty())
      status = detail::bind_regions(run, state_->output_epoch_pending ? state_->program.get() : nullptr);
    else if (state_->output_epoch_pending)
      status = detail::rebind_descriptors(run, state_->program->values, state_->program->dispatches);
    else if (detail::origins_changed(run, state_->program->dispatches))
      status = detail::encode_commands(run, state_->program->dispatches);
    state_->bound = status.ok();
  }
  // Release external buffer tokens outside the queue lock; no allocation is
  // needed for this deferred release because the vector was sized cold.
  for (auto &buffer : state_->old_inputs)
    buffer.reset();
  return status;
}

Result<OwnedResult<Results>> Invocation::submit() noexcept {
  return submit_guarded({});
}

Result<OwnedResult<Results>> Invocation::submit_guarded(LeaseUseBatch uses) noexcept {
  if (state_ && state_->composition_preparing)
    return std::unexpected(Status::FailedPrecondition("cold Vulkan composition outputs are not sealed"));
  if (!state_ || !state_->bound)
    return std::unexpected(
        Status::FailedPrecondition("Vulkan invocation inputs are unbound"));
  const auto &runtime = state_->program->runtime;
  detail::collect(runtime, false).IgnoreError();
  auto writer = state_->slot.begin();
  if (!writer)
    return std::unexpected(std::move(writer).error());
  Result<Completion> completion;
  {
    std::lock_guard lock(runtime->mutex);
    for (std::size_t index = 0; index < state_->program->outputs.size();
         ++index) {
      const auto output = state_->program->outputs[index];
      if (state_->run->retained_buffers[output].use_count() !=
          state_->output_owner_counts[index])
        return std::unexpected(Status::FailedPrecondition(
            "Vulkan result tensor is still retained"));
    }
    auto status = writer->emplace(detail::Access::results(state_->results));
    if (!status.ok())
      return std::unexpected(status);
    completion = detail::submit_locked(state_->run);
    if (completion) state_->run->native_uses = std::move(uses);
  }
  if (!completion)
    return std::unexpected(std::move(completion).error());
  return writer->publish(std::move(*completion));
}

Status Invocation::wait_native() noexcept {
  return state_ ? detail::wait_native(state_->run)
      : Status::FailedPrecondition("empty Vulkan native invocation");
}

} // namespace hyle::compute::vulkan
