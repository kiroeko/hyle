#include "../compute_ops/view_alias.h"
#include "vulkan_program_internal.h"
#include "vulkan_tensor.h"

#include "hyle/obs/trace.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <new>
#include <numeric>
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
      auto buffer = allocate_buffer(program.runtime->device, bytes, false);
      if (!buffer)
        return std::move(buffer).error();
      blocks.push_back(std::move(*buffer));
    }
    run.retained_buffers.resize(program.values.size());
    for (std::size_t value = 0; value < program.values.size(); ++value) {
      if (program.values[value].constant)
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

Result<Program> prepare_plan(const Context &context, const Plan &plan,
                             PlanFacts *facts) noexcept {
  HYLE_TRACE_ZONE_NAMED("hyle.vulkan.prepare");
  if (facts != nullptr)
    *facts = {};
  const auto &runtime = Access::runtime(context);
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
          !description.is_plain() || !bytes ||
          *bytes > runtime->device->properties.limits.maxStorageBufferRange ||
          !elements || *elements > (std::numeric_limits<std::int32_t>::max)())
        return std::unexpected(Status::Unimplemented(
            "Vulkan plan requires fixed plain F32/F16/BF16/I32/I64/Bool "
            "tensors within device limits"));
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
          traits.outputs != 1 || !traits.inputs.accepts(step.inputs.size()) ||
          step.output >= available.size() || available[step.output])
        return std::unexpected(Status::Unimplemented(
            "Vulkan requires pure single-output SSA operations"));
      std::vector<TensorDesc> descriptions;
      descriptions.reserve(step.inputs.size());
      for (const auto input : step.inputs) {
        if (input >= available.size() || !available[input])
          return std::unexpected(Status::InvalidArgument(
              "Vulkan plan is not topologically ordered"));
        descriptions.push_back(plan.values[input].description);
      }
      auto inferred = infer_single_output(step.operation, descriptions);
      if (!inferred)
        return std::unexpected(std::move(inferred).error());
      if (*inferred != plan.values[step.output].description)
        return std::unexpected(Status::InvalidArgument(
            "Vulkan plan output disagrees with operation schema"));
      available[step.output] = true;
      produced[step.output] = true;
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
    program->values = plan.values;
    program->inputs = plan.inputs;
    program->outputs = plan.outputs;
    for (std::uint32_t index = 0; index < plan.values.size(); ++index)
      program->aliases.push_back({index, 0});
    for (const auto input : plan.inputs)
      program->input_descriptions.push_back(plan.values[input].description);
    for (const auto output : plan.outputs)
      program->output_descriptions.push_back(plan.values[output].description);
    std::vector<std::pair<std::uint32_t, std::uint32_t>> packs;
    for (const auto &step : plan.steps) {
      if (const auto *view = step.operation.get_if<View>()) {
        HYLE_ASSIGN_OR_RETURN(
            auto offset, compute::detail::view_alias_offset(
                             *view, program->values[step.inputs[0]].description,
                             program->values[step.output].description));
        auto alias = program->aliases[step.inputs[0]];
        alias.byte_offset +=
            offset; // Every derived span is contained in its root.
        program->aliases[step.output] = alias;
        if (const auto &source = program->values[step.inputs[0]].constant;
            source) {
          // Preserve immutable weights for downstream cold packing and dedup.
          HYLE_ASSIGN_OR_RETURN(auto constant, source.view(*view));
          program->values[step.output].constant = std::move(constant);
        }
        continue;
      }
      auto status = lower_operation(*program, step, packs);
      if (!status.ok())
        return std::unexpected(status);
    }
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
      snapshot.native_dispatch_count = program->dispatches.size();
      snapshot.invocation_storage_block_bytes.reserve(
          program->block_bytes.size());
      for (const auto bytes : program->block_bytes) {
        if (!std::in_range<std::size_t>(bytes))
          return std::unexpected(Status::ResourceExhausted(
              "Vulkan report block capacity overflows"));
        snapshot.invocation_storage_block_bytes.push_back(
            static_cast<std::size_t>(bytes));
      }
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
                        std::span<const Argument> arguments) noexcept {
  if (arguments.size() > (std::numeric_limits<std::uint32_t>::max)() - 1)
    return std::unexpected(
        Status::InvalidArgument("too many Vulkan arguments"));
  try {
    detail::Plan plan;
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
    auto output = infer_single_output(operation, descriptions);
    if (!output)
      return std::unexpected(std::move(output).error());
    step.output = static_cast<std::uint32_t>(plan.values.size());
    plan.values.push_back({*output, {}});
    plan.outputs.push_back(step.output);
    plan.steps.push_back(std::move(step));
    return detail::prepare_plan(context, plan);
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("Vulkan argument preparation"));
  }
}

std::span<const Tensor> Results::tensors() const noexcept {
  return values_ ? std::span<const Tensor>(values_->tensors)
                 : std::span<const Tensor>{};
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
Program::create_invocation(mem::AllocatorRef allocator) const noexcept {
  if (!state_)
    return std::unexpected(Status::InvalidArgument("invalid Vulkan program"));
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
    auto invocation = std::make_shared<detail::InvocationState>();
    invocation->program = state_;
    invocation->slot = std::move(*slot);
    auto run = detail::create_run(state_->runtime);
    if (!run)
      return std::unexpected(std::move(run).error());
    invocation->run = std::move(*run);
    invocation->run->recording_owner = state_;
    auto status = detail::allocate_values(*state_, *invocation->run);
    if (!status.ok())
      return std::unexpected(status);
    invocation->results = std::make_shared<detail::ResultValues>();
    for (const auto output : state_->outputs)
      invocation->results->tensors.push_back(
          detail::Access::tensor(state_->values[output].description,
                                 invocation->run->retained_buffers[output]));
    for (const auto output : state_->outputs)
      invocation->output_owner_counts.push_back(
          invocation->run->retained_buffers[output].use_count());
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
    invocation->bound = state_->inputs.empty();
    return detail::Access::invocation(std::move(invocation));
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("Vulkan invocation metadata"));
  }
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
    if (run.pending || run.observers != 0 || run.retiring ||
        run.completion_owner)
      return Status::FailedPrecondition("Vulkan input slot is still in use");
    for (std::size_t index = 0; index < state_->program->outputs.size();
         ++index) {
      const auto output = state_->program->outputs[index];
      if (state_->program->aliases[output].root != output &&
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
              binding.byte_origin, alias.byte_offset));
    }
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
    if (detail::origins_changed(run, state_->program->dispatches))
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
  }
  if (!completion)
    return std::unexpected(std::move(completion).error());
  return writer->publish(std::move(*completion));
}

} // namespace hyle::compute::vulkan
