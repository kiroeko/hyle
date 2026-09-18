#include "vulkan_program_internal.h"

#include <algorithm>
#include <limits>
#include <new>

namespace hyle::compute::vulkan {
namespace {

Status ready(const std::shared_ptr<detail::RecorderState> &state) noexcept {
  if (!state || state->finished)
    return Status::FailedPrecondition("Vulkan recorder is invalid or finished");
  std::lock_guard lock(state->runtime->mutex);
  if (state->runtime->closed)
    return Status::FailedPrecondition("Vulkan context is closed");
  return state->runtime->failure;
}

bool overlap(const Tensor &left, const Tensor &right) noexcept {
  if (left.native_buffer() != right.native_buffer())
    return false;
  const auto left_size = *left.description().required_bytes();
  const auto right_size = *right.description().required_bytes();
  return left.native_offset() < right.native_offset() + right_size &&
         right.native_offset() < left.native_offset() + left_size;
}

} // namespace

Result<Recorder> Recorder::create(const Context &context) noexcept {
  const auto &runtime = detail::Access::runtime(context);
  if (!runtime)
    return std::unexpected(Status::InvalidArgument("invalid Vulkan context"));
  try {
    auto state = std::make_shared<detail::RecorderState>();
    state->runtime = runtime;
    auto status = ready(state);
    if (!status.ok())
      return std::unexpected(status);
    return Recorder(std::move(state));
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("Vulkan recorder metadata"));
  }
}

Status Recorder::append(const Program &program, std::span<const Tensor> inputs,
                        std::span<const Tensor> outputs) noexcept {
  auto status = ready(state_);
  if (!status.ok())
    return status;
  const auto &prepared = detail::Access::program(program);
  if (!prepared || prepared->runtime != state_->runtime ||
      inputs.size() != prepared->inputs.size() ||
      outputs.size() != prepared->outputs.size())
    return Status::InvalidArgument(
        "Vulkan recorder program, context or port arity");
  for (std::size_t index = 0; index < inputs.size(); ++index)
    if (!inputs[index] ||
        inputs[index].description() != prepared->input_descriptions[index] ||
        detail::Access::buffer(inputs[index])->device !=
            state_->runtime->device)
      return Status::InvalidArgument(
          "Vulkan recorder input description or device");
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    if (!outputs[index] ||
        outputs[index].description() != prepared->output_descriptions[index] ||
        detail::Access::buffer(outputs[index])->device !=
            state_->runtime->device)
      return Status::InvalidArgument(
          "Vulkan recorder output description or device");
    const auto value = prepared->outputs[index];
    const auto alias = prepared->aliases[value];
    if (alias.root != value) {
      const auto port = std::ranges::find(prepared->inputs, alias.root);
      const Tensor *root = port != prepared->inputs.end()
                               ? &inputs[static_cast<std::size_t>(
                                     port - prepared->inputs.begin())]
                               : &prepared->values[alias.root].constant;
      if (!*root)
        return Status::Unimplemented(
            "Vulkan recorder alias output needs an externally bound root");
      if (outputs[index].native_buffer() != root->native_buffer() ||
          outputs[index].native_offset() !=
              root->native_offset() + alias.byte_offset)
        return Status::InvalidArgument(
            "Vulkan recorder View output must alias its input range");
      continue;
    }
    if (detail::Access::buffer(outputs[index])->is_view)
      return Status::Unimplemented(
          "Vulkan typed recorded output requires independent padded storage");
    for (const auto &input : inputs)
      if (overlap(outputs[index], input))
        return Status::InvalidArgument("Vulkan typed output aliases an input");
    for (const auto &value : prepared->values)
      if (value.constant && overlap(outputs[index], value.constant))
        return Status::InvalidArgument(
            "Vulkan typed output aliases a captured constant");
    for (std::size_t previous = 0; previous < index; ++previous)
      if (overlap(outputs[index], outputs[previous]))
        return Status::InvalidArgument("Vulkan typed outputs overlap");
  }
  if (prepared->values.size() >
      (std::numeric_limits<std::uint32_t>::max)() - state_->values.size())
    return Status::ResourceExhausted("Vulkan recorder value count overflows");
  try {
    // Caller-owned ports are excluded from scratch allocation. Replan cold
    // because an owning invocation may otherwise reuse a larger intermediate
    // allocation for its final output; a caller buffer need not have that size.
    auto recipe = *prepared;
    for (std::size_t index = 0; index < inputs.size(); ++index)
      recipe.values[recipe.inputs[index]].constant = inputs[index];
    for (std::size_t index = 0; index < outputs.size(); ++index)
      recipe.values[recipe.outputs[index]].constant = outputs[index];
    status = detail::plan_storage(recipe);
    if (!status.ok())
      return status;
    detail::Run resources;
    resources.runtime = state_->runtime;
    status = detail::allocate_values(recipe, resources);
    if (!status.ok())
      return status;
    std::vector<detail::PlanValue> values;
    values.reserve(recipe.values.size());
    for (std::size_t index = 0; index < recipe.values.size(); ++index) {
      const auto &description = recipe.values[index].description;
      values.push_back(
          {description,
           detail::Access::tensor(
               description, std::move(resources.retained_buffers[index]))});
    }
    const auto base = static_cast<std::uint32_t>(state_->values.size());
    for (auto &dispatch : recipe.dispatches)
      for (auto &value : dispatch.values)
        value += base;
    state_->values.reserve(state_->values.size() + values.size());
    state_->dispatches.reserve(state_->dispatches.size() +
                               recipe.dispatches.size());
    for (auto &value : values)
      state_->values.push_back(std::move(value));
    for (auto &dispatch : recipe.dispatches)
      state_->dispatches.push_back(std::move(dispatch));
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted("Vulkan recorder prepared fragment");
  }
}

Status Recorder::dispatch(const Kernel &kernel,
                          std::span<const Tensor> bindings,
                          std::array<std::uint32_t, 3> groups,
                          std::span<const std::byte> push_constants) noexcept {
  auto status = ready(state_);
  if (!status.ok())
    return status;
  const auto &pipeline = detail::Access::pipeline(kernel);
  if (!pipeline || pipeline->device != state_->runtime->device ||
      bindings.size() != pipeline->bindings ||
      push_constants.size() != pipeline->push_bytes ||
      bindings.size() >
          (std::numeric_limits<std::uint32_t>::max)() - state_->values.size())
    return Status::InvalidArgument("Vulkan native dispatch pipeline or arity");
  for (std::size_t axis = 0; axis < 3; ++axis)
    if (groups[axis] == 0 ||
        groups[axis] >
            pipeline->device->properties.limits.maxComputeWorkGroupCount[axis])
      return Status::OutOfRange("Vulkan native dispatch group count");
  for (const auto &tensor : bindings)
    if (!tensor || detail::Access::buffer(tensor)->device != pipeline->device)
      return Status::InvalidArgument("Vulkan native dispatch buffer device");
  try {
    detail::Dispatch dispatch;
    dispatch.pipeline = pipeline;
    dispatch.groups = groups;
    dispatch.push.assign(push_constants.begin(), push_constants.end());
    dispatch.values.reserve(bindings.size());
    const auto base = static_cast<std::uint32_t>(state_->values.size());
    for (std::uint32_t index = 0; index < bindings.size(); ++index)
      dispatch.values.push_back(base + index);
    state_->values.reserve(state_->values.size() + bindings.size());
    state_->dispatches.reserve(state_->dispatches.size() + 1);
    for (const auto &tensor : bindings)
      state_->values.push_back({tensor.description(), tensor});
    state_->dispatches.push_back(std::move(dispatch));
    return Status::Ok();
  } catch (const std::bad_alloc &) {
    return Status::ResourceExhausted(
        "Vulkan recorder native dispatch metadata");
  }
}

Result<RecordedProgram> Recorder::finish() noexcept {
  auto status = ready(state_);
  if (!status.ok())
    return std::unexpected(status);
  try {
    auto run = detail::create_run(state_->runtime);
    if (!run)
      return std::unexpected(std::move(run).error());
    (*run)->retained_buffers.reserve(state_->values.size());
    for (const auto &value : state_->values)
      (*run)->retained_buffers.push_back(
          detail::Access::buffer(value.constant));
    (*run)->recording_owner = state_;
    status = detail::record(*run, state_->values, state_->dispatches);
    if (!status.ok())
      return std::unexpected(status);
    state_->finished = true;
    return RecordedProgram(std::move(*run));
  } catch (const std::bad_alloc &) {
    return std::unexpected(
        Status::ResourceExhausted("Vulkan recorder finalization"));
  }
}

Result<Completion> RecordedProgram::submit() noexcept {
  if (!state_)
    return std::unexpected(
        Status::FailedPrecondition("invalid Vulkan recorded program"));
  detail::collect(state_->runtime, false).IgnoreError();
  std::lock_guard lock(state_->runtime->mutex);
  return detail::submit_locked(state_);
}

} // namespace hyle::compute::vulkan
