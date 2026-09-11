#include "hyle/graph/executable.h"

#include <array>
#include <new>
#include <span>
#include <utility>

#include "plan_state.h"

namespace hyle::graph {

Executable::Executable(const Executable& other) noexcept : state_(other.state_) {
  detail::retain(state_);
}

Executable& Executable::operator=(const Executable& other) noexcept {
  if (state_ != other.state_) {
    detail::retain(other.state_);
    detail::release(state_);
    state_ = other.state_;
  }
  return *this;
}

Executable::Executable(Executable&& other) noexcept : state_(other.state_) {
  other.state_ = nullptr;
}

Executable& Executable::operator=(Executable&& other) noexcept {
  if (this != &other) {
    detail::release(state_);
    state_ = other.state_;
    other.state_ = nullptr;
  }
  return *this;
}

Executable::~Executable() { detail::release(state_); }

std::size_t Executable::input_count() const noexcept {
  return state_ != nullptr ? state_->input_count : 0;
}

Result<ValueId> Executable::input_id(std::size_t index) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("executable is invalid"));
  }
  if (index >= state_->input_count) {
    return std::unexpected(
        Status::OutOfRange("executable input index is out of range"));
  }
  return state_->input_ids[index];
}

Result<ValueId> Executable::output_id(std::size_t index) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("executable is invalid"));
  }
  if (index != 0) {
    return std::unexpected(
        Status::OutOfRange("executable output index is out of range"));
  }
  return state_->output_id;
}

Result<Invocation> Executable::create_invocation() const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("executable is invalid"));
  }
  return Invocation(state_);
}

Invocation::Invocation(detail::PlanState* state) noexcept : state_(state) {
  detail::retain(state_);
}

Invocation::Invocation(Invocation&& other) noexcept
    : state_(other.state_), inputs_(other.inputs_), output_(other.output_),
      input_mask_(other.input_mask_), output_bound_(other.output_bound_) {
  other.state_ = nullptr;
  other.input_mask_ = 0;
  other.output_bound_ = false;
}

Invocation& Invocation::operator=(Invocation&& other) noexcept {
  if (this != &other) {
    detail::release(state_);
    state_ = other.state_;
    inputs_ = other.inputs_;
    output_ = other.output_;
    input_mask_ = other.input_mask_;
    output_bound_ = other.output_bound_;
    other.state_ = nullptr;
    other.input_mask_ = 0;
    other.output_bound_ = false;
  }
  return *this;
}

Invocation::~Invocation() { detail::release(state_); }

Status Invocation::bind(ValueId value,
                        compute::ConstTensorView tensor) noexcept {
  if (state_ == nullptr) return Status::FailedPrecondition("invocation is invalid");
  if (!tensor.valid()) return Status::InvalidArgument("tensor binding is invalid");
  for (std::size_t index = 0; index < state_->input_count; ++index) {
    if (value == state_->input_ids[index]) {
      if (tensor.description() != state_->input_descs[index]) {
        return Status::InvalidArgument(
            "input tensor does not match the executable description");
      }
      inputs_[index] = tensor;
      input_mask_ = static_cast<std::uint8_t>(input_mask_ | (1U << index));
      return Status::Ok();
    }
  }
  if (value == state_->output_id) {
    return Status::InvalidArgument("graph output requires mutable tensor storage");
  }
  return Status::InvalidArgument("value does not belong to this executable");
}

Status Invocation::bind(ValueId value, compute::TensorView tensor) noexcept {
  if (state_ == nullptr) return Status::FailedPrecondition("invocation is invalid");
  if (!tensor.valid()) return Status::InvalidArgument("tensor binding is invalid");
  if (value == state_->output_id) {
    if (tensor.description() != state_->output_desc) {
      return Status::InvalidArgument(
          "output tensor does not match the executable description");
    }
    output_ = tensor;
    output_bound_ = true;
    return Status::Ok();
  }
  return bind(value, tensor.as_const());
}

void Invocation::reset() noexcept {
  inputs_ = {};
  output_ = {};
  input_mask_ = 0;
  output_bound_ = false;
}

Result<compute::Completion> Invocation::submit(
    compute::Queue& queue) const noexcept {
  if (state_ == nullptr) {
    return std::unexpected(Status::FailedPrecondition("invocation is invalid"));
  }
  const std::uint8_t required =
      static_cast<std::uint8_t>((1U << state_->input_count) - 1U);
  if (input_mask_ != required || !output_bound_) {
    return std::unexpected(
        Status::FailedPrecondition("invocation bindings are incomplete"));
  }
  return state_->prepared.submit(
      queue,
      std::span<const compute::ConstTensorView>(inputs_.data(),
                                                state_->input_count),
      std::span<const compute::TensorView>(&output_, 1));
}

Result<CompiledGraph> compile(const Graph& graph,
                              const compute::Context& context) noexcept {
  if (!graph.valid()) {
    return std::unexpected(Status::InvalidArgument("graph is invalid"));
  }
  std::array<compute::TensorDesc, 2> input_descs{};
  std::array<ValueId, 2> input_ids{};
  for (std::size_t index = 0; index < graph.input_count(); ++index) {
    auto description = graph.input_description(index);
    if (!description) return std::unexpected(std::move(description).error());
    auto id = graph.input_id(index);
    if (!id) return std::unexpected(std::move(id).error());
    input_descs[index] = *description;
    input_ids[index] = *id;
  }
  auto output_desc = graph.output_description();
  if (!output_desc) return std::unexpected(std::move(output_desc).error());
  auto output_id = graph.output_id();
  if (!output_id) return std::unexpected(std::move(output_id).error());
  auto operation = graph.operation();
  if (!operation) return std::unexpected(std::move(operation).error());

  std::array<compute::TensorDesc, 1> outputs{*output_desc};
  auto prepared = compute::prepare(
      context, *operation,
      std::span<const compute::TensorDesc>(input_descs.data(),
                                           graph.input_count()),
      outputs);
  if (!prepared) return std::unexpected(std::move(prepared).error());

  mem::AllocatorRef allocator = context.metadata_allocator();
  if (!allocator.valid()) {
    return std::unexpected(
        Status::FailedPrecondition("compute context is invalid"));
  }
  void* storage =
      allocator.allocate(sizeof(detail::PlanState), alignof(detail::PlanState));
  if (storage == nullptr) {
    return std::unexpected(
        Status::ResourceExhausted("graph executable allocation failed"));
  }
  auto* state = ::new (storage) detail::PlanState(
      allocator, std::move(*prepared),
      std::span<const ValueId>(input_ids.data(), graph.input_count()), *output_id,
      std::span<const compute::TensorDesc>(input_descs.data(),
                                           graph.input_count()),
      *output_desc);

  CompiledGraph result;
  result.executable = detail::ExecutableAccess::adopt(state);
  result.report = CompileReport{
      .backend = context.device().backend(),
      .provider = compute::ProviderKind::kHyleNative,
      .workspace_bytes = 0,
      .copy_bytes = 0,
      .synchronization_count = 0,
      .used_fallback = false,
  };
  return result;
}

}  // namespace hyle::graph
