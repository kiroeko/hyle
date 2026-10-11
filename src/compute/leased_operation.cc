#include "hyle/compute/leased_operation.h"

#include <cstdint>
#include <new>
#include <vector>

#include "hyle/core/assert.h"
#include "hyle/compute/detail/retained_completion.h"
#include "hyle/mem/std_allocator.h"

namespace hyle::compute {
namespace detail {
struct LeasedOperationState final {
  explicit LeasedOperationState(mem::AllocatorRef selected) noexcept
      : allocator(selected), inputs(mem::StdAllocator<ConstTensorView>(selected)),
        outputs(mem::StdAllocator<TensorView>(selected)) {}
  mem::AllocatorRef allocator;
  WriteLease workspace{};
  OperationInvocation invocation{};
  LeaseUsePool uses{};
  RetainedCompletion completion{};
  bool closed = false;
  std::vector<ConstTensorView, mem::StdAllocator<ConstTensorView>> inputs;
  std::vector<TensorView, mem::StdAllocator<TensorView>> outputs;
};

void destroy_leased_operation(LeasedOperationState *state) noexcept {
  if (!state) return;
  const auto allocator = state->allocator;
  std::destroy_at(state);
  allocator.deallocate(state, sizeof(LeasedOperationState), alignof(LeasedOperationState));
}
} // namespace detail

LeasedOperationInvocation::LeasedOperationInvocation(LeasedOperationInvocation &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}
LeasedOperationInvocation &LeasedOperationInvocation::operator=(LeasedOperationInvocation &&other) noexcept {
  if (this != &other) {
    detail::destroy_leased_operation(state_);
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}
LeasedOperationInvocation::~LeasedOperationInvocation() { detail::destroy_leased_operation(state_); }

Result<LeasedOperationInvocation> LeasedOperationInvocation::create(
    const PreparedOperation &prepared, WriteLease workspace, LeasedOperationOptions options) noexcept {
  if (!prepared.valid() || !options.allocator.valid())
    return std::unexpected(Status::InvalidArgument("invalid leased operation/allocator"));
  const auto inputs = prepared.input_count(), outputs = prepared.output_count();
  if (inputs > UINT32_MAX - 1 || outputs > UINT32_MAX - 1 - inputs)
    return std::unexpected(Status::ResourceExhausted("leased operation ports exceed metadata capacity"));
  const auto required = prepared.workspace_requirement();
  mem::BufferView workspace_view;
  if (workspace.valid()) {
    HYLE_ASSIGN_OR_RETURN(workspace_view, workspace.view());
  } else if (required.size != 0) {
    return std::unexpected(Status::InvalidArgument("leased operation requires owned workspace"));
  }
  HYLE_ASSIGN_OR_RETURN(auto invocation, prepared.create_invocation(workspace_view));
  HYLE_ASSIGN_OR_RETURN(auto uses, LeaseUsePool::create({
      .max_batches = 1,
      .max_resources_per_batch = static_cast<std::uint32_t>(inputs + outputs + 1),
      .allocator = options.allocator, .resource_budget = options.resource_budget}));
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  auto *storage = allocator.allocate(sizeof(detail::LeasedOperationState), alignof(detail::LeasedOperationState));
  if (!storage) return std::unexpected(Status::ResourceExhausted("leased operation state metadata"));
  auto *state = std::construct_at(static_cast<detail::LeasedOperationState *>(storage), allocator);
  LeasedOperationInvocation result(state);
  HYLE_ASSIGN_OR_RETURN(state->completion, detail::RetainedCompletion::create(allocator));
  try {
    state->inputs.resize(inputs);
    state->outputs.resize(outputs);
  } catch (...) {
    return std::unexpected(Status::ResourceExhausted("leased operation view metadata"));
  }
  state->workspace = std::move(workspace);
  state->invocation = std::move(invocation);
  state->uses = std::move(uses);
  return result;
}

bool LeasedOperationInvocation::in_flight() const noexcept {
  return state_ && (state_->invocation.in_flight() || state_->uses.active_batches() != 0);
}
ProviderIdentity LeasedOperationInvocation::provider() const noexcept {
  return state_ ? state_->invocation.provider() : ProviderIdentity{};
}

Result<Completion> LeasedOperationInvocation::submit(Queue &queue,
    std::span<const LeasedInput> inputs, std::span<const LeasedOutput> outputs) noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("leased operation is empty"));
  if (state_->closed) return std::unexpected(Status::FailedPrecondition("leased operation is closed"));
  if (inputs.size() != state_->inputs.size() || outputs.size() != state_->outputs.size())
    return std::unexpected(Status::InvalidArgument("leased operation port counts differ"));
  if (in_flight()) return std::unexpected(Status::Unavailable("leased operation is in flight"));
  if (!state_->completion.available())
    return std::unexpected(Status::Unavailable("old leased operation completion retains its slot"));

  // Capture borrowed views before write-use admission blocks mutable access.
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const auto &binding = inputs[index];
    if (binding.read.valid() == (binding.exclusive != nullptr))
      return std::unexpected(Status::InvalidArgument("leased input needs exactly one read permission"));
    if (binding.exclusive) {
      HYLE_ASSIGN_OR_RETURN(auto view, binding.exclusive->tensor(binding.description));
      state_->inputs[index] = view.as_const();
    } else {
      HYLE_ASSIGN_OR_RETURN(state_->inputs[index], binding.read.tensor(binding.description));
    }
  }
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    if (!outputs[index].lease)
      return std::unexpected(Status::InvalidArgument("leased output permission is missing"));
    HYLE_ASSIGN_OR_RETURN(state_->outputs[index], outputs[index].lease->tensor(outputs[index].description));
  }
  HYLE_ASSIGN_OR_RETURN(auto use, state_->uses.begin());
  for (const auto &binding : inputs) {
    HYLE_RETURN_IF_ERROR(binding.exclusive ? use.write(*binding.exclusive) : use.read(binding.read));
  }
  for (const auto &binding : outputs) HYLE_RETURN_IF_ERROR(use.write(*binding.lease));
  if (state_->workspace.valid()) HYLE_RETURN_IF_ERROR(use.write(state_->workspace));
  auto completion = state_->invocation.submit(queue, state_->inputs, state_->outputs);
  if (!completion) return std::unexpected(std::move(completion).error());
  // The original successful submit result is an accepted, valid Completion.
  // The fresh reserved retention cannot fail or lose an accepted prefix.
  HYLE_FATAL_ASSERT(completion->valid());
  const auto attached = use.attach(*completion);
  HYLE_FATAL_ASSERT(attached.ok());
  return state_->completion.publish(std::move(*completion));
}
Status LeasedOperationInvocation::close() noexcept {
  if (!state_) return Status::Ok();
  state_->closed = true;
  state_->uses.close().IgnoreError();
  return in_flight() ? Status::Unavailable("leased operation has native work") : Status::Ok();
}
Status LeasedOperationInvocation::drain() noexcept {
  if (!state_) return Status::Ok();
  close().IgnoreError();
  const auto terminal = state_->completion.wait();
  if (!terminal.ok() && state_->uses.active_batches() != 0) return terminal;
  HYLE_RETURN_IF_ERROR(state_->uses.drain());
  return terminal;
}
} // namespace hyle::compute
