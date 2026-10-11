#include "hyle/compute/apple_memory.h"

#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "hyle/compute/detail/native_lease_inputs.h"
#include "hyle/mem/std_allocator.h"

namespace hyle::compute::apple {
namespace detail {
struct LeasedInvocationState final {
  explicit LeasedInvocationState(mem::AllocatorRef selected)
      : allocator(selected), inputs(selected) {}
  mem::AllocatorRef allocator;
  compute::detail::NativeLeaseInputs inputs;
  Invocation invocation;
  const MemoryResource *memory = nullptr;
  LeaseImportOptions options;
};
void destroy_leased(LeasedInvocationState *state) noexcept {
  if (!state) return;
  const auto allocator = state->allocator;
  std::destroy_at(state);
  allocator.deallocate(state, sizeof(LeasedInvocationState), alignof(LeasedInvocationState));
}
}
LeasedInvocation::LeasedInvocation(LeasedInvocation &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}
LeasedInvocation &LeasedInvocation::operator=(LeasedInvocation &&other) noexcept {
  if (this != &other) {
    if (state_) { close().IgnoreError(); detail::destroy_leased(state_); }
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}
LeasedInvocation::~LeasedInvocation() {
  if (state_) { close().IgnoreError(); detail::destroy_leased(state_); }
}
Result<LeasedInvocation> LeasedInvocation::create(const Program &program, const MemoryResource &memory,
    LeaseImportOptions options) noexcept {
  if (!program.valid() || !memory.valid() || !options.allocator.valid())
    return std::unexpected(Status::InvalidArgument("invalid apple leased invocation"));
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  HYLE_ASSIGN_OR_RETURN(auto native, program.create_invocation(allocator));
  auto *storage = allocator.allocate(sizeof(detail::LeasedInvocationState), alignof(detail::LeasedInvocationState));
  if (!storage) return std::unexpected(Status::ResourceExhausted("apple leased state metadata"));
  auto *state = std::construct_at(static_cast<detail::LeasedInvocationState *>(storage), allocator);
  LeasedInvocation result(state);
  HYLE_RETURN_IF_ERROR(state->inputs.prepare(program.input_count(), options));
  state->invocation = std::move(native);
  state->memory = &memory;
  state->options = options;
  return result;
}
bool LeasedInvocation::in_flight() const noexcept {
  return state_ && state_->inputs.active();
}
Status LeasedInvocation::bind(std::span<const LeasedInput> inputs) noexcept {
  if (!state_ || state_->inputs.closed())
    return Status::FailedPrecondition("apple leased invocation is empty/closed");
  if (in_flight()) return Status::Unavailable("apple leased invocation has native work");
  if (inputs.size() != state_->inputs.size())
    return Status::InvalidArgument("apple leased input count differs");
  try {
    std::vector<mlx::core::array, mem::StdAllocator<mlx::core::array>> views{
        mem::StdAllocator<mlx::core::array>(state_->allocator)};
    views.reserve(inputs.size());
    for (const auto &input : inputs) {
      HYLE_ASSIGN_OR_RETURN(auto view,
          state_->memory->binding(input.source, input.description, state_->options));
      views.push_back(std::move(view));
    }
    HYLE_RETURN_IF_ERROR(state_->invocation.bind(std::span<const mlx::core::array>(views)));
    for (std::size_t index = 0; index < inputs.size(); ++index)
      state_->inputs.retain(index, inputs[index].source);
    return Status::Ok();
  } catch (...) { return Status::ResourceExhausted("apple leased binding metadata"); }
}
Result<OwnedResult<Results>> LeasedInvocation::submit() noexcept {
  if (!state_ || state_->inputs.closed())
    return std::unexpected(Status::FailedPrecondition("apple leased invocation is empty/closed"));
  HYLE_ASSIGN_OR_RETURN(auto batch, state_->inputs.admit());
  return state_->invocation.submit_guarded(std::move(batch));
}
Status LeasedInvocation::close() noexcept {
  return state_ ? state_->inputs.close() : Status::Ok();
}
Status LeasedInvocation::drain() noexcept {
  if (!state_) return Status::Ok();
  close().IgnoreError();
  Status terminal;
  if (state_->inputs.active()) {
    terminal = state_->invocation.wait_native();
    if (!terminal.ok() && state_->inputs.active()) return terminal;
  }
  HYLE_RETURN_IF_ERROR(state_->inputs.drain());
  return terminal;
}
} // namespace hyle::compute::apple

