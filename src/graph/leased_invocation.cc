#include "hyle/graph/leased_invocation.h"

#include <atomic>
#include <array>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/retained_completion.h"
#include "hyle/core/assert.h"
#include "hyle/mem/std_allocator.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/spin_lock.h"
#include "plan_state.h"
#include "submission_internal.h"

namespace hyle::graph {
namespace detail {
struct LeaseBinding final {
  ValueId value{};
  compute::TensorDesc description{};
  compute::ReadLease read{};
  compute::WriteLease write{};
  bool exclusive = false;
  bool input = false;
};

struct LeasedInvocationState final : util::RefCounted<LeasedInvocationState> {
  explicit LeasedInvocationState(mem::AllocatorRef selected) noexcept
      : allocator(selected), bindings(mem::StdAllocator<LeaseBinding>(selected)),
        retentions(mem::StdAllocator<compute::detail::CompletionRetention>(selected)) {}
  void destroy() noexcept {
    const auto selected = allocator;
    std::destroy_at(this);
    selected.deallocate(this, sizeof(LeasedInvocationState), alignof(LeasedInvocationState));
  }
  mem::AllocatorRef allocator;
  // Native invocation destruction precedes its external payload owners.
  std::vector<LeaseBinding, mem::StdAllocator<LeaseBinding>> bindings;
  Invocation invocation{};
  compute::LeaseUsePool uses{};
  std::vector<compute::detail::CompletionRetention,
              mem::StdAllocator<compute::detail::CompletionRetention>> retentions;
  compute::LeaseUseBatch batch{};
  compute::detail::RetainedCompletion completion{};
  util::SpinLock lock{};
  Status first_error{};
  std::atomic<std::size_t> remaining{0}, results{0};
  std::atomic<bool> busy{false};
  bool accepted = false, closed = false;
};

void release_owner(LeasedInvocationState *state) noexcept {
  bool last;
  {
    std::lock_guard guard(state->lock);
    last = state->is_unique();
    if (!last) state->decref();
  }
  if (last) state->decref();
}

LeaseBinding *find_binding(LeasedInvocationState *state, ValueId value) noexcept {
  for (auto &binding : state->bindings) if (binding.value == value) return &binding;
  return nullptr;
}
Status control_ready(LeasedInvocationState *state) noexcept {
  if (!state) return Status::FailedPrecondition("leased graph invocation is empty");
  if (state->closed) return Status::FailedPrecondition("leased graph invocation is closed");
  if (state->busy.load(std::memory_order_acquire) || state->invocation.in_flight())
    return Status::Unavailable("leased graph invocation is in flight");
  if (state->results.load(std::memory_order_acquire) != 0)
    return Status::Unavailable("leased graph results still retain this invocation");
  if (!state->completion.available())
    return Status::Unavailable("old leased graph completion retains its slot");
  return Status::Ok();
}
void record_error(LeasedInvocationState *state, const Status &error) noexcept {
  if (!error.ok()) {
    std::lock_guard guard(state->lock);
    state->first_error.Update(error);
  }
}
void child_terminal(void *opaque, const Status &terminal) noexcept {
  record_error(static_cast<LeasedInvocationState *>(opaque), terminal);
}
void release_episode(void *opaque) noexcept {
  auto *state = static_cast<LeasedInvocationState *>(opaque);
  if (state->remaining.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
  Status terminal;
  {
    std::lock_guard guard(state->lock);
    terminal = state->first_error;
  }
  if (state->accepted) {
    InvocationAccess::retire(state->invocation, terminal);
    auto retired = compute::Completion::completed(terminal);
    HYLE_FATAL_ASSERT(state->batch.attach(retired).ok());
  } else {
    state->batch = {}; // rejection before native acceptance does not poison
  }
  bool last;
  {
    // Include callback cleanup in drain's lock barrier. Its caller still has
    // a root reference; only an abandoned adapter can take the last branch.
    std::lock_guard guard(state->lock);
    last = state->is_unique();
    if (!last) state->decref();
    state->busy.store(false, std::memory_order_release);
    state->busy.notify_all();
  }
  if (last) state->decref();
}
void retain_accepted(void *opaque, std::size_t segment,
                     compute::Completion &completion) noexcept {
  auto *state = static_cast<LeasedInvocationState *>(opaque);
  auto &node = state->retentions[segment];
  HYLE_FATAL_ASSERT(!node.own.valid());
  state->accepted = true;
  state->remaining.fetch_add(1, std::memory_order_relaxed);
  node.own = {state, &child_terminal, &release_episode};
  HYLE_FATAL_ASSERT(compute::detail::CompletionAccess::attach_retention(completion, node).ok());
}

Result<compute::ReadLease> output_storage(LeasedInvocationState *state,
    compute::ConstTensorView tensor) noexcept {
  const auto wanted = tensor.storage();
  HYLE_ASSIGN_OR_RETURN(const auto bytes, tensor.description().required_bytes());
  for (auto &binding : state->bindings) {
    if (!binding.read.valid()) continue;
    HYLE_ASSIGN_OR_RETURN(auto source, binding.read.view());
    if (source.resource() != wanted.resource() || source.native_handle() != wanted.native_handle() ||
        wanted.offset() < source.offset()) continue;
    const auto offset = wanted.offset() - source.offset();
    if (offset <= source.size() && bytes <= source.size() - offset)
      return binding.read.subview(offset, bytes);
  }
  return std::unexpected(Status::FailedPrecondition("output uses original private Graph storage"));
}
} // namespace detail

LeasedGraphOutput::LeasedGraphOutput(detail::LeasedInvocationState *state,
    compute::ConstTensorView tensor) noexcept : state_(state), tensor_(tensor) {
  state_->incref();
  state_->results.fetch_add(1, std::memory_order_relaxed);
}
LeasedGraphOutput::LeasedGraphOutput(const LeasedGraphOutput &other) noexcept
    : state_(other.state_), tensor_(other.tensor_) {
  if (state_) {
    state_->incref(); state_->results.fetch_add(1, std::memory_order_relaxed);
  }
}
LeasedGraphOutput &LeasedGraphOutput::operator=(const LeasedGraphOutput &other) noexcept {
  if (this != &other) { LeasedGraphOutput copy(other); *this = std::move(copy); }
  return *this;
}
LeasedGraphOutput::LeasedGraphOutput(LeasedGraphOutput &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)), tensor_(other.tensor_) {}
LeasedGraphOutput &LeasedGraphOutput::operator=(LeasedGraphOutput &&other) noexcept {
  if (this != &other) { reset(); state_ = std::exchange(other.state_, nullptr); tensor_ = other.tensor_; }
  return *this;
}
LeasedGraphOutput::~LeasedGraphOutput() { reset(); }
void LeasedGraphOutput::reset() noexcept {
  if (auto *state = std::exchange(state_, nullptr)) {
    state->results.fetch_sub(1, std::memory_order_release);
    detail::release_owner(state);
  }
  tensor_ = {};
}
Result<compute::ConstTensorView> LeasedGraphOutput::tensor() const noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("leased graph output is empty"));
  return tensor_;
}
Result<compute::ReadLease> LeasedGraphOutput::storage_lease() const noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("leased graph output is empty"));
  return detail::output_storage(state_, tensor_);
}

LeasedInvocation::LeasedInvocation(LeasedInvocation &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}
LeasedInvocation &LeasedInvocation::operator=(LeasedInvocation &&other) noexcept {
  if (this != &other) {
    if (state_) { close().IgnoreError(); state_->completion.reset(); detail::release_owner(state_); }
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}
LeasedInvocation::~LeasedInvocation() {
  if (state_) { close().IgnoreError(); state_->completion.reset(); detail::release_owner(state_); }
}
Result<LeasedInvocation> LeasedInvocation::create(const Executable &executable,
    LeasedInvocationOptions options, std::span<const ShapeBinding> shape) noexcept {
  if (!executable.valid() || !options.allocator.valid())
    return std::unexpected(Status::InvalidArgument("invalid leased graph/allocator"));
  HYLE_ASSIGN_OR_RETURN(auto invocation, executable.create_invocation(shape));
  const auto actual = invocation.executable();
  const auto inputs = actual.input_count(), outputs = actual.output_count();
  if (inputs > UINT32_MAX - 1 || outputs > UINT32_MAX - 1 - inputs)
    return std::unexpected(Status::ResourceExhausted("leased graph binding capacity"));
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  HYLE_ASSIGN_OR_RETURN(auto uses, compute::LeaseUsePool::create({.max_batches = 1,
      .max_resources_per_batch = static_cast<std::uint32_t>(inputs + outputs + 1),
      .allocator = options.allocator, .resource_budget = options.resource_budget}));
  auto *storage = allocator.allocate(sizeof(detail::LeasedInvocationState), alignof(detail::LeasedInvocationState));
  if (!storage) return std::unexpected(Status::ResourceExhausted("leased graph state metadata"));
  auto *state = std::construct_at(static_cast<detail::LeasedInvocationState *>(storage), allocator);
  LeasedInvocation result(state);
  HYLE_ASSIGN_OR_RETURN(state->completion, compute::detail::RetainedCompletion::create(allocator));
  state->invocation = std::move(invocation); state->uses = std::move(uses);
  try {
    state->bindings.resize(inputs);
    state->bindings.reserve(inputs + outputs);
    state->retentions.resize(actual.segment_count() + 1);
    for (std::size_t index = 0; index < inputs; ++index) {
      HYLE_ASSIGN_OR_RETURN(state->bindings[index].value, actual.input_id(index));
      state->bindings[index].input = true;
    }
    for (std::size_t index = 0; index < outputs; ++index) {
      HYLE_ASSIGN_OR_RETURN(auto ownership, actual.output_ownership(index));
      if (ownership != OutputOwnership::kCaller) continue;
      HYLE_ASSIGN_OR_RETURN(auto value, actual.output_id(index));
      if (!detail::find_binding(state, value)) state->bindings.push_back({.value = value});
    }
  } catch (...) {
    return std::unexpected(Status::ResourceExhausted("leased graph fixed metadata"));
  }
  return result;
}
bool LeasedInvocation::in_flight() const noexcept {
  return state_ && (state_->busy.load(std::memory_order_acquire) || state_->invocation.in_flight());
}
Status LeasedInvocation::bind(ValueId value, compute::TensorDesc description,
                             compute::ReadLease lease) noexcept {
  HYLE_RETURN_IF_ERROR(detail::control_ready(state_));
  auto *binding = detail::find_binding(state_, value);
  if (!binding || !binding->input) return Status::InvalidArgument("read lease must bind a graph input");
  HYLE_ASSIGN_OR_RETURN(auto tensor, lease.tensor(description));
  HYLE_RETURN_IF_ERROR(state_->invocation.bind(value, tensor));
  binding->write.reset(); binding->read = std::move(lease);
  binding->description = description; binding->exclusive = false;
  return Status::Ok();
}
Status LeasedInvocation::bind(ValueId value, compute::TensorDesc description,
                             compute::WriteLease &&lease) noexcept {
  HYLE_RETURN_IF_ERROR(detail::control_ready(state_));
  auto *binding = detail::find_binding(state_, value);
  if (!binding) return Status::InvalidArgument("write lease must bind an external graph value");
  HYLE_ASSIGN_OR_RETURN(auto tensor, lease.tensor(description));
  HYLE_RETURN_IF_ERROR(state_->invocation.bind(value, tensor));
  binding->read.reset(); binding->write = std::move(lease);
  binding->description = description; binding->exclusive = true;
  return Status::Ok();
}
Status LeasedInvocation::prepare_replay(std::span<const QueueBinding> queues,
                                      ReplayOptions options) noexcept {
  HYLE_RETURN_IF_ERROR(detail::control_ready(state_));
  return state_->invocation.prepare_replay(queues, options);
}
ReplayRoute LeasedInvocation::replay_route() const noexcept {
  return state_ ? state_->invocation.replay_route() : ReplayRoute::kUnprepared;
}
Result<ReplayReport> LeasedInvocation::replay_report() const noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("leased graph is empty"));
  return state_->invocation.replay_report();
}
Status LeasedInvocation::reset() noexcept {
  HYLE_RETURN_IF_ERROR(detail::control_ready(state_));
  HYLE_RETURN_IF_ERROR(state_->invocation.reset());
  for (auto &binding : state_->bindings) {
    binding.read.reset(); binding.write.reset(); binding.exclusive = false;
  }
  return Status::Ok();
}
Result<compute::Completion> LeasedInvocation::submit() noexcept {
  HYLE_RETURN_IF_ERROR(detail::control_ready(state_));
  if (state_->invocation.replay_route() == ReplayRoute::kUnprepared)
    HYLE_RETURN_IF_ERROR(state_->invocation.prepare_replay({}));
  return submit(std::span<const QueueBinding>{});
}
Result<compute::Completion> LeasedInvocation::submit(compute::Queue &queue) noexcept {
  HYLE_RETURN_IF_ERROR(detail::control_ready(state_));
  const auto executable = state_->invocation.executable();
  if (executable.required_queue_count() != 1)
    return std::unexpected(Status::InvalidArgument("leased graph needs its full queue bindings"));
  for (std::size_t index = 0; index < executable.lane_count(); ++index) {
    const auto lane = ExecutionLaneId{static_cast<std::uint32_t>(index)};
    HYLE_ASSIGN_OR_RETURN(auto required, executable.requires_queue(lane));
    if (!required) continue;
    const std::array bindings{QueueBinding{lane, &queue}};
    if (state_->invocation.replay_route() == ReplayRoute::kUnprepared)
      HYLE_RETURN_IF_ERROR(state_->invocation.prepare_replay(bindings));
    return submit(bindings);
  }
  return std::unexpected(Status::Internal("leased graph active queue is missing"));
}
Result<compute::Completion> LeasedInvocation::submit(std::span<const QueueBinding> queues) noexcept {
  HYLE_RETURN_IF_ERROR(detail::control_ready(state_));
  for (auto &binding : state_->bindings) {
    if (binding.exclusive && binding.read.valid()) {
      HYLE_ASSIGN_OR_RETURN(binding.write, binding.read.try_write());
    }
  }
  HYLE_ASSIGN_OR_RETURN(auto batch, state_->uses.begin());
  for (const auto &binding : state_->bindings)
    HYLE_RETURN_IF_ERROR(binding.exclusive ? batch.write(binding.write) : batch.read(binding.read));
  HYLE_ASSIGN_OR_RETURN(auto cursor, detail::SubmissionCursor::begin(state_->invocation, queues));
  {
    std::lock_guard guard(state_->lock);
    state_->first_error = Status::Ok(); state_->accepted = false;
    state_->batch = std::move(batch);
    state_->remaining.store(1, std::memory_order_relaxed);
    state_->busy.store(true, std::memory_order_release);
    state_->incref();
  }
  while (!cursor.done()) {
    auto advanced = cursor.advance(true, &detail::retain_accepted, state_);
    if (!advanced) cursor.reject_successors(std::move(advanced).error());
  }
  detail::record_error(state_, cursor.submission_error());
  auto completion = cursor.finish(true);
  if (completion) {
    detail::retain_accepted(state_, state_->retentions.size() - 1, *completion);
  } else {
    detail::record_error(state_, completion.error());
  }
  detail::release_episode(state_); // admission sentinel, never GPU preemption
  if (!completion) return std::unexpected(std::move(completion).error());
  return state_->completion.publish(std::move(*completion));
}
Result<LeasedGraphOutput> LeasedInvocation::output(ValueId value) noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("leased graph is empty"));
  if (in_flight()) return std::unexpected(Status::Unavailable("leased graph output is not retired"));
  HYLE_ASSIGN_OR_RETURN(auto tensor, state_->invocation.output(value));
  for (auto &binding : state_->bindings) {
    if (binding.write.valid()) {
      HYLE_ASSIGN_OR_RETURN(binding.read, binding.write.publish());
    }
  }
  return LeasedGraphOutput(state_, tensor);
}
Status LeasedInvocation::close() noexcept {
  if (!state_) return Status::Ok();
  state_->closed = true;
  return in_flight() ? Status::Unavailable("leased graph still has native work") : Status::Ok();
}
Status LeasedInvocation::drain() noexcept {
  if (!state_) return Status::Ok();
  close().IgnoreError();
  const auto observed = state_->completion.wait();
  // An observed error can already have handed a prefix to its native reaper.
  // Drain still waits the actual-retirement latch, preserving that error;
  // observing the public terminal is not proof that all children retired.
  while (state_->busy.load(std::memory_order_acquire)) state_->busy.wait(true, std::memory_order_acquire);
  std::lock_guard guard(state_->lock);
  state_->first_error.Update(observed);
  return state_->first_error;
}
} // namespace hyle::graph
