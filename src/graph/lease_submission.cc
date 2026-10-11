#include "hyle/graph/detail/lease_submission.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/core/assert.h"
#include "hyle/mem/std_allocator.h"
#include "hyle/util/spin_lock.h"

namespace hyle::graph::detail {
struct LeasePrefix final {
  LeaseSubmissionState *owner = nullptr;
  compute::detail::CompletionRetention retention;
};
struct LeaseSubmissionState final {
  explicit LeaseSubmissionState(mem::AllocatorRef selected)
      : allocator(selected), reads(mem::StdAllocator<compute::ReadLease>(selected)),
        prefixes(mem::StdAllocator<LeasePrefix>(selected)) {}
  mem::AllocatorRef allocator;
  std::vector<compute::ReadLease, mem::StdAllocator<compute::ReadLease>> reads;
  std::vector<LeasePrefix, mem::StdAllocator<LeasePrefix>> prefixes;
  compute::LeaseUsePool uses;
  compute::LeaseUseBatch batch;
  std::atomic<std::size_t> remaining{0};
  std::size_t next_prefix = 0;
  void *parent = nullptr;
  LeaseSubmission::ReferenceFn retain = nullptr, release = nullptr;
  LeaseSubmission::TerminalFn terminal = nullptr;
  util::SpinLock lock;
  Status first_error;
  bool accepted = false;
};
namespace {
void destroy(LeaseSubmissionState *state) noexcept {
  if (!state) return;
  const auto allocator = state->allocator;
  std::destroy_at(state);
  allocator.deallocate(state, sizeof(LeaseSubmissionState), alignof(LeaseSubmissionState));
}
void record_error(LeaseSubmissionState &state, const Status &terminal) noexcept {
  if (!terminal.ok()) {
    std::lock_guard guard(state.lock);
    state.first_error.Update(terminal);
  }
}
void release_episode(LeaseSubmissionState &state) noexcept {
  if (state.remaining.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
  Status terminal;
  { std::lock_guard guard(state.lock); terminal = state.first_error; }
  if (state.accepted) compute::detail::LeaseAccess::retire_native(state.batch, terminal);
  else state.batch = {};
  state.terminal(state.parent, std::move(terminal), state.accepted);
}
void prefix_terminal(void *opaque, const Status &terminal) noexcept {
  record_error(*static_cast<LeasePrefix *>(opaque)->owner, terminal);
}
void prefix_release(void *opaque) noexcept {
  auto *state = static_cast<LeasePrefix *>(opaque)->owner;
  const auto parent = state->parent;
  const auto release = state->release;
  release_episode(*state);
  release(parent); // may destroy this prefix; no access to it afterwards
}
} // namespace

LeaseSubmission::LeaseSubmission(LeaseSubmission &&other) noexcept : state_(std::exchange(other.state_, nullptr)) {}
LeaseSubmission &LeaseSubmission::operator=(LeaseSubmission &&other) noexcept {
  if (this != &other) { destroy(state_); state_ = std::exchange(other.state_, nullptr); }
  return *this;
}
LeaseSubmission::~LeaseSubmission() { destroy(state_); }
Result<LeaseSubmission> LeaseSubmission::create(std::size_t inputs, std::size_t prefixes,
    compute::LeaseImportOptions options) noexcept {
  if (!options.allocator.valid() || inputs >= UINT32_MAX || prefixes == 0)
    return std::unexpected(Status::InvalidArgument("invalid composition lease capacity/allocator"));
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  HYLE_ASSIGN_OR_RETURN(auto uses, compute::LeaseUsePool::create({.max_batches = 1,
      .max_resources_per_batch = static_cast<std::uint32_t>((std::max)(inputs, std::size_t{1})),
      .allocator = options.allocator, .resource_budget = options.resource_budget}));
  auto *storage = allocator.allocate(sizeof(LeaseSubmissionState), alignof(LeaseSubmissionState));
  if (!storage) return std::unexpected(Status::ResourceExhausted("composition lease metadata"));
  auto *state = std::construct_at(static_cast<LeaseSubmissionState *>(storage), allocator);
  LeaseSubmission result(state);
  try {
    state->reads.resize(inputs); state->prefixes.resize(prefixes);
    for (auto &prefix : state->prefixes) prefix.owner = state;
  } catch (...) { return std::unexpected(Status::ResourceExhausted("composition lease tables")); }
  state->uses = std::move(uses);
  return result;
}
void LeaseSubmission::bind(std::size_t input, compute::ReadLease source) noexcept {
  HYLE_FATAL_ASSERT(state_ && input < state_->reads.size());
  state_->reads[input] = std::move(source);
}
void LeaseSubmission::clear(std::size_t input) noexcept {
  if (state_) { HYLE_FATAL_ASSERT(input < state_->reads.size()); state_->reads[input].reset(); }
}
void LeaseSubmission::reset() noexcept { if (state_) for (auto &read : state_->reads) read.reset(); }
Status LeaseSubmission::begin(void *owner, ReferenceFn retain, ReferenceFn release, TerminalFn terminal) noexcept {
  HYLE_FATAL_ASSERT(state_ && owner && retain && release && terminal);
  HYLE_ASSIGN_OR_RETURN(auto batch, state_->uses.begin());
  for (const auto &read : state_->reads) if (read.valid()) HYLE_RETURN_IF_ERROR(batch.read(read));
  state_->parent = owner; state_->retain = retain; state_->release = release; state_->terminal = terminal;
  state_->batch = std::move(batch); state_->next_prefix = 0; state_->accepted = false;
  { std::lock_guard guard(state_->lock); state_->first_error = Status::Ok(); }
  state_->remaining.store(1, std::memory_order_release); // body acceptance sentinel
  return Status::Ok();
}
Status LeaseSubmission::protect(const compute::Completion &completion) noexcept {
  HYLE_FATAL_ASSERT(state_ && state_->remaining.load(std::memory_order_acquire) != 0);
  HYLE_FATAL_ASSERT(state_->next_prefix < state_->prefixes.size());
  auto &prefix = state_->prefixes[state_->next_prefix++];
  state_->accepted = true;
  state_->retain(state_->parent);
  state_->remaining.fetch_add(1, std::memory_order_relaxed);
  prefix.retention.own = {&prefix, &prefix_terminal, &prefix_release};
  // Only the mutable Completion fields change; both local submit results and
  // OwnedResult's original Completion remain the owning native handles.
  HYLE_FATAL_ASSERT(compute::detail::CompletionAccess::attach_retention(
      const_cast<compute::Completion &>(completion), prefix.retention).ok());
  return Status::Ok();
}
void LeaseSubmission::finish(Status terminal, bool accepted) noexcept {
  HYLE_FATAL_ASSERT(state_);
  state_->accepted = state_->accepted || accepted;
  record_error(*state_, terminal);
  release_episode(*state_);
}
Status LeaseSubmission::observed_error() const noexcept {
  if (!state_) return Status::Ok();
  std::lock_guard guard(state_->lock);
  return state_->first_error;
}
} // namespace hyle::graph::detail
