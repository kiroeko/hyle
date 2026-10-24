#include "hyle/graph/detail/lease_submission.h"
#include "submission_internal.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/compute/detail/append_lease.h"
#include "hyle/core/assert.h"
#include "hyle/mem/std_allocator.h"
#include "hyle/util/spin_lock.h"

namespace hyle::graph::detail {
Status LeasePermissionSet::reserve(std::size_t capacity) noexcept {
  try { entries_.reserve(capacity); }
  catch (...) { return Status::ResourceExhausted("graph permission scratch metadata"); }
  return Status::Ok();
}
Status LeasePermissionSet::add(const compute::ReadLease &read) noexcept {
  if (!read.valid()) return Status::FailedPrecondition("graph read permission is empty");
  if (entries_.size() == entries_.capacity())
    return Status::ResourceExhausted("graph permission scratch capacity");
  entries_.push_back({&read, nullptr});
  return Status::Ok();
}
Status LeasePermissionSet::add(const compute::WriteLease &write) noexcept {
  if (!write.valid()) return Status::FailedPrecondition("graph write permission is empty");
  if (entries_.size() == entries_.capacity())
    return Status::ResourceExhausted("graph permission scratch capacity");
  entries_.push_back({nullptr, &write});
  return Status::Ok();
}
Status LeasePermissionSet::add(const compute::detail::AppendLease &append) noexcept {
  if (!append.valid()) return Status::FailedPrecondition("graph append permission is empty");
  if (entries_.size() == entries_.capacity())
    return Status::ResourceExhausted("graph permission scratch capacity");
  entries_.push_back({nullptr, nullptr, &append});
  return Status::Ok();
}
Status LeasePermissionSet::admit(compute::LeaseUseBatch &uses) noexcept {
  struct Reset final {
    LeasePermissionSet &set;
    ~Reset() { set.reset(); }
  } reset{*this};
  const auto identity = [](const Entry &entry) {
    return entry.read ? compute::detail::LeaseAccess::identity(*entry.read)
        : entry.write ? compute::detail::LeaseAccess::identity(*entry.write)
                      : compute::detail::LeaseAccess::identity(*entry.append);
  };
  const auto generation = [](const Entry &entry) {
    return entry.read ? entry.read->generation()
        : entry.write ? entry.write->generation() : entry.append->generation();
  };
  std::sort(entries_.begin(), entries_.end(), [&](const Entry &a, const Entry &b) {
    return std::less<const void *>{}(identity(a), identity(b));
  });
  // Validate the entire group before using the trusted no-rescan F1 entries.
  // Backing identity cannot replace block identity: one arena has many blocks.
  for (std::size_t first = 0; first < entries_.size();) {
    auto end = first + 1;
    while (end < entries_.size() && identity(entries_[end]) == identity(entries_[first])) ++end;
    const compute::detail::AppendLease *append = nullptr;
    bool read = false, write = false;
    for (auto i = first; i < end; ++i) {
      const auto &entry = entries_[i];
      if (generation(entry) != generation(entries_[first]))
        return Status::FailedPrecondition("graph permission generation changed");
      read |= entry.read != nullptr;
      write |= entry.write != nullptr;
      if (entry.append) {
        if (append && append != entry.append)
          return Status::InvalidArgument("graph repeats an original append authority");
        append = entry.append;
      }
    }
    if (write && (read || append))
      return Status::InvalidArgument("one use cannot mix ordinary lease permissions");
    if (append) for (auto i = first; i < end; ++i)
      if (entries_[i].read)
        HYLE_RETURN_IF_ERROR(compute::detail::LeaseAccess::validate_prefix(*append, *entries_[i].read));
    first = end;
  }
  for (std::size_t first = 0; first < entries_.size();) {
    auto end = first + 1;
    const Entry *selected = &entries_[first];
    while (end < entries_.size() && identity(entries_[end]) == identity(entries_[first])) {
      if (entries_[end].append) selected = &entries_[end];
      ++end;
    }
    HYLE_RETURN_IF_ERROR(selected->append
        ? compute::detail::LeaseAccess::append_unique(uses, *selected->append)
        : selected->read ? compute::detail::LeaseAccess::read_unique(uses, *selected->read)
                         : compute::detail::LeaseAccess::write_unique(uses, *selected->write, false));
    first = end;
  }
  return Status::Ok();
}

struct LeasePrefix final {
  LeaseSubmissionState *owner = nullptr;
  compute::detail::CompletionRetention retention;
};
struct LeasePageRange final {
  std::size_t offset = 0, count = 0;
};
struct LeaseSubmissionState final {
  explicit LeaseSubmissionState(mem::AllocatorRef selected)
      : allocator(selected), reads(mem::StdAllocator<compute::ReadLease>(selected)),
        pages(mem::StdAllocator<compute::ReadLease>(selected)),
        page_ranges(mem::StdAllocator<LeasePageRange>(selected)),
        permissions(selected),
        prefixes(mem::StdAllocator<LeasePrefix>(selected)) {}
  mem::AllocatorRef allocator;
  std::vector<compute::ReadLease, mem::StdAllocator<compute::ReadLease>> reads;
  std::vector<compute::ReadLease, mem::StdAllocator<compute::ReadLease>> pages;
  std::vector<LeasePageRange, mem::StdAllocator<LeasePageRange>> page_ranges;
  LeasePermissionSet permissions;
  std::size_t page_count = 0;
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
void replace_pages(LeaseSubmissionState &state, std::size_t input,
    std::span<const compute::ReadLease> source) noexcept {
  if (state.page_ranges.empty()) return;
  auto &range = state.page_ranges[input];
  const auto old_count = range.count;
  const auto tail = state.pages.begin() + range.offset + old_count;
  const auto end = state.pages.begin() + state.page_count;
  const auto next_count = state.page_count - old_count + source.size();
  if (source.size() > old_count) {
    std::move_backward(tail, end, state.pages.begin() + next_count);
  } else if (source.size() < old_count) {
    std::move(tail, end, state.pages.begin() + range.offset + source.size());
    for (auto i = next_count; i < state.page_count; ++i) state.pages[i].reset();
  }
  std::copy(source.begin(), source.end(), state.pages.begin() + range.offset);
  for (auto i = input + 1; i < state.page_ranges.size(); ++i)
    state.page_ranges[i].offset = state.page_ranges[i].offset - old_count + source.size();
  range.count = source.size();
  state.page_count = next_count;
}
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
    LeaseSubmissionOptions options) noexcept {
  if (!options.allocator.valid() || inputs >= UINT32_MAX || prefixes == 0 ||
      options.max_page_reads > UINT32_MAX - inputs)
    return std::unexpected(Status::InvalidArgument("invalid composition lease capacity/allocator"));
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  HYLE_ASSIGN_OR_RETURN(auto uses, compute::LeaseUsePool::create({.max_batches = 1,
      .max_resources_per_batch = static_cast<std::uint32_t>(
          (std::max)(inputs + options.max_page_reads, std::size_t{1})),
      .allocator = options.allocator, .resource_budget = options.resource_budget}));
  auto *storage = allocator.allocate(sizeof(LeaseSubmissionState), alignof(LeaseSubmissionState));
  if (!storage) return std::unexpected(Status::ResourceExhausted("composition lease metadata"));
  auto *state = std::construct_at(static_cast<LeaseSubmissionState *>(storage), allocator);
  LeaseSubmission result(state);
  try {
    state->reads.resize(inputs); state->prefixes.resize(prefixes);
    if (options.max_page_reads != 0) {
      state->pages.resize(options.max_page_reads);
      state->page_ranges.resize(inputs);
      HYLE_RETURN_IF_ERROR(state->permissions.reserve(inputs + options.max_page_reads));
    }
    for (auto &prefix : state->prefixes) prefix.owner = state;
  } catch (...) { return std::unexpected(Status::ResourceExhausted("composition lease tables")); }
  state->uses = std::move(uses);
  return result;
}
void LeaseSubmission::bind(std::size_t input, compute::ReadLease source) noexcept {
  HYLE_FATAL_ASSERT(state_ && input < state_->reads.size());
  replace_pages(*state_, input, {});
  state_->reads[input] = std::move(source);
}
Status LeaseSubmission::bind_pages(std::size_t input,
    std::span<const compute::ReadLease> source) noexcept {
  if (!state_ || input >= state_->reads.size())
    return Status::InvalidArgument("composition page input is out of range");
  if (state_->remaining.load(std::memory_order_acquire) != 0)
    return Status::Unavailable("composition native prefixes have not retired");
  const auto old_count = state_->page_ranges.empty() ? 0 : state_->page_ranges[input].count;
  if (source.size() > state_->pages.size() - (state_->page_count - old_count))
    return Status::ResourceExhausted("composition exact-page capacity exhausted");
  for (const auto &read : source)
    if (!read.valid()) return Status::InvalidArgument("composition exact-page guard is empty");
  replace_pages(*state_, input, source);
  state_->reads[input].reset();
  return Status::Ok();
}
void LeaseSubmission::clear(std::size_t input) noexcept {
  if (state_) {
    HYLE_FATAL_ASSERT(input < state_->reads.size());
    replace_pages(*state_, input, {});
    state_->reads[input].reset();
  }
}
void LeaseSubmission::reset() noexcept {
  if (!state_) return;
  for (auto &read : state_->reads) read.reset();
  for (auto &read : state_->pages) read.reset();
  for (auto &range : state_->page_ranges) range = {};
  state_->page_count = 0;
}
Status LeaseSubmission::begin(void *owner, ReferenceFn retain, ReferenceFn release, TerminalFn terminal) noexcept {
  HYLE_FATAL_ASSERT(state_ && owner && retain && release && terminal);
  HYLE_ASSIGN_OR_RETURN(auto batch, state_->uses.begin());
  if (state_->permissions.enabled()) {
    auto &permissions = state_->permissions;
    permissions.reset();
    for (const auto &read : state_->reads)
      if (read.valid()) HYLE_RETURN_IF_ERROR(permissions.add(read));
    for (std::size_t i = 0; i < state_->page_count; ++i)
      HYLE_RETURN_IF_ERROR(permissions.add(state_->pages[i]));
    HYLE_RETURN_IF_ERROR(permissions.admit(batch));
  } else {
    for (const auto &read : state_->reads)
      if (read.valid()) HYLE_RETURN_IF_ERROR(batch.read(read));
  }
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
