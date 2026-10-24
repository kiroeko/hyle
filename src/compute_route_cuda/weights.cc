#include "weights_internal.h"

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <utility>

#include "../compute/weight_assembly.h"
#include "../compute_backend_cuda/resource_admission.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/context_ref.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/container/small_vector.h"
#include "hyle/util/ref_counted.h"

namespace hyle::compute::cuda {
namespace {
// Each independently bound bank must satisfy the native provider port contract,
// even when its scalar description needs less alignment than the allocation.
constexpr std::size_t kWeightBankAlignment = 256;
} // namespace
namespace detail {

struct WeightBacking final : util::RefCounted<WeightBacking> {
  mem::AllocatorRef allocator{};
  compute::ResourceBudget::Reservation host_charge{}, device_charge{};
  mem::Buffer host{}, device{};
  std::byte *host_pointer = nullptr;
  std::byte *device_pointer = nullptr;
  compute::detail::CompletionLifetime lifetime{};
  void destroy() noexcept {
    const auto owner = allocator;
    std::destroy_at(this);
    owner.deallocate(this, sizeof(WeightBacking), alignof(WeightBacking));
  }
};

struct SelectedWeight final {
  WeightSpec spec{};
  std::size_t offset = 0, bytes = 0;
};

// Completion owns a separate backing reference, never a reference to this
// state which contains that Completion. Native abandonment cannot form a cycle.
struct WeightState final : util::RefCounted<WeightState> {
  explicit WeightState(mem::AllocatorRef owner) noexcept
      : allocator(owner), catalog(owner), selected(owner) {}
  mem::AllocatorRef allocator{};
  compute::detail::ContextRef context{};
  container::SmallVector<WeightSpec, 4, mem::AllocatorRef> catalog;
  container::SmallVector<SelectedWeight, 4, mem::AllocatorRef> selected;
  std::size_t selected_count = 0, max_fragments = 0;
  WeightBacking *backing = nullptr;
  compute::Completion upload{};
  compute::WeightUploadReport report{};
  std::mutex mutex{};
  void destroy() noexcept {
    const auto owner = allocator;
    auto *buffers = backing;
    std::destroy_at(this); // abandons upload with its real native retention
    if (buffers != nullptr)
      buffers->decref();
    owner.deallocate(this, sizeof(WeightState), alignof(WeightState));
  }
};

namespace {
Result<compute::CompletionPoll> observe_upload(WeightState &state) noexcept {
  auto result = state.upload.poll();
  if (!result) state.report.copied_bytes_unknown = true;
  else if (*result == compute::CompletionPoll::kComplete && state.report.submitted_bytes != 0) {
    state.report.completed_bytes = state.report.submitted_bytes;
    state.report.copied_bytes_unknown = false;
  }
  return result;
}
Status wait_upload(WeightState &state) noexcept {
  auto result = state.upload.wait();
  if (!result.ok()) state.report.copied_bytes_unknown = true;
  else if (state.report.submitted_bytes != 0) {
    state.report.completed_bytes = state.report.submitted_bytes;
    state.report.copied_bytes_unknown = false;
  }
  return result;
}
} // namespace

void retain_weight(WeightState *state) noexcept {
  if (state != nullptr)
    state->incref();
}
void release_weight(WeightState *state) noexcept {
  if (state != nullptr)
    state->decref();
}
Status wait_weight(WeightState *state) noexcept {
  if (state == nullptr)
    return Status::FailedPrecondition("invalid weight batch");
  std::lock_guard lock(state->mutex);
  return wait_upload(*state);
}
Result<compute::CompletionPoll> poll_weight(WeightState *state) noexcept {
  if (!state) return std::unexpected(Status::FailedPrecondition("invalid weight batch"));
  std::lock_guard lock(state->mutex);
  return observe_upload(*state);
}
Result<bool> weight_device_route(WeightState *state,
    const compute::detail::QueueRef *queue) noexcept {
  if (!state) return std::unexpected(Status::FailedPrecondition("invalid weight batch"));
  if (!queue || queue->device().backend_id() != kBackendId) return false;
  if (!state->context.owns(*queue))
    return std::unexpected(Status::InvalidArgument("weight queue Context differs"));
  return true;
}
compute::detail::EnqueueResult enqueue_weight(WeightState *state,
    compute::detail::QueueSequence &sequence) noexcept {
  if (!state) return compute::detail::EnqueueResult::rejected(
      Status::FailedPrecondition("invalid weight batch"));
  std::lock_guard lock(state->mutex);
  auto ready = observe_upload(*state);
  if (!ready) return compute::detail::EnqueueResult::rejected(ready.error());
  if (*ready == compute::CompletionPoll::kComplete) return {};
  return sequence.enqueue_resolved_wait(compute::detail::CompletionAccess::dependency(state->upload));
}
Status consumed_weight_status(WeightState *state) noexcept {
  if (!state) return Status::FailedPrecondition("invalid weight batch");
  std::lock_guard lock(state->mutex);
  auto ready = observe_upload(*state);
  if (!ready) return ready.error();
  if (*ready == compute::CompletionPoll::kComplete) return Status::Ok();
  // The successful ordered consumer proves the upload's exact event was
  // reached. Only a broken observation can leave it pending here; preserve
  // that diagnostic through the original native wait/retirement path.
  return wait_upload(*state);
}
Result<compute::Completion> order_weight(WeightState *state,
                                         compute::Queue &queue) noexcept {
  if (state == nullptr)
    return std::unexpected(Status::FailedPrecondition("invalid weight batch"));
  std::lock_guard lock(state->mutex);
  if (!state->context.owns(queue))
    return std::unexpected(Status::InvalidArgument("weight queue Context differs"));
  HYLE_ASSIGN_OR_RETURN(auto sequence,
                       compute::detail::QueueSequence::begin(queue));
  // Observe the immutable upload itself: waiting on a derived Queue prefix
  // does not settle this Completion. Ready batches need no new native fence.
  HYLE_ASSIGN_OR_RETURN(auto ready, observe_upload(*state));
  if (ready == compute::CompletionPoll::kComplete)
    return sequence.finish();
  sequence.record(sequence.enqueue_wait(state->upload));
  return sequence.finish();
}

} // namespace detail
namespace {

using detail::WeightBacking;
using detail::WeightState;

Result<WeightState *> create_state(const compute::Context &context,
                                  mem::MemoryResourceRef host,
                                  mem::MemoryResourceRef device,
                                  std::span<const WeightSpec> catalog,
                                  std::size_t capacity,
                                  std::size_t max_fragments,
                                  mem::AllocatorRef allocator,
                                  const compute::ResourceBudget &budget) noexcept {
  compute::ResourceBudget::Reservation host_charge, device_charge;
  if (budget) {
    HYLE_ASSIGN_OR_RETURN(host_charge, budget.reserve(host, capacity,
        compute::ResourceBudgetCategory::kMigration));
    HYLE_ASSIGN_OR_RETURN(device_charge, budget.reserve(device, capacity,
        compute::ResourceBudgetCategory::kMigration));
  }
  std::size_t alignment = kWeightBankAlignment;
  for (const auto &entry : catalog)
    alignment = std::max(alignment, entry.description.required_alignment());
  HYLE_ASSIGN_OR_RETURN(auto host_buffer,
                       mem::Buffer::create(host, capacity, alignment));
  HYLE_ASSIGN_OR_RETURN(auto device_buffer,
                       mem::Buffer::create(device, capacity, alignment));
  HYLE_ASSIGN_OR_RETURN(auto host_pointer,
                       compute::cuda::pointer<std::byte>(host_buffer.view()));
  HYLE_ASSIGN_OR_RETURN(auto device_pointer,
                       compute::cuda::pointer<std::byte>(device_buffer.view()));
  HYLE_ASSIGN_OR_RETURN(auto retained_context,
                       compute::detail::ContextRef::acquire(context));
  auto *memory = allocator.allocate(sizeof(WeightState), alignof(WeightState));
  if (memory == nullptr)
    return std::unexpected(Status::ResourceExhausted("weight slot metadata"));
  auto *state = std::construct_at(static_cast<WeightState *>(memory), allocator);
  state->max_fragments = max_fragments;
  state->context = std::move(retained_context);
  if (!state->catalog.try_resize(catalog.size()) ||
      !state->selected.try_resize(catalog.size())) {
    state->decref();
    return std::unexpected(Status::ResourceExhausted("weight catalog metadata"));
  }
  std::copy(catalog.begin(), catalog.end(), state->catalog.begin());
  std::sort(state->catalog.begin(), state->catalog.end(),
            [](const auto &a, const auto &b) { return a.id < b.id; });
  memory = allocator.allocate(sizeof(WeightBacking), alignof(WeightBacking));
  if (memory == nullptr) {
    state->decref();
    return std::unexpected(Status::ResourceExhausted("weight backing metadata"));
  }
  state->backing = std::construct_at(static_cast<WeightBacking *>(memory));
  state->backing->allocator = allocator;
  state->backing->host_charge = std::move(host_charge);
  state->backing->device_charge = std::move(device_charge);
  state->backing->host = std::move(host_buffer);
  state->backing->device = std::move(device_buffer);
  state->backing->host_pointer = host_pointer;
  state->backing->device_pointer = device_pointer;
  return state;
}

// The caller holds the slot mutex and its unique state reference. Preserve
// the original nonblocking upload/backing reuse gates before changing metadata.
Status begin_upload(WeightState &state, compute::Queue &queue) noexcept {
  if (!state.context.owns(queue))
    return Status::InvalidArgument("upload queue Context differs");
  if (state.upload.valid()) {
    HYLE_ASSIGN_OR_RETURN(auto previous, detail::observe_upload(state));
    if (previous == compute::CompletionPoll::kPending)
      return Status::FailedPrecondition("weight upload is in flight");
  }
  if (!state.backing->is_unique())
    return Status::FailedPrecondition("native upload retains backing");
  state.upload = {};
  state.report = {};
  state.selected_count = 0;
  return Status::Ok();
}

Status submit_upload(WeightState &state, std::size_t used,
                     std::size_t count, compute::Queue &queue) noexcept {
  auto *backing = state.backing;
  compute::Completion completion = compute::Completion::completed();
  if (used != 0) {
    bool entered = false;
    auto submitted = compute::cuda::submit(queue,
        [backing, used, &entered, &report = state.report](cudaStream_t stream) noexcept {
          entered = true;
          const auto copied = cudaMemcpyAsync(backing->device_pointer,
              backing->host_pointer, used, cudaMemcpyHostToDevice, stream);
          if (copied == cudaSuccess) report.submitted_bytes = used;
          // A successful enqueue establishes the requested extent, not actual
          // traffic. Only this upload's successful retirement clears unknown.
          report.copied_bytes_unknown = true;
          return copied;
        });
    if (!submitted) {
      // If submission reached CUDA, even a failed terminal reservation cannot
      // establish the actual copied prefix. Retain that verdict in the slot.
      state.report.copied_bytes_unknown |= entered;
      state.upload = compute::Completion::completed(submitted.error());
      return submitted.error();
    }
    completion = std::move(*submitted);
  }
  backing->incref();
  backing->lifetime = {backing, [](void *, const Status &) noexcept {},
                       [](void *opaque) noexcept {
                         static_cast<WeightBacking *>(opaque)->decref();
                       }};
  const auto retained = compute::detail::CompletionAccess::attach_lifetime(
      completion, backing->lifetime);
  if (!retained.ok()) {
    const auto retired = completion.wait();
    if (!retired.ok()) state.report.copied_bytes_unknown = true;
    else if (state.report.submitted_bytes != 0) {
      state.report.completed_bytes = state.report.submitted_bytes;
      state.report.copied_bytes_unknown = false;
    }
    state.upload = compute::Completion::completed(retained);
    return retained;
  }
  state.upload = std::move(completion);
  state.selected_count = count;
  return Status::Ok();
}

bool overlaps(const void *left, std::size_t left_size,
              const void *right, std::size_t right_size) noexcept {
  const auto a = reinterpret_cast<std::uintptr_t>(left);
  const auto b = reinterpret_cast<std::uintptr_t>(right);
  return left_size != 0 && right_size != 0 &&
      (a <= b ? b - a < left_size : a - b < right_size);
}

Status validate_slice_aliases(const WeightState &state,
    std::span<const compute::WeightSlice> slices) noexcept {
  // Identity is established by the shared validator, not inferred from a
  // temporary host shadow. Complete every mapping check before staging writes.
  for (const auto &slice : slices) {
    HYLE_ASSIGN_OR_RETURN(auto source, slice.source.storage().map_host());
    const auto bytes = source.bytes();
    const bool alias = overlaps(bytes.data(), bytes.size(),
        state.backing->host_pointer, state.backing->host.size()) ||
        (state.backing->device.properties().has(mem::MemoryProperty::kHostAccessible) &&
         overlaps(bytes.data(), bytes.size(), state.backing->device_pointer,
                  state.backing->device.size()));
    HYLE_RETURN_IF_ERROR(source.close());
    if (alias)
      return Status::InvalidArgument("weight source aliases upload staging/backing");
  }
  return Status::Ok();
}

} // namespace

WeightBatch::WeightBatch(WeightBatch &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}
WeightBatch &WeightBatch::operator=(WeightBatch &&other) noexcept {
  if (this != &other) {
    detail::release_weight(state_);
    state_ = std::exchange(other.state_, nullptr);
  }
  return *this;
}
WeightBatch::~WeightBatch() { detail::release_weight(state_); }
Result<compute::ConstTensorView>
WeightBatch::binding(std::uint64_t id) const noexcept {
  if (state_ == nullptr)
    return std::unexpected(Status::FailedPrecondition("invalid weight batch"));
  for (std::size_t i = 0; i < state_->selected_count; ++i) {
    const auto &entry = state_->selected[i];
    if (entry.spec.id == id) {
      HYLE_ASSIGN_OR_RETURN(auto view,
                           state_->backing->device.view().subview(entry.offset,
                                                                 entry.bytes));
      return compute::ConstTensorView::create(entry.spec.description, view);
    }
  }
  return std::unexpected(Status::NotFound("weight ID is not selected"));
}
Result<compute::CompletionPoll> WeightBatch::poll() const noexcept {
  return detail::poll_weight(state_);
}
Status WeightBatch::wait() const noexcept { return detail::wait_weight(state_); }
Result<compute::WeightUploadReport> WeightBatch::report() const noexcept {
  if (state_ == nullptr)
    return std::unexpected(Status::FailedPrecondition("invalid weight batch"));
  std::lock_guard lock(state_->mutex);
  if (state_->upload.valid()) (void)detail::observe_upload(*state_);
  return state_->report;
}
Result<compute::Completion>
WeightBatch::order(compute::Queue &queue) const noexcept {
  return detail::order_weight(state_, queue);
}

WeightSlots::WeightSlots(WeightSlots &&other) noexcept : closed_(other.closed_) {
  for (std::size_t i = 0; i < 2; ++i)
    states_[i] = std::exchange(other.states_[i], nullptr);
}
WeightSlots &WeightSlots::operator=(WeightSlots &&other) noexcept {
  if (this != &other) {
    for (std::size_t i = 0; i < 2; ++i) {
      detail::release_weight(states_[i]);
      states_[i] = std::exchange(other.states_[i], nullptr);
    }
    closed_ = other.closed_;
  }
  return *this;
}
WeightSlots::~WeightSlots() {
  for (auto *state : states_)
    detail::release_weight(state);
}
Result<WeightSlots> WeightSlots::create(
    const compute::Context &context,
    const compute::cuda::MemoryResource &host,
    const compute::cuda::MemoryResource &device,
    std::span<const WeightSpec> catalog, std::size_t capacity,
    WeightSlotOptions options) noexcept {
  auto allocator = options.allocator;
  if (options.resource_budget) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  if (!allocator.valid() || catalog.empty() || capacity == 0 || options.max_fragments == 0)
    return std::unexpected(Status::InvalidArgument("invalid weight slot capacity"));
  HYLE_RETURN_IF_ERROR(compute::cuda::detail::validate_composition_resources(
      context, host, device));
  for (std::size_t i = 0; i < catalog.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto bytes, catalog[i].description.required_bytes());
    if (bytes > capacity)
      return std::unexpected(Status::ResourceExhausted("weight exceeds capacity"));
    for (std::size_t j = 0; j < i; ++j)
      if (catalog[i].id == catalog[j].id)
        return std::unexpected(Status::InvalidArgument("duplicate weight catalog ID"));
  }
  WeightSlots result;
  for (auto &state : result.states_) {
    HYLE_ASSIGN_OR_RETURN(state, create_state(context, host.ref(), device.ref(),
                                             catalog, capacity, options.max_fragments, allocator,
                                             options.resource_budget));
  }
  return result;
}

Result<WeightBatch> WeightSlots::upload(
    std::size_t slot, std::span<const WeightFragment> selection,
    compute::Queue &queue) noexcept {
  if (closed_ || slot >= 2 || states_[slot] == nullptr)
    return std::unexpected(Status::FailedPrecondition("invalid or closed weight slot"));
  auto &state = *states_[slot];
  if (!state.is_unique())
    return std::unexpected(Status::FailedPrecondition("weight batch is still retained"));
  std::lock_guard lock(state.mutex);
  HYLE_RETURN_IF_ERROR(begin_upload(state, queue));
  if (selection.size() > state.selected.size())
    return std::unexpected(Status::InvalidArgument("too many selected weights"));
  const auto capacity = state.backing->device.size();
  std::size_t used = 0, logical_bytes = 0;
  // Validate every ID/format/range before any copy or native acceptance. The
  // vectors were sized cold, so this is bounded selection work, not allocation.
  for (std::size_t i = 0; i < selection.size(); ++i) {
    const auto &fragment = selection[i];
    const auto found = std::lower_bound(
        state.catalog.begin(), state.catalog.end(), fragment.id,
        [](const WeightSpec &spec, std::uint64_t id) { return spec.id < id; });
    if (found == state.catalog.end() || found->id != fragment.id ||
        !fragment.source.valid() || found->description != fragment.source.description() ||
        !fragment.source.storage().resource().properties().has(
            mem::MemoryProperty::kHostAccessible)) {
      return std::unexpected(Status::InvalidArgument("weight ID/format/source differs"));
    }
    for (std::size_t j = 0; j < i; ++j)
      if (fragment.id == selection[j].id)
        return std::unexpected(Status::InvalidArgument("duplicate selected weight ID"));
    HYLE_ASSIGN_OR_RETURN(auto bytes, found->description.required_bytes());
    const auto alignment = std::max(kWeightBankAlignment,
                                    found->description.required_alignment());
    const auto padding = (alignment - used % alignment) % alignment;
    if (padding > capacity - used || bytes > capacity - used - padding)
      return std::unexpected(Status::ResourceExhausted("selected weights exceed capacity"));
    state.selected[i] = {*found, used + padding, bytes};
    used += padding + bytes;
    logical_bytes += bytes;
  }
  state.report.logical_bytes = logical_bytes;
  std::size_t end = 0;
  for (std::size_t i = 0; i < selection.size(); ++i) {
    const auto &entry = state.selected[i];
    std::fill(state.backing->host_pointer + end,
              state.backing->host_pointer + entry.offset, std::byte{});
    HYLE_ASSIGN_OR_RETURN(auto source, selection[i].source.storage().subview(0, entry.bytes));
    const auto staged = source.read(
        {state.backing->host_pointer + entry.offset, entry.bytes});
    if (!staged.ok()) {
      state.report.copied_bytes_unknown = true;
      return std::unexpected(staged);
    }
    state.report.staged_bytes += entry.bytes;
    end = entry.offset + entry.bytes;
  }
  HYLE_RETURN_IF_ERROR(submit_upload(state, used, selection.size(), queue));
  state.incref(); // the returned batch; bound Graph consumers may retain it
  return WeightBatch(&state);
}

Result<WeightBatch> WeightSlots::upload_slices(
    std::size_t slot, std::span<const compute::WeightSlice> selection,
    compute::Queue &queue) noexcept {
  if (closed_ || slot >= 2 || states_[slot] == nullptr)
    return std::unexpected(Status::FailedPrecondition("invalid or closed weight slot"));
  auto &state = *states_[slot];
  if (!state.is_unique())
    return std::unexpected(Status::FailedPrecondition("weight batch is still retained"));
  std::lock_guard lock(state.mutex);
  HYLE_RETURN_IF_ERROR(begin_upload(state, queue));
  if (selection.size() > state.max_fragments)
    return std::unexpected(Status::ResourceExhausted("weight fragment metadata capacity exceeded"));
  HYLE_ASSIGN_OR_RETURN(const auto size, compute::detail::prepare_weight_slices(
      std::span<const WeightSpec>(state.catalog.data(), state.catalog.size()), selection,
      std::span<detail::SelectedWeight>(state.selected.data(), state.selected.size()),
      state.backing->device.size(), kWeightBankAlignment));
  state.report.logical_bytes = size.logical_bytes;
  HYLE_RETURN_IF_ERROR(validate_slice_aliases(state, selection));
  const std::span<const detail::SelectedWeight> entries(state.selected.data(), size.banks);
  // Clear only private inter-bank padding, not payload that a later source
  // read has yet to populate. Every source alias was rejected above.
  std::size_t end = 0;
  for (const auto &entry : entries) {
    std::fill(state.backing->host_pointer + end,
              state.backing->host_pointer + entry.offset, std::byte{});
    end = entry.offset + entry.bytes;
  }
  for (const auto &slice : selection) {
    HYLE_ASSIGN_OR_RETURN(const auto base, compute::detail::weight_slice_offset(entries, slice.id));
    HYLE_ASSIGN_OR_RETURN(const auto bytes, slice.source.description().required_bytes());
    HYLE_ASSIGN_OR_RETURN(auto source, slice.source.storage().subview(0, bytes));
    const auto staged = source.read(
        {state.backing->host_pointer + base + slice.destination_offset, bytes});
    if (!staged.ok()) {
      state.report.copied_bytes_unknown = true;
      return std::unexpected(staged);
    }
    state.report.staged_bytes += bytes;
  }
  HYLE_RETURN_IF_ERROR(submit_upload(state, size.physical_bytes, size.banks, queue));
  state.incref();
  return WeightBatch(&state);
}

Result<compute::WeightUploadReport> WeightSlots::report(std::size_t slot) const noexcept {
  if (slot >= 2 || states_[slot] == nullptr)
    return std::unexpected(Status::FailedPrecondition("invalid weight slot"));
  auto &state = *states_[slot];
  std::lock_guard lock(state.mutex);
  if (state.upload.valid()) (void)detail::observe_upload(state);
  return state.report;
}

Status WeightSlots::close() noexcept {
  closed_ = true;
  Status terminal;
  for (auto *state : states_) {
    if (state == nullptr)
      continue;
    std::lock_guard lock(state->mutex);
    if (state->upload.valid()) {
      const auto status = detail::wait_upload(*state);
      if (terminal.ok())
        terminal = status;
    }
  }
  return terminal;
}

} // namespace hyle::compute::cuda
