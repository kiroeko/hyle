#include "weights_internal.h"

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <utility>

#include "../compute_backend_cuda/resource_admission.h"
#include "hyle/compute/detail/completion_access.h"
#include "hyle/compute/detail/context_ref.h"
#include "hyle/compute/detail/queue_sequence.h"
#include "hyle/container/small_vector.h"
#include "hyle/util/ref_counted.h"

namespace hyle::compute::cuda {
namespace detail {

struct WeightBacking final : util::RefCounted<WeightBacking> {
  mem::AllocatorRef allocator{};
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
  std::uint64_t id = 0;
  compute::TensorDesc description{};
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
  std::size_t selected_count = 0;
  WeightBacking *backing = nullptr;
  compute::Completion upload{};
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
  return state->upload.wait();
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
                                  mem::AllocatorRef allocator) noexcept {
  std::size_t alignment = 256;
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
  state->backing->host = std::move(host_buffer);
  state->backing->device = std::move(device_buffer);
  state->backing->host_pointer = host_pointer;
  state->backing->device_pointer = device_pointer;
  return state;
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
    if (entry.id == id) {
      HYLE_ASSIGN_OR_RETURN(auto view,
                           state_->backing->device.view().subview(entry.offset,
                                                                 entry.bytes));
      return compute::ConstTensorView::create(entry.description, view);
    }
  }
  return std::unexpected(Status::NotFound("weight ID is not selected"));
}
Result<compute::CompletionPoll> WeightBatch::poll() const noexcept {
  if (state_ == nullptr)
    return std::unexpected(Status::FailedPrecondition("invalid weight batch"));
  std::lock_guard lock(state_->mutex);
  return state_->upload.poll();
}
Status WeightBatch::wait() const noexcept { return detail::wait_weight(state_); }
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
    mem::AllocatorRef allocator) noexcept {
  if (!allocator.valid() || catalog.empty() || capacity == 0)
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
                                             catalog, capacity, allocator));
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
  if (!state.context.owns(queue))
    return std::unexpected(Status::InvalidArgument("upload queue Context differs"));
  if (state.upload.valid()) {
    HYLE_ASSIGN_OR_RETURN(auto previous, state.upload.poll());
    if (previous == compute::CompletionPoll::kPending)
      return std::unexpected(Status::FailedPrecondition("weight upload is in flight"));
  }
  if (!state.backing->is_unique())
    return std::unexpected(Status::FailedPrecondition("native upload retains backing"));
  if (selection.size() > state.selected.size())
    return std::unexpected(Status::InvalidArgument("too many selected weights"));
  const auto capacity = state.backing->device.size();
  std::size_t used = 0;
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
    const auto alignment = found->description.required_alignment();
    const auto padding = (alignment - used % alignment) % alignment;
    if (padding > capacity - used || bytes > capacity - used - padding)
      return std::unexpected(Status::ResourceExhausted("selected weights exceed capacity"));
    state.selected[i] = {fragment.id, found->description, used + padding, bytes};
    used += padding + bytes;
  }
  std::size_t end = 0;
  for (std::size_t i = 0; i < selection.size(); ++i) {
    const auto &entry = state.selected[i];
    std::fill(state.backing->host_pointer + end,
              state.backing->host_pointer + entry.offset, std::byte{});
    HYLE_RETURN_IF_ERROR(selection[i].source.storage().read(
        {state.backing->host_pointer + entry.offset, entry.bytes}));
    end = entry.offset + entry.bytes;
  }
  auto *backing = state.backing;
  compute::Completion completion = compute::Completion::completed();
  if (used != 0) {
    HYLE_ASSIGN_OR_RETURN(auto submitted,
        compute::cuda::submit(queue, [backing, used](cudaStream_t stream) noexcept {
          return cudaMemcpyAsync(backing->device_pointer, backing->host_pointer,
                                 used, cudaMemcpyHostToDevice, stream);
        }));
    completion = std::move(submitted);
  }
  backing->incref();
  backing->lifetime = {backing, [](void *, const Status &) noexcept {},
                       [](void *opaque) noexcept {
                         static_cast<WeightBacking *>(opaque)->decref();
                       }};
  const auto retained = compute::detail::CompletionAccess::attach_lifetime(
      completion, backing->lifetime);
  if (!retained.ok()) {
    completion.wait().IgnoreError();
    return std::unexpected(retained);
  }
  state.upload = std::move(completion);
  state.selected_count = selection.size();
  state.incref(); // the returned batch; bound Graph consumers may retain it
  return WeightBatch(&state);
}

Status WeightSlots::close() noexcept {
  closed_ = true;
  Status terminal;
  for (auto *state : states_) {
    if (state == nullptr)
      continue;
    std::lock_guard lock(state->mutex);
    if (state->upload.valid()) {
      const auto status = state->upload.wait();
      if (terminal.ok())
        terminal = status;
    }
  }
  return terminal;
}

} // namespace hyle::compute::cuda
