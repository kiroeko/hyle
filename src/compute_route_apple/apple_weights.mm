#include "hyle/compute/apple_weights.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <utility>

#include <mlx/allocator.h>

#include "apple_internal.h"
#include "completion_internal.h"
#include "hyle/mem/std_allocator.h"

namespace hyle::compute::apple {
namespace mx = mlx::core;
namespace cd = compute::detail;
namespace detail {

struct SelectedWeight final {
  WeightSpec spec;
  std::size_t offset = 0, bytes = 0;
};

// The native command callback retains only backing, never the slot containing
// the command. Arrays retain the slot, which never contains those arrays.
struct WeightBacking final {
  ResourceBudget::Reservation staging_charge, destination_charge;
  mem::Buffer staging, destination;
};

struct WeightCompletion;
struct WeightProgress final {
  std::mutex mutex;
  bool done = true;
  Status terminal;
  WeightCompletion *abandoned = nullptr;
};

struct WeightState final {
  explicit WeightState(mem::AllocatorRef owner)
      : allocator(owner), catalog(mem::StdAllocator<WeightSpec>(owner)),
        selected(mem::StdAllocator<SelectedWeight>(owner)) {}
  ~WeightState() {
    [command release];
    [event release];
    [queue release];
  }
  mem::AllocatorRef allocator;
  ResourceBudget budget;
  std::vector<WeightSpec, mem::StdAllocator<WeightSpec>> catalog;
  std::vector<SelectedWeight, mem::StdAllocator<SelectedWeight>> selected;
  std::shared_ptr<WeightBacking> backing;
  std::shared_ptr<WeightProgress> progress;
  std::size_t count = 0, capacity = 0;
  id<MTLCommandQueue> queue = nil;
  id<MTLSharedEvent> event = nil;
  id<MTLCommandBuffer> command = nil;
  std::uint64_t generation = 0;
};

struct WeightArrayOwner final {
  ResourceBudget::Reservation private_metadata;
  std::shared_ptr<WeightState> slot;
};

Status metal_status(id<MTLCommandBuffer> command, id<MTLSharedEvent> event,
                    std::uint64_t generation) noexcept {
  if (command.status == MTLCommandBufferStatusCompleted) {
    return event.signaledValue >= generation ? Status::Ok()
        : Status::Internal("Metal weight event did not complete");
  }
  try {
    const auto error = command.error;
    const char *message = error.localizedDescription.UTF8String;
    return Status::Internal("Metal weight blit error " + std::to_string(error.code) + ": " +
                            (message != nullptr ? message : "unknown native error"));
  } catch (...) { return native_exception(); }
}

Result<CompletionPoll> poll_weight(const WeightState &state) noexcept {
  std::lock_guard lock(state.progress->mutex);
  if (!state.progress->done) return CompletionPoll::kPending;
  if (!state.progress->terminal.ok()) return std::unexpected(state.progress->terminal);
  return CompletionPoll::kComplete;
}
Status wait_weight(const WeightState &state) noexcept {
  if (state.command != nil) [state.command waitUntilCompleted];
  const auto result = poll_weight(state);
  return result ? Status::Ok() : result.error();
}

struct WeightCompletion final {
  mem::AllocatorRef allocator;
  std::shared_ptr<WeightState> state;
  cd::CompletionLifetime lifetime{};
  Status hint{};
  WeightCompletion *next = nullptr;
  void destroy() noexcept {
    const auto owner = allocator;
    std::destroy_at(this);
    owner.deallocate(this, sizeof(WeightCompletion), alignof(WeightCompletion));
  }
};

void deliver_weight(WeightCompletion *completion, Status status) noexcept {
  if (status.ok() && !completion->hint.ok()) status = completion->hint;
  if (completion->lifetime.valid()) {
    completion->lifetime.terminal(completion->lifetime.state, status);
    completion->lifetime.release(completion->lifetime.state);
  }
  completion->destroy();
}
void finish_weight(WeightProgress &progress, Status status) noexcept {
  WeightCompletion *abandoned;
  {
    std::lock_guard lock(progress.mutex);
    progress.terminal = status;
    progress.done = true;
    abandoned = std::exchange(progress.abandoned, nullptr);
  }
  while (abandoned != nullptr) {
    auto *next = abandoned->next;
    deliver_weight(abandoned, status);
    abandoned = next;
  }
}

const cd::CompletionVTable weight_completion_vtable{
    .destroy = [](void *opaque, cd::CompletionLifetime lifetime,
                  Status hint) noexcept {
      auto *completion = static_cast<WeightCompletion *>(opaque);
      auto &progress = *completion->state->progress;
      completion->lifetime = lifetime;
      completion->hint = hint;
      Status status;
      {
        std::lock_guard lock(progress.mutex);
        if (!progress.done) {
          // Native completion was registered before commit. Transfer this
          // reference to that handler; do not register a post-commit handler.
          completion->next = progress.abandoned;
          progress.abandoned = completion;
          return;
        }
        status = progress.terminal;
      }
      deliver_weight(completion, status);
    },
    .poll = [](void *opaque) noexcept {
      return poll_weight(*static_cast<WeightCompletion *>(opaque)->state);
    },
    .wait = [](void *opaque) noexcept {
      return wait_weight(*static_cast<WeightCompletion *>(opaque)->state);
    },
    .request_cancel = [](void *) noexcept {
      return Status::Unimplemented("an accepted Metal blit cannot be preempted");
    }};

} // namespace detail

WeightBatch::~WeightBatch() = default;
WeightSlots::~WeightSlots() = default;

Result<CompletionPoll> WeightBatch::poll() const noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("empty Apple weight batch"));
  return detail::poll_weight(*state_);
}
Status WeightBatch::wait() const noexcept {
  if (!state_) return Status::FailedPrecondition("empty Apple weight batch");
  return detail::wait_weight(*state_);
}
Result<mx::array> WeightBatch::binding(std::uint64_t id) const noexcept {
  HYLE_ASSIGN_OR_RETURN(auto ready, poll());
  if (ready != CompletionPoll::kComplete)
    return std::unexpected(Status::FailedPrecondition("Apple weight bytes are not ready"));
  for (std::size_t i = 0; i < state_->count; ++i) {
    const auto &entry = state_->selected[i];
    if (entry.spec.id != id) continue;
    HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(entry.spec.description));
    try {
      auto retained = std::allocate_shared<detail::WeightArrayOwner>(
          mem::StdAllocator<detail::WeightArrayOwner>(state_->allocator));
      retained->slot = state_;
      if (state_->budget.valid()) {
        HYLE_ASSIGN_OR_RETURN(retained->private_metadata,
                             state_->budget.reserve_unattributed(std::nullopt));
      }
      mx::array result(spec.shape, spec.dtype, nullptr, {});
      mx::array base(mx::Shape{1}, mx::uint8, nullptr, {});
      base.set_data(mx::allocator::Buffer(state_->backing->destination.native_handle()),
                    [owner = std::move(retained)](mx::allocator::Buffer) {});
      result.copy_shared_buffer(base, result.strides(), result.flags(), result.size(),
                                static_cast<std::int64_t>(entry.offset / result.itemsize()));
      result.set_status(mx::array::available);
      return result;
    } catch (...) { return std::unexpected(detail::native_exception()); }
  }
  return std::unexpected(Status::NotFound("Apple weight ID is not selected"));
}
Result<Completion> WeightBatch::completion() const noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("empty Apple weight batch"));
  const auto allocator = state_->allocator;
  auto *memory = allocator.allocate(sizeof(detail::WeightCompletion), alignof(detail::WeightCompletion));
  if (memory == nullptr)
    return std::unexpected(Status::ResourceExhausted("Apple weight completion metadata"));
  auto *owner = std::construct_at(static_cast<detail::WeightCompletion *>(memory), allocator, state_);
  return cd::CompletionAccess::make(&detail::weight_completion_vtable, owner);
}

Result<WeightSlots> WeightSlots::create(
    const Context &context, const MemoryResource &memory,
    std::span<const WeightSpec> catalog, std::size_t capacity,
    WeightSlotOptions options) noexcept {
  if (!context.valid() || !memory.valid() || catalog.empty() || capacity == 0 ||
      capacity > static_cast<std::size_t>(INT64_MAX))
    return std::unexpected(Status::InvalidArgument("invalid Apple weight slot capacity/context"));
  auto allocator = options.allocator;
  if (options.resource_budget.valid()) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  if (!allocator.valid())
    return std::unexpected(Status::InvalidArgument("invalid Apple weight metadata allocator"));
  std::size_t alignment = alignof(std::max_align_t);
  for (std::size_t i = 0; i < catalog.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto spec, input_spec(catalog[i].description));
    (void)spec;
    HYLE_ASSIGN_OR_RETURN(auto bytes, catalog[i].description.required_bytes());
    if (bytes > capacity)
      return std::unexpected(Status::ResourceExhausted("Apple weight exceeds slot capacity"));
    alignment = std::max(alignment, catalog[i].description.required_alignment());
    for (std::size_t j = 0; j < i; ++j)
      if (catalog[i].id == catalog[j].id)
        return std::unexpected(Status::InvalidArgument("duplicate Apple weight catalog ID"));
  }
  try {
    WeightSlots result;
    for (auto &state : result.states_) {
      state = std::allocate_shared<detail::WeightState>(mem::StdAllocator<detail::WeightState>(allocator), allocator);
      state->budget = options.resource_budget;
      state->catalog.assign(catalog.begin(), catalog.end());
      std::sort(state->catalog.begin(), state->catalog.end(),
                [](const auto &a, const auto &b) { return a.id < b.id; });
      state->selected.resize(catalog.size());
      state->capacity = capacity;
      state->progress = std::allocate_shared<detail::WeightProgress>(mem::StdAllocator<detail::WeightProgress>(allocator));
      state->backing = std::allocate_shared<detail::WeightBacking>(mem::StdAllocator<detail::WeightBacking>(allocator));
      auto &backing = *state->backing;
      if (options.resource_budget.valid()) {
        HYLE_ASSIGN_OR_RETURN(backing.staging_charge, options.resource_budget.reserve(memory.ref(), (capacity + 3) & ~std::size_t{3}, ResourceBudgetCategory::kMigration));
        HYLE_ASSIGN_OR_RETURN(backing.destination_charge, options.resource_budget.reserve(memory.ref(), (capacity + 3) & ~std::size_t{3}, ResourceBudgetCategory::kMigration));
      }
      const auto native_capacity = (capacity + 3) & ~std::size_t{3};
      HYLE_ASSIGN_OR_RETURN(backing.staging, mem::Buffer::create(memory.ref(), native_capacity, alignment));
      HYLE_ASSIGN_OR_RETURN(backing.destination, mem::Buffer::create(memory.ref(), native_capacity, alignment));
      id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)backing.destination.native_handle();
      state->queue = [buffer.device newCommandQueue];
      state->event = [buffer.device newSharedEvent];
      if (state->queue == nil || state->event == nil)
        return std::unexpected(Status::ResourceExhausted("Metal weight queue/event allocation failed"));
    }
    return result;
  } catch (...) { return std::unexpected(detail::native_exception()); }
}

Result<WeightBatch> WeightSlots::upload(
    std::size_t slot, std::span<const WeightFragment> selection) noexcept {
  if (closed_ || slot >= 2 || !states_[slot])
    return std::unexpected(Status::FailedPrecondition("invalid or closed Apple weight slot"));
  auto &owner = states_[slot];
  if (owner.use_count() != 1)
    return std::unexpected(Status::FailedPrecondition("Apple weight selection is still retained"));
  auto &state = *owner;
  HYLE_ASSIGN_OR_RETURN(auto previous, detail::poll_weight(state));
  if (previous == CompletionPoll::kPending || state.backing.use_count() != 1)
    return std::unexpected(Status::FailedPrecondition("Apple weight transfer is still in flight"));
  if (selection.size() > state.selected.size())
    return std::unexpected(Status::InvalidArgument("too many selected Apple weights"));
  if (state.generation == UINT64_MAX)
    return std::unexpected(Status::OutOfRange("Apple weight event generation exhausted"));
  const auto capacity = state.capacity;
  std::size_t used = 0;
  // Validate the entire bounded selection before host copy or Metal admission.
  for (std::size_t i = 0; i < selection.size(); ++i) {
    const auto &fragment = selection[i];
    const auto found = std::lower_bound(state.catalog.begin(), state.catalog.end(), fragment.id,
        [](const WeightSpec &spec, std::uint64_t id) { return spec.id < id; });
    if (found == state.catalog.end() || found->id != fragment.id ||
        !fragment.source.valid() || found->description != fragment.source.description() ||
        !fragment.source.storage().resource().properties().has(mem::MemoryProperty::kHostAccessible))
      return std::unexpected(Status::InvalidArgument("Apple weight ID/encoding/host source differs"));
    for (std::size_t j = 0; j < i; ++j)
      if (fragment.id == selection[j].id)
        return std::unexpected(Status::InvalidArgument("duplicate selected Apple weight ID"));
    HYLE_ASSIGN_OR_RETURN(auto bytes, found->description.required_bytes());
    // All native carrier scalar widths divide max_align_t. Formats can require
    // larger alignment, so retain their exact physical constraint as well.
    const auto alignment = std::max(alignof(std::max_align_t), found->description.required_alignment());
    const auto padding = (alignment - used % alignment) % alignment;
    if (padding > capacity - used || bytes > capacity - used - padding)
      return std::unexpected(Status::ResourceExhausted("selected Apple weights exceed capacity"));
    state.selected[i] = {*found, used + padding, bytes};
    used += padding + bytes;
  }
  id<MTLBuffer> source = (__bridge id<MTLBuffer>)state.backing->staging.native_handle();
  id<MTLBuffer> destination = (__bridge id<MTLBuffer>)state.backing->destination.native_handle();
  auto *pointer = static_cast<std::byte *>(source.contents);
  std::size_t end = 0;
  for (std::size_t i = 0; i < selection.size(); ++i) {
    const auto &entry = state.selected[i];
    std::fill(pointer + end, pointer + entry.offset, std::byte{});
    HYLE_RETURN_IF_ERROR(selection[i].source.storage().read({pointer + entry.offset, entry.bytes}));
    end = entry.offset + entry.bytes;
  }
  // macOS Metal blits require four-byte extents even for byte carriers. The
  // cold backings include this padding; tensor views preserve exact encoding.
  const auto transfer_bytes = (used + 3) & ~std::size_t{3};
  std::fill(pointer + used, pointer + transfer_bytes, std::byte{});
  @autoreleasepool {
    id<MTLCommandBuffer> command = [state.queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (command == nil || blit == nil)
      return std::unexpected(Status::ResourceExhausted("Metal weight command allocation failed"));
    if (transfer_bytes != 0)
      [blit copyFromBuffer:source sourceOffset:0 toBuffer:destination destinationOffset:0 size:transfer_bytes];
    [blit endEncoding];
    [command encodeSignalEvent:state.event value:state.generation + 1];
    // Capture backing and progress, never the slot owning this command. An
    // abandoned Completion joins progress's fixed intrusive list until this
    // already-registered callback breaks its pending lifetime attachment.
    __block auto retained = state.backing;
    __block auto progress = state.progress;
    id<MTLSharedEvent> event = state.event;
    const auto generation = state.generation + 1;
    [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
      detail::finish_weight(*progress, detail::metal_status(completed, event, generation));
      retained.reset();
      progress.reset();
    }];
    [state.command release];
    state.command = [command retain];
    ++state.generation;
    state.count = selection.size();
    {
      std::lock_guard lock(state.progress->mutex);
      state.progress->done = false;
      state.progress->terminal = Status::Ok();
    }
    [command commit];
  }
  return WeightBatch(owner);
}

Status WeightSlots::close() noexcept {
  closed_ = true;
  Status status;
  for (const auto &state : states_) {
    if (!state) continue;
    const auto terminal = detail::wait_weight(*state);
    if (status.ok()) status = terminal;
  }
  return status;
}

} // namespace hyle::compute::apple
