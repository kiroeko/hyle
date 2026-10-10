#include "hyle/compute/vulkan_weights.h"
#include "vulkan_transfer_internal.h"

#include <algorithm>
#include <mutex>
#include <new>
#include <vector>

namespace hyle::compute::vulkan {
namespace detail {
struct SelectedWeight final { WeightSpec spec; std::size_t offset = 0, bytes = 0; };
struct WeightState final {
  explicit WeightState(mem::AllocatorRef owner)
      : allocator(owner), catalog(mem::StdAllocator<WeightSpec>(owner)),
        selected(mem::StdAllocator<SelectedWeight>(owner)),
        parts(mem::StdAllocator<UploadPart>(owner)) {}
  mem::AllocatorRef allocator;
  std::shared_ptr<Runtime> runtime;
  std::vector<WeightSpec, mem::StdAllocator<WeightSpec>> catalog;
  std::vector<SelectedWeight, mem::StdAllocator<SelectedWeight>> selected;
  std::vector<UploadPart, mem::StdAllocator<UploadPart>> parts;
  Tensor destination;
  TransferSlot transfer;
  Completion completion;
  std::mutex completion_mutex;
  std::size_t capacity = 0, count = 0;
};
} // namespace detail

WeightBatch::~WeightBatch() = default;
WeightSlots::~WeightSlots() = default;
Result<TensorDesc> WeightBatch::description(std::uint64_t id) const noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("empty Vulkan weight batch"));
  for (std::size_t i = 0; i < state_->count; ++i)
    if (state_->selected[i].spec.id == id) return state_->selected[i].spec.description;
  return std::unexpected(Status::NotFound("Vulkan weight ID is not selected"));
}
Result<CompletionPoll> WeightBatch::poll() const noexcept {
  if (!state_) return std::unexpected(Status::FailedPrecondition("empty Vulkan weight batch"));
  std::lock_guard lock(state_->completion_mutex);
  return state_->completion.poll();
}
Status WeightBatch::wait() const noexcept {
  if (!state_) return Status::FailedPrecondition("empty Vulkan weight batch");
  std::lock_guard lock(state_->completion_mutex);
  return state_->completion.wait();
}
Result<Tensor> WeightBatch::binding(std::uint64_t id) const noexcept {
  HYLE_ASSIGN_OR_RETURN(auto ready, poll());
  if (ready != CompletionPoll::kComplete)
    return std::unexpected(Status::FailedPrecondition("Vulkan weight bytes are not ready"));
  for (std::size_t i = 0; i < state_->count; ++i) {
    const auto &entry = state_->selected[i];
    if (entry.spec.id != id) continue;
    try {
      auto buffer = std::allocate_shared<detail::Buffer>(mem::StdAllocator<detail::Buffer>(state_->allocator));
      const auto &root = detail::Access::buffer(state_->destination);
      detail::bind_view_buffer(*buffer, root, entry.offset);
      buffer->device = root->device;
      buffer->external_owner = state_; // The state never owns this alias.
      return detail::Access::tensor(entry.spec.description, std::move(buffer));
    } catch (const std::bad_alloc &) {
      return std::unexpected(Status::ResourceExhausted("Vulkan weight alias metadata"));
    }
  }
  return std::unexpected(Status::NotFound("Vulkan weight ID is not selected"));
}

Result<WeightSlots> WeightSlots::create(
    const Context &context, const MemoryResource &staging, const MemoryResource &device,
    std::span<const WeightSpec> catalog, std::size_t capacity, WeightSlotOptions options) noexcept {
  if (!context.valid() || !staging.valid() || !device.valid() || catalog.empty() || capacity == 0 ||
      capacity > static_cast<std::size_t>(INT64_MAX) ||
      detail::MemoryAccess::runtime(staging) != detail::Access::runtime(context) ||
      detail::MemoryAccess::runtime(device) != detail::Access::runtime(context) ||
      !staging.ref().properties().has(mem::MemoryProperty::kHostAccessible) ||
      device.ref().properties().has(mem::MemoryProperty::kHostAccessible))
    return std::unexpected(Status::InvalidArgument("invalid Vulkan weight resources/Device/family/capacity"));
  for (std::size_t i = 0; i < catalog.size(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto bytes, catalog[i].description.required_bytes());
    if (bytes > capacity) return std::unexpected(Status::OutOfRange("Vulkan catalog weight exceeds capacity"));
    for (std::size_t j = 0; j < i; ++j)
      if (catalog[i].id == catalog[j].id)
        return std::unexpected(Status::InvalidArgument("duplicate Vulkan catalog ID"));
  }
  auto allocator = options.allocator;
  if (options.resource_budget) {
    HYLE_ASSIGN_OR_RETURN(allocator, options.resource_budget.metadata_allocator());
  }
  if (!allocator.valid()) return std::unexpected(Status::InvalidArgument("invalid Vulkan weight metadata allocator"));
  HYLE_ASSIGN_OR_RETURN(auto raw, TensorDesc::dense(DataType::kUInt8,
      std::array<std::int64_t, 1>{static_cast<std::int64_t>(capacity)}));
  try {
    WeightSlots slots;
    for (auto &state : slots.states_) {
      state = std::allocate_shared<detail::WeightState>(mem::StdAllocator<detail::WeightState>(allocator), allocator);
      state->runtime = detail::Access::runtime(context); state->capacity = capacity;
      state->catalog.assign(catalog.begin(), catalog.end());
      std::sort(state->catalog.begin(), state->catalog.end(), [](const auto &a, const auto &b) { return a.id < b.id; });
      state->selected.resize(catalog.size()); state->parts.resize(catalog.size());
      HYLE_ASSIGN_OR_RETURN(state->destination, device.allocate(raw,
          {.resource_budget = options.resource_budget, .category = ResourceBudgetCategory::kMigration}));
      HYLE_ASSIGN_OR_RETURN(state->transfer, TransferSlot::create(context, staging, capacity,
          {.resource_budget = options.resource_budget, .category = ResourceBudgetCategory::kMigration}));
    }
    return slots;
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("Vulkan weight pool metadata"));
  }
}
Result<WeightBatch> WeightSlots::upload(std::size_t slot, std::span<const WeightFragment> selection) noexcept {
  if (closed_ || slot >= 2 || !states_[slot])
    return std::unexpected(Status::FailedPrecondition("invalid or closed Vulkan weight slot"));
  auto &owner = states_[slot]; auto &state = *owner;
  if (owner.use_count() != 1)
    return std::unexpected(Status::FailedPrecondition("Vulkan weight batch/Tensor/owner is still retained"));
  std::lock_guard lock(state.completion_mutex);
  if (state.completion.valid()) {
    HYLE_ASSIGN_OR_RETURN(auto ready, state.completion.poll());
    if (ready == CompletionPoll::kPending)
      return std::unexpected(Status::FailedPrecondition("Vulkan weight upload is pending"));
  }
  if (selection.size() > state.selected.size())
    return std::unexpected(Status::InvalidArgument("too many Vulkan weight fragments"));
  std::size_t used = 0;
  for (std::size_t i = 0; i < selection.size(); ++i) {
    const auto &fragment = selection[i];
    const auto found = std::lower_bound(state.catalog.begin(), state.catalog.end(), fragment.id,
        [](const auto &spec, std::uint64_t id) { return spec.id < id; });
    if (found == state.catalog.end() || found->id != fragment.id || !fragment.source.valid() ||
        fragment.source.description() != found->description ||
        !fragment.source.storage().resource().properties().has(mem::MemoryProperty::kHostAccessible))
      return std::unexpected(Status::InvalidArgument("Vulkan weight ID/encoding/host source differs"));
    for (std::size_t j = 0; j < i; ++j)
      if (selection[j].id == fragment.id) return std::unexpected(Status::InvalidArgument("duplicate selected Vulkan weight ID"));
    HYLE_ASSIGN_OR_RETURN(auto bytes, found->description.required_bytes());
    const auto alignment = std::max<std::size_t>({4, found->description.required_alignment(),
        static_cast<std::size_t>(state.runtime->device->properties.limits.minStorageBufferOffsetAlignment)});
    const auto padding = (alignment - used % alignment) % alignment;
    if (padding > state.capacity - used || bytes > state.capacity - used - padding)
      return std::unexpected(Status::ResourceExhausted("selected Vulkan weights exceed capacity"));
    state.selected[i] = {*found, used + padding, bytes};
    state.parts[i] = {fragment.source, used + padding};
    used += padding + bytes;
  }
  HYLE_ASSIGN_OR_RETURN(auto completion, detail::TransferAccess::upload(state.transfer, state.destination,
      std::span(state.parts.data(), selection.size()), used));
  state.completion = std::move(completion); state.count = selection.size();
  return WeightBatch(owner);
}
Status WeightSlots::close() noexcept {
  closed_ = true; Status status;
  for (const auto &state : states_) if (state && state->completion.valid()) {
    std::lock_guard lock(state->completion_mutex);
    const auto terminal = state->completion.wait();
    if (status.ok()) status = terminal;
  }
  return status;
}
} // namespace hyle::compute::vulkan
