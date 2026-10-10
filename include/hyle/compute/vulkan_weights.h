#pragma once

#include "hyle/compute/vulkan_memory.h"

#if defined(HYLE_COMPUTE_VULKAN)
#include <cstddef>
#include <cstdint>
#include <span>

namespace hyle::compute::vulkan {
namespace detail { struct WeightState; }
struct WeightSpec final { std::uint64_t id = 0; TensorDesc description{}; };
struct WeightFragment final { std::uint64_t id = 0; ConstTensorView source{}; };
struct WeightSlotOptions final {
  mem::AllocatorRef allocator = mem::system_allocator();
  ResourceBudget resource_budget{};
};

// Immutable selection and its real native upload prefix. Metadata can be read
// while pending; binding rejects pending/failed bytes. Native Tensor copies,
// views and native_owner exports retain the selection and forbid slot reuse.
class WeightBatch final {
public:
  WeightBatch() noexcept = default;
  WeightBatch(const WeightBatch &) noexcept = default;
  WeightBatch &operator=(const WeightBatch &) noexcept = default;
  WeightBatch(WeightBatch &&) noexcept = default;
  WeightBatch &operator=(WeightBatch &&) noexcept = default;
  HYLE_COMPUTE_ROUTE_VULKAN_API ~WeightBatch();
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<TensorDesc> description(std::uint64_t id) const noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<Tensor> binding(std::uint64_t id) const noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<CompletionPoll> poll() const noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status wait() const noexcept;
private:
  friend class WeightSlots;
  explicit WeightBatch(std::shared_ptr<detail::WeightState> state) noexcept : state_(std::move(state)) {}
  std::shared_ptr<detail::WeightState> state_;
};

// Two cold native staging/device slots, fixed catalog and command capacity.
// upload copies selected host bytes into mapped staging and submits native
// VkBuffer copy/barriers on the original exact-Device queue. A fence proves the
// ordered native prefix, not an independent DMA-engine completion. No source
// conversion, routing search, cache/eviction policy or private thread pool.
// Thread-compatible; resources/allocator outlive late work and retained arrays.
// Context collect/close uses the existing retirement owner for abandonment.
class WeightSlots final {
public:
  WeightSlots() noexcept = default;
  WeightSlots(const WeightSlots &) = delete;
  WeightSlots &operator=(const WeightSlots &) = delete;
  WeightSlots(WeightSlots &&) noexcept = default;
  WeightSlots &operator=(WeightSlots &&) noexcept = default;
  HYLE_COMPUTE_ROUTE_VULKAN_API ~WeightSlots();
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<WeightSlots>
  create(const Context &context, const MemoryResource &staging, const MemoryResource &device,
         std::span<const WeightSpec> catalog, std::size_t capacity_bytes,
         WeightSlotOptions options = {}) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<WeightBatch>
  upload(std::size_t slot, std::span<const WeightFragment> selection) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status close() noexcept;
private:
  std::shared_ptr<detail::WeightState> states_[2];
  bool closed_ = false;
};
} // namespace hyle::compute::vulkan
#endif
