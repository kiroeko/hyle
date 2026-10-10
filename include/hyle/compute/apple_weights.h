#pragma once

#include "hyle/compute/apple_memory.h"

#if defined(HYLE_COMPUTE_APPLE)
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>

#include "hyle/compute/completion.h"
#include "hyle/compute/tensor.h"

namespace hyle::compute::apple {
namespace detail { struct WeightState; }

struct WeightSpec final {
  std::uint64_t id = 0;
  TensorDesc description{};
};

// Host-accessible bytes with exactly the registered physical encoding. The
// source is borrowed only during upload's synchronous staging copy.
struct WeightFragment final {
  std::uint64_t id = 0;
  ConstTensorView source{};
};

struct WeightSlotOptions final {
  mem::AllocatorRef allocator = mem::system_allocator();
  ResourceBudget resource_budget{};
};

// Owns the selected native backing and real Metal transfer. binding() is a
// cold native view construction and rejects pending/failed transfers. The
// returned array retains this selection, so all its aliases prevent reuse.
// MLX descriptors are opaque; binding under a supplied budget requires its
// explicit allow_unattributed policy. Known buffers/metadata remain charged.
// No generic Queue or implicit CPU/GPU dispatch enters this native route.
class WeightBatch final {
public:
  WeightBatch() noexcept = default;
  WeightBatch(const WeightBatch &) = delete;
  WeightBatch &operator=(const WeightBatch &) = delete;
  WeightBatch(WeightBatch &&) noexcept = default;
  WeightBatch &operator=(WeightBatch &&) noexcept = default;
  HYLE_COMPUTE_ROUTE_APPLE_API ~WeightBatch();
  bool valid() const noexcept { return state_ != nullptr; }
  HYLE_COMPUTE_ROUTE_APPLE_API Result<mlx::core::array>
  binding(std::uint64_t id) const noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Result<CompletionPoll> poll() const noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status wait() const noexcept;
  // Cold completion owner; abandoning it keeps attached lifetimes until the
  // native command terminates. Cancellation cannot preempt an accepted blit.
  HYLE_COMPUTE_ROUTE_APPLE_API Result<Completion> completion() const noexcept;
private:
  friend class WeightSlots;
  explicit WeightBatch(std::shared_ptr<detail::WeightState> state) noexcept
      : state_(std::move(state)) {}
  std::shared_ptr<detail::WeightState> state_;
};

// Two fixed staging/destination allocations and fixed catalog metadata.
// Upload submits a real Metal blit on an owned native transfer queue, separate
// from Context's compute stream. Shared physical memory still requires native
// ordering. CPU-resident weights need not use these buffers until selected.
// Handles are thread-compatible. Resource/allocator outlive every batch,
// returned array and late command. Destruction does not wait; close drains
// transfers only. The engine owns expert selection, caching and eviction.
class WeightSlots final {
public:
  WeightSlots() noexcept = default;
  WeightSlots(const WeightSlots &) = delete;
  WeightSlots &operator=(const WeightSlots &) = delete;
  WeightSlots(WeightSlots &&) noexcept = default;
  WeightSlots &operator=(WeightSlots &&) noexcept = default;
  HYLE_COMPUTE_ROUTE_APPLE_API ~WeightSlots();
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<WeightSlots>
  create(const Context &context, const MemoryResource &memory,
         std::span<const WeightSpec> catalog, std::size_t capacity_bytes,
         WeightSlotOptions options = {}) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Result<WeightBatch>
  upload(std::size_t slot, std::span<const WeightFragment> selection) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status close() noexcept;
private:
  std::shared_ptr<detail::WeightState> states_[2];
  bool closed_ = false;
};

} // namespace hyle::compute::apple
#endif
