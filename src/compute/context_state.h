#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "backend_internal.h"
#include "hyle/compute/device.h"
#include "hyle/util/ref_counted.h"

namespace hyle::compute::detail {

class AdmissionGate final {
public:
  bool try_enter() noexcept {
    std::uint32_t value = state_.load(std::memory_order_acquire);
    while ((value & kClosed) == 0) {
      if ((value & kCountMask) == kCountMask)
        return false;
      if (state_.compare_exchange_weak(value, value + 1,
                                       std::memory_order_acquire,
                                       std::memory_order_relaxed)) {
        return true;
      }
    }
    return false;
  }

  void leave() noexcept {
    const std::uint32_t previous =
        state_.fetch_sub(1, std::memory_order_release);
    if ((previous & kCountMask) == 1)
      state_.notify_all();
  }

  void close() noexcept {
    state_.fetch_or(kClosed, std::memory_order_acq_rel);
    state_.notify_all();
  }

  bool is_open() const noexcept {
    return (state_.load(std::memory_order_acquire) & kClosed) == 0;
  }

  void wait_idle() const noexcept {
    std::uint32_t value = state_.load(std::memory_order_acquire);
    while ((value & kCountMask) != 0) {
      state_.wait(value, std::memory_order_acquire);
      value = state_.load(std::memory_order_acquire);
    }
  }

private:
  static constexpr std::uint32_t kClosed = std::uint32_t{1} << 31;
  static constexpr std::uint32_t kCountMask = kClosed - 1;
  mutable std::atomic<std::uint32_t> state_{0};
};

struct ContextState final : util::RefCounted<ContextState> {
  struct ProviderSlot final {
    Provider provider{};
    void *state = nullptr;
    ProviderInstanceId instance{};
  };

  ContextState(Device device_value, mem::AllocatorRef allocator_value,
               std::size_t allocation_size_value,
               std::size_t provider_offset_value,
               std::size_t provider_capacity_value,
               const BackendVTable *backend_vtable_value) noexcept
      : device(device_value), allocator(allocator_value),
        allocation_size(allocation_size_value),
        provider_offset(provider_offset_value),
        provider_capacity(provider_capacity_value),
        backend_vtable(backend_vtable_value) {}

  ProviderSlot *providers() noexcept {
    auto *bytes = reinterpret_cast<std::byte *>(this);
    return reinterpret_cast<ProviderSlot *>(bytes + provider_offset);
  }
  const ProviderSlot *providers() const noexcept {
    auto *bytes = reinterpret_cast<const std::byte *>(this);
    return reinterpret_cast<const ProviderSlot *>(bytes + provider_offset);
  }

  void destroy() noexcept;

  Device device{};
  mem::AllocatorRef allocator{};
  std::size_t allocation_size = 0;
  std::size_t provider_offset = 0;
  std::size_t provider_capacity = 0;
  std::size_t provider_count = 0;
  const BackendVTable *backend_vtable = nullptr;
  void *backend_state = nullptr;
  bool backend_initialized = false;
  AdmissionGate admission{};
};

struct QueueState final : util::RefCounted<QueueState> {
  QueueState(ContextState *context_value, void *backend_state_value) noexcept
      : context(context_value), backend_state(backend_state_value) {
    context->incref();
  }

  void destroy() noexcept;

  ContextState *context = nullptr;
  void *backend_state = nullptr;
  AdmissionGate admission{};
  // Submission calls on one ordered Queue are a single-producer domain.  The
  // flag is held only while recording/enqueuing a command sequence, never for
  // asynchronous device execution.
  std::atomic_flag submitting = ATOMIC_FLAG_INIT;
};

struct ContextAccess final {
  using BackendQueueCreateFn = Status (*)(
      void *backend_context, mem::AllocatorRef allocator,
      void *call_context, void **backend_queue) noexcept;

  static ContextState *state(const Context &context) noexcept {
    return context.state_;
  }
  static QueueState *state(const Queue &queue) noexcept { return queue.state_; }
  static void *backend_context(const Context &context) noexcept {
    return context.state_ != nullptr ? context.state_->backend_state : nullptr;
  }
  static void *backend_queue(const Queue &queue) noexcept {
    return queue.state_ != nullptr ? queue.state_->backend_state : nullptr;
  }

  // Internal typed-backend seam for an imported/adopted queue whose creation
  // needs vendor arguments not present in the backend-neutral QueueOptions.
  // It preserves the exact allocation, admission and lifetime path used by
  // Context::create_queue; the callback must not publish its backend object on
  // failure.
  HYLE_COMPUTE_API static Result<Queue>
  create_queue(const Context &context, BackendQueueCreateFn create,
               void *call_context) noexcept;
};

// Cross-factor ownership bridges. Keep final destruction inside the Compute
// binary: an implementation DLL may retain Context/Queue state, but must not
// instantiate a call to the factor-private destroy() symbol.
HYLE_COMPUTE_API void retain(ContextState *state) noexcept;
HYLE_COMPUTE_API void release(ContextState *state) noexcept;
HYLE_COMPUTE_API void retain(QueueState *state) noexcept;
HYLE_COMPUTE_API void release(QueueState *state) noexcept;

} // namespace hyle::compute::detail
