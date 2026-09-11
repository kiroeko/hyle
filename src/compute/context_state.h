#pragma once

#include <array>
#include <utility>

#include "hyle/compute/device.h"
#include "hyle/util/ref_counted.h"

namespace hyle::compute::detail {

struct ContextState final : util::RefCounted<ContextState> {
  ContextState(Device device_value, mem::AllocatorRef allocator_value) noexcept
      : device(device_value), allocator(allocator_value) {}

  void destroy() noexcept;

  struct ProviderSlot final {
    Provider provider{};
    void *state = nullptr;
  };

  Device device{};
  mem::AllocatorRef allocator{};
  std::array<ProviderSlot, kMaxContextProviders> providers{};
  std::size_t provider_count = 0;
  bool open = true;
};

struct ContextAccess final {
  static ContextState* state(const Context& context) noexcept {
    return context.state_;
  }

  static ContextState* state(const Queue& queue) noexcept {
    return queue.state_;
  }

  static bool queue_open(const Queue& queue) noexcept { return queue.open_; }
};

inline void retain(ContextState* state) noexcept {
  if (state != nullptr) state->incref();
}

inline void release(ContextState* state) noexcept {
  if (state != nullptr) state->decref();
}

}  // namespace hyle::compute::detail
