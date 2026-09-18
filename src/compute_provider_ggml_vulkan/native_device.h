#pragma once

#include <functional>
#include <memory>

#include "metadata_arena.h"
#include "native_bridge.h"
#include "hyle/mem/quota_allocator.h"

namespace hyle::compute::ggml_vulkan::detail {

struct NativeOptions final {
  const char *loader_path = nullptr;
  std::uint32_t device_index = UINT32_MAX;
  bool allow_software = false;
  bool validation = false;
  mem::QuotaAllocator *quota = nullptr; // If supplied, outlives the device.
};

// One isolated loader/instance/device owner. Copies retain it. Upstream host
// calls must be inside a Scope, on the calling thread; native GPU execution
// does not use the thread-local pointer. A scope borrows its owning handle.
// Preparation on one owner is thread-compatible; independent owners may be
// prepared concurrently. with_queue serializes native queue host access.
class NativeDevice final {
public:
  class Scope final {
  public:
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
    HYLE_GGML_VULKAN_INTERNAL_API ~Scope();
  private:
    friend class NativeDevice;
    explicit Scope(void *previous) noexcept : previous_(previous) {}
    void *previous_;
  };

  NativeDevice() noexcept = default;
  HYLE_GGML_VULKAN_INTERNAL_API static Result<NativeDevice>
  create(NativeOptions options = {}) noexcept;
  explicit operator bool() const noexcept { return state_ != nullptr; }
  HYLE_GGML_VULKAN_INTERNAL_API const bridge::Native &native() const noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API Scope activate() const noexcept;
  // Private region builders own returned ggml handles and free them under a
  // Scope before releasing this device. Y3 adds the asynchronous slot owner.
  HYLE_GGML_VULKAN_INTERNAL_API Result<ggml_backend *>
  create_backend() const noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API Result<ggml_backend_buffer *>
  allocate_context(ggml_context *) const noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API Result<ggml_backend_buffer *>
  allocate_buffer(std::size_t) const noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API Result<ggml_backend_buffer *>
  borrow(VkBuffer, std::size_t, std::shared_ptr<const void>) const noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API VkBuffer
  buffer_handle(ggml_backend_buffer *) const noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API Status
  with_queue(const std::function<Status(VkQueue)> &) const noexcept;

private:
  struct State;
  std::shared_ptr<State> state_;
};

} // namespace hyle::compute::ggml_vulkan::detail
