#pragma once

#include <functional>
#include <memory>

#include "metadata_arena.h"
#include "native_bridge.h"
#include "hyle/mem/quota_allocator.h"

namespace hyle::compute::ggml_vulkan::detail {
class NativeWork;

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
  HYLE_GGML_VULKAN_INTERNAL_API Result<NativeWork> create_work() const noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API Result<ggml_backend_buffer *>
  allocate_buffer(std::size_t) const noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API Result<ggml_backend_buffer *>
  borrow(VkBuffer, std::size_t, std::shared_ptr<const void>) const noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API VkBuffer
  buffer_handle(ggml_backend_buffer *) const noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API Status
  with_queue(const std::function<Status(VkQueue)> &) const noexcept;

private:
  friend class NativeGraph;
  HYLE_GGML_VULKAN_INTERNAL_API Status allocate_graph(
      ggml_cgraph *, bool owns_imports, bridge::Storage **,
      std::size_t *bytes) const noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API void free_graph(bridge::Storage *) const noexcept;
  struct State;
  std::shared_ptr<State> state_;
};

struct WorkState final {
  Status status;
  bool pending = false;
  bool device_lost = false;
};

// Private region execution slot. Host calls on one slot are thread-compatible;
// shared-device encoding uses the route admission lock. The graph is validated
// cold and all its tensors are resident on this device. A submit lease owns its
// graph metadata, buffers and bound inputs until native retirement is proved.
class NativeWork final {
public:
  NativeWork() noexcept = default;
  NativeWork(const NativeWork &) = delete;
  NativeWork &operator=(const NativeWork &) = delete;
  HYLE_GGML_VULKAN_INTERNAL_API NativeWork(NativeWork &&) noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API NativeWork &operator=(NativeWork &&) noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API ~NativeWork();
  HYLE_GGML_VULKAN_INTERNAL_API WorkState
  submit(ggml_cgraph *, std::shared_ptr<const void> resource_lease) noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API WorkState poll() noexcept;
  HYLE_GGML_VULKAN_INTERNAL_API WorkState wait() noexcept;

private:
  friend class NativeDevice;
  WorkState observe(bool wait) noexcept;
  void reset() noexcept;
  NativeDevice device_;
  bridge::Work *work_ = nullptr;
  std::shared_ptr<const void> lease_;
  bool pending_ = false;
};

} // namespace hyle::compute::ggml_vulkan::detail
