#pragma once

// Private C++17 seam compiled into the locked upstream runtime. None of these
// declarations or ggml types are part of the installed HYLE interface.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <vulkan/vulkan_core.h>

struct ggml_backend_buffer;
struct ggml_backend;
struct ggml_context;

namespace hyle::compute::ggml_vulkan::bridge {
struct Runtime;
struct Options {
  PFN_vkGetInstanceProcAddr get_instance_proc_addr = nullptr;
  std::uint32_t device_index = UINT32_MAX;
  bool allow_software = false;
  bool validation = false;
  void *quota = nullptr;
  bool (*reserve)(void *, std::size_t, void **) noexcept = nullptr;
  void (*release)(void *) noexcept = nullptr;
};
struct Native {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  std::uint32_t queue_family = 0;
  std::uint32_t loader_version = 0;
  bool buffer_device_address = false;
  bool storage_update_after_bind = false;
  PFN_vkGetInstanceProcAddr get_instance_proc_addr = nullptr;
};
enum class Error { None, OutOfRange, Exhausted, Unavailable, Native, Internal };
struct Failure {
  Error error = Error::None;
  VkResult native = VK_SUCCESS;
};

struct Charge {
  void *token = nullptr;
  void (*release)(void *) noexcept = nullptr;
  Charge() = default;
  Charge(const Charge &) = delete;
  Charge &operator=(const Charge &) = delete;
  ~Charge() { if (token) release(token); }
  void reserve(const Options &options, std::size_t bytes) {
    if (options.reserve) {
      release = options.release;
      if (!options.reserve(options.quota, bytes, &token))
        throw std::bad_alloc();
    }
  }
};

Failure create(const Options &, Runtime **, Native *) noexcept;
void destroy(Runtime *) noexcept;
void *enter(Runtime *) noexcept;
void leave(void *) noexcept;
Failure with_queue(Runtime *, VkResult (*)(void *, VkQueue), void *) noexcept;
Failure create_backend(Runtime *, ggml_backend **) noexcept;
Failure allocate_context(Runtime *, ggml_context *, ggml_backend_buffer **) noexcept;
Failure allocate_buffer(Runtime *, std::size_t, ggml_backend_buffer **) noexcept;
Failure borrow_buffer(Runtime *, VkBuffer, std::size_t,
                      std::shared_ptr<const void>, ggml_backend_buffer **) noexcept;
ggml_backend_buffer *borrow(VkBuffer, std::size_t,
                            std::shared_ptr<const void>);
VkBuffer buffer_handle(ggml_backend_buffer *) noexcept;
} // namespace hyle::compute::ggml_vulkan::bridge
