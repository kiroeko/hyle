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
struct ggml_cgraph;

namespace hyle::compute::ggml_vulkan::bridge {
struct Runtime;
struct Work;
struct Storage;
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
  std::uint64_t storage_buffer_alignment = 1;
  bool synchronization2 = false;
  bool timeline_semaphore = false;
  bool scalar_block_layout = false;
  bool shader_float16 = false;
  bool storage_buffer16 = false;
  bool shader_int64 = false;
  bool portability_subset = false;
  std::uint32_t queue_create_flags = 0;
  bool shader_int16 = false;
};
enum class Error {
  None,
  InvalidArgument,
  InvalidState,
  OutOfRange,
  Exhausted,
  Unavailable,
  Unsupported,
  Native,
  Internal
};
struct Failure {
  Error error = Error::None;
  VkResult native = VK_SUCCESS;
};
// Failure does not imply quiescence. A pending report retains every resource
// even when its status already records an encoding/submission/observation
// error.
struct WorkReport {
  Failure failure;
  bool pending = false;
  bool device_lost = false;
};

struct Charge {
  void *token = nullptr;
  void (*release)(void *) noexcept = nullptr;
  Charge() = default;
  Charge(const Charge &) = delete;
  Charge &operator=(const Charge &) = delete;
  ~Charge() {
    if (token)
      release(token);
  }
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
void note_device_lost(Runtime *) noexcept;
Failure with_queue(Runtime *, VkResult (*)(void *, VkQueue), void *) noexcept;
Failure create_backend(Runtime *, ggml_backend **) noexcept;
Failure allocate_buffer(Runtime *, std::size_t,
                        ggml_backend_buffer **) noexcept;
Failure borrow_buffer(Runtime *, VkBuffer, std::size_t,
                      std::shared_ptr<const void>, bool address_capable,
                      ggml_backend_buffer **) noexcept;
ggml_backend_buffer *borrow(VkBuffer, std::size_t, std::shared_ptr<const void>,
                            bool address_capable);
VkBuffer buffer_handle(ggml_backend_buffer *) noexcept;
Failure binding_buffer(Runtime *, std::size_t, std::shared_ptr<const void>,
                       ggml_backend_buffer **) noexcept;
void bind_buffer(ggml_backend_buffer *, VkBuffer, std::size_t,
                 std::shared_ptr<const void>) noexcept;
Failure create_work(Runtime *, Work **) noexcept;
// Last-resort blocking retirement. Ordinary route abandonment retains Work in
// its existing pending owner instead of reaching this destructor while busy.
void destroy_work(Work *) noexcept;
WorkReport submit_work(Work *, ggml_cgraph *) noexcept;
WorkReport observe_work(Work *, bool wait) noexcept;
VkFence work_fence(const Work *) noexcept;
Failure create_storage(Runtime *, ggml_cgraph *, bool owns_imports, Storage **,
                       std::size_t *bytes) noexcept;
void destroy_storage(Storage *) noexcept;
} // namespace hyle::compute::ggml_vulkan::bridge
