#pragma once
#include "hyle/core/environment.h"

#include <array>
#include <atomic>
#include <mutex>
#include <utility>
#include <vector>

#include "native_device.h"

// Original C headers use C99 aggregate initializers accepted by MSVC. Keep
// this vendor-specific warning boundary private to the provider implementation.
#if HYLE_COMPILER_MSVC || HYLE_COMPILER_CLANG_CL
#pragma warning(push, 0)
#pragma warning(disable : 4576)
#endif
#include "iree/async/frontier_tracker.h"
#include "iree/async/operations/semaphore.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/vulkan/api.h"
#include "iree/hal/drivers/vulkan/buffer.h"
#include "iree/modules/hal/module.h"
#include "iree/runtime/api.h"
#include "iree/vm/bytecode/module.h"
#if HYLE_COMPILER_MSVC || HYLE_COMPILER_CLANG_CL
#pragma warning(pop)
#endif

namespace hyle::compute::iree_vulkan {
namespace detail {
template <class T, void (*Release)(T *)> struct ReleaseObject {
  void operator()(T *value) const noexcept {
    if (value)
      Release(value);
  }
};
template <class T, void (*Release)(T *)>
using Owner = std::unique_ptr<T, ReleaseObject<T, Release>>;

Status consume_status(iree_status_t status) noexcept;

// Only explicit quotas pay for the allocation table and its lock. A token
// covers an actual VkDeviceMemory allocation (including original staging and
// pool slabs), from before allocation until after its native free.
struct Accounting final {
  struct Allocation {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    mem::QuotaAllocator::Reservation reservation;
  };
  mem::QuotaAllocator &quota;
  std::mutex mutex;
  std::vector<Allocation> allocations;
  explicit Accounting(mem::QuotaAllocator &value) noexcept : quota(value) {}
  ~Accounting() { HYLE_FATAL_ASSERT(allocations.empty()); }
  static iree_status_t allocate(void *, uint64_t, VkDeviceMemory *,
                                iree_hal_vulkan_memory_allocate_action_t,
                                void *) noexcept;
  static void free(void *, VkDeviceMemory, iree_hal_vulkan_memory_free_action_t,
                   void *) noexcept;
};
} // namespace detail

struct NativeDevice::State final {
  ggml_vulkan::detail::NativeDevice native;
  std::unique_ptr<detail::Accounting> accounting;
  detail::Owner<iree_async_proactor_pool_t, iree_async_proactor_pool_release>
      pool;
  detail::Owner<iree_hal_vulkan_syms_t, iree_hal_vulkan_syms_release> syms;
  detail::Owner<iree_hal_device_t, iree_hal_device_release> device;
  detail::Owner<iree_hal_semaphore_t, iree_hal_semaphore_release> retired;
  std::atomic<bool> stopping = false;
  std::mutex stop_mutex;
  Status stop_status;
  Status request_stop() noexcept;
  detail::Owner<iree_runtime_instance_t, iree_runtime_instance_release>
      instance;
  detail::Owner<iree_async_frontier_tracker_t,
                iree_async_frontier_tracker_release>
      tracker;
  detail::Owner<iree_hal_device_group_t, iree_hal_device_group_release> group;
  detail::Owner<iree_vm_module_t, iree_vm_module_release> hal;
  std::array<detail::Owner<iree_vm_module_t, iree_vm_module_release>, 7>
      programs;
};
} // namespace hyle::compute::iree_vulkan
