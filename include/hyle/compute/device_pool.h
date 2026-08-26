#pragma once

#include <cstddef>

#include "hyle/compute/device.h"
#include "hyle/error/result.h"
#include "hyle/core/compiler.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_COMPUTE_API
#elif defined(HYLE_COMPUTE_BUILD)
#  define HYLE_COMPUTE_API HYLE_EXPORT
#else
#  define HYLE_COMPUTE_API HYLE_IMPORT
#endif

// compute-02c: device (VRAM) sub-allocator -- "预分配大块换性能" for the device side.
// Ported wholesale from ggml's two CUDA pools (user: "ggml 很 SOTA 抄全一点"):
//
//   VMM pool  (ggml_cuda_pool_vmm, ggml-cuda.cu:478): reserve a huge (32 GB)
//     CONTIGUOUS virtual address range ONCE (cuMemAddressReserve), then map physical
//     pages on demand (cuMemCreate/cuMemMap) as the pool grows -- allocation is an
//     O(1) bump of pool_used; physical memory grows on demand and is never shrunk
//     (cached for reuse). CONSTRAINT: strictly LIFO (reverse-order) free. This is the
//     textbook "reserve big address space, commit physical on demand". Requires the
//     device to support CUDA VMM (cuDeviceGetAttribute) + the driver API.
//
//   leg pool  (ggml_cuda_pool_leg, ggml-cuda.cu:361): the no-VMM fallback -- a 256-
//     slot cache of freed cudaMalloc blocks; alloc does BEST-FIT over the cache
//     (smallest block >= request), reusing it if found, else a real cudaMalloc with
//     LOOK-AHEAD over-allocation (1.05x rounded up to 256B) so a slightly larger later
//     request still hits; OOM triggers sync + flush cache + retry. free returns the
//     block to a slot (real cudaFree only when the cache is full). Supports ARBITRARY
//     (out-of-order) free.
//
// WHY: a kernel's transient/intermediate device buffers (matmul scratch, dequant
// buffers, ...) are allocated + freed constantly; a raw cudaMalloc per intermediate is
// slow (it synchronizes). The pool reuses device memory so those allocs are near-free.
// This is the DEVICE-side analogue of compute-02b's pinned host pool -- but device VRAM
// does NOT go through mem/ (mem/ is host RAM only; VRAM ownership is compute's).
//
// CUDA picks VMM vs leg per device at pool creation (VMM if supported); Vulkan uses
// best-fit reuse of complete VkBuffer+VkDeviceMemory blocks. The pool handle is opaque;
// DevicePool holds {vt, handle} inline (no pimpl), same as Buffer/Stream.

namespace hyle::compute {

struct ComputeBackend;  // internal vtable (backend.h); forward-declared, not exposed

// A device-memory sub-allocator on one Device. Create once (per device / per stream
// context), then alloc()/free() transient device buffers cheaply. Move-only (owns the
// backend pool: the reserved VMM range or the leg block cache). Thread-safety: NOT
// thread-safe -- one pool per owner (per stream / per worker in the shard-per-core
// model), mirroring ggml's per-stream pool. The VMM backend additionally requires
// LIFO free order (alloc a, b, c -> free c, b, a); the leg backend allows any order.
// Use PoolAlloc (below) for scoped RAII borrows, which naturally gives LIFO.
class DevicePool {
 public:
  DevicePool() noexcept = default;
  DevicePool(const DevicePool&) = delete;
  DevicePool& operator=(const DevicePool&) = delete;
  HYLE_COMPUTE_API DevicePool(DevicePool&& o) noexcept;
  HYLE_COMPUTE_API DevicePool& operator=(DevicePool&& o) noexcept;
  HYLE_COMPUTE_API ~DevicePool();

  // Create a pool on `device`. CUDA picks VMM when supported, otherwise its legacy
  // cached-block backend; Vulkan uses cached native blocks. External-boundary failure
  // returns Result; an invalid Device returns InvalidArgument.
  HYLE_COMPUTE_API static Result<DevicePool> create(const Device& device) noexcept;

  bool valid() const noexcept { return vt_ != nullptr; }
  // True if this pool uses the VMM backend (=> LIFO free required). Informational /
  // for tests; callers should just use PoolAlloc which is LIFO by scope.
  HYLE_COMPUTE_API bool is_vmm() const noexcept;

  // Allocate `bytes` of device memory from the pool. Returns a device pointer (NOT
  // host-dereferenceable) or an external-boundary failure. The returned block may be
  // larger than requested (look-ahead / alignment); the pointer is what a kernel
  // launch uses. Pair every alloc with a free (VMM: in reverse order).
  HYLE_COMPUTE_API Result<void*> alloc(std::size_t bytes) noexcept;
  // Return a block from alloc() to the pool. `bytes` MUST be the SAME value passed to
  // alloc() (the pool tracks sizes by it -- ggml contract). VMM: must be the most
  // recent still-live alloc (LIFO). The memory is NOT returned to the driver (it stays
  // cached in the pool for reuse); the pool frees it to the driver on destruction.
  HYLE_COMPUTE_API void free(void* ptr, std::size_t bytes) noexcept;

 private:
  const ComputeBackend* vt_ = nullptr;  // backend that owns handle_ (null = invalid)
  void* handle_ = nullptr;              // opaque backend pool (VMM or leg), pimpl-ish
  int ordinal_ = 0;
};

// RAII scoped borrow from a DevicePool (mirrors ggml_cuda_pool_alloc<T>). Allocates
// on construction, frees on destruction -- so nested PoolAllocs free in reverse
// (LIFO), satisfying the VMM backend's constraint automatically. Move-only.
// `get()` is the device pointer to hand a kernel; `elements` is sized in T.
template <class T>
class PoolAlloc {
 public:
  PoolAlloc() noexcept = default;
  PoolAlloc(DevicePool& pool, std::size_t elements) noexcept
      : pool_(&pool), bytes_(elements * sizeof(T)) {
    auto r = pool.alloc(bytes_);
    ptr_ = r.has_value() ? static_cast<T*>(*r) : nullptr;
    if (ptr_ == nullptr) bytes_ = 0;  // alloc failed -> empty (valid()==false)
  }
  PoolAlloc(const PoolAlloc&) = delete;
  PoolAlloc& operator=(const PoolAlloc&) = delete;
  PoolAlloc(PoolAlloc&& o) noexcept
      : pool_(o.pool_), ptr_(o.ptr_), bytes_(o.bytes_) {
    o.pool_ = nullptr;
    o.ptr_ = nullptr;
    o.bytes_ = 0;
  }
  PoolAlloc& operator=(PoolAlloc&& o) noexcept {
    if (this != &o) {
      release();
      pool_ = o.pool_;
      ptr_ = o.ptr_;
      bytes_ = o.bytes_;
      o.pool_ = nullptr;
      o.ptr_ = nullptr;
      o.bytes_ = 0;
    }
    return *this;
  }
  ~PoolAlloc() { release(); }

  bool valid() const noexcept { return ptr_ != nullptr; }
  T* get() const noexcept { return ptr_; }        // device pointer for a kernel arg
  std::size_t size_bytes() const noexcept { return bytes_; }

 private:
  void release() noexcept {
    if (ptr_ != nullptr) pool_->free(ptr_, bytes_);
    ptr_ = nullptr;
    bytes_ = 0;
  }
  DevicePool* pool_ = nullptr;
  T* ptr_ = nullptr;
  std::size_t bytes_ = 0;
};

}  // namespace hyle::compute
