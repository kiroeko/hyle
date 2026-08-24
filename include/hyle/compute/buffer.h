#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/compute/device.h"
#include "hyle/compute/types.h"
#include "hyle/error/result.h"
#include "hyle/macros.h"
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_COMPUTE_API
#elif defined(HYLE_COMPUTE_BUILD)
#  define HYLE_COMPUTE_API HYLE_EXPORT
#else
#  define HYLE_COMPUTE_API HYLE_IMPORT
#endif

// compute-02: device Buffer -- ownership of ONE block of device (or pinned-host)
// memory + host<->device copy with byte offsets (compute-design "模块形状" buffer.h).
// Mirrors ggml's buffer_type/buffer model (trait薄包 + ownership自造). This is the
// ONLY device-memory + host<->device copy API; compute-01's spine-proof Stream::
// device_alloc/memcpy_async minimum has been removed in favor of it.
//
// WHAT A Buffer IS (and is NOT): a Buffer owns one backend allocation plus its size.
// `native_handle()` is opaque and only the owning backend may interpret it (CUDA: device
// address; Vulkan: VkMem* record). `host_data()` is the separate CPU mapping for a
// host-visible allocation. It is NOT a tensor -- no shape, dtype, or sub-views.
//
// THE OFFSET WRITE IS THE POINT (verified against llama.cpp): the KV-cache is one
// persistent large buffer written repeatedly at per-token row offsets
// (llama-kv-cache.cpp:1343). set(offset,size)/get(offset,size) are that primitive.
//
// L0/L1 BOUNDARY: Buffer provides owned allocation, explicit host mapping/coherence,
// offset copy, and max-size traits. Async staging requires Stream + Event + backpressure;
// it is not represented by a raw allocator or a commented-out signature.

namespace hyle::compute {

// Internal backend vtable (src/compute/backend.h). Forward-declared so BufferType/
// Buffer can hold a `const ComputeBackend*` without the public header seeing any
// backend-native type (same pattern as stream.h).
struct ComputeBackend;

// A category of memory on a Device (mirrors ggml buffer_type_i): the factory for
// Buffers of one BufferKind + its traits. Cheap value type {vt, ordinal, kind};
// get it from Device::buffer_type()/host_buffer_type().
class BufferType {
 public:
  BufferType() noexcept = default;

  bool valid() const noexcept { return vt_ != nullptr; }
  BufferKind kind() const noexcept { return kind_; }

  // Allocation alignment for this memory kind (CUDA device = 128; host = max_align).
  HYLE_COMPUTE_API std::size_t alignment() const noexcept;
  // Max bytes in a SINGLE allocation (SIZE_MAX when unbounded). Drives the caller's
  // decision to split a >max_size logical buffer into multiple Buffers (ggml
  // multi-buffer). CUDA device is effectively unbounded; some backends cap it.
  HYLE_COMPUTE_API std::size_t max_size() const noexcept;
  // Does this kind provide a CPU mapping through Buffer::host_data()?
  // kHostPinned => true; kDevice on a GPU => false.
  HYLE_COMPUTE_API bool is_host() const noexcept;

 private:
  friend class Buffer;
  friend HYLE_COMPUTE_API BufferType device_buffer_type(const Device&) noexcept;
  friend HYLE_COMPUTE_API BufferType host_buffer_type(const Device&) noexcept;
  BufferType(const ComputeBackend* vt, int ordinal, BufferKind kind) noexcept
      : vt_(vt), ordinal_(ordinal), kind_(kind) {}

  const ComputeBackend* vt_ = nullptr;
  int ordinal_ = 0;
  BufferKind kind_ = BufferKind::kDevice;
};

// Get a Device's memory factories (free functions, not Device members, so device.h
// need not depend on buffer.h -- the dispatch layer resolves the Device's backend).
//   device_buffer_type: device-local memory (VRAM on a GPU; kDevice).
//   host_buffer_type  : host-visible backend memory (CUDA page-locked; Vulkan mapped).
// Both are always returned (a GPU-less backend uses plain RAM for both kinds).
HYLE_COMPUTE_API BufferType device_buffer_type(const Device& device) noexcept;
HYLE_COMPUTE_API BufferType host_buffer_type(const Device& device) noexcept;

// One owned backend allocation. Move-only; destructor frees through the same backend.
// NO pimpl / heap for the handle itself: native handle and optional CPU mapping are stored
// separately inline. Never dereference native_handle(); use host_data() only when is_host().
// IN-FLIGHT CONTRACT: an allocation referenced by launch/launch_batch/replay must remain owned
// and unchanged until that Stream/Event completes. Destroying or move-assigning over its owning
// Buffer while GPU work is pending frees storage still referenced by the backend.
class Buffer {
 public:
  Buffer() noexcept = default;
  HYLE_DISABLE_COPY(Buffer);
  HYLE_COMPUTE_API Buffer(Buffer&& o) noexcept;
  HYLE_COMPUTE_API Buffer& operator=(Buffer&& o) noexcept;
  HYLE_COMPUTE_API ~Buffer();

  // Allocate `bytes` from `type`. An invalid/default BufferType returns
  // InvalidArgument; external-boundary failure returns Result
  // (Unavailable/ResourceExhausted). `bytes` must be <= type.max_size() (a larger
  // logical buffer is the caller's split into multiple Buffers).
  HYLE_COMPUTE_API static Result<Buffer> create(const BufferType& type,
                                                std::size_t bytes) noexcept;

  bool valid() const noexcept { return vt_ != nullptr; }

  // Does this allocation belong to `device`? Invalid/default buffers or devices return
  // false. The typed kernel-launch shell uses this before exposing native_handle() to a
  // backend: CUDA device pointers and Vulkan VkMem* are not interchangeable even when
  // both devices represent the same physical GPU.
  HYLE_COMPUTE_API bool belongs_to(const Device& device) const noexcept;

  // Opaque backend-native allocation handle. KernelArgs passes this value back to the
  // same backend; callers must never dereference or perform pointer arithmetic on it.
  void* native_handle() const noexcept { return native_; }

  std::size_t size() const noexcept { return size_; }
  bool is_host() const noexcept { return host_ != nullptr; }

  // Direct CPU mapping for a host-visible buffer. Caller contract: is_host()==true.
  // After CPU writes call flush_host() before device use; before CPU reads data written
  // by the device call invalidate_host() after completion. Both are no-ops on coherent
  // CUDA/host memory but are required for non-coherent Vulkan memory.
  void* host_data() noexcept {
    HYLE_FATAL_ASSERT(vt_ != nullptr && host_ != nullptr);
    return host_;
  }
  const void* host_data() const noexcept {
    HYLE_FATAL_ASSERT(vt_ != nullptr && host_ != nullptr);
    return host_;
  }
  HYLE_COMPUTE_API Status flush_host(std::size_t offset, std::size_t size) noexcept;
  HYLE_COMPUTE_API Status invalidate_host(std::size_t offset,
                                          std::size_t size) noexcept;

  // ---- synchronous host<->device copy at a byte offset (compute-02 core) -------
  // These return only after the data is in place (the CUDA backend issues an async
  // memcpy on an internal stream then synchronizes -- same "sync semantics" as ggml
  // buffer set_tensor/get_tensor). CONTRACT: offset+size <= size() (caller bug ->
  // fatal assert). `src`/`dst` are host buffers the caller owns. A zero-size operation
  // is a backend-independent no-op and may use null src/dst.
  //
  // These operations do NOT establish ordering with an unrelated async launch Stream.
  // Before set/get/copy/clear or direct host access, the affected Buffer must have no
  // incomplete GPU use (await its Event/Stream first). flush/invalidate maintain mapped
  // cache visibility only; they are not completion primitives.
  HYLE_COMPUTE_API Status set(std::size_t offset, const void* src,
                              std::size_t size) noexcept;  // H2D partial write
  HYLE_COMPUTE_API Status get(std::size_t offset, void* dst,
                              std::size_t size) const noexcept;  // D2H partial read

  // Copy `size` bytes from `src` (at src_offset) into this buffer (at dst_offset).
  // Handles D2D / D2H / H2D and performs the required mapped-cache flush/invalidate
  // around the synchronous copy. This does not wait for a producer kernel: both buffers
  // must have no incomplete GPU use. Both buffers must share a backend; ranges must fit.
  HYLE_COMPUTE_API Status copy_from(const Buffer& src, std::size_t dst_offset,
                                    std::size_t src_offset,
                                    std::size_t size) noexcept;

  // Set `size` bytes at `offset` to byte `value` (cudaMemset / memset). KV-cache
  // zero-init. CONTRACT: offset+size <= size().
  HYLE_COMPUTE_API Status clear(std::size_t offset, std::uint8_t value,
                                std::size_t size) noexcept;

 private:
  const ComputeBackend* vt_ = nullptr;  // backend that owns native_ (null = invalid)
  void* native_ = nullptr;              // opaque backend allocation handle
  void* host_ = nullptr;                // CPU mapping; null for non-host-visible kind
  std::size_t size_ = 0;
  int ordinal_ = 0;                     // backend-local device (for free)
  BufferKind kind_ = BufferKind::kDevice;
};

}  // namespace hyle::compute
