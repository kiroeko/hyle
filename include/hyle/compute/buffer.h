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
// Mirrors ggml's buffer_type/buffer model (trait薄包 + ownership自造). Supersedes
// compute-01's Stream::device_alloc/memcpy_async (those were the spine-proof minimum).
//
// WHAT A Buffer IS (and is NOT): a Buffer is "a base pointer + size" you can copy
// into/out of at an offset. It is NOT a tensor -- no shape, no dtype, no sub-views.
// Offset slicing is the CALLER's job (kernel launch just needs device_ptr(); the
// caller computes base+offset itself -- exactly as ggml folds base+offset into
// tensor->data at alloc time, so the buffer isn't touched at launch). This keeps
// the buffer an L0 primitive; tensors are L1 (the inference engine's job).
//
// THE OFFSET WRITE IS THE POINT (verified against llama.cpp): the KV-cache is one
// persistent large buffer written repeatedly at per-token row offsets
// (llama-kv-cache.cpp:1343). set(offset,size)/get(offset,size) are that primitive.
//
// L0/L1 BOUNDARY: "stream a buffer bigger than RAM into VRAM in tiles + batch the
// submits" is an L1 STRATEGY, not this milestone. compute-02 only provides the
// PRIMITIVES it needs (offset writes + pinned staging + async copy + max_size);
// ggml/llama.cpp likewise build tiled streaming by looping set_tensor over a pinned
// ring in the model loader, NOT inside the buffer layer.

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
  // Is this memory host-dereferenceable (can you read *device_ptr() on the CPU)?
  // kHostPinned => true; kDevice on a GPU => false. Decides copy dispatch.
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
//   host_buffer_type  : pinned host staging (kHostPinned; cudaMallocHost, pooled).
// Both are always returned (a GPU-less backend's host_buffer_type is plain RAM).
HYLE_COMPUTE_API BufferType device_buffer_type(const Device& device) noexcept;
HYLE_COMPUTE_API BufferType host_buffer_type(const Device& device) noexcept;

// One owned block of device (or pinned-host) memory. Move-only (single owner;
// destructor frees). NO pimpl / NO heap for the handle itself -- the state is
// {vt, ptr, size, ordinal, kind, is_host} inline, so device_ptr()/set()/get() are
// one vtable indirect with no extra dereference (same perf discipline as
// Stream/Event). The base pointer is opaque void* -- for kDevice on a GPU it is a
// device pointer you must NOT dereference on the host (use set/get); for kHostPinned
// it is real host RAM.
class Buffer {
 public:
  Buffer() noexcept = default;
  HYLE_DISABLE_COPY(Buffer);
  HYLE_COMPUTE_API Buffer(Buffer&& o) noexcept;
  HYLE_COMPUTE_API Buffer& operator=(Buffer&& o) noexcept;
  HYLE_COMPUTE_API ~Buffer();

  // Allocate `bytes` from `type`. External-boundary failure returns Result
  // (Unavailable/ResourceExhausted). `bytes` must be <= type.max_size() (a larger
  // logical buffer is the caller's split into multiple Buffers).
  HYLE_COMPUTE_API static Result<Buffer> create(const BufferType& type,
                                                std::size_t bytes) noexcept;

  bool valid() const noexcept { return vt_ != nullptr; }
  // The base pointer -- what a kernel launch (compute-04) needs. The caller adds its
  // own offset. For kDevice on a GPU this is NOT host-dereferenceable.
  void* device_ptr() const noexcept { return ptr_; }
  std::size_t size() const noexcept { return size_; }
  bool is_host() const noexcept { return is_host_; }

  // ---- synchronous host<->device copy at a byte offset (compute-02 core) -------
  // These return only after the data is in place (the CUDA backend issues an async
  // memcpy on an internal stream then synchronizes -- same "sync semantics" as ggml
  // buffer set_tensor/get_tensor). CONTRACT: offset+size <= size() (caller bug ->
  // fatal assert). `src`/`dst` are host buffers the caller owns.
  HYLE_COMPUTE_API Status set(std::size_t offset, const void* src,
                              std::size_t size) noexcept;  // H2D partial write
  HYLE_COMPUTE_API Status get(std::size_t offset, void* dst,
                              std::size_t size) const noexcept;  // D2H partial read

  // Copy `size` bytes from `src` (at src_offset) into this buffer (at dst_offset).
  // Handles D2D / D2H / H2D (backend picks the cudaMemcpy kind). Both buffers must
  // belong to the same backend. CONTRACT: offsets+size within both buffers' sizes.
  HYLE_COMPUTE_API Status copy_from(const Buffer& src, std::size_t dst_offset,
                                    std::size_t src_offset,
                                    std::size_t size) noexcept;

  // Set `size` bytes at `offset` to byte `value` (cudaMemset / memset). KV-cache
  // zero-init. CONTRACT: offset+size <= size().
  HYLE_COMPUTE_API Status clear(std::size_t offset, std::uint8_t value,
                                std::size_t size) noexcept;

  // ---- async, stream-bound copy (compute-03; SIGNATURES RESERVED) -------------
  // The async variants issue the copy on `stream` WITHOUT synchronizing; completion
  // is observed via an Event (co_await-able once compute-03 wires GPU completion to
  // the fiber scheduler). Declared here so the interface is fixed ONCE (compute-02
  // will implement them synchronously = enqueue + sync, and compute-03 makes them
  // truly async without changing this surface). Commented out until compute-03 to
  // avoid shipping a half-wired API.
  //   Status set_async(std::size_t offset, const void* src, std::size_t size, Stream& s);
  //   Status get_async(std::size_t offset, void* dst, std::size_t size, Stream& s) const;

 private:
  const ComputeBackend* vt_ = nullptr;  // backend that owns ptr_ (null = invalid)
  void* ptr_ = nullptr;                 // base pointer (device or pinned-host)
  std::size_t size_ = 0;
  int ordinal_ = 0;                     // backend-local device (for free)
  BufferKind kind_ = BufferKind::kDevice;
  bool is_host_ = false;
};

}  // namespace hyle::compute
