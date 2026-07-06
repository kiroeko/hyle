#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/compute/types.h"
#include "hyle/error/status.h"

// INTERNAL header (src/compute only -- not shipped). The manual vtable that each
// compute backend fills in. HYLE does NOT use virtual inheritance anywhere; a
// backend is a plain function-pointer struct (same style as error_handler_t and
// mem::Allocator), and EACH backend is an independent composition unit
// (coding-conventions §10 "组合优于包含"): host_backend.cc defines host_vtable(),
// cuda_backend.cu defines cuda_vtable() -- there is NO god-object switch(tag) on a
// hot path. The public Device/Stream/Event just hold a `const ComputeBackend*` and
// forward through it.
//
// Backend selection is a COLD path (compute-design §CUDA vs shader): pick the
// vtable once at enumerate()/default_device() time. On a hot path (compute-03's
// event poll) the call is one indirect through a resolved pointer -- the same cost
// model as the io proactor reap, acceptable off the innermost kernel loop.
//
// Native handles are opaque void* (a cudaStream_t / cudaEvent_t on the CUDA side,
// a small host struct on the fallback). The public pimpl (StreamImpl/EventImpl
// below) pairs a handle with the vtable that understands it, so a Stream created by
// the CUDA backend always reaps through the CUDA backend.

namespace hyle::compute {

// One backend's operation table. All pointers are non-null for a valid backend
// (host + cuda both fill every slot). Signatures use the public value types
// (Device/DeviceCaps/EventStatus/Status) but never expose a backend-native type.
struct ComputeBackend {
  // Identity (for diagnostics / selecting).
  const char* name;  // "cuda" / "host"

  // ---- device enumeration ----
  // How many devices THIS backend exposes (0 for CUDA when no GPU / driver).
  std::size_t (*device_count)() noexcept;
  // Describe device `ordinal` (< device_count): its type + a stable name pointer.
  // `out_name` receives a process-stable string (backend-owned); never null on true.
  DeviceType (*device_type)(int ordinal) noexcept;
  const char* (*device_name)(int ordinal) noexcept;
  DeviceCaps (*device_caps)(int ordinal) noexcept;
  void (*device_memory)(int ordinal, std::uint64_t* free,
                        std::uint64_t* total) noexcept;

  // ---- stream lifecycle ----
  // Create a backend-native stream on `ordinal`; returns the opaque handle in
  // *out_native (non-null on success) or fills *out_status with the failure.
  Status (*stream_create)(int ordinal, void** out_native) noexcept;
  void (*stream_destroy)(void* native) noexcept;
  Status (*stream_sync)(void* native) noexcept;

  // ---- event lifecycle ----
  Status (*event_create)(int ordinal, void** out_native) noexcept;
  void (*event_destroy)(void* native) noexcept;
  // Record `event_native` on `stream_native` (cudaEventRecord). Caller has already
  // checked both handles are valid + same device (fatal-assert in the public layer).
  void (*event_record)(void* stream_native, void* event_native) noexcept;
  EventStatus (*event_query)(void* event_native) noexcept;  // cudaEventQuery
  void (*event_sync)(void* event_native) noexcept;          // cudaEventSynchronize

  // ---- device memory + async copy (minimal completion-proof ops) ----
  Status (*device_alloc)(int ordinal, std::size_t bytes, void** out_ptr) noexcept;
  void (*device_free)(int ordinal, void* ptr) noexcept;
  // Enqueue an async copy on `stream_native`. Direction inferred by the backend.
  Status (*memcpy_async)(void* stream_native, void* dst, const void* src,
                         std::size_t bytes) noexcept;

  // ---- buffer (compute-02) --------------------------------------------------
  // A Buffer owns one block of memory of a given BufferKind on `ordinal`. The
  // public Buffer stores {vt, ptr, size, ordinal, kind, is_host}; these fn-ptrs do
  // the backend work. kDevice = cudaMalloc (VRAM, not host-deref); kHostPinned =
  // cudaMallocHost (page-locked host RAM, from a mem::MemoryArena pool on the CUDA
  // backend; the host backend just uses plain RAM for both kinds).
  //
  // buffer_type trait queries (per kind, cheap constants): alignment / max single-
  // allocation size / whether the memory is host-dereferenceable. Mirror ggml
  // buffer_type_i {get_alignment, get_max_size, is_host}.
  std::size_t (*buffer_alignment)(BufferKind kind) noexcept;
  std::size_t (*buffer_max_size)(int ordinal, BufferKind kind) noexcept;
  bool (*buffer_is_host)(BufferKind kind) noexcept;

  // Allocate `bytes` of `kind` memory on `ordinal`; returns the base pointer in
  // *out_ptr (non-null on success) or the failure Status. free returns it.
  Status (*buffer_alloc)(int ordinal, BufferKind kind, std::size_t bytes,
                         void** out_ptr) noexcept;
  void (*buffer_free)(int ordinal, BufferKind kind, void* ptr) noexcept;

  // SYNCHRONOUS host<->device copy into/out of a buffer at a byte `offset` (the
  // compute-02 core: partial writes, KV-cache-style). The backend infers direction
  // (dst is device => H2D; src is device => D2H). "Synchronous" = the CUDA backend
  // issues cudaMemcpyAsync on an internal stream then synchronizes, so the data is
  // in place on return (same contract as ggml buffer set_tensor/get_tensor). The
  // async, stream-bound, co_await-able variants are compute-03 (event-gated).
  Status (*buffer_set)(void* dst_base, std::size_t offset, const void* src,
                       std::size_t size) noexcept;  // H2D partial write
  Status (*buffer_get)(const void* src_base, std::size_t offset, void* dst,
                       std::size_t size) noexcept;  // D2H partial read
  // Copy `size` bytes from src_base+src_off to dst_base+dst_off. Buffers may be on
  // the same or different memory kinds (D2D / D2H / H2D); backend picks the cudaMemcpy
  // kind. Both buffers belong to THIS backend (cross-backend copy is a later concern).
  Status (*buffer_copy)(void* dst_base, std::size_t dst_off, const void* src_base,
                        std::size_t src_off, std::size_t size) noexcept;
  // Set `size` bytes at base+offset to byte `value` (cudaMemset / memset). KV-cache
  // zero-init uses this.
  Status (*buffer_clear)(void* base, std::size_t offset, std::uint8_t value,
                         std::size_t size) noexcept;
};

// Each backend exposes its singleton vtable through one of these. host is ALWAYS
// present; cuda only when HYLE_COMPUTE_CUDA is compiled in (else the symbol isn't
// defined and compute.cc's #if omits the reference).
const ComputeBackend* host_vtable() noexcept;
#if defined(HYLE_COMPUTE_CUDA)
const ComputeBackend* cuda_vtable() noexcept;
#endif

}  // namespace hyle::compute
