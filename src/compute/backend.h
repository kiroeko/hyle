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
};

// Each backend exposes its singleton vtable through one of these. host is ALWAYS
// present; cuda only when HYLE_COMPUTE_CUDA is compiled in (else the symbol isn't
// defined and compute.cc's #if omits the reference).
const ComputeBackend* host_vtable() noexcept;
#if defined(HYLE_COMPUTE_CUDA)
const ComputeBackend* cuda_vtable() noexcept;
#endif

}  // namespace hyle::compute
