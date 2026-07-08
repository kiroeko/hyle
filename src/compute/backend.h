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

  // ---- device (VRAM) sub-allocator pool (compute-02c) -----------------------
  // A stateful per-device pool (VMM reserved range or leg block cache -- ported from
  // ggml). pool_create returns an opaque handle in *out_handle; the other fns take it.
  // pool_alloc returns the device pointer in *out_ptr and the ACTUAL (possibly larger)
  // block size in *out_actual. pool_free takes the size ALLOC returned/was asked for
  // (ggml contract: free's size == alloc's requested size). pool_is_vmm reports which
  // backend was chosen (VMM => LIFO free required). These are backend-native (CUDA
  // driver+runtime API); the host fallback provides a trivial malloc-cache pool.
  Status (*pool_create)(int ordinal, void** out_handle) noexcept;
  void (*pool_destroy)(void* handle) noexcept;
  bool (*pool_is_vmm)(void* handle) noexcept;
  Status (*pool_alloc)(void* handle, std::size_t size, void** out_ptr,
                       std::size_t* out_actual) noexcept;
  void (*pool_free)(void* handle, void* ptr, std::size_t size) noexcept;

  // ---- kernel launch (compute-04) -------------------------------------------
  // A kernel is {module, func} handles the backend fills. compile: NVRTC source ->
  // PTX -> load (CUDA); load_ptx: given PTX -> load; load_host: register a host fn
  // (host backend). All return the two handles in *out_module / *out_func (a CUmodule
  // + CUfunction on CUDA; nullptr module + the HostFn in func on host). destroy frees
  // the module. launch submits to the stream (cuLaunchKernel / host fn call).
  //
  // `entry` is the kernel name (extern "C") or a C++ name expression NVRTC lowers.
  // Compile failure returns a Status with the NVRTC log. host_fn is only used by
  // load_host (the HostFn pointer, stashed as func). launch's `config`/`args` are the
  // public LaunchConfig + the cuLaunchKernel void** args array.
  Status (*kernel_compile)(int ordinal, const char* source, const char* entry,
                           void** out_module, void** out_func) noexcept;
  Status (*kernel_load_ptx)(int ordinal, const char* ptx, const char* entry,
                            void** out_module, void** out_func) noexcept;
  Status (*kernel_load_host)(int ordinal, void* host_fn, void** out_module,
                             void** out_func) noexcept;
  void (*kernel_destroy)(void* module, void* func) noexcept;
  // Launch: `config` is a const LaunchConfig* (void* to keep backend.h free of the
  // public header include order; the .cc casts it), `stream_native` the backend
  // stream, `args` a const KernelArgs* (unified ABI: ordered buffers + scalar blob --
  // CUDA lowers to void** args, Vulkan to descriptor set + push constants). Returns a
  // Status (launch/config error).
  Status (*kernel_launch)(void* stream_native, void* module, void* func,
                          const void* config, const KernelArgs* args) noexcept;
  // BATCHED launch (compute-06 finding: batching = the measured perf lever -- N ops in
  // ONE submit beats N separate submits,省 N-1 次 submit+fence, compute-05 finding 4).
  // `modules`/`funcs`/`configs`/`args_arr` are `count` parallel arrays (item i = the
  // i-th kernel + its config + its args). Vulkan records all N dispatches (with barriers
  // between) into ONE command buffer + ONE submit; CUDA/host just loop kernel_launch (a
  // CUDA stream already orders sequential submits with no extra per-submit cost, so their
  // "batch" is the loop). Returns first error or OK.
  Status (*kernel_launch_batch)(void* stream_native, const void* const* modules,
                                const void* const* funcs, const LaunchConfig* configs,
                                const KernelArgs* args_arr, unsigned count) noexcept;

  // ---- GPU timing (compute-06b, checklist #25) ------------------------------
  // An OPTIONAL timing-enabled event, the device-time analog of the completion event
  // above. It measures GPU-side elapsed time between two records -- the CUDA
  // cudaEventElapsedTime model, mirrored on Vulkan via timestamp queries. This is a
  // first-class profiling capability (not debug-only): adaptive kernel selection,
  // perf-regression monitoring, and compute benchmarks (the "prove SOTA" prerequisite)
  // all consume it. Kept SEPARATE from the plain event so the completion hot path stays
  // on the cheap no-timing event (cudaEventDisableTiming).
  //
  // event_create_timing: like event_create but the event records a device timestamp when
  // record()'d (CUDA: a timing cudaEvent; Vulkan: a VkQueryPool timestamp slot + a fence).
  // event_elapsed_ns: GPU nanoseconds between two timing events that have BOTH completed
  // (start recorded before stop on the same stream). Returns the elapsed ns in *out_ns.
  // On the host backend timing events are inline (elapsed = 0). NULL slot (backend without
  // timing support) -> the public layer reports Unimplemented.
  // A timing event has its OWN full lifecycle (create/destroy/record/query/sync) SEPARATE
  // from the plain completion event. Why separate rather than reusing the plain slots: a
  // backend's timing event may have a DIFFERENT native layout than its plain event (Vulkan:
  // plain = bare VkFence, timing = a {VkFence, VkQueryPool} struct), so the plain
  // destroy/query/sync would mis-cast a timing native. The public Event routes ALL of these
  // by its `timing_` flag, so a timing event always stays on the timing slots. On CUDA the
  // timing event is just a cudaEvent_t (timing-enabled), so its timing slots forward to the
  // same cuda impls as the plain ones -- the split costs nothing there.
  Status (*event_create_timing)(int ordinal, void** out_native) noexcept;
  void (*event_destroy_timing)(void* native) noexcept;
  void (*event_record_timing)(void* stream_native, void* event_native) noexcept;
  EventStatus (*event_query_timing)(void* native) noexcept;
  void (*event_sync_timing)(void* native) noexcept;
  // GPU nanoseconds between two completed timing events. Returns the elapsed ns in *out_ns.
  Status (*event_elapsed_ns)(void* start_native, void* stop_native,
                             std::uint64_t* out_ns) noexcept;
};

// Each backend exposes its singleton vtable through one of these. host is ALWAYS
// present; cuda only when HYLE_COMPUTE_CUDA is compiled in (else the symbol isn't
// defined and compute.cc's #if omits the reference); vulkan only when
// HYLE_COMPUTE_VULKAN (compute-06: volk + glslang fetched, loader present at runtime).
const ComputeBackend* host_vtable() noexcept;
#if defined(HYLE_COMPUTE_CUDA)
const ComputeBackend* cuda_vtable() noexcept;
#endif
#if defined(HYLE_COMPUTE_VULKAN)
// May return nullptr if no Vulkan loader/device is present at runtime (volkInitialize
// fails) -- compute.cc's add_backend() tolerates null (that backend contributes 0 devices).
const ComputeBackend* vulkan_vtable() noexcept;
#endif

}  // namespace hyle::compute
