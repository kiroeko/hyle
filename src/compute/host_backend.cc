#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

#include "backend.h"
#include "hyle/compute/types.h"
#include "hyle/error/status.h"

// The HOST fallback backend (coding-conventions §10: an independent composition
// unit, one TU, filling the manual vtable). It is ALWAYS present -- it lets the
// whole compute completion contract compile and run on any platform with no GPU
// (mac, CI, a machine with CUDA not built) and is the DETERMINISTIC backend
// compute-03 will use to test fiber wakeup without a real device.
//
// Semantics: a host "stream" runs work INLINE at enqueue time (there is no async
// engine), so an Event recorded on it is ALWAYS immediately kReady. This honors the
// same record/query/sync API the CUDA backend does -- the caller's code path is
// identical -- it just never observes NotReady. No per-object heap allocation: the
// stream/event native handles are non-null SENTINELS (there is no per-instance state
// to keep -- inline execution means an event carries no pending work).

namespace hyle::compute {
namespace {

// Non-null sentinels: the dispatch layer only checks native != null for validity
// and passes it back to us verbatim; the host backend keeps no per-handle state,
// so a distinct address per kind is all we need (and it's free -- no allocation).
char g_host_stream_tag;
char g_host_event_tag;
void* const kHostStream = &g_host_stream_tag;
void* const kHostEvent = &g_host_event_tag;

std::size_t host_device_count() noexcept { return 1; }

DeviceType host_device_type(int) noexcept { return DeviceType::kCpu; }

const char* host_device_name(int) noexcept { return "cpu"; }

DeviceCaps host_device_caps(int) noexcept {
  // No real asynchrony, but host memory is directly usable (host_buffer) and the
  // event contract is honored synchronously (events=true).
  return DeviceCaps{/*async=*/false, /*host_buffer=*/true, /*events=*/true};
}

void host_device_memory(int, std::uint64_t* free, std::uint64_t* total) noexcept {
  // Host RAM is not meaningfully "device memory"; report 0 rather than probing the
  // OS (informational only, and compute-01 doesn't need it). A later milestone can
  // fill this from platform sysinfo if a consumer wants it.
  if (free != nullptr) *free = 0;
  if (total != nullptr) *total = 0;
}

Status host_stream_create(int, void** out_native) noexcept {
  *out_native = kHostStream;
  return Status();  // OK
}

void host_stream_destroy(void*) noexcept {}  // sentinel: nothing to free

Status host_stream_sync(void*) noexcept { return Status(); }  // inline: already done

Status host_event_create(int, void** out_native) noexcept {
  *out_native = kHostEvent;
  return Status();
}

void host_event_destroy(void*) noexcept {}

void host_event_record(void*, void*) noexcept {}  // inline: nothing pending to mark

EventStatus host_event_query(void*) noexcept {
  return EventStatus::kReady;  // inline execution => always complete
}

void host_event_sync(void*) noexcept {}  // already complete

// Timing events on the host backend: work runs INLINE (no GPU, no async), so a "timing"
// event carries no device time -- create returns the same sentinel, record is a no-op,
// and elapsed is always 0. Exists so the timing API compiles + runs on any platform (same
// role as the host completion event).
Status host_event_create_timing(int, void** out_native) noexcept {
  *out_native = kHostEvent;
  return Status();
}
void host_event_record_timing(void*, void*) noexcept {}  // inline: nothing to timestamp
Status host_event_elapsed_ns(void*, void*, std::uint64_t* out_ns) noexcept {
  *out_ns = 0;  // inline execution -> zero GPU time
  return Status();
}

// ---- buffer (compute-02) ----------------------------------------------------
// On the host backend every memory kind is just plain RAM (there is no separate
// device, and no page-locking benefit without a GPU DMA engine). is_host()=true for
// both -- native handle and CPU mapping are the same plain-RAM pointer.
// set/get/copy/clear are memcpy/memset; flush/invalidate are no-ops.

bool host_buffer_supported(int, BackendMemoryKind) noexcept { return true; }

std::size_t host_buffer_alignment(BackendMemoryKind) noexcept {
  return alignof(std::max_align_t);
}

std::size_t host_buffer_max_size(int, BackendMemoryKind) noexcept {
  return SIZE_MAX;  // bounded only by the system allocator
}

bool host_buffer_is_host(BackendMemoryKind) noexcept {
  return true;  // plain RAM either way -- directly dereferenceable
}

bool host_buffer_is_coherent(BackendMemoryKind) noexcept { return true; }

Status host_buffer_alloc(int, BackendMemoryKind, std::size_t bytes,
                         void** out_native, void** out_host) noexcept {
  void* p = std::malloc(bytes == 0 ? 1 : bytes);
  if (p == nullptr) return Status::ResourceExhausted("host: buffer_alloc");
  *out_native = p;
  *out_host = p;
  return Status();
}

void host_buffer_free(int, BackendMemoryKind, void* ptr) noexcept {
  std::free(ptr);
}
void* host_buffer_host_data(BackendMemoryKind, void* ptr) noexcept {
  return ptr;
}
Status host_buffer_flush(void*, std::size_t, std::size_t) noexcept { return Status(); }
Status host_buffer_invalidate(void*, std::size_t, std::size_t) noexcept { return Status(); }

Status host_buffer_set(void* dst_base, std::size_t offset, const void* src,
                       std::size_t size) noexcept {
  std::memcpy(static_cast<char*>(dst_base) + offset, src, size);
  return Status();
}

Status host_buffer_get(const void* src_base, std::size_t offset, void* dst,
                       std::size_t size) noexcept {
  std::memcpy(dst, static_cast<const char*>(src_base) + offset, size);
  return Status();
}

Status host_buffer_copy(void* dst_base, std::size_t dst_off, const void* src_base,
                        std::size_t src_off, std::size_t size) noexcept {
  std::memcpy(static_cast<char*>(dst_base) + dst_off,
              static_cast<const char*>(src_base) + src_off, size);
  return Status();
}

Status host_buffer_clear(void* base, std::size_t offset, std::uint8_t value,
                         std::size_t size) noexcept {
  std::memset(static_cast<char*>(base) + offset, value, size);
  return Status();
}

// ---- device pool (compute-02c): host fallback = leg-style malloc cache -------
// The host backend has no VRAM; its "device pool" is a small best-fit cache of
// malloc'd blocks (the leg strategy, minus CUDA), so the reuse semantics
// (alloc-after-free returns a cached block, avoiding a fresh malloc) are testable on
// any platform. Arbitrary free order (not VMM/LIFO). Single-owner (not thread-safe),
// same contract as the CUDA pools.
struct HostPool {
  static constexpr int kMaxBuffers = 256;
  struct Block { void* ptr = nullptr; std::size_t size = 0; };
  Block cache[kMaxBuffers] = {};

  void* alloc(std::size_t size, std::size_t* actual) noexcept {
    // best-fit over cached freed blocks (smallest >= size); exact match short-circuits.
    std::size_t best_diff = SIZE_MAX;
    int ibest = -1;
    for (int i = 0; i < kMaxBuffers; ++i) {
      Block& b = cache[i];
      if (b.ptr != nullptr && b.size >= size) {
        std::size_t diff = b.size - size;
        if (diff < best_diff) {
          best_diff = diff;
          ibest = i;
          if (diff == 0) break;
        }
      }
    }
    if (ibest >= 0) {
      Block& b = cache[ibest];
      void* p = b.ptr;
      *actual = b.size;
      b.ptr = nullptr;
      b.size = 0;
      return p;
    }
    std::size_t n = size == 0 ? 1 : size;
    void* p = std::malloc(n);
    *actual = n;
    return p;  // nullptr on OOM -> pool_alloc surfaces ResourceExhausted
  }

  void free(void* ptr, std::size_t size) noexcept {
    for (int i = 0; i < kMaxBuffers; ++i) {
      if (cache[i].ptr == nullptr) {
        cache[i].ptr = ptr;
        cache[i].size = size;
        return;
      }
    }
    std::free(ptr);  // cache full -> really free
  }

  ~HostPool() {
    for (auto& b : cache) if (b.ptr != nullptr) std::free(b.ptr);
  }
};

Status host_pool_create(int, void** out_handle) noexcept {
  auto* p = new (std::nothrow) HostPool();
  if (p == nullptr) return Status::ResourceExhausted("host: pool_create");
  *out_handle = p;
  return Status();
}

void host_pool_destroy(void* handle) noexcept { delete static_cast<HostPool*>(handle); }

bool host_pool_is_vmm(void*) noexcept { return false; }  // host = leg-style, not VMM

Status host_pool_alloc(void* handle, std::size_t size, void** out_ptr,
                       std::size_t* out_actual) noexcept {
  void* p = static_cast<HostPool*>(handle)->alloc(size, out_actual);
  if (p == nullptr) return Status::ResourceExhausted("host: pool_alloc");
  *out_ptr = p;
  return Status();
}

void host_pool_free(void* handle, void* ptr, std::size_t size) noexcept {
  if (ptr != nullptr) static_cast<HostPool*>(handle)->free(ptr, size);
}

// ---- kernel (compute-04): host fallback = registered host fn-ptr ------------
// The host backend can't compile CUDA source or load PTX; a "kernel" here is a host
// function the caller registered via load_host. compile/load_ptx return Unimplemented
// (a GPU-less test uses load_host to exercise the launch abstraction). launch calls
// the fn synchronously with the config + KernelArgs (the fn reads args.buffers /
// args.scalars itself).

// The host kernel signature (matches Kernel::HostFn in kernel.h without including it:
// LaunchConfig + KernelArgs live in types.h). Stored as `func`; module is unused (nullptr).
using HostKernelFn = void (*)(const LaunchConfig&, const KernelArgs&) noexcept;

Status host_kernel_compile(int, const char*, const char*, void**, void**) noexcept {
  return Status::Unimplemented("host backend: no NVRTC (use load_host)");
}

Status host_kernel_load_ptx(int, const char*, unsigned, const char*, const SpecConstant*,
                            unsigned, const void*, unsigned, void**, void**) noexcept {
  return Status::Unimplemented("host backend: no PTX loader (use load_host)");
}

Status host_kernel_load_host(int, void* host_fn, void** out_module,
                             void** out_func) noexcept {
  *out_module = nullptr;   // host has no module
  *out_func = host_fn;     // stash the HostFn pointer as func
  return Status();
}

void host_kernel_destroy(void*, void*) noexcept {}  // nothing owned (fn-ptr)

Status host_kernel_launch(void* /*stream_native*/, void* /*module*/, void* func,
                          const void* config, const KernelArgs* args) noexcept {
  if (func == nullptr) return Status::InvalidArgument("host launch: null kernel");
  auto fn = reinterpret_cast<HostKernelFn>(func);
  fn(*static_cast<const LaunchConfig*>(config), *args);  // synchronous inline call
  return Status();
}

Status host_kernel_launch_batch(void* stream_native, const void* const* modules,
                                const void* const* funcs, const LaunchConfig* configs,
                                const KernelArgs* args_arr, unsigned count) noexcept {
  (void)modules;
  for (unsigned i = 0; i < count; ++i) {
    Status s = host_kernel_launch(stream_native, nullptr, const_cast<void*>(funcs[i]),
                                  &configs[i], &args_arr[i]);
    if (!s.ok()) return s;
  }
  return Status();
}

constexpr ComputeBackend kHostBackend = {
    /*name=*/"host",
    /*device_count=*/host_device_count,
    /*device_type=*/host_device_type,
    /*device_name=*/host_device_name,
    /*device_caps=*/host_device_caps,
    /*device_memory=*/host_device_memory,
    /*stream_create=*/host_stream_create,
    /*stream_destroy=*/host_stream_destroy,
    /*stream_sync=*/host_stream_sync,
    /*event_create=*/host_event_create,
    /*event_destroy=*/host_event_destroy,
    /*event_record=*/host_event_record,
    /*event_query=*/host_event_query,
    /*event_sync=*/host_event_sync,
    /*buffer_supported=*/host_buffer_supported,
    /*buffer_alignment=*/host_buffer_alignment,
    /*buffer_max_size=*/host_buffer_max_size,
    /*buffer_is_host=*/host_buffer_is_host,
    /*buffer_is_coherent=*/host_buffer_is_coherent,
    /*buffer_alloc=*/host_buffer_alloc,
    /*buffer_free=*/host_buffer_free,
    /*buffer_host_data=*/host_buffer_host_data,
    /*buffer_flush_host=*/host_buffer_flush,
    /*buffer_invalidate_host=*/host_buffer_invalidate,
    /*buffer_set=*/host_buffer_set,
    /*buffer_get=*/host_buffer_get,
    /*buffer_copy=*/host_buffer_copy,
    /*buffer_clear=*/host_buffer_clear,
    /*pool_create=*/host_pool_create,
    /*pool_destroy=*/host_pool_destroy,
    /*pool_is_vmm=*/host_pool_is_vmm,
    /*pool_alloc=*/host_pool_alloc,
    /*pool_free=*/host_pool_free,
    /*kernel_compile=*/host_kernel_compile,
    /*kernel_load_ptx=*/host_kernel_load_ptx,
    /*kernel_load_host=*/host_kernel_load_host,
    /*kernel_destroy=*/host_kernel_destroy,
    /*kernel_launch=*/host_kernel_launch,
    /*kernel_launch_batch=*/host_kernel_launch_batch,
    /*event_create_timing=*/host_event_create_timing,
    /*event_destroy_timing=*/host_event_destroy,   // sentinel: same no-op as plain
    /*event_record_timing=*/host_event_record_timing,
    /*event_query_timing=*/host_event_query,       // inline: always Ready
    /*event_sync_timing=*/host_event_sync,         // inline: no-op
    /*event_elapsed_ns=*/host_event_elapsed_ns,
    // record/replay (#16): NULL -> Unimplemented. The host backend runs kernels inline at
    // launch; there is no command buffer to record or resubmit, so recording gives nothing.
    /*recorded_create=*/nullptr,
    /*recorded_replay=*/nullptr,
    /*recorded_destroy=*/nullptr,
};

}  // namespace

const ComputeBackend* host_vtable() noexcept { return &kHostBackend; }

}  // namespace hyle::compute
