#include <cuda_runtime.h>
#include <cuda.h>  // driver API (cuMemAddressReserve/cuMemCreate/cuMemMap) for the VMM pool
#include <nvrtc.h>  // compute-04: runtime CUDA C -> PTX compilation

#include <cstdio>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend.h"
#include "hyle/compute/types.h"
#include "hyle/error/status.h"
#include "hyle/macros.h"  // HYLE_FATAL_ASSERT (VMM pool LIFO-free invariant)

// The CUDA backend (coding-conventions §10: an independent composition unit -- one
// TU, filling the manual vtable -- NOT a switch in a god object). Compiled by nvcc
// (CUDA language TU) ONLY when HYLE_COMPUTE_CUDA is set (CUDAToolkit found); the
// dispatch layer references cuda_vtable() under the same #if.
//
// This is the compute pillar's real payload: mapping HYLE's device/stream/event
// onto cudaStream_t/cudaEvent_t, and -- the whole point of compute-01 -- the event
// POLL primitive cudaEventQuery (compute-design §2 reap model, mirrors CCCL
// event_ref::is_done). No callbacks: a worker will (compute-03) poll query() at its
// reap boundary, exactly like io_uring completions.
//
// Native handles: the cudaStream_t / cudaEvent_t are pointer-sized opaque handles,
// stored directly as the backend's void* native -- no wrapper allocation. On an
// error creating one, we translate the cudaError_t into a HYLE Status (external
// boundary) and never leak cudaError_t across the public API.

namespace hyle::compute {
namespace {

Status native_failure(StatusCode code, NativeErrorDomain native_domain,
                      const char* domain, const char* operation,
                      const char* name, int value,
                      const char* description) noexcept {
  const char* op = operation != nullptr ? operation : "unknown operation";
  const char* error_name = name != nullptr ? name : "UNKNOWN";
  char message[512];
  if (description != nullptr && description[0] != '\0') {
    std::snprintf(message, sizeof(message), "%s %s failed: %s (%d): %s",
                  domain, op, error_name, value, description);
  } else {
    std::snprintf(message, sizeof(message), "%s %s failed: %s (%d)", domain,
                  op, error_name, value);
  }
  return Status::FromNative(code, {native_domain, value}, message);
}

// Translate a cudaError_t (external boundary failure) into a HYLE Status. Uses the
// canonical taxonomy: no-device/driver -> Unavailable, OOM -> ResourceExhausted,
// bad arg -> InvalidArgument, else Internal. Preserve operation, native name,
// numeric value and vendor description in the message.
Status from_cuda(cudaError_t e, const char* what) noexcept {
  if (e == cudaSuccess) return Status();
  StatusCode code = StatusCode::Internal;
  switch (e) {
    case cudaErrorNoDevice:
    case cudaErrorInsufficientDriver:
      code = StatusCode::Unavailable;
      break;
    case cudaErrorNotSupported:
      code = StatusCode::Unimplemented;
      break;
    case cudaErrorMemoryAllocation:
      code = StatusCode::ResourceExhausted;
      break;
    case cudaErrorInvalidValue:
    case cudaErrorInvalidDevice:
      code = StatusCode::InvalidArgument;
      break;
    default:
      break;
  }
  return native_failure(code, NativeErrorDomain::CudaRuntime, "cuda", what,
                        cudaGetErrorName(e),
                        static_cast<int>(e), cudaGetErrorString(e));
}

// Cached device count (cudaGetDeviceCount is cheap but this is queried at table
// build only; a bad driver returns 0 -> compute falls back to host, no crash).
std::size_t cuda_device_count() noexcept {
  int n = 0;
  cudaError_t e = cudaGetDeviceCount(&n);
  if (e != cudaSuccess || n < 0) return 0;
  return static_cast<std::size_t>(n);
}

DeviceType cuda_device_type(int) noexcept {
  // A CUDA device is a discrete GPU (dedicated VRAM). Integrated/unified detection
  // (cudaDeviceProp::integrated -> kIGpu, drives compute-02's copy elision) is
  // wired when compute-02 needs the distinction; compute-01 reports kGpu.
  return DeviceType::kGpu;
}

// Per-device name cache. cudaGetDeviceProperties returns a 256-char name; we keep a
// small process-stable copy per device so device_name() can hand back a stable
// pointer (the public contract). Filled lazily on first query.
constexpr int kMaxCachedDevices = 16;
char g_name_cache[kMaxCachedDevices][256];
bool g_name_filled[kMaxCachedDevices] = {};

const char* cuda_device_name(int ordinal) noexcept {
  if (ordinal < 0 || ordinal >= kMaxCachedDevices) return "cuda";
  if (!g_name_filled[ordinal]) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, ordinal) == cudaSuccess) {
      // prop.name is NUL-terminated <= 256; copy verbatim.
      std::size_t i = 0;
      for (; i < sizeof(g_name_cache[ordinal]) - 1 && prop.name[i] != '\0'; ++i)
        g_name_cache[ordinal][i] = prop.name[i];
      g_name_cache[ordinal][i] = '\0';
    } else {
      g_name_cache[ordinal][0] = '\0';
    }
    g_name_filled[ordinal] = true;
  }
  return g_name_cache[ordinal][0] != '\0' ? g_name_cache[ordinal] : "cuda";
}

DeviceCaps cuda_device_caps(int) noexcept {
  // A real GPU: async streams, pinned host buffers (cudaMallocHost), events.
  return DeviceCaps{/*async=*/true, /*host_buffer=*/true, /*events=*/true};
}

// Set the calling thread's current device to `ordinal`, but ONLY if it isn't
// already -- cudaSetDevice is not free, and (verified against ggml_cuda_set_device,
// ggml-cuda.cu:109) the redundant-set guard is a real win, especially on Windows
// where the driver skips this check itself. Returns the cudaError_t of the set (or
// cudaSuccess if already current).
//
// WHEN a set is needed (verified against ggml + CCCL source): the CURRENT device
// governs CREATION (cudaMalloc / cudaStreamCreate / cudaEventCreate) and KERNEL
// LAUNCH only. Stream/event OPERATIONS (record / query / synchronize / free) do NOT
// depend on the current device -- a CUDA stream/event handle carries its context
// intrinsically. ggml's event record/wait/sync set no device (ggml-cuda.cu:4527/
// 4533/5510); CCCL uses the driver API for stream ops precisely so they never touch
// the current device (stream_ref.h:189 "cudaStreamWaitEvent would push dev 0"). So
// this is called at CREATION sites ONLY; the reap/poll path stays set-free. THIS IS
// WHY fiber migration across workers is safe without pinning a stream to a core --
// an op reaped on a different worker than it was submitted on is still correct.
cudaError_t set_device(int ordinal) noexcept {
  int current = -1;
  cudaError_t e = cudaGetDevice(&current);
  if (e != cudaSuccess) return e;
  if (current == ordinal) return cudaSuccess;
  return cudaSetDevice(ordinal);
}

void cuda_device_memory(int ordinal, std::uint64_t* free,
                        std::uint64_t* total) noexcept {
  std::size_t f = 0, t = 0;
  // cudaMemGetInfo reports for the CURRENT device, so set it first. Best-effort:
  // leave zeros on error (informational query, never fails the caller).
  if (set_device(ordinal) == cudaSuccess) {
    cudaMemGetInfo(&f, &t);
  }
  if (free != nullptr) *free = f;
  if (total != nullptr) *total = t;
}

Status cuda_stream_create(int ordinal, void** out_native) noexcept {
  cudaError_t e = set_device(ordinal);
  if (e != cudaSuccess) return from_cuda(e, "cudaSetDevice");
  cudaStream_t s = nullptr;
  e = cudaStreamCreate(&s);
  if (e != cudaSuccess) return from_cuda(e, "cudaStreamCreate");
  *out_native = static_cast<void*>(s);
  return Status();
}

void cuda_stream_destroy(void* native) noexcept {
  if (native != nullptr) cudaStreamDestroy(static_cast<cudaStream_t>(native));
}

Status cuda_stream_sync(void* native) noexcept {
  cudaError_t e = cudaStreamSynchronize(static_cast<cudaStream_t>(native));
  return from_cuda(e, "cudaStreamSynchronize");
}

Status cuda_event_create(int ordinal, void** out_native) noexcept {
  cudaError_t e = set_device(ordinal);
  if (e != cudaSuccess) return from_cuda(e, "cudaSetDevice");
  cudaEvent_t ev = nullptr;
  // cudaEventDisableTiming: HYLE's completion model only ever query()s/sync()s an event
  // for done-ness (GpuPoller polls it every sweep, compute-03) -- it NEVER reads elapsed
  // time (zero cudaEventElapsedTime in the codebase). Disabling timing skips the per-record
  // timestamp write and uses a lighter sync path, so record/query are cheaper on the hot
  // poll loop. (A timing event would only be needed for profiling, which HYLE doesn't do here.)
  e = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
  if (e != cudaSuccess) return from_cuda(e, "cudaEventCreateWithFlags");
  *out_native = static_cast<void*>(ev);
  return Status();
}

void cuda_event_destroy(void* native) noexcept {
  if (native != nullptr) cudaEventDestroy(static_cast<cudaEvent_t>(native));
}

void cuda_event_record(void* stream_native, void* event_native) noexcept {
  // Record the event on the stream. Errors here are effectively unreachable for
  // valid handles on the same device (checked in the public layer); a failure would
  // surface at the subsequent query()/sync(). cudaEventRecord's own return is
  // dropped intentionally (the vtable slot is void, matching CUDA's fire-and-check
  // model where completion is observed via the event, not the record call).
  cudaEventRecord(static_cast<cudaEvent_t>(event_native),
                  static_cast<cudaStream_t>(stream_native));
}

EventStatus cuda_event_query(void* event_native) noexcept {
  // THE poll primitive (compute-design §2): cudaSuccess => all prior stream work
  // done; cudaErrorNotReady => still running. Any other error (a genuine device
  // fault) is reported as Ready here so the caller unblocks and observes the fault
  // via stream sync / the op result -- query() must stay a cheap non-blocking bool.
  cudaError_t e = cudaEventQuery(static_cast<cudaEvent_t>(event_native));
  return (e == cudaErrorNotReady) ? EventStatus::kNotReady : EventStatus::kReady;
}

void cuda_event_sync(void* event_native) noexcept {
  cudaEventSynchronize(static_cast<cudaEvent_t>(event_native));
}

// ---- GPU timing (compute-06b): timing-enabled event + cudaEventElapsedTime -------
// A timing event is a plain cudaEvent WITHOUT cudaEventDisableTiming, so it records a
// device timestamp. record is identical to the completion path (cudaEventRecord); elapsed
// reads cudaEventElapsedTime (milliseconds) and converts to ns for the common unit.
Status cuda_event_create_timing(int ordinal, void** out_native) noexcept {
  cudaError_t e = set_device(ordinal);
  if (e != cudaSuccess) return from_cuda(e, "cudaSetDevice");
  cudaEvent_t ev = nullptr;
  e = cudaEventCreate(&ev);  // default flags = timing ENABLED (vs event_create's DisableTiming)
  if (e != cudaSuccess) return from_cuda(e, "cudaEventCreate(timing)");
  *out_native = static_cast<void*>(ev);
  return Status();
}
void cuda_event_record_timing(void* stream_native, void* event_native) noexcept {
  cudaEventRecord(static_cast<cudaEvent_t>(event_native),
                  static_cast<cudaStream_t>(stream_native));  // same as non-timing record
}
Status cuda_event_elapsed_ns(void* start_native, void* stop_native,
                             std::uint64_t* out_ns) noexcept {
  float ms = 0.0f;
  cudaError_t e = cudaEventElapsedTime(&ms, static_cast<cudaEvent_t>(start_native),
                                       static_cast<cudaEvent_t>(stop_native));
  if (e != cudaSuccess) return from_cuda(e, "cudaEventElapsedTime");
  *out_ns = static_cast<std::uint64_t>(static_cast<double>(ms) * 1e6);  // ms -> ns
  return Status();
}

// ---- buffer (compute-02) ----------------------------------------------------
// kDevice = cudaMalloc (VRAM, native handle only). kHostPinned = cudaMallocHost
// (page-locked host RAM; native handle and CPU mapping happen to be the same pointer).
// set/get/copy/clear are synchronous. flush/invalidate are no-ops because CUDA pinned
// host memory is coherent; async staging still requires a separate Event-owned ring.

std::size_t cuda_buffer_alignment(BufferKind kind) noexcept {
  // Device: 128 (matches ggml cuda buffer_type, coalesced access + tensor-core
  // alignment). Pinned host: max_align is enough (CPU-side RAM).
  return kind == BufferKind::kDevice ? 128 : alignof(std::max_align_t);
}

std::size_t cuda_buffer_max_size(int, BufferKind) noexcept {
  return SIZE_MAX;  // bounded only by free VRAM; ggml cuda buft is likewise SIZE_MAX
}

bool cuda_buffer_is_host(BufferKind kind) noexcept {
  return kind == BufferKind::kHostPinned;  // pinned = host RAM; device = VRAM
}

Status cuda_buffer_alloc(int ordinal, BufferKind kind, std::size_t bytes,
                         void** out_native, void** out_host) noexcept {
  cudaError_t e = set_device(ordinal);
  if (e != cudaSuccess) return from_cuda(e, "cudaSetDevice");
  std::size_t n = bytes == 0 ? 1 : bytes;
  void* p = nullptr;
  if (kind == BufferKind::kDevice) {
    e = cudaMalloc(&p, n);
    if (e != cudaSuccess) return from_cuda(e, "cudaMalloc");
  } else {
    // Page-locked host memory. No silent fallback to malloc: the free path is fixed
    // by kind, so a malloc fallback would mismatch cudaFreeHost.
    e = cudaMallocHost(&p, n);
    if (e != cudaSuccess) return from_cuda(e, "cudaMallocHost");
  }
  *out_native = p;
  *out_host = kind == BufferKind::kHostPinned ? p : nullptr;
  return Status();
}

void cuda_buffer_free(int, BufferKind kind, void* ptr) noexcept {
  if (ptr == nullptr) return;
  if (kind == BufferKind::kDevice) {
    cudaFree(ptr);
  } else {
    cudaFreeHost(ptr);
  }
}
Status cuda_buffer_flush(void*, std::size_t, std::size_t) noexcept { return Status(); }
Status cuda_buffer_invalidate(void*, std::size_t, std::size_t) noexcept { return Status(); }

Status cuda_buffer_set(void* dst_base, std::size_t offset, const void* src,
                       std::size_t size) noexcept {
  void* dst = static_cast<char*>(dst_base) + offset;  // H2D partial write at offset
  cudaError_t e = cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault,
                                  cudaStreamPerThread);
  if (e != cudaSuccess) return from_cuda(e, "cudaMemcpyAsync(set)");
  return from_cuda(cudaStreamSynchronize(cudaStreamPerThread), "sync(set)");
}

Status cuda_buffer_get(const void* src_base, std::size_t offset, void* dst,
                       std::size_t size) noexcept {
  const void* src = static_cast<const char*>(src_base) + offset;
  cudaError_t e = cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault,
                                  cudaStreamPerThread);
  if (e != cudaSuccess) return from_cuda(e, "cudaMemcpyAsync(get)");
  return from_cuda(cudaStreamSynchronize(cudaStreamPerThread), "sync(get)");
}

Status cuda_buffer_copy(void* dst_base, std::size_t dst_off, const void* src_base,
                        std::size_t src_off, std::size_t size) noexcept {
  void* dst = static_cast<char*>(dst_base) + dst_off;
  const void* src = static_cast<const char*>(src_base) + src_off;
  cudaError_t e = cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault,
                                  cudaStreamPerThread);
  if (e != cudaSuccess) return from_cuda(e, "cudaMemcpyAsync(copy)");
  return from_cuda(cudaStreamSynchronize(cudaStreamPerThread), "sync(copy)");
}

Status cuda_buffer_clear(void* base, std::size_t offset, std::uint8_t value,
                         std::size_t size) noexcept {
  void* dst = static_cast<char*>(base) + offset;
  cudaError_t e = cudaMemsetAsync(dst, value, size, cudaStreamPerThread);
  if (e != cudaSuccess) return from_cuda(e, "cudaMemsetAsync");
  return from_cuda(cudaStreamSynchronize(cudaStreamPerThread), "sync(clear)");
}

// ---- device (VRAM) sub-allocator pool (compute-02c) -------------------------
// Ported wholesale from ggml (user: "ggml 很 SOTA 抄全一点"). Two backends: VMM
// (driver-API virtual memory: reserve big address space + map physical on demand,
// LIFO bump) and leg (cached-block best-fit, no-VMM fallback). Adapted to HYLE's
// error model: ggml's CUDA_CHECK/GGML_ASSERT abort; HYLE returns Status where a
// call can fail operationally, and marks a construction failure by leaving the pool
// in an invalid state (create returns the Status). A cu-driver error is translated
// like a runtime error.

// Translate a CUresult (driver API) into a HYLE Status. cuGetErrorString gives text.
Status from_cu(CUresult r, const char* what) noexcept {
  if (r == CUDA_SUCCESS) return Status();
  const char* name = nullptr;
  const char* msg = nullptr;
  cuGetErrorName(r, &name);
  cuGetErrorString(r, &msg);
  StatusCode code = StatusCode::Internal;
  if (r == CUDA_ERROR_OUT_OF_MEMORY) code = StatusCode::ResourceExhausted;
  if (r == CUDA_ERROR_NOT_SUPPORTED) code = StatusCode::Unimplemented;
  if (r == CUDA_ERROR_INVALID_VALUE || r == CUDA_ERROR_INVALID_DEVICE)
    code = StatusCode::InvalidArgument;
  return native_failure(code, NativeErrorDomain::CudaDriver, "cuda-driver",
                        what, name, static_cast<int>(r), msg);
}

// A pool is one of these two, tagged so pool_is_vmm / dispatch can tell them apart.
// Base holds the tag; the concrete structs follow ggml's layout.
enum class PoolKind : std::uint8_t { kLeg = 0, kVmm = 1 };

struct CudaPoolBase {
  PoolKind kind;
  int device;
  explicit CudaPoolBase(PoolKind k, int d) noexcept : kind(k), device(d) {}
};

// leg pool -- ggml_cuda_pool_leg (ggml-cuda.cu:361). Best-fit over a 256-slot cache
// of freed cudaMalloc blocks; look-ahead over-allocation; OOM -> sync+flush+retry.
struct CudaPoolLeg : CudaPoolBase {
  static constexpr int kMaxBuffers = 256;
  struct Block { void* ptr = nullptr; std::size_t size = 0; };
  Block buffer_pool[kMaxBuffers] = {};
  std::size_t pool_size = 0;

  explicit CudaPoolLeg(int d) noexcept : CudaPoolBase(PoolKind::kLeg, d) {}

  void clear_pool() noexcept {
    cudaSetDevice(device);
    for (auto& b : buffer_pool) {
      if (b.ptr != nullptr) {
        cudaFree(b.ptr);
        pool_size -= b.size;
        b.ptr = nullptr;
        b.size = 0;
      }
    }
  }
  ~CudaPoolLeg() { clear_pool(); }

  Status alloc(std::size_t size, void** out_ptr, std::size_t* actual) noexcept {
    // best-fit over cached freed blocks; exact match short-circuits.
    std::size_t best_diff = 1ull << 36;
    int ibest = -1;
    for (int i = 0; i < kMaxBuffers; ++i) {
      Block& b = buffer_pool[i];
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
      Block& b = buffer_pool[ibest];
      *out_ptr = b.ptr;
      *actual = b.size;
      b.ptr = nullptr;
      b.size = 0;
      return Status();
    }
    // miss: real cudaMalloc with look-ahead over-allocation (1.05x, round up 256B).
    std::size_t look_ahead = static_cast<std::size_t>(1.05 * static_cast<double>(size));
    look_ahead = 256 * ((look_ahead + 255) / 256);
    if (look_ahead == 0) look_ahead = 256;
    cudaSetDevice(device);
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, look_ahead);
    if (err == cudaErrorMemoryAllocation) {
      // flush the cache and retry once (ggml strategy).
      (void)cudaGetLastError();
      cudaDeviceSynchronize();
      clear_pool();
      err = cudaMalloc(&ptr, look_ahead);
    }
    if (err != cudaSuccess) return from_cuda(err, "pool cudaMalloc");
    *out_ptr = ptr;
    *actual = look_ahead;
    pool_size += look_ahead;
    return Status();
  }

  void free(void* ptr, std::size_t size) noexcept {
    for (auto& b : buffer_pool) {
      if (b.ptr == nullptr) {
        b.ptr = ptr;
        b.size = size;
        return;
      }
    }
    // cache full -> really free (rare; ggml logs + frees).
    cudaSetDevice(device);
    cudaFree(ptr);
    pool_size -= size;
  }
};

// VMM pool -- ggml_cuda_pool_vmm (ggml-cuda.cu:478). Reserve 32GB virtual address
// once, map physical pages on demand, O(1) bump alloc, LIFO free.
struct CudaPoolVmm : CudaPoolBase {
  static constexpr std::size_t kMaxSize = 1ull << 35;  // 32 GB

  CUdeviceptr pool_addr = 0;
  std::size_t pool_used = 0;
  std::size_t pool_size = 0;
  std::size_t granularity;

  CudaPoolVmm(int d, std::size_t gran) noexcept
      : CudaPoolBase(PoolKind::kVmm, d), granularity(gran) {}

  ~CudaPoolVmm() {
    if (pool_addr != 0) {
      cuMemUnmap(pool_addr, pool_size);
      cuMemAddressFree(pool_addr, kMaxSize);
    }
  }

  Status alloc(std::size_t size, void** out_ptr, std::size_t* actual) noexcept {
    // round up to 128B so every allocation is aligned for all data types.
    const std::size_t alignment = 128;
    size = alignment * ((size + alignment - 1) / alignment);
    if (size == 0) size = alignment;

    std::size_t avail = pool_size - pool_used;
    if (size > avail) {
      std::size_t reserve = size - avail;
      reserve = granularity * ((reserve + granularity - 1) / granularity);
      if (pool_size + reserve > kMaxSize)
        return Status::ResourceExhausted("VMM pool exceeds 32GB reservation");

      // create physical memory
      CUmemAllocationProp prop = {};
      prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      prop.location.id = device;
      CUmemGenericAllocationHandle handle;
      if (CUresult r = cuMemCreate(&handle, reserve, &prop, 0); r != CUDA_SUCCESS)
        return from_cu(r, "cuMemCreate");

      // reserve virtual address space once
      if (pool_addr == 0) {
        if (CUresult r = cuMemAddressReserve(&pool_addr, kMaxSize, 0, 0, 0);
            r != CUDA_SUCCESS) {
          cuMemRelease(handle);
          return from_cu(r, "cuMemAddressReserve");
        }
      }
      // map at the end of the pool
      CUdeviceptr start = reinterpret_cast<CUdeviceptr>(
          reinterpret_cast<char*>(pool_addr) + pool_size);
      if (CUresult r = cuMemMap(start, reserve, 0, handle, 0); r != CUDA_SUCCESS) {
        cuMemRelease(handle);
        return from_cu(r, "cuMemMap");
      }
      cuMemRelease(handle);  // handle no longer needed after mapping

      // grant read/write access
      CUmemAccessDesc access = {};
      access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      access.location.id = device;
      access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      if (CUresult r = cuMemSetAccess(start, reserve, &access, 1); r != CUDA_SUCCESS)
        return from_cu(r, "cuMemSetAccess");

      pool_size += reserve;
    }

    // O(1) bump.
    *out_ptr = reinterpret_cast<void*>(
        reinterpret_cast<char*>(pool_addr) + pool_used);
    *actual = size;
    pool_used += size;
    return Status();
  }

  void free(void* ptr, std::size_t size) noexcept {
    // round the size the same way alloc did, then LIFO-pop.
    const std::size_t alignment = 128;
    size = alignment * ((size + alignment - 1) / alignment);
    if (size == 0) size = alignment;
    pool_used -= size;
    // contract: LIFO -- ptr must be the most recent live allocation. (ggml asserts;
    // HYLE fatal-asserts the same invariant since a violation is a caller bug.)
    HYLE_FATAL_ASSERT(ptr == reinterpret_cast<void*>(
                                 reinterpret_cast<char*>(pool_addr) + pool_used));
  }
};

// Does `device` support CUDA virtual memory management? (ggml-cuda.cu:233)
bool device_supports_vmm(int device, std::size_t* out_gran) noexcept {
  CUdevice dev;
  if (cuDeviceGet(&dev, device) != CUDA_SUCCESS) return false;
  int vmm = 0;
  if (cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                           dev) != CUDA_SUCCESS)
    return false;
  if (!vmm) return false;
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  std::size_t gran = 0;
  if (cuMemGetAllocationGranularity(&gran, &prop,
                                    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS)
    return false;
  *out_gran = gran;
  return true;
}

Status cuda_pool_create(int ordinal, void** out_handle) noexcept {
  if (cudaError_t e = set_device(ordinal); e != cudaSuccess)
    return from_cuda(e, "cudaSetDevice");
  // The driver must be initialized before cu* calls; a prior runtime call (set_device)
  // implicitly does cuInit for the primary context, so the driver API is usable here.
  std::size_t gran = 0;
  CudaPoolBase* pool = nullptr;
  if (device_supports_vmm(ordinal, &gran)) {
    pool = new (std::nothrow) CudaPoolVmm(ordinal, gran);
  } else {
    pool = new (std::nothrow) CudaPoolLeg(ordinal);
  }
  if (pool == nullptr) return Status::ResourceExhausted("pool_create");
  *out_handle = pool;
  return Status();
}

void cuda_pool_destroy(void* handle) noexcept {
  auto* base = static_cast<CudaPoolBase*>(handle);
  if (base == nullptr) return;
  if (base->kind == PoolKind::kVmm) delete static_cast<CudaPoolVmm*>(base);
  else delete static_cast<CudaPoolLeg*>(base);
}

bool cuda_pool_is_vmm(void* handle) noexcept {
  return static_cast<CudaPoolBase*>(handle)->kind == PoolKind::kVmm;
}

Status cuda_pool_alloc(void* handle, std::size_t size, void** out_ptr,
                       std::size_t* out_actual) noexcept {
  auto* base = static_cast<CudaPoolBase*>(handle);
  return base->kind == PoolKind::kVmm
             ? static_cast<CudaPoolVmm*>(base)->alloc(size, out_ptr, out_actual)
             : static_cast<CudaPoolLeg*>(base)->alloc(size, out_ptr, out_actual);
}

void cuda_pool_free(void* handle, void* ptr, std::size_t size) noexcept {
  auto* base = static_cast<CudaPoolBase*>(handle);
  if (base->kind == PoolKind::kVmm) static_cast<CudaPoolVmm*>(base)->free(ptr, size);
  else static_cast<CudaPoolLeg*>(base)->free(ptr, size);
}

// ---- kernel launch (compute-04): NVRTC runtime compile + driver launch ------
// The full B-tier chain: CUDA C source -> NVRTC -> PTX -> cuModuleLoadData ->
// cuModuleGetFunction -> cuLaunchKernel. NVRTC sequence ported from CCCL
// (nvrtcc_build.h:80-115 + command_list.h name-expression handling); ggml has zero
// nvrtc so nothing to copy there. Launch engineering (grid/block, cudaGetLastError
// fail-fast) from ggml. A kernel's {module,func} = {CUmodule, CUfunction}.

// Translate an nvrtcResult into a HYLE Status (compile-time failures).
Status from_nvrtc(nvrtcResult r, const char* what) noexcept {
  if (r == NVRTC_SUCCESS) return Status();
  StatusCode code = StatusCode::Internal;
  if (r == NVRTC_ERROR_OUT_OF_MEMORY) code = StatusCode::ResourceExhausted;
  if (r == NVRTC_ERROR_INVALID_INPUT || r == NVRTC_ERROR_INVALID_OPTION ||
      r == NVRTC_ERROR_INVALID_PROGRAM)
    code = StatusCode::InvalidArgument;
  return native_failure(code, NativeErrorDomain::Nvrtc, "nvrtc", what,
                        nvrtcGetErrorString(r), static_cast<int>(r), nullptr);
}

// In-process PTX compile cache: hash(source) ^ arch -> PTX. Avoids recompiling the
// same kernel source (NVRTC is slow). Cold path (only on compile), so a plain mutex +
// std::unordered_map is fine -- this is internal compilation bookkeeping, NOT a hot-
// path production container (the std-container discipline targets hot/ABI surfaces).
// FNV-1a hash (not XXH3) to avoid pulling SIMD headers into the nvcc TU.
std::uint64_t fnv1a(const char* s) noexcept {
  std::uint64_t h = 1469598103934665603ull;
  for (; *s; ++s) { h ^= static_cast<unsigned char>(*s); h *= 1099511628211ull; }
  return h;
}
std::mutex g_ptx_cache_mu;
std::unordered_map<std::uint64_t, std::string> g_ptx_cache;

// Probe device compute capability -> "-arch=sm_XX". Uses the driver API (the pool
// already inits it). Falls back to sm_50 (the CUDA 12/13 floor) if the query fails.
void arch_flag(int ordinal, char* out, std::size_t cap) noexcept {
  int major = 5, minor = 0;
  CUdevice dev;
  if (cuDeviceGet(&dev, ordinal) == CUDA_SUCCESS) {
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
  }
  std::snprintf(out, cap, "--gpu-architecture=sm_%d%d", major, minor);
}

// NVRTC-compile `source` for `ordinal`, resolving `entry` (extern "C" name used
// verbatim, or a C++ name expression -> lowered name). Fills *out_ptx and, for a C++
// entry, *out_lowered (the mangled launch name). Caches by hash(source)^arch.
Status nvrtc_to_ptx(int ordinal, const char* source, const char* entry,
                    std::string* out_ptx, std::string* out_lowered) noexcept {
  char arch[64];
  arch_flag(ordinal, arch, sizeof(arch));

  // Cache probe (source + arch). Lowered name is NOT cached (cheap to re-derive only
  // matters for C++ entries; we skip cache when a lowered name is needed so the
  // expression re-resolves -- simpler + correct). extern "C" entry has no lowering.
  std::uint64_t key = fnv1a(source) ^ fnv1a(arch);
  {
    std::lock_guard<std::mutex> lk(g_ptx_cache_mu);
    auto it = g_ptx_cache.find(key);
    if (it != g_ptx_cache.end()) {
      *out_ptx = it->second;
      *out_lowered = entry;  // cached path assumes extern "C" (verbatim name)
      return Status();
    }
  }

  nvrtcProgram prog;
  if (nvrtcResult r = nvrtcCreateProgram(&prog, source, "hyle_kernel.cu", 0, nullptr,
                                         nullptr);
      r != NVRTC_SUCCESS)
    return from_nvrtc(r, "nvrtcCreateProgram");

  // Register `entry` as a name expression so a C++ (mangled) kernel can be launched
  // by its lowered name. Harmless for an extern "C" name too. (CCCL command_list.h:234)
  nvrtcAddNameExpression(prog, entry);

  const char* opts[] = {arch, "--std=c++17"};
  nvrtcResult compile = nvrtcCompileProgram(prog, 2, opts);

  // Always fetch the log (compile errors carry the diagnostic here). CCCL:259-267.
  std::size_t log_size = 0;
  nvrtcGetProgramLogSize(prog, &log_size);
  std::string log;
  if (log_size > 1) {
    log.resize(log_size);
    nvrtcGetProgramLog(prog, log.data());
  }
  if (compile != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&prog);
    // Surface the NVRTC log as the Status message (truncated by Status if long).
    return Status::Internal(log.empty() ? "NVRTC compile failed" : log.c_str());
  }

  // Lowered (mangled) launch name for the C++ entry expression.
  const char* lowered = nullptr;
  if (nvrtcGetLoweredName(prog, entry, &lowered) == NVRTC_SUCCESS && lowered != nullptr)
    *out_lowered = lowered;
  else
    *out_lowered = entry;  // extern "C" / fallback: name is verbatim

  // Get PTX.
  std::size_t ptx_size = 0;
  if (nvrtcResult r = nvrtcGetPTXSize(prog, &ptx_size); r != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&prog);
    return from_nvrtc(r, "nvrtcGetPTXSize");
  }
  out_ptx->resize(ptx_size);
  nvrtcGetPTX(prog, out_ptx->data());
  nvrtcDestroyProgram(&prog);

  // Cache the PTX (keyed on source+arch). We only reach here on a cache miss.
  {
    std::lock_guard<std::mutex> lk(g_ptx_cache_mu);
    g_ptx_cache.emplace(key, *out_ptx);
  }
  return Status();
}

// Load PTX text into a CUmodule + resolve `entry` to a CUfunction.
Status load_module(int ordinal, const char* ptx, const char* entry,
                   void** out_module, void** out_func) noexcept {
  if (cudaError_t e = set_device(ordinal); e != cudaSuccess)
    return from_cuda(e, "cudaSetDevice");
  // The driver API (cuModuleLoadData) needs a CURRENT context on this thread. The
  // runtime creates the primary context LAZILY -- cudaSetDevice alone may not have
  // realized it, so a driver call gets CUDA_ERROR_INVALID_CONTEXT. Force the runtime
  // to instantiate + bind its primary context to this thread with a trivial runtime
  // op (cudaFree(0) is the canonical "init current context" idiom); the driver API
  // then uses that current primary context. (ggml/CCCL run under an already-current
  // context; HYLE's kernel load is the first driver-API entry needing this nudge.)
  cudaFree(nullptr);
  CUmodule module = nullptr;
  if (CUresult r = cuModuleLoadData(&module, ptx); r != CUDA_SUCCESS)
    return from_cu(r, "cuModuleLoadData");
  CUfunction func = nullptr;
  if (CUresult r = cuModuleGetFunction(&func, module, entry); r != CUDA_SUCCESS) {
    cuModuleUnload(module);
    return from_cu(r, "cuModuleGetFunction");
  }
  *out_module = static_cast<void*>(module);
  *out_func = static_cast<void*>(func);
  return Status();
}

Status cuda_kernel_compile(int ordinal, const char* source, const char* entry,
                           void** out_module, void** out_func) noexcept {
  std::string ptx, lowered;
  if (Status st = nvrtc_to_ptx(ordinal, source, entry, &ptx, &lowered); !st.ok())
    return st;
  return load_module(ordinal, ptx.c_str(), lowered.c_str(), out_module, out_func);
}

Status cuda_kernel_load_ptx(int ordinal, const char* ptx, unsigned code_size,
                            const char* entry, const SpecConstant* /*specs*/,
                            unsigned spec_count, const void* /*spec_data*/,
                            unsigned /*spec_data_size*/, void** out_module,
                            void** out_func) noexcept {
  // A non-zero code_size means BINARY input = a SPIR-V module from load_spirv (PTX is text,
  // passed with code_size 0). CUDA can't consume SPIR-V; say so plainly instead of feeding it
  // to the PTX parser (which would fail with a misleading parse error). load_spirv is Vulkan.
  if (code_size > 0)
    return Status::Unimplemented("cuda backend: load_spirv is Vulkan-only (use load_ptx/compile)");
  // Specialization constants are a Vulkan/SPIR-V concept; CUDA's build-time analog is NVRTC
  // macros at compile() -> reject here too.
  if (spec_count > 0)
    return Status::Unimplemented("cuda backend: no spec constants (use compile() macros)");
  return load_module(ordinal, ptx, entry, out_module, out_func);
}

Status cuda_kernel_load_host(int, void*, void**, void**) noexcept {
  // A CUDA device can't run a host fn as a kernel; that's the host backend's path.
  return Status::Unimplemented("cuda backend: load_host is host-only");
}

void cuda_kernel_destroy(void* module, void* /*func*/) noexcept {
  if (module != nullptr) cuModuleUnload(static_cast<CUmodule>(module));
}

Status cuda_kernel_launch(void* stream_native, void* /*module*/, void* func,
                          const void* config, const KernelArgs* args) noexcept {
  const auto* cfg = static_cast<const LaunchConfig*>(config);
  // Rebuild cuLaunchKernel's void** kernelParams from the unified KernelArgs. The typed
  // shell supplies `order` so mixed signatures retain their exact source order; raw
  // KernelArgs without it keep the original buffers-then-scalars convention.
  // Bounded stack array: total args = buffers + scalar fields (cap mirrors the shell).
  void* params[64];
  unsigned np = 0;
  const unsigned total = args->buffer_count + args->scalar_count;
  if (total > 64) return Status::InvalidArgument("cuda launch: >64 kernel args");
  const auto* blob = static_cast<const std::byte*>(args->scalars);
  auto append_buffer = [&](unsigned i) {
    params[np++] =
        const_cast<void*>(static_cast<const void*>(&args->buffers[i]));
  };
  auto append_scalar = [&](unsigned i) {
    params[np++] = const_cast<void*>(static_cast<const void*>(
        blob + args->fields[i].offset));
  };
  if (args->order != nullptr) {
    if (args->arg_count != total)
      return Status::InvalidArgument("cuda launch: invalid argument-order count");
    for (unsigned i = 0; i < args->arg_count; ++i) {
      const KernelArgRef ref = args->order[i];
      if (ref.kind == KernelArgKind::kBuffer) {
        if (ref.index >= args->buffer_count)
          return Status::InvalidArgument("cuda launch: invalid buffer argument index");
        append_buffer(ref.index);
      } else if (ref.kind == KernelArgKind::kScalar) {
        if (ref.index >= args->scalar_count || blob == nullptr || args->fields == nullptr)
          return Status::InvalidArgument("cuda launch: invalid scalar argument index");
        append_scalar(ref.index);
      } else {
        return Status::InvalidArgument("cuda launch: invalid argument kind");
      }
    }
  } else {
    for (unsigned i = 0; i < args->buffer_count; ++i) append_buffer(i);
    for (unsigned i = 0; i < args->scalar_count; ++i) append_scalar(i);
  }
  CUresult r = cuLaunchKernel(
      static_cast<CUfunction>(func), cfg->grid.x, cfg->grid.y, cfg->grid.z,
      cfg->block.x, cfg->block.y, cfg->block.z, cfg->shared_mem,
      static_cast<CUstream>(stream_native), np ? params : nullptr, nullptr);
  if (r != CUDA_SUCCESS) return from_cu(r, "cuLaunchKernel");
  // ggml fail-fast contract: check the launch config error immediately (async exec
  // errors surface later via stream sync / the recorded event).
  return from_cuda(cudaGetLastError(), "kernel launch");
}

// Batched launch: a CUDA stream already ORDERS sequential submits with no extra per-submit
// cost (unlike Vulkan's per-submit fence), so CUDA's "batch" is just launching each kernel
// on the same stream in order -- no separate cmdbuf/fence to amortize. Loop kernel_launch.
Status cuda_kernel_launch_batch(void* stream_native, const void* const* modules,
                                const void* const* funcs, const LaunchConfig* configs,
                                const KernelArgs* args_arr, unsigned count) noexcept {
  for (unsigned i = 0; i < count; ++i) {
    Status s = cuda_kernel_launch(stream_native, const_cast<void*>(modules[i]),
                                  const_cast<void*>(funcs[i]), &configs[i], &args_arr[i]);
    if (!s.ok()) return s;
  }
  return Status();
}

constexpr ComputeBackend kCudaBackend = {
    /*name=*/"cuda",
    /*device_count=*/cuda_device_count,
    /*device_type=*/cuda_device_type,
    /*device_name=*/cuda_device_name,
    /*device_caps=*/cuda_device_caps,
    /*device_memory=*/cuda_device_memory,
    /*stream_create=*/cuda_stream_create,
    /*stream_destroy=*/cuda_stream_destroy,
    /*stream_sync=*/cuda_stream_sync,
    /*event_create=*/cuda_event_create,
    /*event_destroy=*/cuda_event_destroy,
    /*event_record=*/cuda_event_record,
    /*event_query=*/cuda_event_query,
    /*event_sync=*/cuda_event_sync,
    /*buffer_alignment=*/cuda_buffer_alignment,
    /*buffer_max_size=*/cuda_buffer_max_size,
    /*buffer_is_host=*/cuda_buffer_is_host,
    /*buffer_alloc=*/cuda_buffer_alloc,
    /*buffer_free=*/cuda_buffer_free,
    /*buffer_flush_host=*/cuda_buffer_flush,
    /*buffer_invalidate_host=*/cuda_buffer_invalidate,
    /*buffer_set=*/cuda_buffer_set,
    /*buffer_get=*/cuda_buffer_get,
    /*buffer_copy=*/cuda_buffer_copy,
    /*buffer_clear=*/cuda_buffer_clear,
    /*pool_create=*/cuda_pool_create,
    /*pool_destroy=*/cuda_pool_destroy,
    /*pool_is_vmm=*/cuda_pool_is_vmm,
    /*pool_alloc=*/cuda_pool_alloc,
    /*pool_free=*/cuda_pool_free,
    /*kernel_compile=*/cuda_kernel_compile,
    /*kernel_load_ptx=*/cuda_kernel_load_ptx,
    /*kernel_load_host=*/cuda_kernel_load_host,
    /*kernel_destroy=*/cuda_kernel_destroy,
    /*kernel_launch=*/cuda_kernel_launch,
    /*kernel_launch_batch=*/cuda_kernel_launch_batch,
    /*event_create_timing=*/cuda_event_create_timing,
    /*event_destroy_timing=*/cuda_event_destroy,   // cudaEvent_t: same destroy as plain
    /*event_record_timing=*/cuda_event_record_timing,
    /*event_query_timing=*/cuda_event_query,       // same cudaEventQuery
    /*event_sync_timing=*/cuda_event_sync,         // same cudaEventSynchronize
    /*event_elapsed_ns=*/cuda_event_elapsed_ns,
    // record/replay (#16): NULL -> Unimplemented. A CUDA launch has no separate "record" cost
    // to amortize (it just enqueues to the stream), so there's nothing to save; the caller uses
    // launch() in its loop, already optimal here. Recording is a Vulkan-specific optimization.
    /*recorded_create=*/nullptr,
    /*recorded_replay=*/nullptr,
    /*recorded_destroy=*/nullptr,
};

}  // namespace

const ComputeBackend* cuda_vtable() noexcept { return &kCudaBackend; }

}  // namespace hyle::compute
