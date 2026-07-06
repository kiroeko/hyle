#pragma once

#include <cstddef>
#include <cstdint>

#include "hyle/compute/device.h"
#include "hyle/compute/stream.h"
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

// compute-04: kernel launch -- submit an ARBITRARY kernel to a GPU stream. This is
// ggml's net-new gap (ggml is a fixed op set + graph_compute, all compile-time
// <<<>>>; it has no "launch any kernel" API). The base is the DRIVER cuLaunchKernel
// (CUfunction + grid/block/shmem/stream + void** args) -- the only path that takes an
// arbitrary kernel (compile-time <<<>>> can't) and is isomorphic with compute-06's
// SPIR-V "module + entry + args" model.
//
// B-TIER (full): the whole runtime-compile chain -- CUDA C source string -> NVRTC ->
// PTX -> cuModuleLoadData -> cuModuleGetFunction -> cuLaunchKernel. The NVRTC chain is
// ported from CCCL (ggml has zero nvrtc; CCCL's c/parallel is the reference). Launch
// engineering (grid/block, launch_bounds, cudaGetLastError fail-fast) is from ggml.
//
// ASYNC: launch submits and returns (like io submit / ggml <<<>>>); completion is
// observed via compute-03's `co_await StreamComplete{&stream}`. Kernel scratch (a
// kernel's transient device buffers) uses compute-02c's DevicePool -- that pool's
// real consumer is here.
//
// HOST FALLBACK: a Kernel on the host backend is a registered host function pointer;
// launch calls it directly (no GPU) so the abstraction compiles + tests anywhere.

namespace hyle::compute {

struct ComputeBackend;  // internal vtable (backend.h); forward-declared, not exposed

// Dim3 / LaunchConfig live in types.h (POD, shared with the CUDA backend TU without
// pulling std::expected). Blocks needed to cover `n` elements at `block_size`
// threads/block -- the (n + bs - 1) / bs idiom ggml hand-writes at every launch site
// (clamp.cu:20 etc.); HYLE provides it once. constexpr: usable in a LaunchConfig init.
constexpr unsigned blocks_for(std::size_t n, unsigned block_size) noexcept {
  return block_size == 0 ? 0 : static_cast<unsigned>((n + block_size - 1) / block_size);
}

// A loaded, launchable kernel. Backend-opaque handles held inline ({vt, module,
// func} -- no pimpl, same discipline as Stream/Buffer). Move-only (owns the backend
// module). The public header never names CUmodule/CUfunction (void* handles).
class Kernel {
 public:
  Kernel() noexcept = default;
  HYLE_DISABLE_COPY(Kernel);
  HYLE_COMPUTE_API Kernel(Kernel&& o) noexcept;
  HYLE_COMPUTE_API Kernel& operator=(Kernel&& o) noexcept;
  HYLE_COMPUTE_API ~Kernel();

  // ① Runtime-compile a CUDA C `source` string into a launchable kernel (NVRTC ->
  // PTX -> module load). `entry` is the kernel function name: an `extern "C"` name
  // (used verbatim) or a C++ name expression (resolved via NVRTC lowered-name, so
  // templated/mangled kernels work). The device's compute capability is auto-probed
  // for -arch=sm_XX. A same-source recompile hits an in-process cache (XXH3(source) ^
  // arch). Compile failure returns a Status carrying the NVRTC log.
  //
  // On the host backend `source` compilation isn't meaningful -> returns Unimplemented
  // (host kernels are registered via load_host, below); a GPU-less test uses load_host.
  HYLE_COMPUTE_API static Result<Kernel> compile(const Device& device,
                                                 const char* source,
                                                 const char* entry) noexcept;

  // ② Load an already-compiled PTX string (skips NVRTC), for a consumer that has PTX.
  // `entry` is the kernel name in the PTX (extern "C" / already-lowered).
  HYLE_COMPUTE_API static Result<Kernel> load_ptx(const Device& device,
                                                  const char* ptx,
                                                  const char* entry) noexcept;

  // ③ Host fallback: register a host function as a "kernel". `fn` is called by launch
  // with the args array (fn interprets void** itself). Only meaningful on the host
  // backend; on a GPU device returns Unimplemented (use compile/load_ptx there). This
  // is what lets kernel launch be exercised with no GPU.
  using HostFn = void (*)(const LaunchConfig&, void** args) noexcept;
  HYLE_COMPUTE_API static Result<Kernel> load_host(const Device& device,
                                                   HostFn fn) noexcept;

  bool valid() const noexcept { return vt_ != nullptr; }

 private:
  friend HYLE_COMPUTE_API Status launch(Stream&, const Kernel&, const LaunchConfig&,
                                        void**) noexcept;
  const ComputeBackend* vt_ = nullptr;  // backend that owns the handles (null=invalid)
  void* module_ = nullptr;              // CUmodule (CUDA) / unused (host)
  void* func_ = nullptr;                // CUfunction (CUDA) / HostFn (host)
  int ordinal_ = 0;
};

// Submit `kernel` to `stream` with `config` and `args` -- driver cuLaunchKernel
// (CUfunction, grid, block, shmem, stream, args, extra=0). `args` is an array of
// pointers to the kernel's arguments (each buffer's device_ptr() address / each
// scalar's address), the cuLaunchKernel void** ABI; its length/types must match the
// kernel signature (caller contract). ASYNC: returns after submit; observe completion
// with `co_await StreamComplete{&stream}` (compute-03). A launch-config error surfaces
// as Status (checked via cudaGetLastError after submit, ggml fail-fast contract). On
// the host backend, calls the registered HostFn synchronously.
HYLE_COMPUTE_API Status launch(Stream& stream, const Kernel& kernel,
                               const LaunchConfig& config, void** args) noexcept;

}  // namespace hyle::compute
