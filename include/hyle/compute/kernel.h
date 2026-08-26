#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>       // memcpy (typed shell packs scalars into a stack blob)
#include <type_traits>   // is_same_v / is_trivially_copyable_v (compile-time arg classify)
#include <utility>       // forward

#include "hyle/compute/buffer.h"  // Buffer -- the typed shell classifies args as Buffer vs scalar
#include "hyle/compute/device.h"
#include "hyle/compute/stream.h"
#include "hyle/compute/types.h"
#include "hyle/error/result.h"
#include "hyle/core/compiler.h"
#include "hyle/mem/alignment.h"   // align_up (scalar blob field alignment)
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
  Kernel(const Kernel&) = delete;
  Kernel& operator=(const Kernel&) = delete;
  HYLE_COMPUTE_API Kernel(Kernel&& o) noexcept;
  HYLE_COMPUTE_API Kernel& operator=(Kernel&& o) noexcept;
  HYLE_COMPUTE_API ~Kernel();

  // ① Runtime-compile a CUDA C `source` string into a launchable kernel (NVRTC ->
  // PTX -> module load). `entry` is the kernel function name: an `extern "C"` name
  // (used verbatim) or a C++ name expression (resolved via NVRTC lowered-name, so
  // templated/mangled kernels work). The device's compute capability is auto-probed
  // for -arch=sm_XX. A same-source recompile hits an in-process cache (XXH3(source) ^
  // arch). Compile failure returns a Status carrying the NVRTC log.
  // An invalid Device returns InvalidArgument.
  //
  // On the host backend `source` compilation isn't meaningful -> returns Unimplemented
  // (host kernels are registered via load_host, below); a GPU-less test uses load_host.
  HYLE_COMPUTE_API static Result<Kernel> compile(const Device& device,
                                                 const char* source,
                                                 const char* entry) noexcept;

  // ② Load an already-compiled PTX string (skips NVRTC), for a consumer that has PTX (CUDA).
  // `entry` is the kernel name in the PTX (extern "C" / already-lowered). PTX is text, so no
  // length is needed. For a Vulkan SPIR-V module use load_spirv() instead.
  // An invalid Device returns InvalidArgument.
  HYLE_COMPUTE_API static Result<Kernel> load_ptx(const Device& device,
                                                  const char* ptx,
                                                  const char* entry) noexcept;

  // ②' Load a pre-compiled SPIR-V module (Vulkan). `blob` is self-describing (words + byte
  // size), obtained EITHER at compile time from a tools/hyle_shader.py-generated header
  // (`inline constexpr SpirvBlob`, #include'd) OR at runtime by reading a .spv file and
  // pointing a SpirvBlob at the bytes -- the SAME call serves both, so the loading strategy
  // is the consumer's choice, not fixed by us. `entry` is the entry-point name ("main" for
  // glslc/glslang output). On a non-Vulkan backend this returns Unimplemented (CUDA takes
  // PTX via load_ptx). SPIR-V is not "PTX": the split keeps each backend's format explicit.
  // An invalid Device returns InvalidArgument.
  HYLE_COMPUTE_API static Result<Kernel> load_spirv(const Device& device,
                                                    const SpirvBlob& blob,
                                                    const char* entry = "main") noexcept;

  // ②'' load_spirv WITH SPECIALIZATION CONSTANTS (checklist #2/#3): shader constants resolved
  // at pipeline-BUILD time (baked into the compiled ISA: dead-code elimination, loop
  // unrolling, a workgroup size the driver knows), NOT per-launch push constants. `specs`
  // maps each GLSL `constant_id` to a slice of `spec_data` (id + byte offset/size),
  // `spec_count` of them. The values are FROZEN into this Kernel's pipeline; changing them
  // needs a NEW Kernel (that immutability is the point). Vulkan-only (CUDA's build-time analog
  // is compile()'s NVRTC macros -> Unimplemented when spec_count > 0).
  HYLE_COMPUTE_API static Result<Kernel> load_spirv(const Device& device,
                                                    const SpirvBlob& blob, const char* entry,
                                                    const SpecConstant* specs,
                                                    unsigned spec_count,
                                                    const void* spec_data,
                                                    unsigned spec_data_size) noexcept;

  // ③ Host fallback: register a host function as a "kernel". `fn` is called by launch
  // with the config + KernelArgs (fn reads args.buffers / args.scalars itself). Only
  // meaningful on the host backend; on a GPU device returns Unimplemented (use compile/
  // load_ptx there); an invalid Device returns InvalidArgument. This is what lets kernel
  // launch be exercised with no GPU.
  using HostFn = void (*)(const LaunchConfig&, const KernelArgs&) noexcept;
  HYLE_COMPUTE_API static Result<Kernel> load_host(const Device& device,
                                                   HostFn fn) noexcept;

  bool valid() const noexcept { return vt_ != nullptr; }

 private:
  friend HYLE_COMPUTE_API Status launch(Stream&, const Kernel&, const LaunchConfig&,
                                        const KernelArgs&) noexcept;
  friend HYLE_COMPUTE_API Status launch_batch(Stream&, const struct BatchItem*,
                                              unsigned) noexcept;
  friend class RecordedLaunch;  // record() reads module_/func_/vt_ (same as launch)
  const ComputeBackend* vt_ = nullptr;  // backend that owns the handles (null=invalid)
  void* module_ = nullptr;              // CUmodule (CUDA) / unused (host)
  void* func_ = nullptr;                // CUfunction (CUDA) / HostFn (host)
  int ordinal_ = 0;
};

// ---- C-ABI CORE (exported, one compiled copy) -------------------------------
// Submit `kernel` to `stream` with `config` and pre-built `args` (KernelArgs, types.h)
// -- the UNIFIED launch ABI that lowers to BOTH backends: on CUDA `order` preserves an
// optional mixed pointer/scalar source order (omitting it retains the raw API's legacy
// buffers-then-scalars convention); on Vulkan the buffers bind to descriptor slots 0..N
// and the scalar blob becomes push constants. `config.grid` is the grid / workgroup
// count (same ceil(n/block) math both sides); `config.block` is the CUDA block dim
// (Vulkan's local_size lives in the shader). The KernelArgs' arrays are caller-owned
// (zero heap) and must outlive this call. ⚠️ LIFETIME: the kernel runs ASYNCHRONOUSLY --
// the Kernel object and every backing device Buffer must stay alive until the launch
// COMPLETES (co_await StreamComplete / stream.sync()), NOT merely until launch() returns.
// On Vulkan the GPU reads these objects after launch() has returned; destroying a Kernel or
// Buffer right after launch is a use-after-free. ASYNC: launch returns after submit; observe
// completion with `co_await StreamComplete{&stream}` (compute-03). A
// launch/config error surfaces as Status (CUDA: cudaGetLastError; Vulkan: submit result).
// On the host backend, calls the registered HostFn synchronously.
//
// ⚠️ CONTRACT (caller bugs). How a violation surfaces differs by backend, because their
// underlying guarantees differ -- this is documented honestly, not a uniform promise:
//   - config.grid (workgroup count) must be <= the device's max per dim (Vulkan
//     maxComputeWorkGroupCount / CUDA maxGridSize). grid = ceil(n / block_size); use
//     blocks_for(). VULKAN: exceeding it is undefined behavior with no driver guard, so the
//     backend HYLE_FATAL_ASSERTs (fail-fast). CUDA: the driver validates the launch config and
//     an over-limit grid comes back as an error Status (cudaGetLastError), not an assert.
//   - the packed scalar blob must be <= the device push-constant limit. The typed shell
//     caps its stack blob at 256 B; a Vulkan device with a smaller limit returns
//     InvalidArgument. CUDA passes scalars as kernel params (no push-constant limit).
//   - relaunching the SAME Kernel must keep the SAME arg SHAPE (buffer count + scalar bytes):
//     the pipeline layout is built once from the first launch and frozen -- Vulkan
//     HYLE_FATAL_ASSERTs a shape mismatch. (CUDA has no frozen layout; args rebuild per call.)
//   - a Stream is single-execution-flow: do not launch on the SAME Stream concurrently from
//     multiple fibers/threads (see stream.h). Use one Stream per fiber for parallelism. This
//     one CANNOT be asserted (a data race is not locally observable) -- it is a hard caller
//     contract on every backend.
//   - raw KernelArgs contains opaque native handles, not Buffer ownership metadata: every
//     handle must belong to the Stream's backend device. The typed shell checks this.
//
// Most callers use the typed `launch(stream, kernel, config, args...)` shell below, which
// builds KernelArgs at compile time; this raw form is for a caller that already has one.
HYLE_COMPUTE_API Status launch(Stream& stream, const Kernel& kernel,
                               const LaunchConfig& config, const KernelArgs& args) noexcept;

// ---- BATCHED launch (compute-06 finding: batching = the measured perf lever) ----------
// Submit N (kernel, config, args) items in ONE backend submit instead of N -- on Vulkan
// this is ONE command buffer + ONE fence for all N dispatches (省 N-1 submits+fences, the
// 1.07x+ lever that grows with more/smaller ops, compute-05 finding 4). On CUDA/host it
// loops (their stream already orders sequential submits at no per-submit cost). Each item
// carries a Kernel + its LaunchConfig + pre-built KernelArgs; the args' arrays are caller-
// owned and must outlive the call. The referenced Kernels and Buffers must remain alive until
// the batch completes. ASYNC on Vulkan (co_await stream to await); errors return the first
// item's failure.
struct BatchItem {
  const Kernel* kernel;
  LaunchConfig config;
  KernelArgs args;
};
HYLE_COMPUTE_API Status launch_batch(Stream& stream, const BatchItem* items,
                                     unsigned count) noexcept;

// ---- RECORD / REPLAY (compute-06 checklist #16) ------------------------------
// For an ITERATIVE workload that runs the SAME (kernel, config, args) many times -- an
// iterative solver applying one stencil/SpMV thousands of times, a per-frame compute pass --
// the recorded GPU commands are identical every iteration; only the BUFFER CONTENTS change.
// RecordedLaunch records those commands ONCE (bindings + push constants + dispatch dims frozen)
// and replays them with no re-recording, saving that per-iteration CPU (kompute's record-once /
// eval-many). On Vulkan this reuses one command buffer across submits; on CUDA/host it falls
// back to a plain launch per replay (same result, no CPU saving -- their submit has no record
// cost). Move-only; owns the recorded command buffer + descriptor pool until destroyed.
//
// CONTRACT:
//   - Record AND replay on the SAME Stream (the command buffer belongs to its pool).
//   - The RecordedLaunch MUST be destroyed BEFORE the Stream it was recorded on. Its recorded
//     command buffer + descriptor pool live in that Stream's command pool; ~RecordedLaunch
//     frees them back to that pool, so outliving the Stream is a use-after-free (the pool is
//     already gone). Same ownership rule as an iterator vs its container -- keep the
//     RecordedLaunch in a scope nested inside the Stream's.
//   - The Kernel and the Buffers referenced by `args` must outlive the RecordedLaunch (their
//     handles are frozen into the recording). To feed new data, write into those SAME Buffers
//     between replays; to change a push-constant value or SWAP a buffer, make a NEW recording.
//   - Each replay is ASYNC (like launch). Vulkan permits multiple outstanding replays of the
//     same recording; queue order and recorded barriers preserve their ordering. Await via
//     `co_await StreamComplete` or stream.sync() before host access or lifetime teardown.
class RecordedLaunch {
 public:
  RecordedLaunch() noexcept = default;
  RecordedLaunch(const RecordedLaunch&) = delete;
  RecordedLaunch& operator=(const RecordedLaunch&) = delete;
  HYLE_COMPUTE_API RecordedLaunch(RecordedLaunch&& o) noexcept;
  HYLE_COMPUTE_API RecordedLaunch& operator=(RecordedLaunch&& o) noexcept;
  HYLE_COMPUTE_API ~RecordedLaunch();

  // Record `kernel` with `config` + `args` on `stream` into a reusable command buffer. The
  // pipeline is built (once, lazily) if needed. Returns Unimplemented on a backend with no
  // recording benefit (CUDA/host: their submit has no record cost to amortize) -- the caller
  // should just use launch() in its loop there (there is NO automatic fallback; check the
  // Result). `args` arrays are read during recording only (the backend copies what it needs);
  // the referenced Buffers must stay alive for the RecordedLaunch's lifetime.
  HYLE_COMPUTE_API static Result<RecordedLaunch> record(Stream& stream, const Kernel& kernel,
                                                        const LaunchConfig& config,
                                                        const KernelArgs& args) noexcept;

  // Submit the recorded commands once (async). Call repeatedly for each iteration.
  HYLE_COMPUTE_API Status replay(Stream& stream) noexcept;

  bool valid() const noexcept { return vt_ != nullptr; }

 private:
  const ComputeBackend* vt_ = nullptr;  // backend that owns `recorded_` (null = invalid)
  void* recorded_ = nullptr;            // backend recording handle (VkRecorded* / opaque)
  void* stream_ = nullptr;              // the stream it was recorded on (destroy/replay target)
};

// ---- TYPED SHELL (header-only, compile-time, zero heap) ----------------------
// The ergonomic launch: pass kernel arguments DIRECTLY and in source order, including
// mixed pointer/scalar signatures such as `launch(s,k,cfg,a,n,b,alpha)`.
// A variadic fold classifies each arg at COMPILE TIME (conventions §7 typed shell over a
// type-erased C-ABI core, same as PodVector): a `Buffer` -> its native_handle() appended to an
// ordered buffer list (position = binding index); anything trivially-copyable -> memcpy'd
// into a naturally-aligned byte blob (with a {offset,size} descriptor) and emits an
// argument-order table so CUDA reconstructs the exact original signature. Vulkan pushes
// the whole blob; its shader declaration must match the C++ byte layout (use raw KernelArgs
// for explicitly authored offsets). All storage is STACK; nothing hits the heap.
//
// This resolves the CUDA(void**)-vs-Vulkan(descriptor+push) split ENTIRELY at compile time
// -- the consumer writes ONE launch call that works on every backend (compute-04 CUDA /
// compute-06 Vulkan / compute-08 HIP), which is compute-04's "unified public interface".
namespace detail {
// A scalar arg is anything trivially-copyable that is NOT a Buffer.
template <class T>
inline constexpr bool is_kernel_scalar_v =
    std::is_trivially_copyable_v<std::remove_cvref_t<T>> &&
    !std::is_same_v<std::remove_cvref_t<T>, Buffer>;
// A bounded typed-shell scratch blob, not a portable Vulkan capability claim. The backend
// checks the selected device's actual maxPushConstantsSize (Vulkan guarantees at least
// 128 B) and returns InvalidArgument when this 256 B convenience ceiling is too large for
// that device. Larger parameter blocks belong in a Buffer.
inline constexpr std::size_t kMaxScalarBlob = 256;
}  // namespace detail

// Constrained so a SINGLE KernelArgs arg never matches here (it must bind the C-ABI core
// above) -- without this, `launch(s,k,cfg, ka)` would re-match the template as
// Args=KernelArgs& and recurse forever. Any other arg list (buffers/scalars) uses the shell.
template <class... Args>
  requires (!(sizeof...(Args) == 1 &&
              (std::is_same_v<std::remove_cvref_t<Args>, KernelArgs> && ...)))
Status launch(Stream& stream, const Kernel& kernel, const LaunchConfig& config,
              Args&&... args) noexcept {
  constexpr std::size_t kN = sizeof...(Args);
  void* buffers[kN == 0 ? 1 : kN];             // ordered native handles (stack)
  alignas(16) std::byte blob[detail::kMaxScalarBlob];  // packed scalar bytes (stack)
  ScalarField fields[kN == 0 ? 1 : kN];        // per-scalar {offset,size} (stack)
  KernelArgRef order[kN == 0 ? 1 : kN];        // exact mixed source order (stack)
  unsigned nb = 0, nf = 0, na = 0, blob_off = 0;

  // Fold over the arg pack: classify + pack each, in order (conventions §7 + the
  // renderer-Mesh-validated variadic-fold + byte-pack + descriptor-array pattern).
  auto pack_one = [&](auto&& a) noexcept {
    using A = std::remove_cvref_t<decltype(a)>;
    if constexpr (std::is_same_v<A, Buffer>) {
      HYLE_FATAL_ASSERT(a.valid() && a.belongs_to(stream.device()));
      order[na++] = KernelArgRef{KernelArgKind::kBuffer, nb};
      buffers[nb++] = a.native_handle();          // Buffer -> ordered buffer list
    } else {
      static_assert(detail::is_kernel_scalar_v<A>,
                    "launch arg must be a Buffer or a trivially-copyable scalar");
      static_assert(alignof(A) <= 16,
                    "over-aligned kernel values must be passed through a Buffer");
      // Preserve the C++ value's natural alignment. `sizeof(T)` is not an alignment
      // (12-byte structs are the counterexample) and is not necessarily a power of two.
      const std::size_t sz = sizeof(A);
      constexpr std::size_t al = alignof(A);
      blob_off = static_cast<unsigned>(mem::align_up(blob_off, al));
      HYLE_FATAL_ASSERT(blob_off + sz <= detail::kMaxScalarBlob);  // scalars overflow -> use a Buffer
      std::memcpy(blob + blob_off, &a, sz);
      order[na++] = KernelArgRef{KernelArgKind::kScalar, nf};
      fields[nf++] = ScalarField{blob_off, static_cast<unsigned>(sz)};
      blob_off += static_cast<unsigned>(sz);
    }
  };
  (pack_one(std::forward<Args>(args)), ...);

  KernelArgs ka{buffers, nb, nf ? blob : nullptr, blob_off,
                nf ? fields : nullptr, nf, na ? order : nullptr, na};
  return launch(stream, kernel, config, ka);   // -> C-ABI core (one indirect + backend)
}

}  // namespace hyle::compute
