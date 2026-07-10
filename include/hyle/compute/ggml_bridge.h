#pragma once

#include <cstddef>

#include "hyle/compute/device.h"
#include "hyle/error/result.h"
#include "hyle/error/status.h"
#include "hyle/macros.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/blocking.h"  // BlockingPool, spawn_blocking, JoinHandle
#include "hyle/visibility.h"

#if defined(HYLE_STATIC)
#  define HYLE_COMPUTE_API
#elif defined(HYLE_COMPUTE_BUILD)
#  define HYLE_COMPUTE_API HYLE_EXPORT
#else
#  define HYLE_COMPUTE_API HYLE_IMPORT
#endif

// compute-05: ggml_bridge -- borrow ggml's operator kernels (matmul/softmax/...) to
// run under HYLE completion + scheduling. ggml is an OPERATOR-KERNEL SUPPLIER, not a
// runtime: this header exposes ZERO ggml types (no ggml_context / ggml_tensor /
// ggml_backend_t) -- all ggml coupling is sealed inside ggml_bridge.cc. See
// docs/milestones/compute-05-ggml-bridge.md + docs/architecture/
// borrowing-ggml-memory-model.md + compute-design "compute-05 借 ggml 算力". Positioning:
// 算子借 ggml, 底座替 llama.cpp, 通用 GPU 计算补 ggml 之不能 (compute-04/06).
//
// THE THREE LOAD-BEARING DECISIONS (all in the milestone spec, evidenced from source):
//  ① MEMORY OWNERSHIP = host 归 HYLE / device 归 ggml (borrowing-ggml-memory-model.md,
//     gdb-proven). ggml's CUDA backend STRUCTURALLY rejects external device pointers
//     (buffer_from_host_ptr=false, ggml-cuda.cu:5032) -- injecting a HYLE device_ptr as
//     a compute-tensor's ->data crashes on varied-shape graphs (ggml-cuda.cu:1924, null
//     buffer context from a shared size-0 dummy; see docs/debug/ggml-zerocopy-null-
//     buffer-hunt.md). So ggml OWNS the device buffers its operators compute on (gallocr,
//     mandatory on CUDA -- exactly like llama.cpp). HYLE owns the HOST side: the operator
//     API takes HOST inputs/outputs (HYLE mem, ideally compute-02b pinned staging) fed in
//     via ggml_backend_tensor_set (a bare cudaMemcpyHostToDevice, ggml-cuda.cu:688 -- ZERO
//     constraint on the host source pointer) and read back via ggml_backend_tensor_get.
//     HYLE self-managed VRAM (compute-02 Buffer/DevicePool/VMM, true zero-copy) is the
//     domain of compute-04/06 SELF-AUTHORED kernels + ② CAD/CAE data, NOT borrowed ggml.
//  ② COMPLETION via spawn_blocking (findings 3/8): ggml's CUDA stream is PRIVATE (no
//     inject API) and its only completion is blocking cudaStreamSynchronize, so the
//     whole build-graph+compute+synchronize runs on the sched-11 blocking pool; the
//     fiber suspends and is woken via the JoinHandle -- NOT GpuPoller. This is why the
//     op returns a JoinHandle<Status>, not a co_await stream.
//  ③ CPU THREAD MODEL = batching-split, n_threads CONFIGURABLE (finding 10 + 专章,
//     mirroring llama.cpp llama-context.cpp:2425-2429): a batchable big graph runs
//     ggml at full cores (the HYLE worker parks meanwhile, no contention); many
//     independent small inferences that CANNOT batch run at n_threads=1 with HYLE
//     scheduling N of them. HYLE's fiber scheduler is ALWAYS the sole core arbiter --
//     ggml is only ever a leaf it schedules (never spawns its own runtime uninvited).
//
// THREAD-SAFETY CONTRACT (spec "线程安全契约"): ggml makes NO backend-level thread-
// safety promise (global g_state + per-backend private stream/pool). A single
// GgmlBackend instance must have at most ONE in-flight graph_compute at a time. The
// "high-concurrency independent small inference" path therefore gives EACH fiber its
// OWN GgmlBackend (n_threads=1) -- no shared backend + lock (that would re-serialize
// what should be parallel).

namespace hyle::compute {

// An owned ggml backend bound to one Device (internally a ggml_backend_t + a persistent
// ggml_gallocr_t for compute-graph device buffers; opaque here). Lifetime = create once,
// reuse for many ops, free on destruction (ggml's own convention: gpt-2 example inits a
// backend once and reuses it; llama.cpp keeps one gallocr for all decode graphs).
// Move-only. Serial use only (see thread-safety contract above): for concurrent
// independent work, hold one GgmlBackend PER fiber.
class HYLE_COMPUTE_API GgmlBackend {
 public:
  GgmlBackend() noexcept = default;
  HYLE_DISABLE_COPY(GgmlBackend);
  GgmlBackend(GgmlBackend&& o) noexcept : be_(o.be_), galloc_(o.galloc_) {
    o.be_ = nullptr; o.galloc_ = nullptr;
  }
  GgmlBackend& operator=(GgmlBackend&& o) noexcept;
  ~GgmlBackend();

  // Create a ggml backend on `device`. A GPU device -> ggml CUDA backend; a CPU device
  // -> ggml CPU backend with n_threads set. A persistent gallocr (result/scratch device
  // buffers) is created alongside and reused for every op (llama.cpp model).
  //   n_threads: CPU-backend thread count (③ batching-split). Pass full cores for a
  //   batched big graph; pass 1 for high-concurrency independent small inference.
  //   IGNORED by the CUDA backend (GPU-side stream scheduling). Default = 1 (the
  //   conservative "never oversubscribe" leaf; batched callers pass full cores).
  // External-boundary failure (backend init) -> Result error.
  static Result<GgmlBackend> create(const Device& device, int n_threads = 1) noexcept;

  bool valid() const noexcept { return be_ != nullptr; }

 private:
  friend HYLE_COMPUTE_API sched::JoinHandle<Status> ggml_matmul_f32(
      GgmlBackend&, sched::BlockingPool&, mem::Allocator, const float*, const float*,
      float*, int, int, int) noexcept;
  friend HYLE_COMPUTE_API sched::JoinHandle<Status> ggml_matmul_f32_batched(
      GgmlBackend&, sched::BlockingPool&, mem::Allocator,
      const struct MatmulF32Job*, int) noexcept;
  friend HYLE_COMPUTE_API sched::JoinHandle<Status> ggml_rms_norm_f32(
      GgmlBackend&, sched::BlockingPool&, mem::Allocator, const float*, float*, int, float) noexcept;
  friend HYLE_COMPUTE_API sched::JoinHandle<Status> ggml_silu_f32(
      GgmlBackend&, sched::BlockingPool&, mem::Allocator, const float*, float*, int) noexcept;
  friend HYLE_COMPUTE_API sched::JoinHandle<Status> ggml_add_f32(
      GgmlBackend&, sched::BlockingPool&, mem::Allocator, const float*, const float*, float*, int) noexcept;
  friend HYLE_COMPUTE_API sched::JoinHandle<Status> ggml_mul_f32(
      GgmlBackend&, sched::BlockingPool&, mem::Allocator, const float*, const float*, float*, int) noexcept;
  explicit GgmlBackend(void* be, void* galloc) noexcept : be_(be), galloc_(galloc) {}
  void* raw_() const noexcept { return be_; }
  void* galloc_ptr_() const noexcept { return galloc_; }
  void* be_ = nullptr;      // ggml_backend_t
  void* galloc_ = nullptr;  // ggml_gallocr_t (compute-graph device buffers, persistent)
};

// FIRST CONSUMER = dense F32 matmul, C = A · Bᵀ. Both inputs are [*, k] row-major HOST
// data; the result is host too. This is the ggml-native matmul (ggml_mul_mat contracts
// both operands on ne0): A is [m rows × k], B is [n rows × k], C is [m rows × n] with
// C[i,j] = dot(A_row_i, B_row_j). All three are HOST buffers (HYLE mem, ideally pinned)
// -- ggml owns the DEVICE buffers internally (gallocr) and the bridge copies host->device
// (tensor_set) in and device->host (tensor_get) out. This is the host/device ownership
// boundary of ①, NOT a device-pointer API: on CUDA, ggml cannot compute on an external
// device pointer, so device Buffers are not accepted here (that is compute-04/06's domain).
//
// This is NOT a "matmul-only API" -- it is the minimal proof that "HYLE host mem + HYLE
// fiber scheduling + ggml operator (on ggml VRAM)" compose; later operators are added
// the same way inside the bridge (spec: operator facade is consumer-driven).
//
// ASYNC (②): internally spawn_blocking offloads "build ggml_context + build cgraph +
// gallocr alloc + tensor_set(H2D) + graph_compute + synchronize + tensor_get(D2H) + free"
// onto `pool`; ALL ggml objects live and die on the pool thread. The closure captures
// only the backend/gallocr handles + three HOST pointers + m/k/n (POD, like io/dns.h's
// Job). Returns JoinHandle<Status>: from a fiber, .join() is fiber-aware. `alloc` backs
// the JoinState + job. `be` AND the host buffers a/b/c MUST outlive the join (the pool
// thread reads a/b and writes c through the raw pointers -- caller guarantees).
//
// CONTRACT (caller bug -> fatal assert): be.valid(); a/b/c non-null; m/k/n > 0; a/b/c
// point at host buffers of m·k / n·k / m·n floats.
HYLE_COMPUTE_API sched::JoinHandle<Status> ggml_matmul_f32(
    GgmlBackend& be, sched::BlockingPool& pool, mem::Allocator alloc,
    const float* a, const float* b, float* c, int m, int k, int n) noexcept;

// BATCHING (finding 4, the performance lever): llama.cpp's real concurrency is NOT
// multi-stream -- it merges many requests into ONE graph and submits once. Same here:
// N independent matmuls built into ONE ggml cgraph, ONE graph_compute + ONE synchronize
// (one spawn_blocking offload), instead of N separate submits (N thread hops + N kernel
// launches). Each job is C_i = A_i · B_iᵀ with its own shape (jobs may differ in m/k/n).
//
// A job carries HOST pointers + dims -- POD, captured by value into the spawn_blocking
// closure. The caller owns the host buffers and must keep them alive across the join.
// The `jobs` array itself is COPIED into the closure (so the array need not outlive the
// await, but the buffers a/b/c it points at must).
struct MatmulF32Job {
  const float* a;  // host input A, [m rows × k] row-major
  const float* b;  // host input B, [n rows × k] row-major (C = A·Bᵀ)
  float* c;        // host output C, [m rows × n] row-major
  int m, k, n;
};
// CONTRACT: be.valid(); count in [1, 64]; every job's a/b/c non-null, dims > 0, and the
// host buffers sized for m·k / n·k / m·n floats.
HYLE_COMPUTE_API sched::JoinHandle<Status> ggml_matmul_f32_batched(
    GgmlBackend& be, sched::BlockingPool& pool, mem::Allocator alloc,
    const MatmulF32Job* jobs, int count) noexcept;

// ---- ELEMENTWISE / NORM operators (compute-09 step-1: prove the op face is not a toy) ----
// compute-05 exposed ONLY matmul as the "bridge mechanism proof". These add the next
// transformer-block operators (rms_norm / silu / add / mul) via a SHARED internal graph
// executor -- the whole build-context + gallocr-alloc + H2D + compute + D2H + free boilerplate
// is now written ONCE; each operator contributes only its build-graph line. That is what makes
// "borrow more ggml operators" cheap and non-toy (NOT "one hand-copied function per op"). These
// four are WEIGHTLESS activation-shaped ops (n floats in -> n floats out), so they follow
// matmul's "copy input each call" model with no weight-residency machinery (that lands with the
// decoder block, step-2). Same async contract as matmul: internally spawn_blocking, returns
// JoinHandle<Status>; host buffers must outlive the join. n > 0; buffers non-null (fatal assert).
//
// dst[i] = rms_norm(src)[i] * ... : RMS-normalize `src` over its last dim of `n` elements
// (a single row). eps is the ggml_rms_norm epsilon. src/dst are host float[n]. (No weight
// multiply here -- that is a separate ggml_mul with the norm weight, composed at the block level.)
HYLE_COMPUTE_API sched::JoinHandle<Status> ggml_rms_norm_f32(
    GgmlBackend& be, sched::BlockingPool& pool, mem::Allocator alloc,
    const float* src, float* dst, int n, float eps) noexcept;

// dst[i] = silu(src[i]) = src[i] / (1 + exp(-src[i])). FFN activation. Host float[n].
HYLE_COMPUTE_API sched::JoinHandle<Status> ggml_silu_f32(
    GgmlBackend& be, sched::BlockingPool& pool, mem::Allocator alloc,
    const float* src, float* dst, int n) noexcept;

// dst[i] = a[i] + b[i] (residual). Host float[n] each.
HYLE_COMPUTE_API sched::JoinHandle<Status> ggml_add_f32(
    GgmlBackend& be, sched::BlockingPool& pool, mem::Allocator alloc,
    const float* a, const float* b, float* dst, int n) noexcept;

// dst[i] = a[i] * b[i] (elementwise; norm-weight apply / FFN gate). Host float[n] each.
HYLE_COMPUTE_API sched::JoinHandle<Status> ggml_mul_f32(
    GgmlBackend& be, sched::BlockingPool& pool, mem::Allocator alloc,
    const float* a, const float* b, float* dst, int n) noexcept;

}  // namespace hyle::compute
