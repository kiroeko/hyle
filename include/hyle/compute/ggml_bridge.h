#pragma once

#include <cstddef>

#include "hyle/compute/buffer.h"
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
// run on HYLE-managed VRAM + completion + scheduling. ggml is an OPERATOR-KERNEL
// SUPPLIER, not a runtime: this header exposes ZERO ggml types (no ggml_context /
// ggml_tensor / ggml_backend_t) -- all ggml coupling is sealed inside ggml_bridge.cc.
// See docs/milestones/compute-05-ggml-bridge.md + compute-design "compute-05 借 ggml
// 算力：调研发现汇总" (findings 1-10 + CPU 执行模型专章). Positioning: 算子借 ggml,
// 底座替 llama.cpp, 通用 GPU 计算补 ggml 之不能 (compute-04/06).
//
// THE THREE LOAD-BEARING DECISIONS (all in the milestone spec, evidenced from source):
//  ① ZERO-COPY (finding 2): a HYLE Buffer's device_ptr() is injected straight into a
//     no_alloc ggml tensor's ->data; ggml's ordinary CUDA operators read data without
//     validating buffer ownership, so no extra allocation/copy. F32 only for now.
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

// An owned ggml backend bound to one Device (internally a ggml_backend_t; opaque
// here). Lifetime = create once, reuse for many ops, free on destruction (ggml's own
// convention: gpt-2 example inits a backend once and reuses it). Move-only (owns the
// ggml_backend_t). Serial use only (see thread-safety contract above): for concurrent
// independent work, hold one GgmlBackend PER fiber.
class HYLE_COMPUTE_API GgmlBackend {
 public:
  GgmlBackend() noexcept = default;
  HYLE_DISABLE_COPY(GgmlBackend);
  GgmlBackend(GgmlBackend&& o) noexcept : be_(o.be_) { o.be_ = nullptr; }
  GgmlBackend& operator=(GgmlBackend&& o) noexcept;
  ~GgmlBackend();

  // Create a ggml backend on `device`. A GPU device -> ggml CUDA backend (zero-copy
  // device_ptr path). A CPU device -> ggml CPU backend with n_threads set.
  //   n_threads: CPU-backend thread count (③ batching-split). Pass full cores for a
  //   batched big graph; pass 1 for high-concurrency independent small inference.
  //   IGNORED by the CUDA backend (GPU-side stream scheduling). Default = 1 (the
  //   conservative "never oversubscribe" leaf; batched callers pass full cores).
  // External-boundary failure (backend init) -> Result error.
  static Result<GgmlBackend> create(const Device& device, int n_threads = 1) noexcept;

  bool valid() const noexcept { return be_ != nullptr; }

 private:
  friend HYLE_COMPUTE_API sched::JoinHandle<Status> ggml_matmul_f32(
      GgmlBackend&, sched::BlockingPool&, mem::Allocator, const Buffer&,
      const Buffer&, Buffer&, int, int, int) noexcept;
  explicit GgmlBackend(void* be) noexcept : be_(be) {}
  void* raw_() const noexcept { return be_; }  // for the bridge free functions
  void* be_ = nullptr;  // ggml_backend_t (opaque; real type only in the .cc)
};

// FIRST CONSUMER = dense F32 matmul: C(m×n) = A(m×k) · B(k×n), all three HYLE device
// Buffers (their device_ptr() is zero-copy-injected into ggml tensors, ①). This is
// NOT a "matmul-only API" -- it is the minimal proof that "HYLE VRAM + HYLE fiber
// scheduling + ggml operator" compose; later operators are added the same way inside
// the bridge (spec: operator facade is consumer-driven, not built up front).
//
// ASYNC (②): internally spawn_blocking offloads "build no_alloc ggml_context + build
// cgraph + ggml_backend_graph_compute + synchronize + free context" onto `pool`; ALL
// ggml objects live and die on the pool thread (nothing escapes to the fiber thread).
// The closure captures only the backend handle + three device_ptrs + m/k/n (POD, like
// io/dns.h's POD Job). Returns JoinHandle<Status>: from a fiber, .join() is fiber-
// aware (the worker runs other fibers while the pool thread blocks in ggml). `alloc`
// backs the JoinState + job (use the allocator you join from). `be` MUST outlive the
// join (backend is persistent; caller guarantees).
//
// CONTRACT (caller bug -> fatal assert, not silent): be.valid(); a/b/c valid and sized
// for m·k / k·n / m·n F32 elements; a/b/c on the same device as `be`.
HYLE_COMPUTE_API sched::JoinHandle<Status> ggml_matmul_f32(
    GgmlBackend& be, sched::BlockingPool& pool, mem::Allocator alloc,
    const Buffer& a, const Buffer& b, Buffer& c, int m, int k, int n) noexcept;

}  // namespace hyle::compute
