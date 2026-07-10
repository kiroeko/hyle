// compute-05: ggml_bridge -- ALL ggml coupling is sealed in this one TU (spec ⑤:
// adapter single-point isolation). ggml types (ggml_context/ggml_tensor/
// ggml_backend_t) appear ONLY here; the public header exposes none. A ggml upgrade
// touches this file + the golden-compare tests, nothing else.
//
// See docs/milestones/compute-05-ggml-bridge.md + docs/architecture/
// borrowing-ggml-memory-model.md + compute-design "compute-05 借 ggml 算力". This file
// implements the three load-bearing decisions from ggml_bridge.h: ① host/device memory
// ownership boundary, ② completion via spawn_blocking, ③ CPU thread model = n_threads.

#include "hyle/compute/ggml_bridge.h"

#include <cstdint>
#include <utility>

#include "hyle/macros.h"

// ggml headers -- CONFINED to this .cc (never in ggml_bridge.h). ggml-cuda.h is pure
// C declarations (no CUDA toolkit headers), safe to include from this plain C++ TU.
// ggml does NOT export GGML_USE_CUDA to consumers, so gate on HYLE_COMPUTE_CUDA (set
// PRIVATE on hyle_compute by src/compute/CMakeLists.txt when the toolkit was found).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#if defined(HYLE_COMPUTE_CUDA)
#  include "ggml-cuda.h"
#endif

namespace hyle::compute {

namespace {

// Reinterpret the opaque void* backend handle as ggml's type, local to this TU.
inline ggml_backend_t as_be(void* p) noexcept {
  return static_cast<ggml_backend_t>(p);
}

// One matmul's raw HOST inputs + dims -- the internal POD, mirrors the public
// MatmulF32Job but stays TU-local so the header's struct isn't dragged in here.
struct MmJob {
  const float* a;
  const float* b;
  float* c;
  int m, k, n;
};

// The blocking half: runs ENTIRELY on the sched-11 pool thread (spec ②). Builds ONE
// no_alloc ggml_context holding ALL `count` matmuls, lets ggml's gallocr allocate REAL
// device buffers for every tensor (spec ①: device 归 ggml -- CUDA structurally rejects
// external device pointers, buffer_from_host_ptr=false; see borrowing-ggml-memory-
// model.md), copies HYLE HOST inputs in via ggml_backend_tensor_set (a bare
// cudaMemcpyHostToDevice with ZERO constraint on the host source, ggml-cuda.cu:688),
// builds ONE cgraph over all results, ONE graph_compute + ONE synchronize (finding 4:
// batching = one submit for many ops), then reads results back to HYLE HOST via
// tensor_get. count==1 is the single-matmul case. Nothing ggml escapes this function.
//
// GGML MATMUL SEMANTICS (verified ggml.h:1415-1417): ggml_mul_mat(A,B) with A=[k,m],
// B=[k,n] -> C=[m,n], C = A·Bᵀ (both contract on ne0). Per job: gA=[k,m] host src a,
// gB=[k,n] host src b, gC=[m,n] read back to host c.
//
// MEMORY MODEL (spec ①, gdb-proven -- docs/debug/ggml-zerocopy-null-buffer-hunt.md):
// this is the llama.cpp-faithful path. Inputs are marked ggml_set_input / the result
// ggml_set_output so gallocr allocates them real buffers with valid CUDA context (the
// earlier size-0 dummy-buffer injection nulled that context and crashed varied-shape
// batches at ggml-cuda.cu:1924 -- deleted). ggml owns the device buffers; HYLE owns the
// host data on both ends.
Status matmul_batch_blocking(ggml_backend_t be, ggml_gallocr_t galloc,
                             const MmJob* jobs, int count) noexcept {
  const size_t mem_size =
      ggml_tensor_overhead() * (static_cast<size_t>(count) * 3 + 4) + ggml_graph_overhead();
  ggml_init_params params = {mem_size, nullptr, /*no_alloc=*/true};
  ggml_context* ctx = ggml_init(params);
  if (ctx == nullptr) return Status::ResourceExhausted("ggml_init failed");

  ggml_cgraph* gf = ggml_new_graph(ctx);
  ggml_tensor* inA[64];
  ggml_tensor* inB[64];
  ggml_tensor* results[64];
  for (int i = 0; i < count; ++i) {
    ggml_tensor* gA = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, jobs[i].k, jobs[i].m);
    ggml_tensor* gB = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, jobs[i].k, jobs[i].n);
    ggml_set_input(gA);  // gallocr allocates a real device buffer (valid CUDA context)
    ggml_set_input(gB);
    ggml_tensor* gC = ggml_mul_mat(ctx, gA, gB);
    ggml_set_output(gC);  // gallocr won't alias the result away before tensor_get
    inA[i] = gA;
    inB[i] = gB;
    results[i] = gC;
    ggml_build_forward_expand(gf, gC);
  }

  // ggml owns ALL device buffers for this graph (inputs + results). CUDA-mandatory.
  if (!ggml_gallocr_alloc_graph(galloc, gf)) {
    ggml_free(ctx);
    return Status::ResourceExhausted("ggml_gallocr_alloc_graph failed");
  }

  // Copy HYLE host inputs into ggml's device tensors (H2D). The host source pointer has
  // no ggml constraint (ggml-cuda.cu:688) -- it is HYLE mem (ideally pinned staging).
  for (int i = 0; i < count; ++i) {
    ggml_backend_tensor_set(inA[i], jobs[i].a, 0, ggml_nbytes(inA[i]));
    ggml_backend_tensor_set(inB[i], jobs[i].b, 0, ggml_nbytes(inB[i]));
  }

  ggml_status ec = ggml_backend_graph_compute(be, gf);
  if (ec == GGML_STATUS_SUCCESS) {
    for (int i = 0; i < count; ++i) {
      ggml_backend_tensor_get(results[i], jobs[i].c, 0, ggml_nbytes(results[i]));  // D2H
    }
  }
  ggml_free(ctx);
  if (ec != GGML_STATUS_SUCCESS) {
    return Status::Internal("ggml_backend_graph_compute failed");
  }
  return Status();  // OK
}

// ---- generic elementwise/norm executor (compute-09 step-1) --------------------------
// The SHARED graph executor: everything matmul_batch_blocking does EXCEPT the operator line
// is boilerplate, written once here. A weightless activation-shaped op (n floats -> n floats)
// takes 1 or 2 host inputs of length n and produces 1 host output of length n. The op is
// selected by an enum; the ONLY op-specific code is the `build` switch below (one ggml call
// each). This is what makes adding the next operator cheap -- no 50-line copy per op.
enum class EwOp { kRmsNorm, kSilu, kAdd, kMul };

// One elementwise job's raw HOST buffers + dims (TU-local POD, like MmJob).
struct EwJob {
  const float* a;   // input 0 (host)
  const float* b;   // input 1 (host) or nullptr for unary
  float* out;       // output (host)
  int n;            // element count (single row)
  float eps;        // rms_norm epsilon (ignored by others)
  EwOp op;
};

Status elementwise_blocking(ggml_backend_t be, ggml_gallocr_t galloc, const EwJob& j) noexcept {
  const bool binary = (j.op == EwOp::kAdd || j.op == EwOp::kMul);
  // tensor overhead: 2 inputs (a[,b]) + 1 result, + graph. Small fixed budget.
  const size_t mem_size = ggml_tensor_overhead() * 4 + ggml_graph_overhead();
  ggml_init_params params = {mem_size, nullptr, /*no_alloc=*/true};
  ggml_context* ctx = ggml_init(params);
  if (ctx == nullptr) return Status::ResourceExhausted("ggml_init failed");

  ggml_cgraph* gf = ggml_new_graph(ctx);
  ggml_tensor* gA = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, j.n);
  ggml_set_input(gA);  // gallocr gives it a real device buffer (valid CUDA context)
  ggml_tensor* gB = nullptr;
  if (binary) { gB = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, j.n); ggml_set_input(gB); }

  // THE ONLY OP-SPECIFIC LINE(S): one ggml call per operator.
  ggml_tensor* gOut = nullptr;
  switch (j.op) {
    case EwOp::kRmsNorm: gOut = ggml_rms_norm(ctx, gA, j.eps); break;
    case EwOp::kSilu:    gOut = ggml_silu(ctx, gA); break;
    case EwOp::kAdd:     gOut = ggml_add(ctx, gA, gB); break;
    case EwOp::kMul:     gOut = ggml_mul(ctx, gA, gB); break;
  }
  ggml_set_output(gOut);  // gallocr won't alias the result away before tensor_get
  ggml_build_forward_expand(gf, gOut);

  if (!ggml_gallocr_alloc_graph(galloc, gf)) {
    ggml_free(ctx);
    return Status::ResourceExhausted("ggml_gallocr_alloc_graph failed");
  }
  ggml_backend_tensor_set(gA, j.a, 0, ggml_nbytes(gA));       // H2D input 0
  if (binary) ggml_backend_tensor_set(gB, j.b, 0, ggml_nbytes(gB));  // H2D input 1

  ggml_status ec = ggml_backend_graph_compute(be, gf);
  if (ec == GGML_STATUS_SUCCESS)
    ggml_backend_tensor_get(gOut, j.out, 0, ggml_nbytes(gOut));  // D2H result
  ggml_free(ctx);
  if (ec != GGML_STATUS_SUCCESS) return Status::Internal("ggml_backend_graph_compute failed");
  return Status();  // OK
}

// Shared dispatch for the 4 public elementwise ops: POD-capture + spawn_blocking. Takes the
// already-extracted raw backend/gallocr handles (the public wrapper does the friend access +
// contract checks) so this stays a plain TU-local helper, not a friend of GgmlBackend.
sched::JoinHandle<Status> ew_spawn(void* be_raw, void* galloc, sched::BlockingPool& pool,
                                   mem::Allocator alloc, const float* a, const float* b,
                                   float* out, int n, float eps, EwOp op) noexcept {
  EwJob job{a, b, out, n, eps, op};
  return sched::spawn_blocking(pool, alloc, [be_raw, galloc, job]() noexcept {
    return elementwise_blocking(as_be(be_raw), static_cast<ggml_gallocr_t>(galloc), job);
  });
}

}  // namespace

GgmlBackend& GgmlBackend::operator=(GgmlBackend&& o) noexcept {
  if (this != &o) {
    if (galloc_ != nullptr) ggml_gallocr_free(static_cast<ggml_gallocr_t>(galloc_));
    if (be_ != nullptr) ggml_backend_free(as_be(be_));
    be_ = o.be_; galloc_ = o.galloc_;
    o.be_ = nullptr; o.galloc_ = nullptr;
  }
  return *this;
}

GgmlBackend::~GgmlBackend() {
  if (galloc_ != nullptr) ggml_gallocr_free(static_cast<ggml_gallocr_t>(galloc_));
  if (be_ != nullptr) ggml_backend_free(as_be(be_));
}

Result<GgmlBackend> GgmlBackend::create(const Device& device, int n_threads) noexcept {
  ggml_backend_t be = nullptr;
  if (device.is_gpu()) {
#if defined(HYLE_COMPUTE_CUDA)
    // GPU -> ggml CUDA backend on the device's ordinal.
    be = ggml_backend_cuda_init(device.ordinal());
    if (be == nullptr) return std::unexpected(Status::Unavailable("ggml_backend_cuda_init failed"));
#else
    return std::unexpected(Status::Unavailable("ggml CUDA backend not compiled"));
#endif
  } else {
    // CPU -> ggml CPU backend with the configured thread count (③ batching-split:
    // caller passes full cores for a batched big graph, 1 for independent small work).
    be = ggml_backend_cpu_init();
    if (be == nullptr) return std::unexpected(Status::Unavailable("ggml_backend_cpu_init failed"));
    ggml_backend_cpu_set_n_threads(be, n_threads);
  }
  // One persistent gallocr (llama.cpp model): allocates the real device buffers for
  // every compute graph, created once and reused for all ops. This is spec ①'s "device
  // 归 ggml" -- ggml owns the buffers its operators compute on.
  ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(be);
  ggml_gallocr_t galloc = ggml_gallocr_new(buft);
  if (galloc == nullptr) {
    ggml_backend_free(be);
    return std::unexpected(Status::Unavailable("ggml_gallocr_new failed"));
  }
  return GgmlBackend(static_cast<void*>(be), static_cast<void*>(galloc));
}

sched::JoinHandle<Status> ggml_matmul_f32(GgmlBackend& be, sched::BlockingPool& pool,
                                          mem::Allocator alloc, const float* a,
                                          const float* b, float* c, int m, int k,
                                          int n) noexcept {
  // Caller-contract checks (bug -> fatal assert, not silent -- contract-over-defense).
  HYLE_FATAL_ASSERT(be.valid());
  HYLE_FATAL_ASSERT(a != nullptr && b != nullptr && c != nullptr);
  HYLE_FATAL_ASSERT(m > 0 && k > 0 && n > 0);

  // POD job captured BY VALUE (runs on another thread -- like io/dns.h's Job): the
  // backend + gallocr handles + three HOST pointers + dims. NO ggml type escapes. The
  // host buffers a/b/c must outlive the join (the pool thread reads a/b, writes c).
  void* be_raw = be.raw_();
  void* galloc = be.galloc_ptr_();
  MmJob job{a, b, c, m, k, n};
  return sched::spawn_blocking(pool, alloc, [be_raw, galloc, job]() noexcept {
    return matmul_batch_blocking(as_be(be_raw),
                                 static_cast<ggml_gallocr_t>(galloc), &job, 1);
  });
}

sched::JoinHandle<Status> ggml_matmul_f32_batched(GgmlBackend& be,
                                                  sched::BlockingPool& pool,
                                                  mem::Allocator alloc,
                                                  const MatmulF32Job* jobs,
                                                  int count) noexcept {
  // Caller-contract checks (bug -> fatal assert). Cap 64 keeps matmul_batch_blocking's
  // fixed-size tensor arrays stack-only (no per-batch heap in the pool closure).
  HYLE_FATAL_ASSERT(be.valid());
  HYLE_FATAL_ASSERT(count > 0 && count <= 64);
  HYLE_FATAL_ASSERT(jobs != nullptr);
  // Copy jobs into a fixed POD array captured by value (the caller's array need not
  // outlive the await; the host buffers it points at must -- caller's responsibility).
  struct Batch {
    MmJob jobs[64];
    int count;
  } batch;
  batch.count = count;
  for (int i = 0; i < count; ++i) {
    const MatmulF32Job& j = jobs[i];
    HYLE_FATAL_ASSERT(j.a != nullptr && j.b != nullptr && j.c != nullptr);
    HYLE_FATAL_ASSERT(j.m > 0 && j.k > 0 && j.n > 0);
    batch.jobs[i] = MmJob{j.a, j.b, j.c, j.m, j.k, j.n};
  }
  void* be_raw = be.raw_();
  void* galloc = be.galloc_ptr_();
  return sched::spawn_blocking(pool, alloc, [be_raw, galloc, batch]() noexcept {
    return matmul_batch_blocking(as_be(be_raw),
                                 static_cast<ggml_gallocr_t>(galloc),
                                 batch.jobs, batch.count);
  });
}

// ---- elementwise / norm ops (compute-09 step-1): thin public wrappers over ew_spawn ----
// Each does the contract checks + friend access (raw_/galloc_ptr_) then delegates.
sched::JoinHandle<Status> ggml_rms_norm_f32(GgmlBackend& be, sched::BlockingPool& pool,
                                            mem::Allocator alloc, const float* src,
                                            float* dst, int n, float eps) noexcept {
  HYLE_FATAL_ASSERT(be.valid() && src != nullptr && dst != nullptr && n > 0);
  HYLE_FATAL_ASSERT(eps >= 0.0f);  // ggml asserts this internally (ops.cpp:3805) -- fail-fast in
                                   // the CALLER thread, not deep in the pool thread's ggml abort.
  return ew_spawn(be.raw_(), be.galloc_ptr_(), pool, alloc, src, nullptr, dst, n, eps, EwOp::kRmsNorm);
}
sched::JoinHandle<Status> ggml_silu_f32(GgmlBackend& be, sched::BlockingPool& pool,
                                        mem::Allocator alloc, const float* src,
                                        float* dst, int n) noexcept {
  HYLE_FATAL_ASSERT(be.valid() && src != nullptr && dst != nullptr && n > 0);
  return ew_spawn(be.raw_(), be.galloc_ptr_(), pool, alloc, src, nullptr, dst, n, 0.0f, EwOp::kSilu);
}
sched::JoinHandle<Status> ggml_add_f32(GgmlBackend& be, sched::BlockingPool& pool,
                                       mem::Allocator alloc, const float* a, const float* b,
                                       float* dst, int n) noexcept {
  HYLE_FATAL_ASSERT(be.valid() && a != nullptr && b != nullptr && dst != nullptr && n > 0);
  return ew_spawn(be.raw_(), be.galloc_ptr_(), pool, alloc, a, b, dst, n, 0.0f, EwOp::kAdd);
}
sched::JoinHandle<Status> ggml_mul_f32(GgmlBackend& be, sched::BlockingPool& pool,
                                       mem::Allocator alloc, const float* a, const float* b,
                                       float* dst, int n) noexcept {
  HYLE_FATAL_ASSERT(be.valid() && a != nullptr && b != nullptr && dst != nullptr && n > 0);
  return ew_spawn(be.raw_(), be.galloc_ptr_(), pool, alloc, a, b, dst, n, 0.0f, EwOp::kMul);
}

}  // namespace hyle::compute
