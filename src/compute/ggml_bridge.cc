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
#include <cstring>
#include <new>       // std::nothrow (FfnWeights heap alloc)
#include <utility>

#include "backend.h"
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
  if (!device.valid())
    return std::unexpected(
        Status::InvalidArgument("creating ggml backend on an invalid device"));
  if (n_threads <= 0)
    return std::unexpected(
        Status::InvalidArgument("ggml backend thread count must be positive"));
  ggml_backend_t be = nullptr;
  if (std::strcmp(device.backend(), "cuda") == 0) {
#if defined(HYLE_COMPUTE_CUDA)
    // Device::ordinal() is the HYLE-global table index. ggml expects the
    // CUDA-local ordinal, which differs once other backends are present.
    be = ggml_backend_cuda_init(device_backend_ordinal(device));
    if (be == nullptr) return std::unexpected(Status::Unavailable("ggml_backend_cuda_init failed"));
#else
    return std::unexpected(Status::Unavailable("ggml CUDA backend not compiled"));
#endif
  } else if (std::strcmp(device.backend(), "host") == 0) {
    // CPU -> ggml CPU backend with the configured thread count (③ batching-split:
    // caller passes full cores for a batched big graph, 1 for independent small work).
    be = ggml_backend_cpu_init();
    if (be == nullptr) return std::unexpected(Status::Unavailable("ggml_backend_cpu_init failed"));
    ggml_backend_cpu_set_n_threads(be, n_threads);
  } else {
    return std::unexpected(Status::Unimplemented(
        "ggml bridge supports only HYLE host and CUDA devices"));
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

// ---- FFN block (compute-09 step-2): resident-weight transformer feed-forward sub-block --------
namespace {

// The 4 resident weight tensors (device-resident, built once). TU-local; the header holds a
// void* to a heap-allocated instance of this so ggml types never leak.
struct FfnWeights {
  ggml_tensor* w_norm;  // [d]
  ggml_tensor* w_gate;  // [d, ffn] (ne0=d contract dim, ne1=ffn) -- gate projection
  ggml_tensor* w_up;    // [d, ffn]
  ggml_tensor* w_down;  // [ffn, d] (ne0=ffn, ne1=d) -- down projection
};

// The blocking half of forward: build ONE compute graph that references the RESIDENT weights +
// the fed activation, gallocr-alloc the activation/intermediate device buffers, H2D the input,
// compute, D2H the output. Runs entirely on the pool thread (spec ②). Weights already live in
// wbuf (not touched by gallocr). x = [rows*d] host, out = [rows*d] host.
Status ffn_forward_blocking(ggml_backend_t be, ggml_gallocr_t galloc, const FfnWeights* w,
                            int d, float eps, const float* x, float* out,
                            int rows) noexcept {
  // node budget: input + rms + norm_mul + gate_mm + gate_silu + up_mm + gate_up_mul + down_mm +
  // residual = ~9 tensors; give headroom. Weights are NOT in this ctx (they live in wctx).
  // (ffn is not needed here -- every ffn-shaped intermediate is derived from the resident weight
  // tensors' own ne, e.g. w_gate is [d,ffn]; only d + rows shape the fed activation.)
  const size_t mem_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
  ggml_init_params params = {mem_size, nullptr, /*no_alloc=*/true};
  ggml_context* ctx = ggml_init(params);
  if (ctx == nullptr) return Status::ResourceExhausted("ggml_init failed");

  ggml_cgraph* gf = ggml_new_graph(ctx);
  // activation input x: [d, rows] (ne0=d hidden, ne1=rows). row-major host [rows*d] maps directly.
  ggml_tensor* gx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, rows);
  ggml_set_input(gx);

  // h = rms_norm(x) ⊙ w_norm.  rms_norm over ne0 (d) per row; then broadcast-mul the [d] weight.
  ggml_tensor* norm = ggml_rms_norm(ctx, gx, eps);
  ggml_tensor* h = ggml_mul(ctx, norm, w->w_norm);  // w_norm [d] repeats over rows (ggml broadcast)
  // gate = silu(w_gate · hᵀ) : mul_mat(w_gate[d,ffn], h[d,rows]) -> [ffn, rows]; then silu.
  ggml_tensor* gate = ggml_silu(ctx, ggml_mul_mat(ctx, w->w_gate, h));
  // up = w_up · hᵀ -> [ffn, rows]
  ggml_tensor* up = ggml_mul_mat(ctx, w->w_up, h);
  // gu = gate ⊙ up  -> [ffn, rows]
  ggml_tensor* gu = ggml_mul(ctx, gate, up);
  // ffn_out = w_down · guᵀ : mul_mat(w_down[ffn,d], gu[ffn,rows]) -> [d, rows]
  ggml_tensor* ffn_out = ggml_mul_mat(ctx, w->w_down, gu);
  // out = x + ffn_out (residual)
  ggml_tensor* gout = ggml_add(ctx, gx, ffn_out);
  ggml_set_output(gout);
  ggml_build_forward_expand(gf, gout);

  if (!ggml_gallocr_alloc_graph(galloc, gf)) {
    ggml_free(ctx);
    return Status::ResourceExhausted("ggml_gallocr_alloc_graph failed");
  }
  ggml_backend_tensor_set(gx, x, 0, ggml_nbytes(gx));  // H2D activation only (weights resident)
  ggml_status ec = ggml_backend_graph_compute(be, gf);
  if (ec == GGML_STATUS_SUCCESS)
    ggml_backend_tensor_get(gout, out, 0, ggml_nbytes(gout));  // D2H result
  ggml_free(ctx);
  if (ec != GGML_STATUS_SUCCESS) return Status::Internal("ggml_backend_graph_compute failed");
  return Status();
}

}  // namespace

FfnBlock::FfnBlock(void* be, void* galloc, void* wctx, void* wbuf, void* weights, int d,
                   int ffn, float eps) noexcept
    : be_(be), galloc_(galloc), wctx_(wctx), wbuf_(wbuf), weights_(weights),
      d_(d), ffn_(ffn), eps_(eps) {}

FfnBlock::FfnBlock(FfnBlock&& o) noexcept
    : be_(o.be_), galloc_(o.galloc_), wctx_(o.wctx_), wbuf_(o.wbuf_), weights_(o.weights_),
      d_(o.d_), ffn_(o.ffn_), eps_(o.eps_) {
  o.be_ = nullptr; o.galloc_ = nullptr; o.wctx_ = nullptr; o.wbuf_ = nullptr; o.weights_ = nullptr;
}

FfnBlock& FfnBlock::operator=(FfnBlock&& o) noexcept {
  if (this != &o) {
    reset();
    be_ = o.be_; galloc_ = o.galloc_; wctx_ = o.wctx_; wbuf_ = o.wbuf_; weights_ = o.weights_;
    d_ = o.d_; ffn_ = o.ffn_; eps_ = o.eps_;
    o.be_ = nullptr; o.galloc_ = nullptr; o.wctx_ = nullptr; o.wbuf_ = nullptr; o.weights_ = nullptr;
  }
  return *this;
}

void FfnBlock::reset() noexcept {
  delete static_cast<FfnWeights*>(weights_);                                    // TU-local POD
  if (wbuf_ != nullptr) ggml_backend_buffer_free(static_cast<ggml_backend_buffer_t>(wbuf_));
  if (wctx_ != nullptr) ggml_free(static_cast<ggml_context*>(wctx_));
  if (galloc_ != nullptr) ggml_gallocr_free(static_cast<ggml_gallocr_t>(galloc_));
  if (be_ != nullptr) ggml_backend_free(as_be(be_));
  be_ = galloc_ = wctx_ = wbuf_ = weights_ = nullptr;
}

FfnBlock::~FfnBlock() { reset(); }

Result<FfnBlock> FfnBlock::create(const Device& device, int d, int ffn, float eps,
                                  const float* w_norm, const float* w_gate, const float* w_up,
                                  const float* w_down, int n_threads) noexcept {
  if (!device.valid())
    return std::unexpected(
        Status::InvalidArgument("creating ggml FFN on an invalid device"));
  if (n_threads <= 0)
    return std::unexpected(
        Status::InvalidArgument("ggml FFN thread count must be positive"));
  HYLE_FATAL_ASSERT(d > 0 && ffn > 0 && eps >= 0.0f);
  HYLE_FATAL_ASSERT(w_norm && w_gate && w_up && w_down);

  // 1. compute backend (same as GgmlBackend::create).
  ggml_backend_t be = nullptr;
  if (std::strcmp(device.backend(), "cuda") == 0) {
#if defined(HYLE_COMPUTE_CUDA)
    be = ggml_backend_cuda_init(device_backend_ordinal(device));
#endif
    if (be == nullptr) return std::unexpected(Status::Unavailable("ggml CUDA backend unavailable"));
  } else if (std::strcmp(device.backend(), "host") == 0) {
    be = ggml_backend_cpu_init();
    if (be == nullptr) return std::unexpected(Status::Unavailable("ggml_backend_cpu_init failed"));
    ggml_backend_cpu_set_n_threads(be, n_threads);
  } else {
    return std::unexpected(Status::Unimplemented(
        "ggml FFN supports only HYLE host and CUDA devices"));
  }

  // 2. weight context (no_alloc: metadata only; buffer comes from alloc_ctx_tensors_from_buft).
  ggml_init_params wparams = {ggml_tensor_overhead() * 8, nullptr, /*no_alloc=*/true};
  ggml_context* wctx = ggml_init(wparams);
  if (wctx == nullptr) { ggml_backend_free(be); return std::unexpected(Status::ResourceExhausted("weight ctx")); }

  auto* w = new (std::nothrow) FfnWeights{};
  if (w == nullptr) { ggml_free(wctx); ggml_backend_free(be); return std::unexpected(Status::ResourceExhausted("weights")); }
  // ne0 = contract/input dim (llama.cpp convention).
  w->w_norm = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, d);          // [d]
  w->w_gate = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, d, ffn);     // [d, ffn]
  w->w_up   = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, d, ffn);     // [d, ffn]
  w->w_down = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, ffn, d);     // [ffn, d]

  // 3. RESIDENT weight buffer: allocate real device memory for all wctx tensors, ONCE.
  ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(be);
  ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, buft);
  if (wbuf == nullptr) { delete w; ggml_free(wctx); ggml_backend_free(be); return std::unexpected(Status::ResourceExhausted("weight buffer")); }

  // 4. copy the host weights in ONCE (H2D; resident from here on).
  ggml_backend_tensor_set(w->w_norm, w_norm, 0, ggml_nbytes(w->w_norm));
  ggml_backend_tensor_set(w->w_gate, w_gate, 0, ggml_nbytes(w->w_gate));
  ggml_backend_tensor_set(w->w_up,   w_up,   0, ggml_nbytes(w->w_up));
  ggml_backend_tensor_set(w->w_down, w_down, 0, ggml_nbytes(w->w_down));

  // 5. persistent gallocr for the per-forward activation/intermediate buffers.
  ggml_gallocr_t galloc = ggml_gallocr_new(buft);
  if (galloc == nullptr) {
    ggml_backend_buffer_free(wbuf); delete w; ggml_free(wctx); ggml_backend_free(be);
    return std::unexpected(Status::ResourceExhausted("gallocr"));
  }
  return FfnBlock(static_cast<void*>(be), static_cast<void*>(galloc), static_cast<void*>(wctx),
                  static_cast<void*>(wbuf), static_cast<void*>(w), d, ffn, eps);
}

sched::JoinHandle<Status> FfnBlock::forward(sched::BlockingPool& pool, mem::Allocator alloc,
                                            const float* x, float* out, int rows) noexcept {
  HYLE_FATAL_ASSERT(valid() && x != nullptr && out != nullptr && rows > 0);
  void* be_raw = be_;
  void* galloc = galloc_;
  const FfnWeights* w = static_cast<const FfnWeights*>(weights_);
  int d = d_; float eps = eps_;
  return sched::spawn_blocking(pool, alloc, [be_raw, galloc, w, d, eps, x, out, rows]() noexcept {
    return ffn_forward_blocking(as_be(be_raw), static_cast<ggml_gallocr_t>(galloc), w, d,
                                eps, x, out, rows);
  });
}

}  // namespace hyle::compute
