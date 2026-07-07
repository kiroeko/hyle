// compute-05: ggml_bridge -- ALL ggml coupling is sealed in this one TU (spec ⑤:
// adapter single-point isolation). ggml types (ggml_context/ggml_tensor/
// ggml_backend_t) appear ONLY here; the public header exposes none. A ggml upgrade
// touches this file + the golden-compare tests, nothing else.
//
// See docs/milestones/compute-05-ggml-bridge.md + compute-design "compute-05 借 ggml
// 算力：调研发现汇总" (findings 1-10 + CPU 执行模型专章). This file implements the
// three load-bearing decisions from ggml_bridge.h: ① zero-copy device_ptr injection,
// ② completion via spawn_blocking, ③ CPU thread model = configurable n_threads.

#include "hyle/compute/ggml_bridge.h"

#include <cstdint>
#include <utility>

#include "hyle/macros.h"

// ggml headers -- CONFINED to this .cc (never in ggml_bridge.h). ggml-cuda.h is pure
// C declarations (no CUDA toolkit headers), safe to include from this plain C++ TU.
// ggml does NOT export GGML_USE_CUDA to consumers, so gate on HYLE_COMPUTE_CUDA (set
// PRIVATE on hyle_compute by src/compute/CMakeLists.txt when the toolkit was found).
#include "ggml.h"
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

// The blocking half of a matmul: runs ENTIRELY on the sched-11 pool thread (spec ②).
// Builds a no_alloc ggml_context, injects HYLE device_ptrs as tensor->data (① zero-
// copy), builds the cgraph, computes, synchronizes, frees the context. Nothing ggml
// escapes this function. Returns a HYLE Status (R for spawn_blocking's JoinState).
//
// GGML MATMUL SEMANTICS (verified ggml.h:1415-1417 + simple-backend example numbers):
//   ggml_mul_mat(ctx, A, B): A=[ne0=k, ne1=p], B=[ne0=k, ne1=q] (both contract on
//   ne0), result=[ne0=p, ne1=q], result[j*p + i] = dot(A_row_i, B_row_j). ggml is
//   COLUMN-MAJOR. This first consumer computes C = A · Bᵀ with BOTH inputs [*, k]
//   row-major -- the NATURAL ggml matmul, no transpose copy. Shapes:
//     gA = [k, m]  (HYLE a, m rows of k)      -- data <- a.device_ptr()
//     gB = [k, n]  (HYLE b as n rows of k)    -- data <- b.device_ptr()
//     gC = [m, n]  result                     -- data <- c.device_ptr()
//
// ZERO-COPY, CORRECTED (finding 2 was HALF WRONG -- see docs/debug/ggml-zerocopy-null-
// buffer-hunt.md): the CUDA matmul path DOES dereference tensor->buffer -- src0->buffer
// ->buft for the split check (ggml-cuda.cu:1853) and src1/dst->buffer->context->device
// for the device id (1832-1833). A NULL ->buffer segfaults (0xC0000005, caught on the
// RTX 4070). BUT the kernel's DATA base is still src0/1/dst->data (1931/1951/1973), which
// we control. So the fix is NOT "let ggml own the data + copy": we let ggml ALLOCATE the
// tensors (valid ->buffer metadata: buft + device) via ggml_backend_alloc_ctx_tensors,
// then OVERWRITE ->data with the HYLE device_ptr. ggml's throwaway allocation supplies
// only the buffer struct; the compute reads/writes HYLE memory. Still zero-copy on the
// actual matmul data. (Works for CPU backend too: ->data is host RAM = HYLE host Buffer.)
// One matmul's raw inputs (device pointers + dims) -- the internal POD, mirrors the
// public MatmulF32Job but stays TU-local so the header's struct isn't dragged in here.
struct MmJob {
  const void* a;
  const void* b;
  void* c;
  int m, k, n;
};

// The blocking half: runs ENTIRELY on the sched-11 pool thread (spec ②). Builds ONE
// no_alloc ggml_context holding ALL `count` matmuls, injects HYLE device_ptrs as tensor
// ->data (① zero-copy), builds ONE cgraph over all results, ONE graph_compute + ONE
// synchronize (finding 4: batching = one submit for many ops). count==1 is the single-
// matmul case. Nothing ggml escapes this function.
//
// GGML MATMUL SEMANTICS (verified ggml.h:1415-1417): ggml_mul_mat(A,B) with A=[k,m],
// B=[k,n] -> C=[m,n], C = A·Bᵀ (both contract on ne0). See the single-op history in git
// for the full derivation; per job: gA=[k,m] data=a, gB=[k,n] data=b, gC=[m,n] data=c.
//
// ZERO-COPY via alloc_ctx_tensors + overwrite ->data (finding 2, CORRECTED -- see
// docs/debug/ggml-zerocopy-null-buffer-hunt.md): the CUDA matmul derefs tensor->buffer
// (->buft for split check ggml-cuda.cu:1853, ->context->device 1832-33), so a NULL
// ->buffer segfaults; but the kernel's DATA base is ->data (1931/1951/1973) which we
// control. So we let ggml ALLOCATE tensors (valid buffer metadata) then OVERWRITE ->data
// with HYLE pointers. ggml's throwaway allocation bytes are unused; compute reads/writes
// HYLE memory. Batching just means N triples of tensors in one context.
Status matmul_batch_blocking(ggml_backend_t be, const MmJob* jobs, int count) noexcept {
  // no_alloc context: metadata structs only (data is HYLE memory). Each matmul needs 3
  // tensors; ggml_new_graph + build_forward_expand add graph overhead. Budget generously:
  // 3 tensors/job * count + slack, plus per-graph overhead.
  const size_t mem_size =
      ggml_tensor_overhead() * (static_cast<size_t>(count) * 3 + 4) + ggml_graph_overhead();
  ggml_init_params params = {
      /*.mem_size   =*/ mem_size,
      /*.mem_buffer =*/ nullptr,
      /*.no_alloc   =*/ true,
  };
  ggml_context* ctx = ggml_init(params);
  if (ctx == nullptr) return Status::ResourceExhausted("ggml_init failed");

  // Build all matmul nodes first (tensors created in ctx), remember each result to shadow
  // its ->data after allocation. Fixed cap keeps this stack-only (contract: count <= 64).
  ggml_tensor* results[64];
  ggml_tensor* inA[64];
  ggml_tensor* inB[64];
  for (int i = 0; i < count; ++i) {
    inA[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, jobs[i].k, jobs[i].m);
    inB[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, jobs[i].k, jobs[i].n);
    results[i] = ggml_mul_mat(ctx, inA[i], inB[i]);  // [ne0=m, ne1=n]
  }

  // Give EVERY tensor a valid ggml buffer (non-null ->buffer for the CUDA derefs), then
  // overwrite ->data with HYLE pointers (zero-copy on the real data).
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
  if (buf == nullptr) {
    ggml_free(ctx);
    return Status::ResourceExhausted("ggml_backend_alloc_ctx_tensors failed");
  }
  for (int i = 0; i < count; ++i) {
    inA[i]->data = const_cast<void*>(jobs[i].a);
    inB[i]->data = const_cast<void*>(jobs[i].b);
    results[i]->data = jobs[i].c;
  }

  // ONE graph over all results, ONE compute+sync (the batching win, finding 4).
  ggml_cgraph* gf = ggml_new_graph(ctx);
  for (int i = 0; i < count; ++i) ggml_build_forward_expand(gf, results[i]);
  ggml_status ec = ggml_backend_graph_compute(be, gf);

  ggml_backend_buffer_free(buf);
  ggml_free(ctx);

  if (ec != GGML_STATUS_SUCCESS) {
    return Status::Internal("ggml_backend_graph_compute failed");
  }
  return Status();  // OK
}

// COPY-PATH fallback (finding 2 safety net): ggml OWNS the tensor memory (no ->data
// overwrite), host data copied in via ggml_backend_tensor_set, result out via
// tensor_get. Does NOT depend on the fragile ->data-injection behavior -- works even if
// a ggml upgrade changes it. Same C = A·Bᵀ. Pays 3 copies (H2D a/b, D2H c).
Status matmul_copy_blocking(ggml_backend_t be, const float* ha, const float* hb,
                            float* hc, int m, int k, int n) noexcept {
  const size_t mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
  ggml_init_params params = {mem_size, nullptr, /*no_alloc=*/true};
  ggml_context* ctx = ggml_init(params);
  if (ctx == nullptr) return Status::ResourceExhausted("ggml_init failed");

  ggml_tensor* gA = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
  ggml_tensor* gB = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
  ggml_tensor* gC = ggml_mul_mat(ctx, gA, gB);

  // ggml owns the memory (we do NOT overwrite ->data here -- that's the whole point).
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
  if (buf == nullptr) {
    ggml_free(ctx);
    return Status::ResourceExhausted("ggml_backend_alloc_ctx_tensors failed");
  }
  // Copy host inputs into ggml's tensors (H2D on a GPU backend).
  ggml_backend_tensor_set(gA, ha, 0, ggml_nbytes(gA));
  ggml_backend_tensor_set(gB, hb, 0, ggml_nbytes(gB));

  ggml_cgraph* gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, gC);
  ggml_status ec = ggml_backend_graph_compute(be, gf);

  if (ec == GGML_STATUS_SUCCESS) {
    ggml_backend_tensor_get(gC, hc, 0, ggml_nbytes(gC));  // D2H result out
  }
  ggml_backend_buffer_free(buf);
  ggml_free(ctx);

  if (ec != GGML_STATUS_SUCCESS) {
    return Status::Internal("ggml_backend_graph_compute failed");
  }
  return Status();  // OK
}

}  // namespace

GgmlBackend& GgmlBackend::operator=(GgmlBackend&& o) noexcept {
  if (this != &o) {
    if (be_ != nullptr) ggml_backend_free(as_be(be_));
    be_ = o.be_;
    o.be_ = nullptr;
  }
  return *this;
}

GgmlBackend::~GgmlBackend() {
  if (be_ != nullptr) ggml_backend_free(as_be(be_));
}

Result<GgmlBackend> GgmlBackend::create(const Device& device, int n_threads) noexcept {
  ggml_backend_t be = nullptr;
  if (device.is_gpu()) {
#if defined(HYLE_COMPUTE_CUDA)
    // GPU -> ggml CUDA backend on the device's ordinal (zero-copy device_ptr path).
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
  return GgmlBackend(static_cast<void*>(be));
}

sched::JoinHandle<Status> ggml_matmul_f32(GgmlBackend& be, sched::BlockingPool& pool,
                                          mem::Allocator alloc, const Buffer& a,
                                          const Buffer& b, Buffer& c, int m, int k,
                                          int n) noexcept {
  // Caller-contract checks (bug -> fatal assert, not silent -- contract-over-defense).
  HYLE_FATAL_ASSERT(be.valid());
  HYLE_FATAL_ASSERT(a.valid() && b.valid() && c.valid());
  HYLE_FATAL_ASSERT(m > 0 && k > 0 && n > 0);
  const size_t f32 = sizeof(float);
  HYLE_FATAL_ASSERT(a.size() >= static_cast<size_t>(m) * k * f32);
  HYLE_FATAL_ASSERT(b.size() >= static_cast<size_t>(n) * k * f32);
  HYLE_FATAL_ASSERT(c.size() >= static_cast<size_t>(m) * n * f32);

  // POD job captured BY VALUE (runs on another thread -- like io/dns.h's Job): the
  // backend handle + three device pointers + dims. NO ggml type, NO Buffer reference
  // escapes; device_ptr() is a plain pointer into HYLE-owned memory that outlives the
  // op (caller guarantees, same as backend). spawn_blocking moves this onto the pool.
  void* be_raw = be.raw_();
  MmJob job{a.device_ptr(), b.device_ptr(), c.device_ptr(), m, k, n};
  return sched::spawn_blocking(pool, alloc, [be_raw, job]() noexcept {
    return matmul_batch_blocking(as_be(be_raw), &job, 1);  // count==1: single matmul
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
  const size_t f32 = sizeof(float);
  // Copy jobs into a fixed POD array captured by value (the caller's array need not
  // outlive the await; nothing but device pointers + dims crosses to the pool thread).
  struct Batch {
    MmJob jobs[64];
    int count;
  } batch;
  batch.count = count;
  for (int i = 0; i < count; ++i) {
    const MatmulF32Job& j = jobs[i];
    HYLE_FATAL_ASSERT(j.a != nullptr && j.b != nullptr && j.c != nullptr);
    HYLE_FATAL_ASSERT(j.m > 0 && j.k > 0 && j.n > 0);
    (void)f32;  // sizes are the caller's Buffers' responsibility (raw ptrs here)
    batch.jobs[i] = MmJob{j.a, j.b, j.c, j.m, j.k, j.n};
  }
  void* be_raw = be.raw_();
  return sched::spawn_blocking(pool, alloc, [be_raw, batch]() noexcept {
    return matmul_batch_blocking(as_be(be_raw), batch.jobs, batch.count);
  });
}

sched::JoinHandle<Status> ggml_matmul_f32_copy(GgmlBackend& be,
                                               sched::BlockingPool& pool,
                                               mem::Allocator alloc, const float* ha,
                                               const float* hb, float* hc, int m, int k,
                                               int n) noexcept {
  HYLE_FATAL_ASSERT(be.valid());
  HYLE_FATAL_ASSERT(ha != nullptr && hb != nullptr && hc != nullptr);
  HYLE_FATAL_ASSERT(m > 0 && k > 0 && n > 0);
  // Host pointers are captured by value (the POINTERS, not the data): the pool thread
  // copies through them, so the caller's host buffers MUST outlive the join (contract).
  void* be_raw = be.raw_();
  return sched::spawn_blocking(pool, alloc, [be_raw, ha, hb, hc, m, k, n]() noexcept {
    return matmul_copy_blocking(as_be(be_raw), ha, hb, hc, m, k, n);
  });
}

}  // namespace hyle::compute
