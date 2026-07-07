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
Status matmul_blocking(ggml_backend_t be, const void* a_dev, const void* b_dev,
                       void* c_dev, int m, int k, int n) noexcept {
  // no_alloc context: holds tensor metadata structs; ggml_backend_alloc_ctx_tensors
  // gives them buffers below. Size = per-tensor overhead * a few + graph overhead
  // (example simple-backend.cpp:69). 8 tensors is ample for a 3-tensor matmul.
  const size_t mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
  ggml_init_params params = {
      /*.mem_size   =*/ mem_size,
      /*.mem_buffer =*/ nullptr,  // ggml allocates the metadata arena internally
      /*.no_alloc   =*/ true,     // no data yet; alloc_ctx_tensors provides buffers
  };
  ggml_context* ctx = ggml_init(params);
  if (ctx == nullptr) return Status::ResourceExhausted("ggml_init failed");

  ggml_tensor* gA = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
  ggml_tensor* gB = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
  ggml_tensor* gC = ggml_mul_mat(ctx, gA, gB);  // result = [ne0=m, ne1=n]

  // Give every tensor (gA, gB, gC) a valid ggml buffer so ->buffer->buft/context are
  // non-null (the CUDA matmul derefs them). This makes a throwaway ggml allocation we
  // immediately shadow with the HYLE pointers below.
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
  if (buf == nullptr) {
    ggml_free(ctx);
    return Status::ResourceExhausted("ggml_backend_alloc_ctx_tensors failed");
  }

  // Zero-copy injection: point the kernel's data base at HYLE memory (the ggml buffer
  // metadata stays valid for the split/device checks; the ggml allocation's own bytes
  // are simply unused). See the file-level ZERO-COPY comment for the source evidence.
  gA->data = const_cast<void*>(a_dev);
  gB->data = const_cast<void*>(b_dev);
  gC->data = c_dev;

  ggml_cgraph* gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, gC);

  // Submit + block. graph_compute launches async then synchronize blocks THIS (pool)
  // thread until done (spec ②; ggml's CUDA stream is private, blocking is the only
  // completion). The fiber that spawn_blocking'd this is meanwhile free.
  ggml_status ec = ggml_backend_graph_compute(be, gf);

  ggml_backend_buffer_free(buf);  // free ggml's throwaway allocation (not HYLE memory)
  ggml_free(ctx);                 // free metadata arena

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
  const void* a_dev = a.device_ptr();
  const void* b_dev = b.device_ptr();
  void* c_dev = c.device_ptr();
  return sched::spawn_blocking(pool, alloc, [be_raw, a_dev, b_dev, c_dev, m, k, n]() noexcept {
    return matmul_blocking(as_be(be_raw), a_dev, b_dev, c_dev, m, k, n);
  });
}

}  // namespace hyle::compute
