#include "hyle/compute/ggml_bridge.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "hyle/compute/device.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/blocking.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

// compute-05 test: borrow a ggml matmul operator under HYLE scheduling. The memory model
// is host 归 HYLE / device 归 ggml (docs/architecture/borrowing-ggml-memory-model.md):
// the API takes HOST inputs/outputs (HYLE mem), ggml owns the device buffers (gallocr)
// and the bridge copies H2D (tensor_set) / D2H (tensor_get). Offloaded via spawn_blocking
// so a fiber suspends while the ggml graph computes on a pool thread, woken through the
// JoinHandle. Proves the three load-bearing decisions (ggml_bridge.h ①②③) compose.
//
// GGML MATMUL SHAPE (verified ggml.h:1415-1417): the bridge computes C = A · Bᵀ with
// both inputs [*, k] row-major. gA=[k,m] (m rows of k), gB=[k,n] (n rows of k), result
// gC=[m,n] laid out row-major as n rows of m, where gC[j*m + i] = dot(A_row_i, B_row_j).
// The golden reference below mirrors exactly that indexing.

using hyle::mem::system_allocator;
using hyle::sched::BlockingPool;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using namespace hyle;
using namespace hyle::compute;

// Threadsafe death style on Linux only (re-exec vs fork-after-CUDA-init). Windows death
// tests are always spawn-based (no fork) so the flag is unneeded -- and its symbol
// (testing::FLAGS_gtest_death_test_style) doesn't link against the gtest DLL. See
// contract_test.cc for the full rationale.
#if defined(_WIN32)
#  define HYLE_THREADSAFE_DEATH() ((void)0)
#else
#  define HYLE_THREADSAFE_DEATH() GTEST_FLAG_SET(death_test_style, "threadsafe")
#endif

namespace {

// Golden CPU reference for ggml's C = A·Bᵀ. A is m rows × k, B is n rows × k (both
// row-major). out is n rows × m row-major: out[j*m + i] = sum_l A[i*k+l]*B[j*k+l].
void ref_matmul_abt(const std::vector<float>& a, const std::vector<float>& b,
                    std::vector<float>& out, int m, int k, int n) {
  out.assign(static_cast<size_t>(m) * n, 0.0f);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      float acc = 0.0f;
      for (int l = 0; l < k; ++l) acc += a[i * k + l] * b[j * k + l];
      out[j * m + i] = acc;
    }
  }
}

// Run one ggml matmul on `device` through the bridge: prepare host A/B, call
// ggml_matmul_f32 (spawn_blocking + JoinHandle), read host C, compare to the golden
// reference. Returns 0 on success, negative step code on failure. Runs inside a fiber
// (join() is fiber-aware).
int run_matmul(const Device& device, BlockingPool& pool, int m, int k, int n,
               int n_threads) {
  auto be = GgmlBackend::create(device, n_threads);
  if (!be.has_value()) return -1;

  std::vector<float> ha(static_cast<size_t>(m) * k), hb(static_cast<size_t>(n) * k);
  for (int i = 0; i < m * k; ++i) ha[i] = static_cast<float>((i % 7) - 3) * 0.5f;
  for (int i = 0; i < n * k; ++i) hb[i] = static_cast<float>((i % 5) - 2) * 0.25f;

  std::vector<float> hc(static_cast<size_t>(m) * n, -12345.0f);
  auto h = ggml_matmul_f32(*be, pool, system_allocator(), ha.data(), hb.data(),
                           hc.data(), m, k, n);
  auto r = h.join();  // fiber-aware suspend until the pool thread's ggml compute done
  if (!r.has_value()) return -5;
  if (!r->ok()) return -6;

  std::vector<float> ref;
  ref_matmul_abt(ha, hb, ref, m, k, n);
  for (size_t i = 0; i < ref.size(); ++i) {
    if (std::fabs(hc[i] - ref[i]) > 1e-3f * (1.0f + std::fabs(ref[i]))) return -8;
  }
  return 0;
}

struct MatmulArg {
  Device device;
  BlockingPool* pool;
  int m, k, n, n_threads;
  int result = -100;
  std::atomic<bool> done{false};

  MatmulArg(const Device& d, BlockingPool* p, int m_, int k_, int n_, int nt)
      : device(d), pool(p), m(m_), k(k_), n(n_), n_threads(nt) {}
};
void matmul_fiber(void* p) {
  auto* a = static_cast<MatmulArg*>(p);
  a->result = run_matmul(a->device, *a->pool, a->m, a->k, a->n, a->n_threads);
  a->done.store(true, std::memory_order_release);
}

// Drive one matmul fiber to completion on a scheduler; return its result code.
int drive_one(const Device& device, BlockingPool& pool, int m, int k, int n,
              int n_threads) {
  MatmulArg arg{device, &pool, m, k, n, n_threads};
  auto f = std::make_unique<Fiber>(kDefaultStackSize,
                                   &matmul_fiber, &arg);
  Scheduler sched(2, system_allocator(), 256);
  sched.spawn(f.get());
  sched.drain();
  sched.shutdown();
  EXPECT_TRUE(arg.done.load());
  return arg.result;
}

}  // namespace

// ① + ② + golden compare on the DEFAULT device (GPU if present -> ggml CUDA path with
// gallocr device buffers + tensor_set/get; else host -> ggml CPU). The decisive "HYLE
// host mem + HYLE fiber sched + ggml operator (on ggml VRAM)" end-to-end proof.
TEST(GgmlBridge, MatmulDefaultDevice) {
  BlockingPool pool(system_allocator());
  int rc = drive_one(default_device(), pool, /*m=*/8, /*k=*/16, /*n=*/4,
                     /*n_threads=*/1);
  EXPECT_EQ(rc, 0) << "matmul failed at step " << rc;
}

// ③ CPU thread-model split: the SAME matmul on the ggml CPU backend at n_threads=1
// (independent-small-inference leaf, zero extra threads) AND at n_threads>1 (batched
// full-core path) must both produce the correct result -- proving n_threads is a
// working config knob, not a hardcoded 1. (Host CPU device via enumerate.)
TEST(GgmlBridge, CpuThreadModelSplit) {
  // Find the CPU device (always last in enumerate()).
  Device devs[8];
  std::size_t n = enumerate(devs, 8);
  ASSERT_GT(n, 0u);
  Device cpu = devs[n - 1];
  ASSERT_FALSE(cpu.is_gpu());

  BlockingPool pool(system_allocator());
  EXPECT_EQ(drive_one(cpu, pool, 8, 16, 4, /*n_threads=*/1), 0) << "n_threads=1 leaf";
  EXPECT_EQ(drive_one(cpu, pool, 8, 16, 4, /*n_threads=*/4), 0) << "n_threads=4 batched";
}

// ---- compute-09 step-1: elementwise / norm operators through the SHARED graph executor ----
// Proves the bridge op face extends past matmul cheaply: rms_norm / silu / add / mul each run
// end-to-end (host in -> ggml op on ggml VRAM -> host out) via the same spawn_blocking + gallocr
// path, and match an independent host scalar reference. This is the "not a toy" step: adding an
// operator is now one build-graph line, not a copied 50-line function.
namespace {
// Host scalar references (independent of ggml).
void ref_rms_norm(const std::vector<float>& x, std::vector<float>& out, float eps) {
  double ss = 0.0;
  for (float v : x) ss += double(v) * v;
  const float inv = static_cast<float>(1.0 / std::sqrt(ss / x.size() + eps));
  out.resize(x.size());
  for (size_t i = 0; i < x.size(); ++i) out[i] = x[i] * inv;
}

// Run the 4 elementwise ops on `device`, each compared to its host reference. Returns 0 or a
// negative step code. Runs in a fiber (join() is fiber-aware).
int run_elementwise(const Device& device, BlockingPool& pool) {
  auto be = GgmlBackend::create(device, /*n_threads=*/1);
  if (!be.has_value()) return -1;
  constexpr int kN = 323;  // prime-ish (17*19), NOT a multiple of 8/16/64 -> exercises SIMD tail
  std::vector<float> a(kN), b(kN), out(kN, -12345.0f);
  for (int i = 0; i < kN; ++i) { a[i] = float((i % 23) - 11) * 0.7f; b[i] = float((i % 7) - 3) * 0.5f; }

  // rms_norm
  {
    auto h = ggml_rms_norm_f32(*be, pool, system_allocator(), a.data(), out.data(), kN, 1e-6f);
    auto r = h.join();
    if (!r.has_value() || !r->ok()) return -10;
    std::vector<float> ref; ref_rms_norm(a, ref, 1e-6f);
    for (int i = 0; i < kN; ++i) if (std::fabs(out[i] - ref[i]) > 1e-4f * (1.0f + std::fabs(ref[i]))) return -11;
  }
  // silu: x / (1 + exp(-x))
  {
    auto h = ggml_silu_f32(*be, pool, system_allocator(), a.data(), out.data(), kN);
    auto r = h.join();
    if (!r.has_value() || !r->ok()) return -20;
    for (int i = 0; i < kN; ++i) {
      float ref = a[i] / (1.0f + std::exp(-a[i]));
      if (std::fabs(out[i] - ref) > 1e-4f * (1.0f + std::fabs(ref))) return -21;
    }
  }
  // add
  {
    auto h = ggml_add_f32(*be, pool, system_allocator(), a.data(), b.data(), out.data(), kN);
    auto r = h.join();
    if (!r.has_value() || !r->ok()) return -30;
    for (int i = 0; i < kN; ++i) if (std::fabs(out[i] - (a[i] + b[i])) > 1e-5f) return -31;
  }
  // mul
  {
    auto h = ggml_mul_f32(*be, pool, system_allocator(), a.data(), b.data(), out.data(), kN);
    auto r = h.join();
    if (!r.has_value() || !r->ok()) return -40;
    for (int i = 0; i < kN; ++i) if (std::fabs(out[i] - (a[i] * b[i])) > 1e-5f) return -41;
  }
  return 0;
}
struct EwArg { Device device; BlockingPool* pool; int result = -100; std::atomic<bool> done{false}; };
void elementwise_fiber(void* p) {
  auto* a = static_cast<EwArg*>(p);
  a->result = run_elementwise(a->device, *a->pool);
  a->done.store(true, std::memory_order_release);
}
int drive_elementwise(const Device& device, BlockingPool& pool) {
  EwArg arg{device, &pool};
  auto f = std::make_unique<Fiber>(kDefaultStackSize, &elementwise_fiber, &arg);
  Scheduler sched(2, system_allocator(), 256);
  sched.spawn(f.get());
  sched.drain();
  sched.shutdown();
  EXPECT_TRUE(arg.done.load());
  return arg.result;
}
}  // namespace

TEST(GgmlBridge, ElementwiseOpsDefaultDevice) {
  BlockingPool pool(system_allocator());
  int rc = drive_elementwise(default_device(), pool);
  EXPECT_EQ(rc, 0) << "elementwise op failed at step " << rc;
}
TEST(GgmlBridge, ElementwiseOpsCpu) {
  Device devs[8];
  std::size_t n = enumerate(devs, 8);
  ASSERT_GT(n, 0u);
  Device cpu = devs[n - 1];
  ASSERT_FALSE(cpu.is_gpu());
  BlockingPool pool(system_allocator());
  EXPECT_EQ(drive_elementwise(cpu, pool), 0);
}

// ---- compute-09 step-2: FFN block with RESIDENT weights, end-to-end vs host scalar reference ----
namespace {
// Host scalar reference for the whole LLaMA FFN block. Layouts match FfnBlock::create:
//   x=[rows*d], w_norm=[d], w_gate=[ffn*d] (ffn rows of d), w_up=[ffn*d], w_down=[d*ffn] (d rows of ffn).
void ref_ffn(const std::vector<float>& x, const std::vector<float>& wn,
             const std::vector<float>& wg, const std::vector<float>& wu,
             const std::vector<float>& wd, std::vector<float>& out, int rows, int d, int ffn,
             float eps) {
  out.assign(static_cast<size_t>(rows) * d, 0.0f);
  std::vector<float> h(d), gate(ffn), up(ffn), gu(ffn);
  for (int r = 0; r < rows; ++r) {
    const float* xr = &x[static_cast<size_t>(r) * d];
    double ss = 0.0;
    for (int j = 0; j < d; ++j) ss += double(xr[j]) * xr[j];
    const float inv = static_cast<float>(1.0 / std::sqrt(ss / d + eps));
    for (int j = 0; j < d; ++j) h[j] = xr[j] * inv * wn[j];  // rms_norm ⊙ w_norm
    for (int f = 0; f < ffn; ++f) {
      float acc = 0.0f;
      for (int j = 0; j < d; ++j) acc += h[j] * wg[static_cast<size_t>(f) * d + j];
      gate[f] = acc / (1.0f + std::exp(-acc));  // silu
      float accu = 0.0f;
      for (int j = 0; j < d; ++j) accu += h[j] * wu[static_cast<size_t>(f) * d + j];
      up[f] = accu;
      gu[f] = gate[f] * up[f];
    }
    for (int j = 0; j < d; ++j) {
      float acc = 0.0f;
      for (int f = 0; f < ffn; ++f) acc += gu[f] * wd[static_cast<size_t>(j) * ffn + f];
      out[static_cast<size_t>(r) * d + j] = xr[j] + acc;  // residual
    }
  }
}

int run_ffn(const Device& device, BlockingPool& pool) {
  constexpr int kD = 32, kFfn = 88, kRows = 5;  // ffn/rows non-round to exercise shapes
  // Tolerance is backend-aware: the ggml CPU backend matches the host scalar reference tightly
  // (~1e-3), but the CUDA backend uses fused/reduced-precision matmul accumulation -- GPU vs
  // ggml-CPU on the SAME graph diverge ~0.19% on large values (measured 3.5e-2 abs at |v|~18),
  // which cancellation in the residual amplifies to ~1.5% relative at near-zero outputs. That is
  // GPU fp32-matmul precision, NOT a bug (both backends compute the same graph). So the GPU bar
  // is a looser rel + abs floor; the CPU bar stays tight (real correctness gate for the shapes).
  const float rel = device.is_gpu() ? 3e-2f : 2e-3f;
  const float absf = device.is_gpu() ? 2e-2f : 1e-3f;
  std::vector<float> wn(kD), wg(static_cast<size_t>(kFfn) * kD), wu(static_cast<size_t>(kFfn) * kD),
      wd(static_cast<size_t>(kD) * kFfn), x(static_cast<size_t>(kRows) * kD);
  auto fill = [](std::vector<float>& v, int salt) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = float((int(i) * 13 + salt * 7) % 17 - 8) * 0.05f;
  };
  fill(wn, 1); fill(wg, 2); fill(wu, 3); fill(wd, 4); fill(x, 5);
  for (auto& v : wn) v += 0.9f;  // norm weight near 1 (realistic)

  auto blk = FfnBlock::create(device, kD, kFfn, 1e-5f, wn.data(), wg.data(), wu.data(),
                              wd.data(), /*n_threads=*/1);
  if (!blk.has_value()) return -1;
  // MOVE coverage: move-construct, then run every forward on the moved-into block. Proves the
  // move-ctor transfers weight/buffer/backend ownership (weights stay resident) and the
  // moved-from block's dtor is a safe no-op (nulled handles). blk is dead after this line.
  FfnBlock mvd = std::move(*blk);
  std::vector<float> out(static_cast<size_t>(kRows) * kD, -12345.0f);
  auto h = mvd.forward(pool, system_allocator(), x.data(), out.data(), kRows);
  auto r = h.join();
  if (!r.has_value() || !r->ok()) return -5;
  std::vector<float> ref;
  ref_ffn(x, wn, wg, wu, wd, ref, kRows, kD, kFfn, 1e-5f);
  for (size_t i = 0; i < ref.size(); ++i)
    if (std::fabs(out[i] - ref[i]) > rel * std::fabs(ref[i]) + absf) return -8;

  // RESIDENCY proof: a SECOND forward with a DIFFERENT activation reuses the same resident
  // weights (no re-create) and must still match the reference for the new input.
  std::vector<float> x2(static_cast<size_t>(kRows) * kD), out2(static_cast<size_t>(kRows) * kD, 0.0f);
  fill(x2, 99);
  auto h2 = mvd.forward(pool, system_allocator(), x2.data(), out2.data(), kRows);
  auto r2 = h2.join();
  if (!r2.has_value() || !r2->ok()) return -10;
  std::vector<float> ref2;
  ref_ffn(x2, wn, wg, wu, wd, ref2, kRows, kD, kFfn, 1e-5f);
  for (size_t i = 0; i < ref2.size(); ++i)
    if (std::fabs(out2[i] - ref2[i]) > rel * std::fabs(ref2[i]) + absf) return -11;
  return 0;
}
struct FfnArg { Device device; BlockingPool* pool; int result = -100; std::atomic<bool> done{false}; };
void ffn_fiber(void* p) {
  auto* a = static_cast<FfnArg*>(p);
  a->result = run_ffn(a->device, *a->pool);
  a->done.store(true, std::memory_order_release);
}
int drive_ffn(const Device& device, BlockingPool& pool) {
  FfnArg arg{device, &pool};
  auto f = std::make_unique<Fiber>(kDefaultStackSize, &ffn_fiber, &arg);
  Scheduler sched(2, system_allocator(), 256);
  sched.spawn(f.get());
  sched.drain();
  sched.shutdown();
  EXPECT_TRUE(arg.done.load());
  return arg.result;
}
}  // namespace

TEST(GgmlBridge, FfnBlockDefaultDevice) {
  BlockingPool pool(system_allocator());
  EXPECT_EQ(drive_ffn(default_device(), pool), 0) << "FFN block failed";
}
TEST(GgmlBridge, FfnBlockCpu) {
  Device devs[8];
  std::size_t n = enumerate(devs, 8);
  ASSERT_GT(n, 0u);
  Device cpu = devs[n - 1];
  ASSERT_FALSE(cpu.is_gpu());
  BlockingPool pool(system_allocator());
  EXPECT_EQ(drive_ffn(cpu, pool), 0);
}

// Thread-safety contract (spec 线程安全契约): N fibers each hold their OWN GgmlBackend
// and run independent matmuls concurrently ("high-concurrency independent small
// inference" grid). All must be correct -- no shared backend, no race (WSL TSan clears
// this). Also exercises spawn_blocking not stalling the workers (fibers overlap).
TEST(GgmlBridge, ConcurrentIndependentBackends) {
  constexpr int kFibers = 6;
  Device d = default_device();
  BlockingPool pool(system_allocator());

  std::vector<std::unique_ptr<MatmulArg>> args;
  std::vector<std::unique_ptr<Fiber>> fibers;
  args.reserve(kFibers);
  fibers.reserve(kFibers);
  Scheduler sched(4, system_allocator(), 256);
  for (int i = 0; i < kFibers; ++i) {
    auto a = std::make_unique<MatmulArg>(d, &pool, 8, 16, 4, /*n_threads=*/1);
    auto f = std::make_unique<Fiber>(kDefaultStackSize,
                                     &matmul_fiber, a.get());
    sched.spawn(f.get());
    args.push_back(std::move(a));
    fibers.push_back(std::move(f));
  }
  sched.drain();
  sched.shutdown();
  for (int i = 0; i < kFibers; ++i) {
    EXPECT_TRUE(args[i]->done.load()) << "fiber " << i;
    EXPECT_EQ(args[i]->result, 0) << "fiber " << i << " failed at step " << args[i]->result;
  }
}

namespace {
// Run a BATCH of matmuls (finding 4): N matmuls with varied shapes built into ONE ggml
// graph, ONE submit. Each result golden-compared. Returns 0 / negative step code.
// VARIED SHAPES ARE THE POINT: the earlier size-0 dummy-buffer injection crashed exactly
// here (ggml-cuda.cu:1924, varied shapes take the generic mul_mat path reading a nulled
// buffer context; docs/debug/ggml-zerocopy-null-buffer-hunt.md). The gallocr + tensor_set
// model has valid buffer contexts, so varied-shape batches are correct on CUDA.
int run_batch(const Device& device, BlockingPool& pool, int n_threads) {
  auto be = GgmlBackend::create(device, n_threads);
  if (!be.has_value()) return -1;

  // 3 jobs, deliberately different shapes (proves per-job dims in one graph).
  struct Spec { int m, k, n; };
  const Spec specs[3] = {{8, 16, 4}, {4, 8, 6}, {12, 4, 3}};
  constexpr int kJobs = 3;

  std::vector<float> ha[kJobs], hb[kJobs], hc[kJobs], ref[kJobs];
  MatmulF32Job jobs[kJobs];

  for (int i = 0; i < kJobs; ++i) {
    int m = specs[i].m, k = specs[i].k, n = specs[i].n;
    ha[i].resize(static_cast<size_t>(m) * k);
    hb[i].resize(static_cast<size_t>(n) * k);
    hc[i].assign(static_cast<size_t>(m) * n, -12345.0f);
    for (int j = 0; j < m * k; ++j) ha[i][j] = static_cast<float>((j + i) % 7 - 3) * 0.5f;
    for (int j = 0; j < n * k; ++j) hb[i][j] = static_cast<float>((j + i) % 5 - 2) * 0.25f;
    ref_matmul_abt(ha[i], hb[i], ref[i], m, k, n);
    jobs[i] = MatmulF32Job{ha[i].data(), hb[i].data(), hc[i].data(), m, k, n};
  }

  auto h = ggml_matmul_f32_batched(*be, pool, system_allocator(), jobs, kJobs);
  auto r = h.join();
  if (!r.has_value()) return -5;
  if (!r->ok()) return -6;

  for (int i = 0; i < kJobs; ++i) {
    for (size_t j = 0; j < ref[i].size(); ++j)
      if (std::fabs(hc[i][j] - ref[i][j]) > 1e-3f * (1.0f + std::fabs(ref[i][j]))) return -(100 + i);
  }
  return 0;
}

struct BatchArg {
  Device device;
  BlockingPool* pool;
  int result = -100;
  std::atomic<bool> done{false};
  BatchArg(const Device& d, BlockingPool* p) : device(d), pool(p) {}
};
void batch_fiber(void* p) {
  auto* a = static_cast<BatchArg*>(p);
  a->result = run_batch(a->device, *a->pool, /*n_threads=*/1);
  a->done.store(true, std::memory_order_release);
}
}  // namespace

// 合批 (finding 4): N matmuls of DIFFERENT shapes in ONE graph / ONE submit, each result
// golden-correct. Proves the batching path (one graph_compute for many ops) works AND
// that varied shapes are safe under the gallocr model (the dummy-buffer regression's
// crash case, now correct).
TEST(GgmlBridge, BatchedMatmul) {
  BlockingPool pool(system_allocator());
  BatchArg arg{default_device(), &pool};
  auto f = std::make_unique<Fiber>(kDefaultStackSize,
                                   &batch_fiber, &arg);
  Scheduler sched(2, system_allocator(), 256);
  sched.spawn(f.get());
  sched.drain();
  sched.shutdown();
  EXPECT_TRUE(arg.done.load());
  EXPECT_EQ(arg.result, 0) << "batched matmul failed at step " << arg.result;
}

namespace {
// One F32 matmul helper on an EXISTING backend (host data prepared + golden-checked).
// Unlike run_matmul it does NOT create a backend -- it reuses the caller's, which is the
// whole point of this test. Returns 0 / negative step code.
int one_on(GgmlBackend& be, BlockingPool& pool, int m, int k, int n, int salt) {
  std::vector<float> ha(static_cast<size_t>(m) * k), hb(static_cast<size_t>(n) * k);
  for (int i = 0; i < m * k; ++i) ha[i] = static_cast<float>((i + salt) % 7 - 3) * 0.5f;
  for (int i = 0; i < n * k; ++i) hb[i] = static_cast<float>((i + salt) % 5 - 2) * 0.25f;
  std::vector<float> hc(static_cast<size_t>(m) * n, -12345.0f), ref;
  auto h = ggml_matmul_f32(be, pool, system_allocator(), ha.data(), hb.data(),
                           hc.data(), m, k, n);
  auto r = h.join();
  if (!r.has_value() || !r->ok()) return -1;
  ref_matmul_abt(ha, hb, ref, m, k, n);
  for (size_t i = 0; i < ref.size(); ++i)
    if (std::fabs(hc[i] - ref[i]) > 1e-3f * (1.0f + std::fabs(ref[i]))) return -2;
  return 0;
}

// Reuse ONE persistent GgmlBackend (create-once/reuse-many, llama.cpp model) across many
// ops of DIFFERENT shapes, including a batched call in the middle. This exercises the
// persistent gallocr's reserve/realloc as the graph shape/node-count changes between
// calls -- the exact path the old dummy-buffer injection crashed on (varied shapes on a
// shared galloc). Also moves the backend first, covering move semantics. Returns 0 / code.
int run_reuse(const Device& device, BlockingPool& pool) {
  auto created = GgmlBackend::create(device, /*n_threads=*/1);
  if (!created.has_value()) return -1;
  GgmlBackend be = std::move(*created);  // move-construct: dummy_/galloc_ transfer intact
  if (!be.valid()) return -2;

  // Single ops of changing shapes on the SAME backend (galloc realloc between shapes).
  if (one_on(be, pool, 8, 16, 4, 0) != 0) return -10;
  if (one_on(be, pool, 4, 8, 6, 1) != 0) return -11;   // shape change -> galloc reserve
  if (one_on(be, pool, 32, 32, 8, 2) != 0) return -12;  // bigger -> realloc grow

  // A batched (multi-node, varied-shape) call on the same galloc, then back to single.
  struct Spec { int m, k, n; };
  const Spec specs[2] = {{8, 16, 4}, {12, 4, 3}};
  std::vector<float> ha[2], hb[2], hc[2], ref[2];
  MatmulF32Job jobs[2];
  for (int i = 0; i < 2; ++i) {
    int m = specs[i].m, k = specs[i].k, n = specs[i].n;
    ha[i].resize(static_cast<size_t>(m) * k);
    hb[i].resize(static_cast<size_t>(n) * k);
    hc[i].assign(static_cast<size_t>(m) * n, -12345.0f);
    for (int j = 0; j < m * k; ++j) ha[i][j] = static_cast<float>((j + i) % 7 - 3) * 0.5f;
    for (int j = 0; j < n * k; ++j) hb[i][j] = static_cast<float>((j + i) % 5 - 2) * 0.25f;
    ref_matmul_abt(ha[i], hb[i], ref[i], m, k, n);
    jobs[i] = MatmulF32Job{ha[i].data(), hb[i].data(), hc[i].data(), m, k, n};
  }
  {
    auto h = ggml_matmul_f32_batched(be, pool, system_allocator(), jobs, 2);
    auto r = h.join();
    if (!r.has_value() || !r->ok()) return -20;
    for (int i = 0; i < 2; ++i)
      for (size_t j = 0; j < ref[i].size(); ++j)
        if (std::fabs(hc[i][j] - ref[i][j]) > 1e-3f * (1.0f + std::fabs(ref[i][j]))) return -(30 + i);
  }

  // Back to a single op (graph shrinks 2 nodes -> 1: galloc must handle node-count drop).
  if (one_on(be, pool, 8, 16, 4, 0) != 0) return -40;
  return 0;
}

struct ReuseArg {
  Device device;
  BlockingPool* pool;
  int result = -100;
  std::atomic<bool> done{false};
  ReuseArg(const Device& d, BlockingPool* p) : device(d), pool(p) {}
};
void reuse_fiber(void* p) {
  auto* a = static_cast<ReuseArg*>(p);
  a->result = run_reuse(a->device, *a->pool);
  a->done.store(true, std::memory_order_release);
}
}  // namespace

// Persistent-backend reuse (the create-once/reuse-many design + move semantics): ONE
// GgmlBackend (moved after create) drives single ops of changing shapes, a batched
// varied-shape call, then a single op again -- all sharing the one persistent gallocr.
// This is the regression guard for gallocr reserve/realloc across shape + node-count
// changes (the exact scenario the old dummy-buffer model crashed on).
TEST(GgmlBridge, BackendReuseAcrossShapes) {
  BlockingPool pool(system_allocator());
  ReuseArg arg{default_device(), &pool};
  auto f = std::make_unique<Fiber>(kDefaultStackSize,
                                   &reuse_fiber, &arg);
  Scheduler sched(2, system_allocator(), 256);
  sched.spawn(f.get());
  sched.drain();
  sched.shutdown();
  EXPECT_TRUE(arg.done.load());
  EXPECT_EQ(arg.result, 0) << "backend reuse failed at step " << arg.result;
}

// ---- CONTRACT death tests (ggml_bridge.cc:170-172 / 193-206) ------------------
// The bridge's caller-contract asserts fire SYNCHRONOUSLY on the CALLING thread (before
// spawn_blocking offloads), so a death test triggers them WITHOUT a scheduler/fiber -- a
// BlockingPool is constructed (the signature needs it) but never drained. The CPU ggml
// backend (ggml_backend_cpu_init) is unconditionally available, so we mint a real, valid
// GgmlBackend on the CPU device and then feed it a contract-violating argument.
//
// Matcher "HYLE fatal" proves death came from HYLE_FATAL_ASSERT (contract-over-defense),
// not an incidental crash. Threadsafe style: a GPU machine has CUDA initialized by other
// tests; re-exec avoids fork-after-CUDA-init. Each scenario is a named function (the
// death statement's commas would confuse the EXPECT_DEATH macro).

namespace {

Device cpu_device() {
  Device devs[8];
  std::size_t n = enumerate(devs, 8);
  return devs[n - 1];  // host CPU is always last
}

void matmul_null_output() {
  auto be = GgmlBackend::create(cpu_device(), /*n_threads=*/1);
  BlockingPool pool(system_allocator());
  float ab[16] = {};
  // c == nullptr violates ggml_bridge.cc:171 (a/b/c non-null).
  (void)ggml_matmul_f32(*be, pool, system_allocator(), ab, ab, nullptr, 2, 2, 2);
}

void matmul_bad_dims() {
  auto be = GgmlBackend::create(cpu_device(), /*n_threads=*/1);
  BlockingPool pool(system_allocator());
  float ab[16] = {};
  // k == 0 violates ggml_bridge.cc:172 (m/k/n > 0).
  (void)ggml_matmul_f32(*be, pool, system_allocator(), ab, ab, ab, 2, 0, 2);
}

void batched_count_zero() {
  auto be = GgmlBackend::create(cpu_device(), /*n_threads=*/1);
  BlockingPool pool(system_allocator());
  MatmulF32Job jobs[1] = {};
  // count == 0 violates ggml_bridge.cc:194 (count in [1,64]).
  (void)ggml_matmul_f32_batched(*be, pool, system_allocator(), jobs, 0);
}

void batched_count_over_cap() {
  auto be = GgmlBackend::create(cpu_device(), /*n_threads=*/1);
  BlockingPool pool(system_allocator());
  MatmulF32Job jobs[1] = {};
  // count == 65 > 64 violates ggml_bridge.cc:194.
  (void)ggml_matmul_f32_batched(*be, pool, system_allocator(), jobs, 65);
}

void batched_null_jobs() {
  auto be = GgmlBackend::create(cpu_device(), /*n_threads=*/1);
  BlockingPool pool(system_allocator());
  // jobs == nullptr violates ggml_bridge.cc:195.
  (void)ggml_matmul_f32_batched(*be, pool, system_allocator(), nullptr, 1);
}

void batched_job_bad_dims() {
  auto be = GgmlBackend::create(cpu_device(), /*n_threads=*/1);
  BlockingPool pool(system_allocator());
  float ab[16] = {};
  MatmulF32Job jobs[1] = {MatmulF32Job{ab, ab, ab, 2, 0, 2}};  // k==0 -> :206
  (void)ggml_matmul_f32_batched(*be, pool, system_allocator(), jobs, 1);
}

// compute-09 elementwise contract violations (mirror the matmul death tests).
void silu_null_dst() {
  auto be = GgmlBackend::create(cpu_device(), 1);
  BlockingPool pool(system_allocator());
  float s[8] = {};
  (void)ggml_silu_f32(*be, pool, system_allocator(), s, nullptr, 8);  // dst null -> assert
}
void add_bad_n() {
  auto be = GgmlBackend::create(cpu_device(), 1);
  BlockingPool pool(system_allocator());
  float ab[8] = {};
  (void)ggml_add_f32(*be, pool, system_allocator(), ab, ab, ab, 0);  // n==0 -> assert
}
void add_null_b() {
  auto be = GgmlBackend::create(cpu_device(), 1);
  BlockingPool pool(system_allocator());
  float ab[8] = {};
  (void)ggml_add_f32(*be, pool, system_allocator(), ab, nullptr, ab, 8);  // binary b null -> assert
}
void rms_norm_negative_eps() {
  auto be = GgmlBackend::create(cpu_device(), 1);
  BlockingPool pool(system_allocator());
  float s[8] = {}, d[8] = {};
  (void)ggml_rms_norm_f32(*be, pool, system_allocator(), s, d, 8, -1.0f);  // eps<0 -> assert
}

// compute-09 step-2 FfnBlock::create contract violations (fail-fast in HYLE, so a caller bug
// never sinks into a ggml abort on the pool thread -- the compute-07/09 lesson).
void ffn_bad_dim() {
  float w[16] = {};
  (void)FfnBlock::create(cpu_device(), /*d=*/0, /*ffn=*/4, 1e-5f, w, w, w, w);  // d<=0 -> assert
}
void ffn_null_weight() {
  float w[16] = {};
  (void)FfnBlock::create(cpu_device(), 4, 4, 1e-5f, w, nullptr, w, w);  // w_gate null -> assert
}

}  // namespace

TEST(GgmlBridgeDeathTest, MatmulNullOutput) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(matmul_null_output(), "HYLE fatal");
}

TEST(GgmlBridgeDeathTest, MatmulBadDims) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(matmul_bad_dims(), "HYLE fatal");
}

TEST(GgmlBridgeDeathTest, BatchedCountZero) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(batched_count_zero(), "HYLE fatal");
}

TEST(GgmlBridgeDeathTest, BatchedCountOverCap) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(batched_count_over_cap(), "HYLE fatal");
}

TEST(GgmlBridgeDeathTest, BatchedNullJobs) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(batched_null_jobs(), "HYLE fatal");
}

TEST(GgmlBridgeDeathTest, BatchedJobBadDims) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(batched_job_bad_dims(), "HYLE fatal");
}

// compute-09 elementwise contract death tests (caller-bug fail-fast, like matmul above).
TEST(GgmlBridgeDeathTest, SiluNullDst) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(silu_null_dst(), "HYLE fatal");
}
TEST(GgmlBridgeDeathTest, AddBadN) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(add_bad_n(), "HYLE fatal");
}
TEST(GgmlBridgeDeathTest, AddNullB) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(add_null_b(), "HYLE fatal");
}
TEST(GgmlBridgeDeathTest, RmsNormNegativeEps) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(rms_norm_negative_eps(), "HYLE fatal");
}

TEST(GgmlBridgeDeathTest, FfnBadDim) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(ffn_bad_dim(), "HYLE fatal");
}

TEST(GgmlBridgeDeathTest, FfnNullWeight) {
  HYLE_THREADSAFE_DEATH();
  EXPECT_DEATH(ffn_null_weight(), "HYLE fatal");
}
