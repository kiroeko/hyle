#include "hyle/compute/ggml_bridge.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "hyle/compute/buffer.h"
#include "hyle/compute/device.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/blocking.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

// compute-05 test: borrow a ggml matmul operator to run on HYLE-managed VRAM (zero-
// copy device_ptr injection), offloaded via spawn_blocking so a fiber suspends while
// the ggml graph computes on a pool thread, woken through the JoinHandle. Proves the
// three load-bearing decisions (ggml_bridge.h ①②③) compose end-to-end.
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

// Run one ggml matmul on `device` through the bridge: alloc 3 HYLE buffers, upload
// A/B, call ggml_matmul_f32 (spawn_blocking + JoinHandle), download C, compare to the
// golden reference. Returns 0 on success, negative step code on failure. Runs inside a
// fiber (join() is fiber-aware). `zero_copy` toggles nothing in the API here (the API
// is always zero-copy); the copy-vs-zero-copy golden compare is the same reference.
int run_matmul(const Device& device, BlockingPool& pool, int m, int k, int n,
               int n_threads) {
  auto be = GgmlBackend::create(device, n_threads);
  if (!be.has_value()) return -1;

  std::vector<float> ha(static_cast<size_t>(m) * k), hb(static_cast<size_t>(n) * k);
  for (int i = 0; i < m * k; ++i) ha[i] = static_cast<float>((i % 7) - 3) * 0.5f;
  for (int i = 0; i < n * k; ++i) hb[i] = static_cast<float>((i % 5) - 2) * 0.25f;

  BufferType bt = device_buffer_type(device);
  auto ba = Buffer::create(bt, ha.size() * sizeof(float));
  auto bb = Buffer::create(bt, hb.size() * sizeof(float));
  auto bc = Buffer::create(bt, static_cast<size_t>(m) * n * sizeof(float));
  if (!ba.has_value() || !bb.has_value() || !bc.has_value()) return -2;
  if (!ba->set(0, ha.data(), ha.size() * sizeof(float)).ok()) return -3;
  if (!bb->set(0, hb.data(), hb.size() * sizeof(float)).ok()) return -4;

  auto h = ggml_matmul_f32(*be, pool, system_allocator(), *ba, *bb, *bc, m, k, n);
  auto r = h.join();  // fiber-aware suspend until the pool thread's ggml compute done
  if (!r.has_value()) return -5;
  if (!r->ok()) return -6;

  std::vector<float> hc(static_cast<size_t>(m) * n, -12345.0f);
  if (!bc->get(0, hc.data(), hc.size() * sizeof(float)).ok()) return -7;

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
  auto f = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                   &matmul_fiber, &arg);
  Scheduler sched(2, system_allocator(), 256);
  sched.spawn(f.get());
  sched.drain();
  sched.shutdown();
  EXPECT_TRUE(arg.done.load());
  return arg.result;
}

}  // namespace

// ① + ② + golden compare on the DEFAULT device (GPU if present -> zero-copy CUDA path;
// else host -> ggml CPU). The decisive "HYLE VRAM + HYLE fiber sched + ggml operator"
// end-to-end proof.
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
    auto f = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
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
int run_batch(const Device& device, BlockingPool& pool, int n_threads) {
  auto be = GgmlBackend::create(device, n_threads);
  if (!be.has_value()) return -1;

  // 3 jobs, deliberately different shapes (proves per-job dims in one graph).
  struct Spec { int m, k, n; };
  const Spec specs[3] = {{8, 16, 4}, {4, 8, 6}, {12, 4, 3}};
  constexpr int kJobs = 3;

  std::vector<float> ha[kJobs], hb[kJobs], ref[kJobs];
  std::vector<Buffer> ba, bb, bc;  // keep Buffers alive across the join
  ba.reserve(kJobs); bb.reserve(kJobs); bc.reserve(kJobs);
  MatmulF32Job jobs[kJobs];
  BufferType bt = device_buffer_type(device);

  for (int i = 0; i < kJobs; ++i) {
    int m = specs[i].m, k = specs[i].k, n = specs[i].n;
    ha[i].resize(static_cast<size_t>(m) * k);
    hb[i].resize(static_cast<size_t>(n) * k);
    for (int j = 0; j < m * k; ++j) ha[i][j] = static_cast<float>((j + i) % 7 - 3) * 0.5f;
    for (int j = 0; j < n * k; ++j) hb[i][j] = static_cast<float>((j + i) % 5 - 2) * 0.25f;
    ref_matmul_abt(ha[i], hb[i], ref[i], m, k, n);

    auto a = Buffer::create(bt, ha[i].size() * sizeof(float));
    auto b = Buffer::create(bt, hb[i].size() * sizeof(float));
    auto c = Buffer::create(bt, static_cast<size_t>(m) * n * sizeof(float));
    if (!a.has_value() || !b.has_value() || !c.has_value()) return -2;
    if (!a->set(0, ha[i].data(), ha[i].size() * sizeof(float)).ok()) return -3;
    if (!b->set(0, hb[i].data(), hb[i].size() * sizeof(float)).ok()) return -4;
    jobs[i] = MatmulF32Job{a->device_ptr(), b->device_ptr(), c->device_ptr(), m, k, n};
    ba.push_back(std::move(*a)); bb.push_back(std::move(*b)); bc.push_back(std::move(*c));
  }

  auto h = ggml_matmul_f32_batched(*be, pool, system_allocator(), jobs, kJobs);
  auto r = h.join();
  if (!r.has_value()) return -5;
  if (!r->ok()) return -6;

  for (int i = 0; i < kJobs; ++i) {
    std::vector<float> hc(ref[i].size(), -12345.0f);
    if (!bc[i].get(0, hc.data(), hc.size() * sizeof(float)).ok()) return -7;
    for (size_t j = 0; j < ref[i].size(); ++j)
      if (std::fabs(hc[j] - ref[i][j]) > 1e-3f * (1.0f + std::fabs(ref[i][j]))) return -(100 + i);
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
// golden-correct. Proves the batching path (one graph_compute for many ops) works.
TEST(GgmlBridge, BatchedMatmul) {
  BlockingPool pool(system_allocator());
  BatchArg arg{default_device(), &pool};
  auto f = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                   &batch_fiber, &arg);
  Scheduler sched(2, system_allocator(), 256);
  sched.spawn(f.get());
  sched.drain();
  sched.shutdown();
  EXPECT_TRUE(arg.done.load());
  EXPECT_EQ(arg.result, 0) << "batched matmul failed at step " << arg.result;
}
