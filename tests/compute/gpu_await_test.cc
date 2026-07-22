#include "hyle/compute/gpu_await.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "hyle/compute/buffer.h"
#include "hyle/compute/device.h"
#include "hyle/compute/kernel.h"
#include "hyle/compute/stream.h"
#include "hyle/mem/allocator.h"
#include "hyle/sched/coro.h"
#include "hyle/sched/fiber.h"
#include "hyle/sched/scheduler.h"

// compute-03 test: a fiber co_awaits a GPU stream's completion. On a real GPU the
// stream has async work (device copies) in flight, so the fiber SUSPENDS and the GPU
// poller thread wakes it when the event fires -- proving GPU completion -> fiber wake
// (compute-design §4, same shape as io co_await). On the host fallback the event is
// immediately ready, so co_await completes without suspending (still exercises the
// path). The poller's Waker::wake() from its own thread re-injects the fiber via the
// scheduler's global queue + notify (the cross-thread wake TSan must clear).
//
// COVERAGE NOTE (A2, why host does NOT exercise suspend->poller-sweep->wake): the host
// backend's Event is ALWAYS kReady (inline execution, host_backend.cc:74), so
// StreamComplete::await_ready() returns true and never suspends -- BY DESIGN, not a gap.
// Driving the suspend/sweep/cross-thread-wake chain needs an Event that reports NotReady
// first; but Event's vt_/native_ are private (friend Stream only) with no test seam, so a
// "controllable event" would require changing the production header. We deliberately do
// NOT add a test-only seam: the full suspend->sweep->wake path is already proven on REAL
// hardware here (CoAwaitStreamCompletion / ConcurrentAwaits on a CUDA GPU, cross-thread
// wake cleared by WSL TSan) and on the Vulkan side (vulkan_test CoAwaitCompletion). The
// host path's job is only to prove the awaitable COMPILES + completes with no device.

using hyle::mem::system_allocator;
using hyle::sched::Fiber;
using hyle::sched::kDefaultStackSize;
using hyle::sched::Scheduler;
using hyle::sched::Task;
using namespace hyle;
using namespace hyle::compute;

namespace {

struct Drive {
  Task<int> task;
  std::atomic<bool> finished{false};
};
void drive_entry(void* p) {
  auto* d = static_cast<Drive*>(p);
  d->task.resume();
  d->finished.store(true, std::memory_order_release);
}

bool has_gpu() {
  Device devs[8];
  std::size_t n = enumerate(devs, 8);
  for (std::size_t i = 0; i < n && i < 8; ++i)
    if (devs[i].is_gpu()) return true;
  return false;
}

// A deliberately HEAVY GPU kernel: each thread spins a long accumulation loop before
// writing, so the launch is still IN FLIGHT when the fiber reaches co_await -- that is
// what makes await_ready() return false and drives the real suspend -> GpuPoller sweep
// -> cross-thread wake path (a trivial kernel could finish before the first poll). It
// writes out[i] = i + <loop-invariant> so the result is verifiable.
constexpr const char* kSpinKernel = R"(
extern "C" __global__ void spin(unsigned* out, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    unsigned acc = 0;
    for (int k = 0; k < iters; ++k) acc = acc * 1664525u + 1013904223u;
    out[i] = (unsigned)i + (acc & 0u);  // acc masked to 0: result is exactly i
  }
}
)";

// The host-fallback "kernel": same out[i]=i semantics, runs inline (synchronous), so on
// the host backend co_await completes WITHOUT suspending (event already ready) -- the
// same path host took before, now via a kernel instead of the removed raw memcpy.
void host_spin(const LaunchConfig&, const KernelArgs& args) noexcept {
  auto* out = static_cast<unsigned*>(args.buffers[0]);
  int n = *reinterpret_cast<const int*>(static_cast<const std::byte*>(args.scalars) +
                                        args.fields[0].offset);
  for (int i = 0; i < n; ++i) out[i] = static_cast<unsigned>(i);
}

// A coroutine that launches a kernel producing out[i]=i on a stream, co_awaits the
// stream's completion (SUSPENDS on a real GPU: the heavy kernel is still running), then
// reads the buffer back and verifies. Returns 0 on success, negative step code on the
// first failure. Kernel launch (compute-04) is the async, stream-bound work the removed
// raw memcpy_async used to provide -- Buffer set/get is synchronous so can't leave work
// in flight to observe.
Task<int> gpu_await_coro(mem::Allocator, int* out_result) {
  Device d = default_device();
  const bool gpu = has_gpu();
  auto s = Stream::create(d);
  if (!s.has_value()) { *out_result = -1; co_return -1; }

  constexpr int kN = 4096;
  const std::size_t bytes = kN * sizeof(unsigned);
  auto out = Buffer::create(device_buffer_type(d), bytes);
  if (!out.has_value()) { *out_result = -2; co_return -2; }

  auto k = gpu ? Kernel::compile(d, kSpinKernel, "spin")
               : Kernel::load_host(d, &host_spin);
  // A GPU backend without NVRTC source-compile (e.g. Vulkan wants SPIR-V via load_ptx,
  // covered by vulkan_test's CoAwaitCompletion) can't run this CUDA-C kernel -> SKIP
  // (code 1), not fail. This test's job is the CUDA + host completion->fiber path.
  if (gpu && !k.has_value()) { *out_result = 1; co_return 1; }
  if (!k.has_value()) { *out_result = -3; co_return -3; }

  int n = kN;
  int iters = gpu ? 200000 : 0;  // heavy on GPU so it's still in flight at co_await
  LaunchConfig cfg;
  cfg.block = {256, 1, 1};
  cfg.grid = {blocks_for(kN, 256), 1, 1};
  if (gpu) {
    if (!launch(*s, *k, cfg, *out, n, iters).ok()) { *out_result = -4; co_return -4; }
  } else {
    // Host kernel reads buffers[0] + scalar n via a raw KernelArgs (matches host_spin).
    void* bufs[] = {out->native_handle()};
    ScalarField fields[] = {{0, sizeof(int)}};
    KernelArgs ka{bufs, 1, &n, sizeof(int), fields, 1};
    if (!launch(*s, *k, cfg, ka).ok()) { *out_result = -4; co_return -4; }
  }

  Status st = co_await StreamComplete{&*s};  // SUSPEND until GPU done (poller wakes us)
  if (!st.ok()) { *out_result = -5; co_return -5; }

  // The kernel wrote out[i] = i by the time we resume.
  std::vector<unsigned> host_out(kN, 0xFFFFFFFFu);
  if (!out->get(0, host_out.data(), bytes).ok()) { *out_result = -6; co_return -6; }
  for (int i = 0; i < kN; ++i) {
    if (host_out[i] != static_cast<unsigned>(i)) { *out_result = -7; co_return -7; }
  }
  *out_result = 0;
  co_return 0;
}

}  // namespace

TEST(GpuAwait, CoAwaitStreamCompletion) {
  Scheduler sched(2, system_allocator(), 256);

  int result = -100;
  auto d = std::make_unique<Drive>();
  d->task = gpu_await_coro(system_allocator(), &result);
  Fiber f(kDefaultStackSize, &drive_entry, d.get());
  sched.spawn(&f);
  sched.drain();
  sched.shutdown();

  compute::detail::gpu_poll_shutdown();  // clean poller teardown

  EXPECT_TRUE(d->finished.load());
  if (result == 1) GTEST_SKIP() << "GPU backend has no NVRTC (Vulkan uses SPIR-V; see vulkan_test)";
  EXPECT_EQ(result, 0) << "gpu_await_coro failed at step " << result;
}

// Multiple fibers each co_await their OWN stream concurrently; all must wake and
// finish. Exercises the poller tracking several in-flight events at once.
TEST(GpuAwait, ConcurrentAwaits) {
  constexpr int kFibers = 8;
  Scheduler sched(4, system_allocator(), 256);

  std::vector<std::unique_ptr<Drive>> drives;
  std::vector<std::unique_ptr<Fiber>> fibers;
  std::vector<int> results(kFibers, -100);
  drives.reserve(kFibers);
  fibers.reserve(kFibers);
  for (int i = 0; i < kFibers; ++i) {
    auto d = std::make_unique<Drive>();
    d->task = gpu_await_coro(system_allocator(), &results[i]);
    auto f = std::make_unique<Fiber>(kDefaultStackSize,
                                     &drive_entry, d.get());
    sched.spawn(f.get());
    drives.push_back(std::move(d));
    fibers.push_back(std::move(f));
  }
  sched.drain();
  sched.shutdown();
  compute::detail::gpu_poll_shutdown();

  if (results[0] == 1) GTEST_SKIP() << "GPU backend has no NVRTC (Vulkan uses SPIR-V; see vulkan_test)";
  for (int i = 0; i < kFibers; ++i) {
    EXPECT_TRUE(drives[i]->finished.load()) << "fiber " << i;
    EXPECT_EQ(results[i], 0) << "fiber " << i << " failed at step " << results[i];
  }
}
