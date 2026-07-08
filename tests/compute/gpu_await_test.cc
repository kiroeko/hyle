#include "hyle/compute/gpu_await.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "hyle/compute/buffer.h"
#include "hyle/compute/device.h"
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

// A coroutine that: allocs a device buffer, enqueues async copies on a stream,
// co_awaits the stream's completion, then verifies the data round-tripped. Returns 0
// on success, a negative code on the first failing step (checked by the test).
Task<int> gpu_await_coro(mem::Allocator, int* out_result) {
  Device d = default_device();
  auto s = Stream::create(d);
  if (!s.has_value()) { *out_result = -1; co_return -1; }

  constexpr std::size_t kN = 4096;
  const std::size_t bytes = kN * sizeof(std::uint32_t);
  std::vector<std::uint32_t> host_in(kN), host_out(kN, 0);
  for (std::size_t i = 0; i < kN; ++i) host_in[i] = static_cast<std::uint32_t>(i * 5 + 2);

  auto dptr = s->device_alloc(bytes);
  if (!dptr.has_value()) { *out_result = -2; co_return -2; }

  // Enqueue async H2D then D2H copies on the stream (real GPU work -> event not
  // immediately ready), then co_await the stream draining.
  if (!s->memcpy_async(*dptr, host_in.data(), bytes).ok()) { *out_result = -3; co_return -3; }
  if (!s->memcpy_async(host_out.data(), *dptr, bytes).ok()) { *out_result = -4; co_return -4; }

  Status st = co_await StreamComplete{&*s};  // SUSPEND until GPU done (poller wakes us)
  if (!st.ok()) { *out_result = -5; co_return -5; }

  // Data must have round-tripped through device memory by the time we resume.
  for (std::size_t i = 0; i < kN; ++i) {
    if (host_out[i] != host_in[i]) { *out_result = -6; co_return -6; }
  }
  s->device_free(*dptr);
  *out_result = 0;
  co_return 0;
}

}  // namespace

TEST(GpuAwait, CoAwaitStreamCompletion) {
  Scheduler sched(2, system_allocator(), 256);

  int result = -100;
  auto d = std::make_unique<Drive>();
  d->task = gpu_await_coro(system_allocator(), &result);
  Fiber f(system_allocator(), kDefaultStackSize, &drive_entry, d.get());
  sched.spawn(&f);
  sched.drain();
  sched.shutdown();

  compute::detail::gpu_poll_shutdown();  // clean poller teardown

  EXPECT_TRUE(d->finished.load());
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
    auto f = std::make_unique<Fiber>(system_allocator(), kDefaultStackSize,
                                     &drive_entry, d.get());
    sched.spawn(f.get());
    drives.push_back(std::move(d));
    fibers.push_back(std::move(f));
  }
  sched.drain();
  sched.shutdown();
  compute::detail::gpu_poll_shutdown();

  for (int i = 0; i < kFibers; ++i) {
    EXPECT_TRUE(drives[i]->finished.load()) << "fiber " << i;
    EXPECT_EQ(results[i], 0) << "fiber " << i << " failed at step " << results[i];
  }
}
