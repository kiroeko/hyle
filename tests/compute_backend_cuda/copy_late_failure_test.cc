#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/batch_transfer.h"
#include "hyle/mem/quota_allocator.h"

cudaError_t launch_illegal_address_kernel(cudaStream_t stream) noexcept;
cudaError_t prepare_illegal_address_kernel() noexcept;

namespace {
namespace c = hyle::compute;
namespace cuda = c::cuda;
namespace mem = hyle::mem;

struct Gate final {
  std::atomic<bool> released{false};
  static void CUDART_CB wait(void *opaque) {
    auto &gate = *static_cast<Gate *>(opaque);
    while (!gate.released.load(std::memory_order_acquire))
      gate.released.wait(false, std::memory_order_acquire);
  }
  void release() { released.store(true, std::memory_order_release); released.notify_all(); }
};
struct Drain final {
  Gate &gate; c::Context &context;
  ~Drain() { gate.release(); context.drain().IgnoreError(); }
};

// Device faults poison the primary context. This target contains only this
// journey and runs in a process separate from the original backend fault test.
TEST(CudaCopyLateFailure, ActualProducerFailureKeepsEarlierGroupAndNeverPublishesFailedOutput) {
  auto context = c::Context::create(cuda::device(0).value()).value();
  auto queue = context.create_queue().value(), producer_queue = context.create_queue().value();
  auto host = cuda::MemoryResource::create(context, cuda::MemoryKind::kPinnedHost).value();
  auto device = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  // Loading a first-use CUDA module may synchronize. Finish cold loading
  // before the callback holds this context; this does not execute the fault.
  ASSERT_EQ(prepare_illegal_address_kernel(), cudaSuccess);
  const std::array<c::CopyResourceSpec, 5> specs{{
      {host.ref(), 32}, {device.ref(), 32}, {host.ref(), 32},
      {device.ref(), 32}, {host.ref(), 32}}};
  std::array<c::LeasePool, 5> pools;
  for (unsigned id = 0; id < pools.size(); ++id)
    pools[id] = c::LeasePool::create(specs[id].resource, 1024, {.max_blocks = 1}).value();
  mem::QuotaAllocator metadata(mem::system_allocator());
  Gate gate; Drain drain{gate, context};
  auto program = c::CopyBatchProgram::prepare(context, specs, {.allocator = metadata.get()}).value();
  auto invocation = c::CopyBatchInvocation::create(program, queue).value();
  auto writer = pools[0].allocate(32).value();
  ASSERT_TRUE(writer.view()->fill(std::byte{47}).ok());
  auto source = writer.publish().value();
  ASSERT_TRUE(invocation.bind_source({0}, source).ok()); source.reset();
  for (unsigned id : {1u, 2u, 4u}) {
    auto output = pools[id].allocate(32).value();
    ASSERT_TRUE(invocation.bind_mutable({id}, std::move(output)).ok());
  }
  auto failing_source = pools[3].allocate(32).value();
  auto view = failing_source.view().value();
  auto pointer = cuda::pointer<std::byte>(view).value();
  auto uses = c::LeaseUsePool::create({.max_batches = 1, .max_resources_per_batch = 1}).value();
  auto use = uses.begin().value(); ASSERT_TRUE(use.write(failing_source).ok());
  std::atomic<bool> forced_release{false};
  std::jthread watchdog([&](std::stop_token stop) {
    const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (!stop.stop_requested() && std::chrono::steady_clock::now() < limit)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (!stop.stop_requested()) { forced_release.store(true); gate.release(); }
  });
  std::fprintf(stderr, "F2 late: cold loaded; enqueue producer\n");
  auto producer = cuda::submit(producer_queue, [&](cudaStream_t stream) noexcept {
    auto status = cudaLaunchHostFunc(stream, &Gate::wait, &gate);
    if (status == cudaSuccess) status = cudaMemsetAsync(pointer, 13, 32, stream);
    return status == cudaSuccess ? launch_illegal_address_kernel(stream) : status;
  });
  ASSERT_TRUE(producer.has_value()); ASSERT_TRUE(use.attach(*producer).ok());
  std::fprintf(stderr, "F2 late: producer accepted; bind ranges\n");
  EXPECT_FALSE(failing_source.view().has_value());
  ASSERT_TRUE(invocation.bind_mutable({3}, std::move(failing_source)).ok());
  ASSERT_TRUE(invocation.bind_producer({3}, std::move(*producer)).ok());
  const std::array<c::CopyPhase, 3> phases{{{0, 10}, {1, 10}, {0, 20}}};
  const std::array<c::CopyRange, 3> ranges{{
      {{0}, {1}, {0}, 0, 0, 32}, {{1}, {2}, {1}, 0, 0, 32}, {{3}, {4}, {2}, 0, 0, 32}}};
  ASSERT_TRUE(invocation.prepare_replay(phases, ranges).ok());
  std::fprintf(stderr, "F2 late: submit copy phases\n");
  auto submitted = invocation.submit(); ASSERT_TRUE(submitted.has_value());
  std::fprintf(stderr, "F2 late: copy accepted; observe early group\n");
  c::ReadLease early;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    auto ready = submitted->output({1}, {2}, 0, 32);
    if (ready) { early = std::move(*ready); break; }
    ASSERT_TRUE(ready.error().IsUnavailable());
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(early.valid());
  std::array<std::byte, 32> actual{};
  ASSERT_TRUE(early.view()->read(actual).ok());
  for (auto byte : actual) EXPECT_EQ(byte, std::byte{47});
  EXPECT_FALSE(submitted->output({2}, {4}, 0, 32).has_value());
  EXPECT_TRUE(submitted->in_flight()); EXPECT_EQ(uses.active_batches(), 1u);

  std::fprintf(stderr, "F2 late: early group ready; release actual fault\n");
  gate.release();
  const auto terminal = submitted->wait();
  EXPECT_FALSE(terminal.ok());
  EXPECT_EQ(terminal.native_error().domain, hyle::NativeErrorDomain::CudaRuntime);
  EXPECT_EQ(terminal.native_error().code, static_cast<std::int64_t>(cudaErrorIllegalAddress));
  EXPECT_FALSE(submitted->output({2}, {4}, 0, 32).has_value());
  auto retained = submitted->output({1}, {2}, 0, 32); ASSERT_TRUE(retained.has_value());
  ASSERT_TRUE(retained->view()->read(actual).ok());
  for (auto byte : actual) EXPECT_EQ(byte, std::byte{47});
  retained->reset();
  *submitted = c::CopyBatchSubmission{}; invocation = {}; program.reset();
  context.drain().IgnoreError();
  const auto retired_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while ((metadata.used() != 0 || uses.active_batches() != 0) &&
         std::chrono::steady_clock::now() < retired_deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  EXPECT_EQ(metadata.used(), 0u); EXPECT_EQ(uses.active_batches(), 0u);
  for (unsigned id = 0; id < pools.size(); ++id) {
    EXPECT_EQ(pools[id].stats().active_uses, 0u);
    EXPECT_EQ(pools[id].stats().live_blocks, id == 2 ? 1u : 0u);
  }
  early.reset(); EXPECT_EQ(pools[2].stats().live_blocks, 0u);
  EXPECT_FALSE(forced_release.load());
}
} // namespace
