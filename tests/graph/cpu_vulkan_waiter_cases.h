#pragma once

#include <atomic>
#include <chrono>

#include <hyle/mem/quota_allocator.h>
#include <hyle/platform/thread.h>
#include <hyle/sched/blocking.h>
#include <hyle/sched/sync.h>

// Included after the ordinary public sample, reusing its tensor/Drain helpers.
namespace {
namespace vulkan_waiter_cases {
using hyle::sched::BlockingPool;
using hyle::sched::Event;
using hyle::sched::EventResetMode;
constexpr auto gate_timeout = std::chrono::seconds(10);

Status check(bool condition, const char *message) {
  return condition ? Status::Ok() : Status::Internal(message);
}

struct CpuGate final {
  Event entered{EventResetMode::kManual}, release{EventResetMode::kManual};
  std::atomic<bool> armed{false};
  std::atomic<unsigned> calls{0};
  static void execute(void *opaque, std::size_t count, c::ParallelTaskFn task,
                      void *context) noexcept {
    auto &self = *static_cast<CpuGate *>(opaque);
    self.calls.fetch_add(1, std::memory_order_relaxed);
    if (self.armed.exchange(false, std::memory_order_acq_rel)) {
      self.entered.signal();
      self.release.wait(); // Fiber-aware; the one Scheduler worker stays free.
    }
    for (std::size_t i = 0; i < count; ++i) task(context, i, count);
  }
  void arm() noexcept {
    entered.reset(); release.reset();
    armed.store(true, std::memory_order_release);
  }
};

struct PoolGate final {
  Event entered{EventResetMode::kManual}, release{EventResetMode::kManual};
  Event destroyed{EventResetMode::kManual};
  std::atomic<bool> cancelled{false};
  hyle::sched::detail::BlockingJob job{
      .ctx = this,
      .run_fn = [](void *opaque) noexcept {
        auto &self = *static_cast<PoolGate *>(opaque);
        self.entered.signal(); self.release.wait();
      },
      .cancel_fn = [](void *opaque) noexcept {
        static_cast<PoolGate *>(opaque)->cancelled.store(true, std::memory_order_release);
      },
      .destroy_fn = [](void *opaque) noexcept {
        static_cast<PoolGate *>(opaque)->destroyed.signal();
      }};
};

// Declare after the invocation and all gates. Every early return releases the
// real CPU/pool gates and joins shutdown before those borrowed objects die.
struct ShutdownGuard final {
  BlockingPool &pool;
  hyle::sched::Scheduler &scheduler;
  PoolGate &marker;
  PoolGate *blocker = nullptr;
  CpuGate *cpu = nullptr;
  hyle::platform::Thread thread;
  ~ShutdownGuard() { finish(); }
  Status start() {
    HYLE_ASSIGN_OR_RETURN(thread, hyle::platform::Thread::spawn(
        [](void *opaque) { static_cast<BlockingPool *>(opaque)->shutdown(); }, &pool));
    return Status::Ok();
  }
  void finish() {
    if (cpu) cpu->release.signal();
    marker.release.signal();
    if (blocker) blocker->release.signal();
    if (thread.joinable()) thread.join();
    else pool.shutdown();
    scheduler.drain();
  }
};

void signal_sentinel(void *opaque) noexcept { static_cast<Event *>(opaque)->signal(); }

Status run() {
  CpuGate cpu_gate;
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::cpu::create_context(
      {.executor = c::ParallelExecutorRef{&cpu_gate, 2, &CpuGate::execute}}));
  HYLE_ASSIGN_OR_RETURN(auto vulkan, v::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto device, v::MemoryResource::create(vulkan, v::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging, v::MemoryResource::create(vulkan, v::MemoryKind::kStaging));
  const b::Environment environment{.cpu = &cpu, .vulkan = &vulkan, .device = &device, .staging = &staging};
  hyle::sched::Scheduler scheduler(1, m::system_allocator(), 32);
  Drain drain{scheduler};
  HYLE_ASSIGN_OR_RETURN(auto queue, cpu.create_queue());
  // Dense F32 MatMul uses the ordinary mature oneDNN/ggml routes. Both call
  // ParallelExecutorRef even for one task, so this gate has no grain threshold
  // and does not depend on changing the ordinary provider selection.
  HYLE_ASSIGN_OR_RETURN(auto desc, c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{64, 64}));
  HYLE_ASSIGN_OR_RETURN(auto input, tensor(desc));
  HYLE_ASSIGN_OR_RETURN(auto weight, tensor(desc));
  std::array<float, 4096> values, diagonal{};
  values.fill(1.0f);
  for (std::size_t i = 0; i < 64; ++i) diagonal[64 * i + i] = 2.0f;
  HYLE_RETURN_IF_ERROR(input.buffer.view().write(std::as_bytes(std::span(values))));
  HYLE_RETURN_IF_ERROR(weight.buffer.view().write(std::as_bytes(std::span(diagonal))));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto x, builder.input(desc));
  HYLE_ASSIGN_OR_RETURN(auto w, builder.input(desc));
  HYLE_ASSIGN_OR_RETURN(auto product, builder.add(c::MatMul{}, x, w));
  HYLE_ASSIGN_OR_RETURN(auto negative, builder.add(c::Unary{c::UnaryKind::kNeg}, product));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.add(c::Unary{c::UnaryKind::kAbs}, negative));
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  const std::array placements{
      g::NodePlacement{*graph.node_id(0), b::kCpuTarget, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(1), b::kVulkanTarget, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(2), b::kCpuTarget, g::PlacementStrength::kRequired}};
  HYLE_ASSIGN_OR_RETURN(auto program, b::compile(graph, environment, {.node_placements = placements}));
  const auto slot = [&](BlockingPool &pool) -> Result<b::Invocation> {
    HYLE_ASSIGN_OR_RETURN(auto invocation, program.create_invocation(scheduler, queue, pool));
    HYLE_RETURN_IF_ERROR(invocation.bind(x, input.view.as_const()));
    HYLE_RETURN_IF_ERROR(invocation.bind(w, weight.view.as_const()));
    HYLE_RETURN_IF_ERROR(invocation.prepare_replay());
    return invocation;
  };
  const auto failed_episode = [&](b::Invocation &invocation) -> Status {
    const auto before = cpu_gate.calls.load(std::memory_order_relaxed);
    HYLE_ASSIGN_OR_RETURN(auto work, invocation.submit());
    const auto result = work.wait();
    HYLE_RETURN_IF_ERROR(check(!result && result.error().IsUnavailable(),
        "rejected native waiter did not report admission failure"));
    HYLE_RETURN_IF_ERROR(check(cpu_gate.calls.load(std::memory_order_relaxed) == before,
        "pool rejection admitted the CPU prefix before waiter start"));
    return check(invocation.close().IsUnavailable(), "rejected waiter close lost its status");
  };

  {
    Status status;
    BlockingPool invalid(m::AllocatorRef{}, 1, std::chrono::seconds(1), &status);
    HYLE_RETURN_IF_ERROR(check(status.IsInvalidArgument() && !invalid.valid(),
        "recoverable invalid pool construction succeeded"));
    auto rejected = program.create_invocation(scheduler, queue, invalid);
    HYLE_RETURN_IF_ERROR(check(!rejected && rejected.error().IsInvalidArgument(),
        "Graph accepted an invalid native wait pool"));
  }
  {
    m::QuotaAllocator quota(m::system_allocator(), 0);
    Status status;
    BlockingPool missing(quota.get(), 1, std::chrono::seconds(1), &status);
    HYLE_RETURN_IF_ERROR(check(status.IsResourceExhausted() && !missing.valid(),
        "recoverable cold pool OOM did not fail"));
    auto rejected = program.create_invocation(scheduler, queue, missing);
    HYLE_RETURN_IF_ERROR(check(!rejected && rejected.error().IsInvalidArgument(),
        "Graph accepted a pool with failed cold construction"));
  }
  {
    BlockingPool stopped(m::system_allocator(), 1);
    stopped.shutdown();
    HYLE_ASSIGN_OR_RETURN(auto invocation, slot(stopped));
    HYLE_RETURN_IF_ERROR(failed_episode(invocation));
    // Native waits are absent from a pure CPU plan. A stopped but valid pool
    // must not turn this existing route into a native admission requirement.
    const std::array cpu_placements{
        g::NodePlacement{*graph.node_id(0), b::kCpuTarget, g::PlacementStrength::kRequired},
        g::NodePlacement{*graph.node_id(1), b::kCpuTarget, g::PlacementStrength::kRequired},
        g::NodePlacement{*graph.node_id(2), b::kCpuTarget, g::PlacementStrength::kRequired}};
    HYLE_ASSIGN_OR_RETURN(auto cpu_only, b::compile(graph, environment,
        {.node_placements = cpu_placements}));
    HYLE_ASSIGN_OR_RETURN(auto plain, cpu_only.create_invocation(scheduler, queue, stopped));
    HYLE_RETURN_IF_ERROR(plain.bind(x, input.view.as_const()));
    HYLE_RETURN_IF_ERROR(plain.bind(w, weight.view.as_const()));
    HYLE_RETURN_IF_ERROR(plain.prepare_replay());
    {
      HYLE_ASSIGN_OR_RETURN(auto result, plain.run());
      HYLE_ASSIGN_OR_RETURN(auto actual, result->output(output));
      HYLE_RETURN_IF_ERROR(actual.storage().read(std::as_writable_bytes(std::span(values))));
      for (auto value : values)
        HYLE_RETURN_IF_ERROR(check(value == 2.0f, "stopped pool changed pure CPU execution"));
    }
    HYLE_RETURN_IF_ERROR(plain.close());
  }
  {
    // Pool metadata succeeds; reserve all remaining quota before its first
    // worker record can be allocated. Graph metadata uses its original owner.
    m::QuotaAllocator quota(m::system_allocator(), 1024 * 1024);
    Status status;
    BlockingPool missing_worker(quota.get(), 1, std::chrono::seconds(1), &status);
    HYLE_RETURN_IF_ERROR(status);
    HYLE_ASSIGN_OR_RETURN(auto invocation, slot(missing_worker));
    auto reserved = quota.try_reserve_bytes(quota.remaining());
    HYLE_RETURN_IF_ERROR(check(reserved.has_value(), "could not reserve worker failure quota"));
    HYLE_RETURN_IF_ERROR(failed_episode(invocation));
    HYLE_RETURN_IF_ERROR(check(missing_worker.live_threads() == 0,
        "first-worker OOM unexpectedly created a pool thread"));
  }
  {
    BlockingPool pool(m::system_allocator(), 1);
    m::QuotaAllocator quota(m::system_allocator(), 16 * 1024 * 1024);
    HYLE_ASSIGN_OR_RETURN(auto bounded, b::compile(graph, environment,
        {.node_placements = placements, .allocator = quota.get()}));
    auto reserved = quota.try_reserve_bytes(quota.remaining());
    HYLE_RETURN_IF_ERROR(check(reserved.has_value(), "could not reserve cold invocation quota"));
    auto rejected = bounded.create_invocation(scheduler, queue, pool);
    HYLE_RETURN_IF_ERROR(check(!rejected && rejected.error().IsResourceExhausted(),
        "cold waiter invocation metadata OOM was not recoverable"));
  }
  {
    BlockingPool pool(m::system_allocator(), 1);
    PoolGate blocker, marker;
    Event sentinel{EventResetMode::kManual};
    HYLE_ASSIGN_OR_RETURN(auto invocation, slot(pool));
    ShutdownGuard cleanup{pool, scheduler, marker, &blocker};
    HYLE_RETURN_IF_ERROR(check(pool.submit(&blocker.job), "could not admit occupying pool job"));
    HYLE_RETURN_IF_ERROR(check(blocker.entered.wait_for(gate_timeout), "occupying pool job did not start"));
    HYLE_RETURN_IF_ERROR(check(pool.submit(&marker.job), "could not queue pool shutdown marker"));
    const auto before = cpu_gate.calls.load(std::memory_order_relaxed);
    HYLE_ASSIGN_OR_RETURN(auto work, invocation.submit());
    HYLE_RETURN_IF_ERROR(scheduler.spawn_detached(&signal_sentinel, &sentinel));
    HYLE_RETURN_IF_ERROR(check(sentinel.wait_for(gate_timeout),
        "one Scheduler worker did not run the sentinel while the pool was occupied"));
    HYLE_RETURN_IF_ERROR(cleanup.start());
    HYLE_RETURN_IF_ERROR(check(marker.destroyed.wait_for(gate_timeout) &&
        marker.cancelled.load(std::memory_order_acquire) && !marker.entered.is_signaled(),
        "shutdown did not cancel its proven queued marker"));
    // Public Graph has no waiter-queued accessor. Its root may have enqueued
    // before shutdown or may race admission after shutdown. Both must fail
    // without any CPU/native prefix; this scenario does not label that race.
    const auto result = work.wait();
    HYLE_RETURN_IF_ERROR(check(!result &&
        (result.error().IsCancelled() || result.error().IsUnavailable()),
        "occupied pool shutdown did not resolve Graph admission"));
    HYLE_RETURN_IF_ERROR(check(cpu_gate.calls.load(std::memory_order_relaxed) == before,
        "queued/refused waiter admitted a CPU prefix"));
    const auto closed = invocation.close();
    HYLE_RETURN_IF_ERROR(check(closed.IsCancelled() || closed.IsUnavailable(),
        "queued/refused waiter close lost its failure"));
    cleanup.finish();
  }
  for (bool cancel : {false, true}) {
    BlockingPool pool(m::system_allocator(), 1);
    PoolGate marker;
    Event sentinel{EventResetMode::kManual};
    HYLE_ASSIGN_OR_RETURN(auto invocation, slot(pool));
    ShutdownGuard cleanup{pool, scheduler, marker, nullptr, &cpu_gate};
    cpu_gate.arm();
    HYLE_ASSIGN_OR_RETURN(auto work, invocation.submit());
    HYLE_RETURN_IF_ERROR(check(cpu_gate.entered.wait_for(gate_timeout),
        "real CPU MatMul did not reach the public executor gate"));
    HYLE_RETURN_IF_ERROR(check(pool.submit(&marker.job), "could not queue running-waiter marker"));
    HYLE_RETURN_IF_ERROR(cleanup.start());
    HYLE_RETURN_IF_ERROR(check(marker.destroyed.wait_for(gate_timeout) &&
        marker.cancelled.load(std::memory_order_acquire) && !marker.entered.is_signaled(),
        "resident waiter did not occupy the pool through shutdown"));
    HYLE_RETURN_IF_ERROR(scheduler.spawn_detached(&signal_sentinel, &sentinel));
    HYLE_RETURN_IF_ERROR(check(sentinel.wait_for(gate_timeout),
        "one Scheduler worker stalled while a running waiter was held"));
    if (cancel) HYLE_RETURN_IF_ERROR(work.request_cancel());
    cpu_gate.release.signal();
    auto result = work.wait();
    if (cancel) {
      HYLE_RETURN_IF_ERROR(check(!result && result.error().IsCancelled(),
          "running waiter shutdown/cancel lost the graph cancellation"));
      HYLE_RETURN_IF_ERROR(check(invocation.close().IsCancelled(),
          "running waiter shutdown/cancel close did not drain"));
    } else {
      if (!result) return result.error();
      HYLE_ASSIGN_OR_RETURN(auto actual, result->get()->output(output));
      HYLE_RETURN_IF_ERROR(actual.storage().read(std::as_writable_bytes(std::span(values))));
      for (auto value : values)
        HYLE_RETURN_IF_ERROR(check(value == 2.0f, "running shutdown changed real CPU/native/CPU math"));
      HYLE_RETURN_IF_ERROR(invocation.close());
    }
    cleanup.finish();
  }
  std::puts("G09 Vulkan waiter: invalid/cold/worker OOM, refusal, occupied shutdown, running shutdown/cancel and one-worker sentinels passed");
  return Status::Ok();
}
} // namespace vulkan_waiter_cases
} // namespace
