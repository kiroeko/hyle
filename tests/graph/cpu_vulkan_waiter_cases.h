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
  std::atomic<unsigned> calls{0}, skip{0}, entered_call{0};
  static void execute(void *opaque, std::size_t count, c::ParallelTaskFn task,
                      void *context) noexcept {
    auto &self = *static_cast<CpuGate *>(opaque);
    const auto call = self.calls.fetch_add(1, std::memory_order_relaxed) + 1;
    if (self.armed.load(std::memory_order_acquire)) {
      if (self.skip.load(std::memory_order_relaxed) != 0) {
        self.skip.fetch_sub(1, std::memory_order_relaxed);
      } else if (self.armed.exchange(false, std::memory_order_acq_rel)) {
        self.entered_call.store(call, std::memory_order_release);
        self.entered.signal();
        self.release.wait(); // Actual synchronous provider body on the pool worker.
      }
    }
    for (std::size_t i = 0; i < count; ++i) task(context, i, count);
  }
  void arm(unsigned skip_calls = 0) noexcept {
    entered.reset(); release.reset();
    skip.store(skip_calls, std::memory_order_relaxed);
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
  const Resources environment{cpu, vulkan, device, staging};
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
  HYLE_ASSIGN_OR_RETURN(auto destination, tensor(desc));
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
      g::NodePlacement{*graph.node_id(0), kCpuTarget, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(1), kVulkanTarget, g::PlacementStrength::kRequired},
      g::NodePlacement{*graph.node_id(2), kCpuTarget, g::PlacementStrength::kRequired}};
  const std::array external{g::ExternalResourceContract{x, kHostStorage},
      g::ExternalResourceContract{w, kHostStorage}, g::ExternalResourceContract{output, kHostStorage}};
  HYLE_ASSIGN_OR_RETURN(auto program, compile_graph(graph, environment,
      {.node_placements = placements, .external_resources = external}));
  HYLE_ASSIGN_OR_RETURN(auto queues, cpu_queues(program.report, queue));
  // Confirm the actual selected CPU/native/CPU bodies and separate coarse
  // segments before interpreting an executor gate as prefix or suffix work.
  std::array<std::size_t, 3> node_steps{SIZE_MAX, SIZE_MAX, SIZE_MAX};
  std::array<std::size_t, 3> node_segments{SIZE_MAX, SIZE_MAX, SIZE_MAX};
  for (std::size_t i = 0; i < program.report.step_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto step, program.report.step(i));
    for (std::size_t n = 0; n < step.node_count; ++n) {
      HYLE_ASSIGN_OR_RETURN(auto node, program.report.step_node(i, n));
      for (std::size_t j = 0; j < node_steps.size(); ++j)
        if (node == graph.node_id(j).value()) node_steps[j] = i;
    }
  }
  for (std::size_t i = 0; i < program.report.segment_count(); ++i) {
    HYLE_ASSIGN_OR_RETURN(auto segment, program.report.segment(i));
    for (std::size_t n = 0; n < node_steps.size(); ++n)
      if (node_steps[n] >= segment.step_begin &&
          node_steps[n] < segment.step_begin + segment.step_count) node_segments[n] = i;
  }
  for (std::size_t i = 0; i < node_steps.size(); ++i) {
    HYLE_RETURN_IF_ERROR(check(node_steps[i] != SIZE_MAX && node_segments[i] != SIZE_MAX,
        "controlled CPU/native node is absent from the physical plan"));
    HYLE_ASSIGN_OR_RETURN(auto step, program.report.step(node_steps[i]));
    HYLE_RETURN_IF_ERROR(check(step.target == (i == 1 ? kVulkanTarget : kCpuTarget) &&
        step.runtime.blocks_host == (i != 1) && step.dispatch_count == 1,
        "controlled segment does not have the actual coarse host/device runtime"));
    std::printf("Vulkan pool node=%zu step=%zu segment=%zu blocks_host=%d provider=%.*s\n",
        i, node_steps[i], node_segments[i], int(step.runtime.blocks_host),
        int(step.provider.name.size()), step.provider.name.data());
  }
  HYLE_RETURN_IF_ERROR(check(node_segments[0] != node_segments[2],
      "CPU prefix and suffix were not separate coarse admissions"));
  const auto reset_destination = [&]() -> Status {
    values.fill(99.0F);
    return destination.buffer.view().write(std::as_bytes(std::span(values)));
  };
  const auto slot = [&](BlockingPool &pool) -> Result<g::Invocation> {
    HYLE_ASSIGN_OR_RETURN(auto invocation, program.executable.create_invocation());
    HYLE_RETURN_IF_ERROR(g::prepare_progress(invocation, scheduler, pool));
    HYLE_RETURN_IF_ERROR(invocation.bind(x, input.view.as_const()));
    HYLE_RETURN_IF_ERROR(invocation.bind(w, weight.view.as_const()));
    HYLE_RETURN_IF_ERROR(invocation.bind(output, destination.view));
    HYLE_RETURN_IF_ERROR(invocation.prepare_replay(queues));
    return invocation;
  };
  const auto failed_episode = [&](g::Invocation &invocation) -> Status {
    const auto before = cpu_gate.calls.load(std::memory_order_relaxed);
    HYLE_ASSIGN_OR_RETURN(auto work, invocation.submit(queues));
    const auto result = work.wait();
    HYLE_RETURN_IF_ERROR(check(result.IsCancelled(),
        "refused coarse CPU job did not report its original pool cancellation"));
    HYLE_RETURN_IF_ERROR(check(cpu_gate.calls.load(std::memory_order_relaxed) == before,
        "pool rejection admitted a synchronous CPU body"));
    HYLE_RETURN_IF_ERROR(check(invocation.drain().IsCancelled(), "rejected pool job drain lost its status"));
    return check(!invocation.submit(queues), "drained rejected slot admitted a successor");
  };

  {
    Status status;
    BlockingPool invalid(m::AllocatorRef{}, 1, std::chrono::seconds(1), &status);
    HYLE_RETURN_IF_ERROR(check(status.IsInvalidArgument() && !invalid.valid(),
        "recoverable invalid pool construction succeeded"));
    auto rejected = slot(invalid);
    HYLE_RETURN_IF_ERROR(check(!rejected && rejected.error().IsInvalidArgument(),
        "Graph progress accepted an invalid CPU-work pool"));
  }
  {
    m::QuotaAllocator quota(m::system_allocator(), 0);
    Status status;
    BlockingPool missing(quota.get(), 1, std::chrono::seconds(1), &status);
    HYLE_RETURN_IF_ERROR(check(status.IsResourceExhausted() && !missing.valid(),
        "recoverable cold pool OOM did not fail"));
    auto rejected = slot(missing);
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
        g::NodePlacement{*graph.node_id(0), kCpuTarget, g::PlacementStrength::kRequired},
        g::NodePlacement{*graph.node_id(1), kCpuTarget, g::PlacementStrength::kRequired},
        g::NodePlacement{*graph.node_id(2), kCpuTarget, g::PlacementStrength::kRequired}};
    HYLE_ASSIGN_OR_RETURN(auto cpu_only, compile_graph(graph, environment,
        {.node_placements = cpu_placements, .external_resources = external}));
    HYLE_ASSIGN_OR_RETURN(auto plain, cpu_only.executable.create_invocation());
    HYLE_ASSIGN_OR_RETURN(auto plain_queues, cpu_queues(cpu_only.report, queue));
    // Optional progress is not installed on an ordinary synchronous CPU slot.
    HYLE_RETURN_IF_ERROR(plain.bind(x, input.view.as_const()));
    HYLE_RETURN_IF_ERROR(plain.bind(w, weight.view.as_const()));
    HYLE_RETURN_IF_ERROR(plain.bind(output, destination.view));
    HYLE_RETURN_IF_ERROR(plain.prepare_replay(plain_queues));
    {
      HYLE_ASSIGN_OR_RETURN(auto done, plain.submit(plain_queues));
      HYLE_RETURN_IF_ERROR(done.wait());
      HYLE_ASSIGN_OR_RETURN(auto result, plain.result(output));
      HYLE_ASSIGN_OR_RETURN(auto actual, result.tensor());
      HYLE_RETURN_IF_ERROR(actual.storage().read(std::as_writable_bytes(std::span(values))));
      for (auto value : values)
        HYLE_RETURN_IF_ERROR(check(value == 2.0f, "stopped optional pool changed pure CPU execution"));
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
    const std::array limits{c::ResourceBudgetLimit{m::system_resource(), SIZE_MAX},
        c::ResourceBudgetLimit{device.ref(), SIZE_MAX}, c::ResourceBudgetLimit{staging.ref(), SIZE_MAX}};
    HYLE_ASSIGN_OR_RETURN(auto metadata, c::ResourceBudget::create(
        {.limits = limits, .allocator = quota.get(), .allow_unattributed = true}));
    HYLE_ASSIGN_OR_RETURN(auto bounded, compile_graph(graph, environment,
        {.resource_budget = metadata, .node_placements = placements, .external_resources = external}));
    HYLE_ASSIGN_OR_RETURN(auto cold_slot, bounded.executable.create_invocation());
    auto reserved = quota.try_reserve_bytes(quota.remaining());
    HYLE_RETURN_IF_ERROR(check(reserved.has_value(), "could not reserve cold invocation quota"));
    auto rejected = bounded.executable.create_invocation();
    HYLE_RETURN_IF_ERROR(check(!rejected && rejected.error().IsResourceExhausted(),
        "cold invocation metadata OOM was not recoverable"));
    const auto progress_oom = g::prepare_progress(cold_slot, scheduler, pool);
    HYLE_RETURN_IF_ERROR(check(progress_oom.IsResourceExhausted(),
        "cold progress metadata OOM was not recoverable"));
    reserved->reset();
    HYLE_RETURN_IF_ERROR(g::prepare_progress(cold_slot, scheduler, pool));

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
    HYLE_ASSIGN_OR_RETURN(auto work, invocation.submit(queues));
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
    HYLE_RETURN_IF_ERROR(check(result.IsCancelled(),
        "occupied pool shutdown did not resolve Graph admission"));
    HYLE_RETURN_IF_ERROR(check(cpu_gate.calls.load(std::memory_order_relaxed) == before,
        "queued/refused waiter admitted a CPU prefix"));
    const auto closed = invocation.drain();
    HYLE_RETURN_IF_ERROR(check(closed.IsCancelled(),
        "queued/refused waiter close lost its failure"));
    cleanup.finish();
  }
  {
    BlockingPool pool(m::system_allocator(), 1);
    PoolGate marker;
    HYLE_ASSIGN_OR_RETURN(auto invocation, slot(pool));
    ShutdownGuard cleanup{pool, scheduler, marker, nullptr, &cpu_gate};
    HYLE_RETURN_IF_ERROR(reset_destination());
    const auto before = cpu_gate.calls.load(std::memory_order_relaxed);
    cpu_gate.arm();
    HYLE_ASSIGN_OR_RETURN(auto work, invocation.submit(queues));
    HYLE_RETURN_IF_ERROR(check(cpu_gate.entered.wait_for(gate_timeout) &&
        cpu_gate.entered_call.load(std::memory_order_acquire) == before + 1,
        "real CPU prefix did not reach its first executor body"));
    HYLE_RETURN_IF_ERROR(check(pool.submit(&marker.job), "could not queue prefix shutdown marker"));
    HYLE_RETURN_IF_ERROR(cleanup.start());
    HYLE_RETURN_IF_ERROR(check(marker.destroyed.wait_for(gate_timeout) && marker.cancelled.load() &&
        !marker.entered.is_signaled(), "prefix shutdown did not cancel queued marker"));
    cpu_gate.release.signal();
    HYLE_RETURN_IF_ERROR(check(work.wait().IsCancelled(),
        "stopped pool admitted a new blocking CPU suffix"));
    HYLE_RETURN_IF_ERROR(destination.buffer.view().read(std::as_writable_bytes(std::span(values))));
    HYLE_RETURN_IF_ERROR(check(cpu_gate.calls.load() == before + 1 &&
        std::all_of(values.begin(), values.end(), [](float x) { return x == 99.0F; }),
        "shutdown after accepted prefix ran or wrote the refused CPU suffix"));
    HYLE_RETURN_IF_ERROR(check(invocation.drain().IsCancelled(),
        "accepted prefix shutdown lost its retired cancellation"));
    cleanup.finish();
  }
  for (bool cancel : {false, true}) {
    BlockingPool pool(m::system_allocator(), 1);
    PoolGate marker;
    Event sentinel{EventResetMode::kManual};
    HYLE_ASSIGN_OR_RETURN(auto invocation, slot(pool));
    ShutdownGuard cleanup{pool, scheduler, marker, nullptr, &cpu_gate};
    HYLE_RETURN_IF_ERROR(reset_destination());
    const auto before = cpu_gate.calls.load(std::memory_order_relaxed);
    cpu_gate.arm(1); // One real CPU MatMul fan-out, then the actual CPU Abs body.
    HYLE_ASSIGN_OR_RETURN(auto work, invocation.submit(queues));
    HYLE_RETURN_IF_ERROR(check(cpu_gate.entered.wait_for(gate_timeout) &&
        cpu_gate.entered_call.load(std::memory_order_acquire) == before + 2,
        "selected final CPU segment did not reach its actual executor body"));
    HYLE_RETURN_IF_ERROR(destination.buffer.view().read(std::as_writable_bytes(std::span(values))));
    HYLE_RETURN_IF_ERROR(check(std::all_of(values.begin(), values.end(), [](float x) { return x == 99.0F; }),
        "final CPU output was written before its actual provider body"));
    HYLE_RETURN_IF_ERROR(check(pool.submit(&marker.job), "could not queue running-waiter marker"));
    HYLE_RETURN_IF_ERROR(cleanup.start());
    HYLE_RETURN_IF_ERROR(check(marker.destroyed.wait_for(gate_timeout) &&
        marker.cancelled.load(std::memory_order_acquire) && !marker.entered.is_signaled(),
        "actual final CPU job did not occupy the pool through shutdown"));
    HYLE_RETURN_IF_ERROR(scheduler.spawn_detached(&signal_sentinel, &sentinel));
    HYLE_RETURN_IF_ERROR(check(sentinel.wait_for(gate_timeout),
        "one Scheduler worker stalled while a running waiter was held"));
    if (cancel) HYLE_RETURN_IF_ERROR(work.request_cancel());
    cpu_gate.release.signal();
    auto result = work.wait();
    if (cancel) {
      HYLE_RETURN_IF_ERROR(check(result.IsCancelled(),
          "running waiter shutdown/cancel lost the graph cancellation"));
      HYLE_RETURN_IF_ERROR(check(invocation.drain().IsCancelled(),
          "running waiter shutdown/cancel close did not drain"));
    } else {
      HYLE_RETURN_IF_ERROR(result);
      HYLE_ASSIGN_OR_RETURN(auto observed, invocation.result(output));
      HYLE_ASSIGN_OR_RETURN(auto actual, observed.tensor());
      HYLE_RETURN_IF_ERROR(actual.storage().read(std::as_writable_bytes(std::span(values))));
      for (auto value : values)
        HYLE_RETURN_IF_ERROR(check(value == 2.0f, "running shutdown changed real CPU/native/CPU math"));
      observed = {};
      HYLE_RETURN_IF_ERROR(invocation.close());
    }
    cleanup.finish();
  }
  std::puts("G09 Vulkan coarse progress: invalid/cold/worker OOM, refusal, occupied shutdown, accepted-prefix suffix refusal, final-running shutdown/cancel and one-worker sentinels passed");
  return Status::Ok();
}
} // namespace vulkan_waiter_cases
} // namespace
