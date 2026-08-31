#include "p1_common.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/join.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"
#include "hyle/sched/waker.h"

namespace {

namespace p1 = hyle::bench::sched_p1;
using hyle::mem::Allocator;
using hyle::sched::CancelToken;
using hyle::sched::Channel;
using hyle::sched::JoinHandle;
using hyle::sched::Scheduler;
using hyle::sched::WaitGroup;

struct RequestContext {
  Channel<std::uint64_t>* results = nullptr;
  WaitGroup* done = nullptr;
  const CancelToken* cancel = nullptr;
  std::span<const std::uint64_t> input{};
  std::uint64_t seed = 0;
  p1::WorkerHits* hits = nullptr;
  std::atomic<unsigned>* failures = nullptr;
};

void request_callback(void* raw) noexcept {
  auto* context = static_cast<RequestContext*>(raw);
  context->hits->hit(hyle::sched::current_worker_index());
  if (context->cancel->is_cancelled()) {
    context->failures->fetch_add(1, std::memory_order_relaxed);
  } else {
    const std::uint64_t digest =
        p1::tile_digest(context->input, context->seed);
    if (!context->results->try_send(digest))
      context->failures->fetch_add(1, std::memory_order_relaxed);
  }
  context->done->done();
}

struct RequestRun {
  double wall_ns = 0.0;
  std::uint64_t checksum = 0;
};

RequestRun run_request_replay(Scheduler& scheduler, Allocator allocator,
                              std::span<const std::uint64_t> input,
                              std::size_t count, p1::WorkerHits& hits) {
  Channel<std::uint64_t> results(allocator, count);
  WaitGroup done(static_cast<long>(count));
  const CancelToken cancel = CancelToken::root(allocator);
  std::atomic<unsigned> failures{0};
  std::vector<RequestContext> contexts(count);
  std::uint64_t expected_checksum = 0;
  for (std::size_t i = 0; i < count; ++i) {
    contexts[i] = RequestContext{&results, &done, &cancel, input,
                                 static_cast<std::uint64_t>(i + 1), &hits,
                                 &failures};
    expected_checksum +=
        p1::tile_digest(input, static_cast<std::uint64_t>(i + 1));
  }

  const auto start = p1::Clock::now();
  for (RequestContext& context : contexts) {
    if (!scheduler.schedule(&request_callback, &context).ok())
      p1::fail("HYLE rejected an external request callback");
  }
  done.wait();
  std::uint64_t checksum = 0;
  for (std::size_t i = 0; i < count; ++i) {
    auto value = results.try_recv();
    if (!value)
      p1::fail("HYLE request channel lost a result");
    checksum += *value;
  }
  const auto finish = p1::Clock::now();

  if (failures.load(std::memory_order_relaxed) != 0 ||
      checksum != expected_checksum || results.try_recv()) {
    p1::fail("HYLE request replay violated its cancellation/channel contract");
  }
  return {p1::elapsed_ns(start, finish), checksum};
}

struct StructuredRun {
  double wall_ns = 0.0;
  std::uint64_t checksum = 0;
};

StructuredRun run_structured_body(
    Scheduler& scheduler, Allocator allocator,
    std::span<const std::uint64_t> input, std::size_t rounds,
    std::size_t fanout, p1::WorkerHits& hits) {
  std::vector<JoinHandle<std::uint64_t>> handles;
  handles.reserve(fanout);
  std::uint64_t expected_checksum = 0;
  for (std::size_t round = 0; round < rounds; ++round) {
    for (std::size_t task = 0; task < fanout; ++task) {
      const auto seed = static_cast<std::uint64_t>(round * fanout + task + 1);
      expected_checksum += p1::tile_digest(input, seed);
    }
  }

  std::uint64_t checksum = 0;
  const auto start = p1::Clock::now();
  for (std::size_t round = 0; round < rounds; ++round) {
    handles.clear();
    for (std::size_t task = 0; task < fanout; ++task) {
      const auto seed = static_cast<std::uint64_t>(round * fanout + task + 1);
      handles.push_back(hyle::sched::schedule_task(
          scheduler, allocator, [input, seed, &hits]() noexcept {
            hits.hit(hyle::sched::current_worker_index());
            return p1::tile_digest(input, seed);
          }));
    }
    hyle::sched::when_all(
        std::span<JoinHandle<std::uint64_t>>(handles.data(), handles.size()));
    for (auto& handle : handles) {
      auto result = handle.join();
      if (!result)
        p1::fail("HYLE structured task failed to join");
      checksum += *result;
    }
  }
  const auto finish = p1::Clock::now();
  handles.clear();

  if (checksum != expected_checksum)
    p1::fail("HYLE structured replay produced a wrong result");
  return {p1::elapsed_ns(start, finish), checksum};
}

struct StructuredContext {
  Scheduler* scheduler = nullptr;
  Allocator allocator{};
  std::span<const std::uint64_t> input{};
  std::size_t rounds = 0;
  std::size_t fanout = 0;
  p1::WorkerHits* hits = nullptr;
  WaitGroup done{1};
  StructuredRun result{};
};

void structured_root(void* raw) {
  auto* context = static_cast<StructuredContext*>(raw);
  context->result = run_structured_body(
      *context->scheduler, context->allocator, context->input,
      context->rounds, context->fanout, *context->hits);
  context->done.done();
}

StructuredRun run_structured_replay(
    Scheduler& scheduler, Allocator allocator,
    std::span<const std::uint64_t> input, std::size_t rounds,
    std::size_t fanout, p1::WorkerHits& hits) {
  StructuredContext context{&scheduler, allocator, input, rounds, fanout,
                            &hits};
  if (!scheduler.spawn_detached(&structured_root, &context).ok())
    p1::fail("HYLE rejected the structured root fiber");
  context.done.wait();
  return context.result;
}

struct WakeContext;
struct WakeSlot {
  WakeContext* context = nullptr;
  unsigned worker = 0;
  hyle::sched::Waker waker{};
  std::atomic<std::size_t> published{0};
  std::atomic<std::size_t> completed{0};
};

struct WakeContext {
  std::array<WakeSlot, p1::kWorkerCount>* slots = nullptr;
  WaitGroup* done = nullptr;
  p1::WorkerHits* measured_hits = nullptr;
  std::atomic<unsigned> failures{0};
};

void completion_waiter(void* raw) noexcept {
  auto* slot = static_cast<WakeSlot*>(raw);
  constexpr std::size_t total_episodes =
      p1::kWakeWarmupEpisodes + p1::kWakeMeasuredEpisodes;
  for (std::size_t episode = 0; episode < total_episodes; ++episode) {
    if (hyle::sched::current_worker_index() !=
        static_cast<int>(slot->worker)) {
      slot->context->failures.fetch_add(1, std::memory_order_relaxed);
    }
    slot->waker = hyle::sched::current_waker();
    if (!slot->waker.valid())
      slot->context->failures.fetch_add(1, std::memory_order_relaxed);
    slot->published.store(episode + 1, std::memory_order_release);
    slot->published.notify_one();
    hyle::sched::suspend();
    if (episode >= p1::kWakeWarmupEpisodes)
      slot->context->measured_hits->hit(
          hyle::sched::current_worker_index());
    slot->completed.store(episode + 1, std::memory_order_release);
    slot->completed.notify_one();
  }
  slot->context->done->done();
}

struct WakeRun {
  double wall_ns = 0.0;
  std::uint64_t checksum = 0;
};

WakeRun run_completion_replay(Scheduler& scheduler, p1::WorkerHits& hits) {
  std::array<WakeSlot, p1::kWorkerCount> slots{};
  WaitGroup done(static_cast<long>(p1::kWorkerCount));
  WakeContext context{&slots, &done, &hits};
  for (std::size_t worker = 0; worker < slots.size(); ++worker) {
    slots[worker].context = &context;
    slots[worker].worker = static_cast<unsigned>(worker);
    if (!scheduler
             .spawn_detached_on(static_cast<unsigned>(worker),
                                &completion_waiter, &slots[worker])
             .ok()) {
      p1::fail("HYLE rejected a pinned completion waiter");
    }
  }

  p1::Clock::time_point measured_start{};
  constexpr std::size_t total_episodes =
      p1::kWakeWarmupEpisodes + p1::kWakeMeasuredEpisodes;
  for (std::size_t episode = 0; episode < total_episodes; ++episode) {
    for (const WakeSlot& slot : slots)
      p1::wait_at_least(slot.published, episode + 1);
    if (episode == p1::kWakeWarmupEpisodes)
      measured_start = p1::Clock::now();
    for (const WakeSlot& slot : slots)
      slot.waker.wake();
    for (const WakeSlot& slot : slots)
      p1::wait_at_least(slot.completed, episode + 1);
  }
  const auto measured_finish = p1::Clock::now();
  done.wait();

  if (context.failures.load(std::memory_order_relaxed) != 0 ||
      hits.total() != p1::kWakeMeasuredEpisodes * p1::kWorkerCount) {
    p1::fail("HYLE completion replay violated worker or wake accounting");
  }
  std::uint64_t checksum = 0;
  for (const WakeSlot& slot : slots)
    checksum += slot.completed.load(std::memory_order_relaxed);
  return {p1::elapsed_ns(measured_start, measured_finish), checksum};
}

}  // namespace

int main() {
  const auto cpus = p1::requested_cpus();
  p1::AffinityState affinity(cpus);
  hyle::sched::SchedulerOptions options;
  options.worker_cpu_set = std::span<const unsigned>(cpus);
  options.worker_config_policy = hyle::platform::ThreadConfigPolicy::kRequired;
  options.on_worker_start = +[](void* raw, unsigned worker) noexcept {
    static_cast<p1::AffinityState*>(raw)->record(worker, p1::current_cpu());
  };
  options.worker_hook_context = &affinity;

  const Allocator allocator = hyle::mem::system_allocator();
  hyle::Status scheduler_status;
  Scheduler scheduler(static_cast<unsigned>(p1::kWorkerCount), allocator,
                      32'768, options, &scheduler_status);
  if (!scheduler_status.ok())
    p1::fail("HYLE scheduler construction failed");
  affinity.await_ready();
  if (!affinity.exact())
    p1::fail("HYLE worker placement differs from the requested CPU set");

  std::vector<std::uint64_t> input(p1::kStructuredTileElements);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<std::uint64_t>(i * 17 + 3);

  p1::WorkerHits request_hits;
  (void)run_request_replay(
      scheduler, allocator,
      std::span<const std::uint64_t>(input.data(), p1::kRequestTileElements),
      p1::kRequestWarmup, request_hits);
  request_hits.reset();
  const RequestRun request = run_request_replay(
      scheduler, allocator,
      std::span<const std::uint64_t>(input.data(), p1::kRequestTileElements),
      p1::kRequestCount, request_hits);
  p1::print_measurement(
      "request-fanout-collect",
      static_cast<double>(p1::kRequestCount) * 1e9 / request.wall_ns,
      p1::kRequestCount, 1, p1::kRequestCount, request.checksum,
      request.wall_ns, request_hits, affinity);

  p1::WorkerHits structured_hits;
  (void)run_structured_replay(
      scheduler, allocator, input, 1, p1::kStructuredWarmupFanout,
      structured_hits);
  structured_hits.reset();
  const StructuredRun structured = run_structured_replay(
      scheduler, allocator, input, p1::kStructuredRounds,
      p1::kStructuredFanout, structured_hits);
  const std::uint64_t structured_callbacks =
      p1::kStructuredRounds * p1::kStructuredFanout;
  p1::print_measurement(
      "structured-spawn-join",
      static_cast<double>(structured_callbacks) * 1e9 / structured.wall_ns,
      p1::kStructuredRounds, p1::kStructuredFanout, structured_callbacks,
      structured.checksum, structured.wall_ns, structured_hits, affinity);

  p1::WorkerHits wake_hits;
  const WakeRun wake = run_completion_replay(scheduler, wake_hits);
  const std::uint64_t wake_callbacks =
      p1::kWakeMeasuredEpisodes * p1::kWorkerCount;
  p1::print_measurement(
      "completion-wake", static_cast<double>(wake_callbacks) * 1e9 /
                             wake.wall_ns,
      p1::kWakeMeasuredEpisodes, p1::kWorkerCount, wake_callbacks,
      wake.checksum, wake.wall_ns, wake_hits, affinity);

  scheduler.shutdown();
  p1::print_run_telemetry(cpus);
  return 0;
}
