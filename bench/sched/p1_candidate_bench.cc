#include "p1_common.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"
#include "hyle/sched/waker.h"

namespace {

namespace p1 = hyle::bench::sched_p1;
using hyle::mem::AllocatorRef;
using hyle::sched::CancelToken;
using hyle::sched::Channel;
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

RequestRun run_request_replay(Scheduler& scheduler, AllocatorRef allocator,
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
    for (WakeSlot& slot : slots)
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

  const AllocatorRef allocator = hyle::mem::system_allocator();
  hyle::Status scheduler_status;
  Scheduler scheduler(static_cast<unsigned>(p1::kWorkerCount), allocator,
                      32'768, options, &scheduler_status);
  if (!scheduler_status.ok())
    p1::fail("HYLE scheduler construction failed");
  affinity.await_ready();
  if (!affinity.exact())
    p1::fail("HYLE worker placement differs from the requested CPU set");

  std::vector<std::uint64_t> input(p1::kRequestTileElements);
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
