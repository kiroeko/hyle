#include "p1_common.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stop_token>
#include <vector>

#include "marl/conditionvariable.h"
#include "marl/event.h"
#include "marl/mutex.h"
#include "marl/scheduler.h"
#include "marl/task.h"
#include "marl/waitgroup.h"

namespace {

namespace p1 = hyle::bench::sched_p1;

thread_local int peer_worker_index = -1;

template <class T>
class PeerChannel {
 public:
  explicit PeerChannel(std::size_t capacity) : storage_(capacity) {}

  bool try_send(T value) {
    {
      marl::lock lock(mutex_);
      if (size_ == storage_.size())
        return false;
      storage_[tail_] = value;
      tail_ = (tail_ + 1) % storage_.size();
      ++size_;
    }
    not_empty_.notify_one();
    return true;
  }

  bool try_recv(T& value) {
    {
      marl::lock lock(mutex_);
      if (size_ == 0)
        return false;
      value = storage_[head_];
      head_ = (head_ + 1) % storage_.size();
      --size_;
    }
    not_full_.notify_one();
    return true;
  }

 private:
  std::vector<T> storage_;
  std::size_t head_ = 0;
  std::size_t tail_ = 0;
  std::size_t size_ = 0;
  marl::mutex mutex_;
  marl::ConditionVariable not_empty_;
  marl::ConditionVariable not_full_;
};

struct RequestContext {
  PeerChannel<std::uint64_t>* results = nullptr;
  marl::WaitGroup* done = nullptr;
  const std::stop_token* cancel = nullptr;
  std::span<const std::uint64_t> input{};
  std::uint64_t seed = 0;
  p1::WorkerHits* hits = nullptr;
  std::atomic<unsigned>* failures = nullptr;
};

void request_callback(RequestContext* context) {
  context->hits->hit(peer_worker_index);
  if (context->cancel->stop_requested()) {
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

RequestRun run_request_replay(marl::Scheduler& scheduler,
                              std::span<const std::uint64_t> input,
                              std::size_t count, p1::WorkerHits& hits) {
  PeerChannel<std::uint64_t> results(count);
  marl::WaitGroup done(static_cast<unsigned>(count));
  std::stop_source cancel_source;
  const std::stop_token cancel = cancel_source.get_token();
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
    scheduler.enqueue(marl::Task([context = &context] {
      request_callback(context);
    }));
  }
  done.wait();
  std::uint64_t checksum = 0;
  for (std::size_t i = 0; i < count; ++i) {
    std::uint64_t value = 0;
    if (!results.try_recv(value))
      p1::fail("Marl request channel lost a result");
    checksum += value;
  }
  const auto finish = p1::Clock::now();

  std::uint64_t extra = 0;
  if (failures.load(std::memory_order_relaxed) != 0 ||
      checksum != expected_checksum || results.try_recv(extra)) {
    p1::fail("Marl request replay violated its cancellation/channel contract");
  }
  return {p1::elapsed_ns(start, finish), checksum};
}

struct WakeContext;
struct WakeSlot {
  WakeContext* context = nullptr;
  unsigned worker = 0;
  marl::Event event{};
  std::atomic<std::size_t> published{0};
  std::atomic<std::size_t> completed{0};
};

struct WakeContext {
  std::array<WakeSlot, p1::kWorkerCount>* slots = nullptr;
  marl::WaitGroup* done = nullptr;
  p1::WorkerHits* measured_hits = nullptr;
  std::array<std::atomic<bool>, p1::kWorkerCount> claimed{};
  std::atomic<unsigned> claimed_count{0};
  std::atomic<unsigned> failures{0};
};

void completion_waiter(WakeSlot* slot) {
  constexpr std::size_t total_episodes =
      p1::kWakeWarmupEpisodes + p1::kWakeMeasuredEpisodes;
  for (std::size_t episode = 0; episode < total_episodes; ++episode) {
    if (peer_worker_index != static_cast<int>(slot->worker))
      slot->context->failures.fetch_add(1, std::memory_order_relaxed);
    slot->published.store(episode + 1, std::memory_order_release);
    slot->published.notify_one();
    slot->event.wait();
    if (episode >= p1::kWakeWarmupEpisodes)
      slot->context->measured_hits->hit(peer_worker_index);
    slot->completed.store(episode + 1, std::memory_order_release);
    slot->completed.notify_one();
  }
  slot->context->done->done();
}

struct WakeRun {
  double wall_ns = 0.0;
  std::uint64_t checksum = 0;
};

WakeRun run_completion_replay(marl::Scheduler& scheduler,
                              p1::WorkerHits& hits) {
  std::array<WakeSlot, p1::kWorkerCount> slots{};
  marl::WaitGroup done(static_cast<unsigned>(p1::kWorkerCount));
  WakeContext context{&slots, &done, &hits};
  for (std::size_t worker = 0; worker < slots.size(); ++worker) {
    slots[worker].context = &context;
    slots[worker].worker = static_cast<unsigned>(worker);
  }

  // Marl has no public enqueue-on-worker operation. Its external enqueue path
  // distributes these bootstrap tasks among the four already-spinning workers;
  // the first task observed on each worker claims that worker's persistent
  // completion fiber, while duplicates return. This setup is outside timing.
  constexpr unsigned bootstrap_tasks = 64;
  marl::WaitGroup bootstrapped(bootstrap_tasks);
  for (unsigned i = 0; i < bootstrap_tasks; ++i) {
    scheduler.enqueue(marl::Task([&] {
      const int worker = peer_worker_index;
      if (worker < 0 || static_cast<std::size_t>(worker) >= p1::kWorkerCount)
        p1::fail("Marl bootstrap ran outside a worker");
      bool expected = false;
      const bool owner = context.claimed[static_cast<std::size_t>(worker)]
                             .compare_exchange_strong(
                                 expected, true, std::memory_order_acq_rel,
                                 std::memory_order_relaxed);
      if (owner) {
        context.claimed_count.fetch_add(1, std::memory_order_release);
        // Hold this OS worker until every other worker has claimed its own
        // persistent fiber. Waiting on Marl Event here would let this worker
        // consume all remaining bootstrap tasks and would fabricate a
        // single-worker completion topology. Setup is outside measurement.
        p1::wait_until(
            [&] {
              return context.claimed_count.load(std::memory_order_acquire) ==
                     p1::kWorkerCount;
            },
            "Marl workers did not claim the completion topology");
      }
      bootstrapped.done();
      if (owner)
        completion_waiter(&slots[static_cast<std::size_t>(worker)]);
    }));
  }
  bootstrapped.wait();
  if (context.claimed_count.load(std::memory_order_acquire) !=
      p1::kWorkerCount) {
    p1::fail("Marl did not place one completion fiber on every worker");
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
      slot.event.signal();
    for (const WakeSlot& slot : slots)
      p1::wait_at_least(slot.completed, episode + 1);
  }
  const auto measured_finish = p1::Clock::now();
  done.wait();

  if (context.failures.load(std::memory_order_relaxed) != 0 ||
      hits.total() != p1::kWakeMeasuredEpisodes * p1::kWorkerCount) {
    p1::fail("Marl completion replay violated worker or wake accounting");
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
  marl::Scheduler::Config config;
  config.setWorkerThreadCount(static_cast<int>(p1::kWorkerCount));
  config.setWorkerThreadInitializer([&](int worker) {
    if (worker < 0 || static_cast<std::size_t>(worker) >= p1::kWorkerCount)
      p1::fail("Marl initialized an out-of-range worker");
    const unsigned cpu = cpus[static_cast<std::size_t>(worker)];
    if (!p1::pin_current_thread(cpu))
      p1::fail("Marl could not pin a worker to the requested CPU");
    peer_worker_index = worker;
    affinity.record(static_cast<std::size_t>(worker), p1::current_cpu());
  });
  marl::Scheduler scheduler(config);
  affinity.await_ready();
  if (!affinity.exact())
    p1::fail("Marl worker placement differs from the requested CPU set");

  std::vector<std::uint64_t> input(p1::kRequestTileElements);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<std::uint64_t>(i * 17 + 3);

  p1::WorkerHits request_hits;
  (void)run_request_replay(
      scheduler,
      std::span<const std::uint64_t>(input.data(), p1::kRequestTileElements),
      p1::kRequestWarmup, request_hits);
  request_hits.reset();
  const RequestRun request = run_request_replay(
      scheduler,
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

  p1::print_run_telemetry(cpus);
  return 0;
}
