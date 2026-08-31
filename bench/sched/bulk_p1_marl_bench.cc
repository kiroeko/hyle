#include "p1_common.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "marl/scheduler.h"
#include "marl/task.h"
#include "marl/waitgroup.h"

namespace {

namespace p1 = hyle::bench::sched_p1;
thread_local int peer_worker_index = -1;

struct BatchRun {
  double wall_ns = 0.0;
  std::uint64_t checksum = 0;
};

BatchRun run_batches(marl::Scheduler& scheduler,
                     std::span<const std::uint64_t> input,
                     std::size_t rounds, std::size_t fanout,
                     p1::WorkerHits& hits) {
  std::vector<std::uint64_t> results(fanout);
  std::uint64_t expected_checksum = 0;
  for (std::size_t round = 0; round < rounds; ++round) {
    for (std::size_t item = 0; item < fanout; ++item) {
      expected_checksum += p1::tile_digest(
          input, static_cast<std::uint64_t>(round * fanout + item + 1));
    }
  }

  std::uint64_t checksum = 0;
  const auto start = p1::Clock::now();
  for (std::size_t round = 0; round < rounds; ++round) {
    marl::WaitGroup done(static_cast<unsigned>(fanout));
    for (std::size_t item = 0; item < fanout; ++item) {
      const auto seed =
          static_cast<std::uint64_t>(round * fanout + item + 1);
      scheduler.enqueue(marl::Task([&, item, seed] {
        hits.hit(peer_worker_index);
        results[item] = p1::tile_digest(input, seed);
        done.done();
      }));
    }
    done.wait();
    for (const std::uint64_t result : results)
      checksum += result;
  }
  const auto finish = p1::Clock::now();
  if (checksum != expected_checksum)
    p1::fail("Marl batch replay produced a wrong result");
  return {p1::elapsed_ns(start, finish), checksum};
}

struct RootContext {
  marl::Scheduler* scheduler = nullptr;
  std::span<const std::uint64_t> input{};
  std::size_t rounds = 0;
  std::size_t fanout = 0;
  p1::WorkerHits* hits = nullptr;
  marl::WaitGroup done{1};
  BatchRun result{};
};

BatchRun run_replay(marl::Scheduler& scheduler,
                    std::span<const std::uint64_t> input,
                    std::size_t rounds, std::size_t fanout,
                    p1::WorkerHits& hits) {
  RootContext context{&scheduler, input, rounds, fanout, &hits};
  scheduler.enqueue(marl::Task([&context] {
    context.result = run_batches(*context.scheduler, context.input,
                                 context.rounds, context.fanout,
                                 *context.hits);
    context.done.done();
  }));
  context.done.wait();
  return context.result;
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

  std::vector<std::uint64_t> input(p1::kStructuredTileElements);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<std::uint64_t>(i * 17 + 3);

  p1::WorkerHits hits;
  (void)run_replay(scheduler, input, 1, p1::kStructuredWarmupFanout, hits);
  hits.reset();
  const BatchRun run = run_replay(
      scheduler, input, p1::kStructuredRounds, p1::kStructuredFanout, hits);
  const std::uint64_t callbacks =
      p1::kStructuredRounds * p1::kStructuredFanout;
  p1::print_measurement(
      "batch-structured", static_cast<double>(callbacks) * 1e9 / run.wall_ns,
      p1::kStructuredRounds, p1::kStructuredFanout, callbacks, run.checksum,
      run.wall_ns, hits, affinity);
  p1::print_run_telemetry(cpus);
  return 0;
}
