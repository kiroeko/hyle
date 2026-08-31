#include "p1_common.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/global_control.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/partitioner.h"
#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/task_scheduler_observer.h"

namespace {

namespace p1 = hyle::bench::sched_p1;
namespace tbb = oneapi::tbb;
thread_local int peer_worker_index = -1;

class PinnedArenaObserver final : public tbb::task_scheduler_observer {
 public:
  PinnedArenaObserver(tbb::task_arena& arena,
                      const std::array<unsigned, p1::kWorkerCount>& cpus,
                      p1::AffinityState& affinity)
      : tbb::task_scheduler_observer(arena),
        cpus_(cpus),
        affinity_(affinity) {}

  void on_scheduler_entry(bool) override {
    const int slot = tbb::this_task_arena::current_thread_index();
    if (slot < 0 || static_cast<std::size_t>(slot) >= p1::kWorkerCount)
      p1::fail("oneTBB entered outside the four-slot arena");
    const unsigned cpu = cpus_[static_cast<std::size_t>(slot)];
    if (!p1::pin_current_thread(cpu))
      p1::fail("oneTBB could not pin an arena worker");
    peer_worker_index = slot;
    affinity_.record_or_validate(static_cast<std::size_t>(slot),
                                 p1::current_cpu());
  }

  void on_scheduler_exit(bool) override { peer_worker_index = -1; }

 private:
  const std::array<unsigned, p1::kWorkerCount>& cpus_;
  p1::AffinityState& affinity_;
};

struct BatchRun {
  double wall_ns = 0.0;
  std::uint64_t checksum = 0;
};

BatchRun run_batches(std::span<const std::uint64_t> input,
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
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, fanout, 1),
        [&, round](const tbb::blocked_range<std::size_t>& range) {
          const int slot = tbb::this_task_arena::current_thread_index();
          if (slot != peer_worker_index)
            p1::fail("oneTBB arena slot changed without observer entry");
          for (std::size_t item = range.begin(); item < range.end(); ++item) {
            hits.hit(slot);
            results[item] = p1::tile_digest(
                input,
                static_cast<std::uint64_t>(round * fanout + item + 1));
          }
        },
        tbb::simple_partitioner{});
    for (const std::uint64_t result : results)
      checksum += result;
  }
  const auto finish = p1::Clock::now();
  if (checksum != expected_checksum)
    p1::fail("oneTBB batch replay produced a wrong result");
  return {p1::elapsed_ns(start, finish), checksum};
}

template <class Fn>
void enqueue_and_wait(tbb::task_arena& arena, Fn&& fn) {
  std::atomic<bool> done{false};
  arena.enqueue([&] {
    fn();
    done.store(true, std::memory_order_release);
    done.notify_one();
  });
  bool observed = done.load(std::memory_order_acquire);
  while (!observed) {
    done.wait(false, std::memory_order_acquire);
    observed = done.load(std::memory_order_acquire);
  }
}

void establish_four_worker_topology(tbb::task_arena& arena) {
  enqueue_and_wait(arena, [] {
    std::atomic<unsigned> entered{0};
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, p1::kWorkerCount, 1),
        [&](const tbb::blocked_range<std::size_t>& range) {
          if (range.size() != 1)
            p1::fail("oneTBB topology warmup did not split to unit work");
          entered.fetch_add(1, std::memory_order_acq_rel);
          p1::wait_until(
              [&] {
                return entered.load(std::memory_order_acquire) ==
                       p1::kWorkerCount;
              },
              "oneTBB did not admit four arena workers");
        },
        tbb::simple_partitioner{});
  });
}

BatchRun run_replay(tbb::task_arena& arena,
                    std::span<const std::uint64_t> input,
                    std::size_t rounds, std::size_t fanout,
                    p1::WorkerHits& hits) {
  BatchRun result;
  enqueue_and_wait(arena, [&] {
    result = run_batches(input, rounds, fanout, hits);
  });
  return result;
}

}  // namespace

int main() {
  const auto cpus = p1::requested_cpus();
  p1::AffinityState affinity(cpus);
  // max_allowed_parallelism reserves one slot for an application thread even
  // though this replay's external thread never enters the arena.  Permit four
  // TBB workers globally, then keep the execution ceiling at exactly four with
  // the arena constraint below.
  tbb::global_control limit(tbb::global_control::max_allowed_parallelism,
                            p1::kWorkerCount + 1);
  // This is an enqueue-only arena: reserve no slot for a master because the
  // external caller deliberately never enters it.
  tbb::task_arena arena(static_cast<int>(p1::kWorkerCount), 0);
  arena.initialize();
  PinnedArenaObserver observer(arena, cpus, affinity);
  observer.observe(true);
  establish_four_worker_topology(arena);
  affinity.await_ready();

  std::vector<std::uint64_t> input(p1::kStructuredTileElements);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<std::uint64_t>(i * 17 + 3);

  p1::WorkerHits hits;
  (void)run_replay(arena, input, 1, p1::kStructuredWarmupFanout, hits);
  hits.reset();
  const BatchRun run = run_replay(
      arena, input, p1::kStructuredRounds, p1::kStructuredFanout, hits);
  const std::uint64_t callbacks =
      p1::kStructuredRounds * p1::kStructuredFanout;
  p1::print_measurement(
      "batch-structured", static_cast<double>(callbacks) * 1e9 / run.wall_ns,
      p1::kStructuredRounds, p1::kStructuredFanout, callbacks, run.checksum,
      run.wall_ns, hits, affinity);

  observer.observe(false);
  arena.terminate();
  p1::print_run_telemetry(cpus);
  return 0;
}
