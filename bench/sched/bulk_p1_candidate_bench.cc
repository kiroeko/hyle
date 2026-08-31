#include "p1_common.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/sched/bulk.h"
#include "hyle/sched/scheduler.h"
#include "hyle/sched/sync.h"

namespace {

namespace p1 = hyle::bench::sched_p1;
using hyle::mem::Allocator;
using hyle::sched::BulkExecuteOptions;
using hyle::sched::Scheduler;
using hyle::sched::WaitGroup;

struct BatchRun {
  double wall_ns = 0.0;
  std::uint64_t checksum = 0;
};

BatchRun run_batches(Scheduler& scheduler,
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

  BulkExecuteOptions options;
  options.grain_size = 1;
  options.max_concurrency = static_cast<unsigned>(p1::kWorkerCount);
  std::uint64_t checksum = 0;
  const auto start = p1::Clock::now();
  for (std::size_t round = 0; round < rounds; ++round) {
    const hyle::Status status = hyle::sched::bulk_execute(
        scheduler, 0, fanout,
        [&, round](std::size_t begin, std::size_t end) noexcept {
          for (std::size_t item = begin; item < end; ++item) {
            hits.hit(hyle::sched::current_worker_index());
            results[item] = p1::tile_digest(
                input,
                static_cast<std::uint64_t>(round * fanout + item + 1));
          }
        },
        options);
    if (!status.ok())
      p1::fail("HYLE bulk_execute rejected the normal batch replay");
    for (const std::uint64_t result : results)
      checksum += result;
  }
  const auto finish = p1::Clock::now();
  if (checksum != expected_checksum)
    p1::fail("HYLE batch replay produced a wrong result");
  return {p1::elapsed_ns(start, finish), checksum};
}

struct RootContext {
  Scheduler* scheduler = nullptr;
  std::span<const std::uint64_t> input{};
  std::size_t rounds = 0;
  std::size_t fanout = 0;
  p1::WorkerHits* hits = nullptr;
  WaitGroup done{1};
  BatchRun result{};
};

void batch_root(void* raw) {
  auto* context = static_cast<RootContext*>(raw);
  context->result = run_batches(*context->scheduler, context->input,
                                context->rounds, context->fanout,
                                *context->hits);
  context->done.done();
}

BatchRun run_replay(Scheduler& scheduler,
                    std::span<const std::uint64_t> input,
                    std::size_t rounds, std::size_t fanout,
                    p1::WorkerHits& hits) {
  RootContext context{&scheduler, input, rounds, fanout, &hits};
  if (!scheduler.spawn_detached(&batch_root, &context).ok())
    p1::fail("HYLE rejected the batch root fiber");
  context.done.wait();
  return context.result;
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

  scheduler.shutdown();
  p1::print_run_telemetry(cpus);
  return 0;
}
