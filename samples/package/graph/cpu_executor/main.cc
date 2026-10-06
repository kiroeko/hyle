#include <hyle/graph/cpu.h>
#include <hyle/graph/executable.h>
#include <hyle/mem/stats.h>
#include <hyle/sched/bulk.h>
#include <hyle/sched/scheduler.h>
#include <hyle/sched/waker.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace {
namespace compute = hyle::compute;
namespace graph = hyle::graph;
namespace mem = hyle::mem;
namespace sched = hyle::sched;

void require(bool condition, const char *what) {
  if (!condition) {
    std::fprintf(stderr, "CPU executor consumer failed: %s\n", what);
    std::exit(1);
  }
}
void check(hyle::Status status) {
  if (!status.ok()) {
    const auto message = status.message();
    std::fprintf(stderr, "CPU executor consumer: %.*s\n",
                 static_cast<int>(message.size()), message.data());
    std::exit(1);
  }
}
template <class T> T take(hyle::Result<T> result) {
  if (!result)
    check(std::move(result).error());
  return std::move(*result);
}

// Only this consumer's observer pays for task counters. They are not part of
// the adapter's required shape and this executable does not measure latency.
struct Observations {
  std::atomic<unsigned> started{0}, stopped{0};
  std::array<std::atomic<std::size_t>, 2> worker_tasks{};
  std::atomic<std::size_t> episodes{0}, nested_episodes{0};

  std::size_t task_count() const noexcept {
    return worker_tasks[0].load() + worker_tasks[1].load();
  }
};

// Borrowed, synchronous, infallible executor over an application-owned runtime.
// The application keeps Scheduler valid AND open until all Compute work ends.
struct Executor {
  sched::Scheduler &scheduler;
  Observations &observations;

  compute::ParallelExecutorRef ref() noexcept {
    return {this, scheduler.num_workers(), &parallel_for, &in_parallel};
  }
  static bool in_parallel(void *) noexcept {
    // Conservatively avoid nested fan-out from any Scheduler worker. bulk also
    // handles raw-worker nesting without waiting for another worker.
    return sched::current_worker_index() >= 0;
  }
  static void parallel_for(void *opaque, std::size_t count,
                           compute::ParallelTaskFn task,
                           void *context) noexcept {
    auto &self = *static_cast<Executor *>(opaque);
    ++self.observations.episodes;
    if (in_parallel(nullptr))
      ++self.observations.nested_episodes;
    const auto status = sched::bulk_execute(
        self.scheduler, 0, count,
        [&](std::size_t begin, std::size_t end) noexcept {
          const int worker = sched::current_worker_index();
          require(worker >= 0 && worker < 2,
                  "task ran outside provider workers");
          for (std::size_t i = begin; i < end; ++i) {
            ++self.observations.worker_tasks[static_cast<std::size_t>(worker)];
            task(context, i, count);
          }
        });
    // Never hide admission failure or replay a potentially completed prefix.
    // With the documented open-runtime precondition and this infallible body,
    // bulk completes every index even if fewer than requested runners enter.
    check(status);
  }
};

constexpr std::size_t kRows = 17, kInner = 129, kHidden = 257;

compute::TensorDesc matrix(std::size_t rows, std::size_t columns) {
  return take(compute::TensorDesc::dense(
      compute::DataType::kFloat32,
      std::array{static_cast<std::int64_t>(rows),
                 static_cast<std::int64_t>(columns)}));
}
struct Tensor {
  mem::Buffer storage;
  compute::TensorView view;
  explicit Tensor(compute::TensorDesc description)
      : storage(take(mem::Buffer::create(
            mem::system_resource(), take(description.required_bytes()), 64))),
        view(take(compute::TensorView::create(description, storage.view()))) {}
  void write(std::span<const float> values) {
    check(storage.view().write(std::as_bytes(values)));
  }
};

std::vector<float> weights(std::size_t count, int salt) {
  std::vector<float> result(count);
  for (std::size_t i = 0; i < count; ++i)
    result[i] =
        static_cast<float>(static_cast<int>((i + salt) % 13) - 6) / 128.0F;
  return result;
}

struct Model {
  std::vector<float> gate_values = weights(kInner * kHidden, 1);
  std::vector<float> up_values = weights(kInner * kHidden, 4);
  std::vector<float> down_values = weights(kHidden * kInner, 7);
  Tensor gate{matrix(kInner, kHidden)};
  Tensor up{matrix(kInner, kHidden)};
  Tensor down{matrix(kHidden, kInner)};
  compute::Context context;
  graph::Graph logical;
  graph::ValueId input_id, output_id;
  graph::CompiledGraph compiled;

  Model(Executor &executor, mem::AllocatorRef allocator) {
    gate.write(gate_values);
    up.write(up_values);
    down.write(down_values);
    context = take(graph::cpu::create_context(
        {.metadata_allocator = allocator, .executor = executor.ref()}));
    auto builder = take(graph::GraphBuilder::create());
    input_id = take(builder.input(matrix(kRows, kInner)));
    const auto gate_id = take(builder.constant(matrix(kInner, kHidden)));
    const auto up_id = take(builder.constant(matrix(kInner, kHidden)));
    const auto down_id = take(builder.constant(matrix(kHidden, kInner)));
    const auto normalized = take(builder.add(compute::RmsNorm{}, input_id));
    const auto g = take(builder.add(compute::MatMul{}, normalized, gate_id));
    const auto u = take(builder.add(compute::MatMul{}, normalized, up_id));
    const auto activated = take(builder.add(
        compute::GatedActivation{.inputs = compute::GatedInputKind::kSeparate},
        g, u));
    output_id = take(builder.add(compute::MatMul{}, activated, down_id));
    check(builder.publish(output_id));
    logical = take(std::move(builder).finish());
    const std::array constants{
        graph::ConstantBinding{gate_id, gate.view.as_const()},
        graph::ConstantBinding{up_id, up.view.as_const()},
        graph::ConstantBinding{down_id, down.view.as_const()}};
    compiled = take(graph::compile(
        logical, context,
        {.constants = constants,
         .provider_replay = graph::ProviderReplayPolicy::kAllowAny}));
    require(compiled.report.node_count() == 5, "unexpected graph shape");
    // Mature fragments keep their own runtime. Region count and prepared vs
    // managed replay are reported facts, not a required legacy FFN route.
  }
};

// Independent double oracle; no HYLE/SIMD/oneDNN computation is used here.
std::vector<double> oracle(const Model &model, std::span<const float> input) {
  std::vector<double> result(kRows * kInner);
  std::array<double, kInner> normalized;
  std::array<double, kHidden> activated;
  for (std::size_t row = 0; row < kRows; ++row) {
    double squared = 0;
    for (std::size_t i = 0; i < kInner; ++i) {
      const double value = input[row * kInner + i];
      squared += value * value;
    }
    const double scale = 1 / std::sqrt(squared / kInner + 1.0e-5);
    for (std::size_t i = 0; i < kInner; ++i)
      normalized[i] = input[row * kInner + i] * scale;
    for (std::size_t col = 0; col < kHidden; ++col) {
      double g = 0, u = 0;
      for (std::size_t i = 0; i < kInner; ++i) {
        g += normalized[i] * model.gate_values[i * kHidden + col];
        u += normalized[i] * model.up_values[i * kHidden + col];
      }
      activated[col] = g / (1 + std::exp(-g)) * u;
    }
    for (std::size_t col = 0; col < kInner; ++col)
      for (std::size_t i = 0; i < kHidden; ++i)
        result[row * kInner + col] +=
            activated[i] * model.down_values[i * kInner + col];
  }
  return result;
}

struct Run {
  Model &model;
  Tensor input{matrix(kRows, kInner)}, output{matrix(kRows, kInner)};
  std::vector<float> input_values = std::vector<float>(kRows * kInner);
  std::vector<float> actual = std::vector<float>(kRows * kInner);
  std::vector<double> expected;
  compute::Queue queue;
  graph::Invocation invocation;

  explicit Run(Model &owner)
      : model(owner), queue(take(owner.context.create_queue())),
        invocation(take(owner.compiled.executable.create_invocation())) {
    check(invocation.bind(owner.input_id, input.view.as_const()));
    check(invocation.bind(owner.output_id, output.view));
    const std::array queues{graph::QueueBinding{
        .lane = graph::ExecutionLaneId{0}, .queue = &queue}};
    check(invocation.prepare_replay(queues));
  }
  void set_input(int seed) {
    for (std::size_t i = 0; i < input_values.size(); ++i)
      input_values[i] =
          static_cast<float>(static_cast<int>((i * 7 + seed) % 31) - 15) /
          16.0F;
    expected = oracle(model, input_values);
    input.write(input_values);
    std::fill(actual.begin(), actual.end(), 12345.0F);
    output.write(actual);
  }
  void execute() noexcept {
    auto done = take(invocation.submit(queue));
    check(done.wait());
  }
  void verify() {
    check(
        output.storage.view().read(std::as_writable_bytes(std::span(actual))));
    for (std::size_t i = 0; i < actual.size(); ++i)
      require(std::isfinite(actual[i]) &&
                  std::abs(actual[i] - expected[i]) <=
                      3.0e-5 * (std::max)(1.0, std::abs(expected[i])),
              "computed output differs from independent oracle");
  }
};

void run_model(Executor &executor) {
  auto stats = take(mem::StatsAllocator::create(mem::system_allocator()));
  {
    Model model(executor, stats.get());
    Run first(model), second(model);
    first.set_input(1);
    second.set_input(5);
    // Ordinary prepared input changes do not change bindings or prepare again.
    const auto before = stats.snapshot();
    const auto episodes_before = executor.observations.episodes.load();
    const auto tasks_before = executor.observations.task_count();
    const auto nested_before = executor.observations.nested_episodes.load();
    for (int seed : {1, 3, 9}) {
      first.set_input(seed);
      first.execute();
      first.verify();
    }
    first.set_input(2);
    second.set_input(11);
    std::jthread a([&] { first.execute(); });
    std::jthread b([&] { second.execute(); });
    a.join();
    b.join();
    first.verify();
    second.verify();

    first.set_input(17);
    check(executor.scheduler.schedule(
        [](void *opaque) noexcept { static_cast<Run *>(opaque)->execute(); },
        &first));
    executor.scheduler.drain();
    first.verify();
    const auto after = stats.snapshot();
    require(after.total_allocs == before.total_allocs &&
                after.reallocs == before.reallocs,
            "prepared replay allocated through Compute's metadata owner");
    const auto episodes =
        executor.observations.episodes.load() - episodes_before;
    const auto tasks = executor.observations.task_count() - tasks_before;
    const auto nested =
        executor.observations.nested_episodes.load() - nested_before;
    require(episodes > 0 && tasks > 0,
            "mature computed route bypassed its executor");
    // The scheduled execution above verifies nested completion and outputs;
    // a provider may handle nesting inline without another bulk episode.
    std::printf("CPU executor PASS: workers=%u route=%s regions=%zu "
                "workspace=%zu replay_metadata_allocations=0 "
                "executor_episodes=%zu worker_tasks=%zu nested=%zu\n",
                executor.scheduler.num_workers(),
                "default mature",
                model.compiled.report.region_count(),
                model.compiled.report.workspace_bytes(), episodes, tasks,
                nested);
  }
  require(stats.balanced(), "Compute metadata outlived all consumers");
}

void run_workers(unsigned workers) {
  Observations observed;
  auto stats = take(mem::StatsAllocator::create(mem::system_allocator()));
  {
    sched::SchedulerOptions options;
    options.idle_policy = sched::SchedulerIdlePolicy::kPowerSaving;
    options.worker_hook_context = &observed;
    options.on_worker_start = [](void *p, unsigned) noexcept {
      ++static_cast<Observations *>(p)->started;
    };
    options.on_worker_stop = [](void *p, unsigned) noexcept {
      ++static_cast<Observations *>(p)->stopped;
    };
    hyle::Status status;
    sched::Scheduler scheduler(workers, stats.get(), 64, options, &status);
    check(status);
    Executor executor{scheduler, observed};
    std::array<std::atomic<unsigned>, 8> visits{};
    executor.ref().parallel_for(
        visits.size(),
        [](void *p, std::size_t index, std::size_t) noexcept {
          ++(*static_cast<decltype(visits) *>(p))[index];
        },
        &visits);
    for (const auto &value : visits)
      require(value.load() == 1, "bulk dropped or duplicated a task");
    run_model(executor);
    scheduler.begin_shutdown();
    require(!scheduler.accepting(), "shutdown did not close admission");
    std::atomic<bool> invoked{false};
    auto rejected = scheduler.schedule(
        [](void *p) noexcept { *static_cast<std::atomic<bool> *>(p) = true; },
        &invoked);
    require(!rejected.ok() && !invoked.load(), "closed runtime accepted work");
    scheduler.shutdown();
  }
  require(stats.balanced(), "Scheduler storage survived destruction");
  require(observed.started == workers && observed.stopped == workers,
          "worker lifecycle mismatch");
  require(observed.task_count() != 0, "no task ran on a real provider worker");
  std::printf("  actual worker tasks=[%zu,%zu], episodes=%zu nested=%zu, "
              "workers_started/stopped=%u/%u\n",
              observed.worker_tasks[0].load(), observed.worker_tasks[1].load(),
              observed.episodes.load(), observed.nested_episodes.load(),
              observed.started.load(), observed.stopped.load());
}
} // namespace

int main() {
  run_workers(1);
  run_workers(2);
  return 0;
}
