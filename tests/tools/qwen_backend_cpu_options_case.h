#pragma once

#include "../../engine/inference/backend.h"
#include "../../engine/inference/readback.h"

#include <array>
#include <atomic>
#include <limits>
#include <string>

namespace hyle::inference::test {

struct CpuOptionsFacts final {
  std::size_t executor_calls = 0, executor_tasks = 0, compared_values = 0;
  std::string configured_provider, default_provider;
};

// Use the same real CPU Graph through the configured and original default
// factory. The external executor backing outlives every derived native use.
inline Result<CpuOptionsFacts> cpu_options_case() {
  namespace c = compute;
  namespace g = graph;
  struct Probe final {
    std::atomic<std::size_t> calls{0}, tasks{0};
  } probe;
  const c::ParallelExecutorRef executor(
      &probe, 2,
      [](void *self, std::size_t count, c::ParallelTaskFn task, void *state) noexcept {
        auto &p = *static_cast<Probe *>(self);
        p.calls.fetch_add(1, std::memory_order_relaxed);
        p.tasks.fetch_add(count, std::memory_order_relaxed);
        for (std::size_t i = 0; i < count; ++i) task(state, i, count);
      });
  const c::cpu::DefaultContextOptions options{.executor = executor, .native_threads = 1};
  // These must fail before any GPU endpoint exists, including CPU-only images.
  const auto no_cpu = Backend::create({.primary = BackendKind::kCuda}, {}, options);
  if (no_cpu || !no_cpu.error().IsInvalidArgument())
    return std::unexpected(Status::Internal("CPU options silently ignored a missing CPU endpoint"));
  for (const auto threads : {std::uint32_t{0}, (std::numeric_limits<std::uint32_t>::max)()}) {
    const c::cpu::DefaultContextOptions invalid{.native_threads = threads};
    const auto rejected = Backend::create({.primary = BackendKind::kCpu}, {}, invalid);
    if (rejected || !rejected.error().IsInvalidArgument())
      return std::unexpected(Status::Internal("CPU options did not preserve native thread validation"));
  }
  HYLE_ASSIGN_OR_RETURN(auto configured, Backend::create({.primary = BackendKind::kCpu}, {}, options));
  HYLE_ASSIGN_OR_RETURN(auto defaulted, Backend::create({.primary = BackendKind::kCpu}));
  HYLE_ASSIGN_OR_RETURN(auto desc, c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{17}));
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto input, builder.input(desc));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.add(c::Unary{c::UnaryKind::kNeg}, input));
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  CpuOptionsFacts facts;
  std::array<std::array<float, 17>, 2> expected{};
  for (std::size_t replay = 0; replay < expected.size(); ++replay)
    for (std::size_t i = 0; i < expected[replay].size(); ++i)
      expected[replay][i] = -(static_cast<float>(static_cast<int>(i) - 8) / 8.F +
          static_cast<float>(replay) / 32.F);
  auto execute = [&](Backend &backend, std::string &provider) -> Status {
    HYLE_ASSIGN_OR_RETURN(auto queue, backend.common_queue());
    HYLE_ASSIGN_OR_RETURN(auto compiled, backend.compile(graph, {}));
    provider = compiled.report.provider().name;
    HYLE_ASSIGN_OR_RETURN(auto readback, PreparedReadback::prepare(compiled, output, backend));
    HYLE_ASSIGN_OR_RETURN(auto storage, mem::Buffer::create(backend.device(), sizeof(expected[0]), 256));
    HYLE_ASSIGN_OR_RETURN(auto run, compiled.executable.create_invocation());
    HYLE_RETURN_IF_ERROR(run.prepare_resources());
    HYLE_ASSIGN_OR_RETURN(auto tensor, c::ConstTensorView::create(desc, storage.view()));
    HYLE_RETURN_IF_ERROR(run.bind(input, tensor));
    HYLE_RETURN_IF_ERROR(readback.bind(run));
    for (std::size_t replay = 0; replay < expected.size(); ++replay) {
      std::array<float, 17> values{}, actual{};
      for (std::size_t i = 0; i < values.size(); ++i) values[i] = -expected[replay][i];
      HYLE_RETURN_IF_ERROR(backend.write(storage.view(), std::as_bytes(std::span(values))));
      HYLE_RETURN_IF_ERROR(run.prepare_replay(backend.queue_bindings(), {.mode = g::ReplayMode::kCommands}));
      HYLE_ASSIGN_OR_RETURN(auto done, run.submit(backend.queue_bindings()));
      HYLE_RETURN_IF_ERROR(done.wait());
      HYLE_RETURN_IF_ERROR(readback.read(backend, run, std::as_writable_bytes(std::span(actual))));
      if (actual != expected[replay])
        return Status::Internal("CPU configured/default Graph values differ from the exact negation");
      facts.compared_values += actual.size();
      if (replay == 0) {
        auto moved = std::move(backend);
        if (backend.kind() != BackendKind::kDefault)
          return Status::Internal("moved CPU Backend retained its old endpoint");
        backend = std::move(moved);
        HYLE_ASSIGN_OR_RETURN(auto same_queue, backend.common_queue());
        if (same_queue != queue)
          return Status::Internal("CPU Backend move changed the prepared Queue address");
      }
    }
    HYLE_RETURN_IF_ERROR(run.drain());
    return backend.collect_retired();
  };
  HYLE_RETURN_IF_ERROR(execute(configured, facts.configured_provider));
  facts.executor_calls = probe.calls.load(std::memory_order_relaxed);
  facts.executor_tasks = probe.tasks.load(std::memory_order_relaxed);
  if (facts.executor_calls == 0 || facts.executor_tasks == 0)
    return std::unexpected(Status::Internal("configured CPU Graph did not dispatch the borrowed executor"));
  HYLE_RETURN_IF_ERROR(execute(defaulted, facts.default_provider));
  if (probe.calls.load(std::memory_order_relaxed) != facts.executor_calls ||
      probe.tasks.load(std::memory_order_relaxed) != facts.executor_tasks)
    return std::unexpected(Status::Internal("default CPU Graph unexpectedly reused another endpoint's executor"));
  return facts;
}

} // namespace hyle::inference::test
