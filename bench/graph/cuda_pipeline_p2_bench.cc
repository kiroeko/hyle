// Same-task cross-domain product replay: CUDA device input -> pinned host ->
// CPU native Unary -> pinned host -> CUDA device output. The Graph and direct
// routes use identical public HYLE commands, resources and completion points.
#include <nanobench.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/transfer.h"
#include "hyle/graph/executable.h"
#include "hyle/mem/buffer.h"

#include "../common/perf_process.h"

namespace {
namespace nb = ankerl::nanobench;
using hyle::Status;
using hyle::compute::Context;
using hyle::compute::DataType;
using hyle::compute::OperationInvocation;
using hyle::compute::Queue;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::compute::Unary;
using hyle::compute::UnaryKind;
using hyle::graph::CompileEnvironment;
using hyle::graph::CompileOptions;
using hyle::graph::ExecutionDomain;
using hyle::graph::ExecutionDomainId;
using hyle::graph::ExternalResourceContract;
using hyle::graph::GraphBuilder;
using hyle::graph::Invocation;
using hyle::graph::NodePlacement;
using hyle::graph::PlacementStrength;
using hyle::graph::QueueBinding;
using hyle::graph::ReplayMode;
using hyle::graph::ReplayRoute;

constexpr std::size_t kElements = 1U << 20U;

[[noreturn]] void fail_setup(std::string_view message) {
  std::fprintf(stderr, "graph CUDA pipeline benchmark setup failed: %.*s\n",
               static_cast<int>(message.size()), message.data());
  std::abort();
}

template <class T>
T take_or_abort(hyle::Result<T> result, std::string_view message) {
  if (!result)
    fail_setup(message);
  return std::move(*result);
}

void require_ok(Status status, std::string_view message) {
  if (!status.ok())
    fail_setup(message);
}

bool environment_present(const char *name) {
#if defined(_WIN32)
  char *value = nullptr;
  std::size_t size = 0;
  const bool present = _dupenv_s(&value, &size, name) == 0 && value != nullptr;
  std::free(value);
  return present;
#else
  return std::getenv(name) != nullptr;
#endif
}

template <class Candidate, class Peer>
void run_pair(nb::Bench &bench, bool peer_first,
              std::string_view candidate_name, Candidate &&candidate,
              std::string_view peer_name, Peer &&peer) {
  if (peer_first) {
    bench.run(peer_name, std::forward<Peer>(peer));
    bench.run(candidate_name, std::forward<Candidate>(candidate));
  } else {
    bench.run(candidate_name, std::forward<Candidate>(candidate));
    bench.run(peer_name, std::forward<Peer>(peer));
  }
}

struct OwnedTensor final {
  hyle::mem::Buffer buffer{};
  TensorView view{};
};

OwnedTensor make_tensor(hyle::mem::MemoryResourceRef resource,
                        const TensorDesc &description) {
  const std::size_t bytes =
      take_or_abort(description.required_bytes(), "tensor byte size");
  OwnedTensor result;
  result.buffer = take_or_abort(
      hyle::mem::Buffer::create(resource, bytes, 256), "tensor allocation");
  result.view = take_or_abort(TensorView::create(description, result.buffer.view()),
                              "tensor view");
  return result;
}

struct GraphRoute final {
  OwnedTensor input{};
  OwnedTensor output{};
  Queue cpu_queue{};
  Queue cuda_queue{};
  Invocation invocation{};
};

struct DirectRoute final {
  OwnedTensor input{};
  OwnedTensor host_input{};
  OwnedTensor host_output{};
  OwnedTensor output{};
  Queue cpu_queue{};
  Queue cuda_queue{};
  OperationInvocation unary{};
};

class Fixture final {
public:
  explicit Fixture(bool peer_allocation_first)
      : description_(take_or_abort(
            TensorDesc::dense(
                DataType::kFloat32,
                std::array{static_cast<std::int64_t>(kElements)}),
            "pipeline tensor description")),
        cpu_context_(take_or_abort(hyle::compute::cpu::create_context(),
                                   "CPU context")),
        cuda_context_(take_or_abort(hyle::compute::Context::create(
                                        take_or_abort(
                                            hyle::compute::cuda::device(0),
                                            "CUDA device")),
                                    "CUDA context")),
        pinned_resource_(take_or_abort(
            hyle::compute::cuda::MemoryResource::create(
                cuda_context_, hyle::compute::cuda::MemoryKind::kPinnedHost),
            "CUDA pinned resource")),
        device_resource_(take_or_abort(
            hyle::compute::cuda::MemoryResource::create(
                cuda_context_, hyle::compute::cuda::MemoryKind::kDevice),
            "CUDA device resource")) {
    if (peer_allocation_first) {
      direct_.emplace(make_direct());
      graph_.emplace(make_graph());
    } else {
      graph_.emplace(make_graph());
      direct_.emplace(make_direct());
    }
    prepare_graph_replay();
    initialize(graph_->input);
    initialize(direct_->input);
    for (int warmup = 0; warmup < 64; ++warmup) {
      if (run_graph() == 0)
        fail_setup("warm Graph cross-domain pipeline");
      if (run_direct() == 0)
        fail_setup("warm direct cross-domain pipeline");
    }
    validate(graph_->output);
    validate(direct_->output);
  }

  std::uint64_t run_graph() noexcept {
    const std::array queues{
        QueueBinding{.domain = ExecutionDomainId{0},
                     .queue = &graph_->cpu_queue},
        QueueBinding{.domain = ExecutionDomainId{1},
                     .queue = &graph_->cuda_queue},
    };
    auto completion = graph_->invocation.submit(queues);
    if (!completion) {
      std::fprintf(stderr, "Graph submit failed: %.*s\n",
                   static_cast<int>(completion.error().message().size()),
                   completion.error().message().data());
      std::abort();
    }
    const Status terminal = completion->wait();
    if (!terminal.ok()) {
      std::fprintf(stderr, "Graph wait failed: %.*s\n",
                   static_cast<int>(terminal.message().size()),
                   terminal.message().data());
      std::abort();
    }
    return 1U;
  }

  std::uint64_t run_direct() noexcept {
    auto inbound = hyle::compute::copy_async(
        direct_->cuda_queue, direct_->host_input.buffer.view(),
        direct_->input.buffer.view());
    if (!inbound) {
      std::fprintf(stderr, "direct inbound submit failed: %.*s\n",
                   static_cast<int>(inbound.error().message().size()),
                   inbound.error().message().data());
      std::abort();
    }
    Status terminal = inbound->wait();
    if (!terminal.ok()) {
      std::fprintf(stderr, "direct inbound wait failed: %.*s\n",
                   static_cast<int>(terminal.message().size()),
                   terminal.message().data());
      std::abort();
    }
    auto transformed = direct_->unary.submit(
        direct_->cpu_queue, direct_->host_input.view.as_const(),
        direct_->host_output.view);
    if (!transformed) {
      std::fprintf(stderr, "direct Unary submit failed: %.*s\n",
                   static_cast<int>(transformed.error().message().size()),
                   transformed.error().message().data());
      std::abort();
    }
    terminal = transformed->wait();
    if (!terminal.ok()) {
      std::fprintf(stderr, "direct Unary wait failed: %.*s\n",
                   static_cast<int>(terminal.message().size()),
                   terminal.message().data());
      std::abort();
    }
    auto outbound = hyle::compute::copy_async(
        direct_->cuda_queue, direct_->output.buffer.view(),
        direct_->host_output.buffer.view());
    if (!outbound) {
      std::fprintf(stderr, "direct outbound submit failed: %.*s\n",
                   static_cast<int>(outbound.error().message().size()),
                   outbound.error().message().data());
      std::abort();
    }
    terminal = outbound->wait();
    if (!terminal.ok()) {
      std::fprintf(stderr, "direct outbound wait failed: %.*s\n",
                   static_cast<int>(terminal.message().size()),
                   terminal.message().data());
      std::abort();
    }
    return 1U;
  }

private:
  GraphRoute make_graph() {
    GraphBuilder builder =
        take_or_abort(GraphBuilder::create(), "graph builder");
    const auto input =
        take_or_abort(builder.input(description_), "graph input");
    const auto output = take_or_abort(
        builder.add(Unary{UnaryKind::kNeg}, input), "graph Unary node");
    require_ok(builder.publish(output), "publish graph output");
    auto logical = take_or_abort(std::move(builder).finish(), "logical graph");

    const std::array domains{
        ExecutionDomain{.context = &cpu_context_,
                        .workspace_resource = pinned_resource_.ref()},
        ExecutionDomain{.context = &cuda_context_,
                        .workspace_resource = device_resource_.ref()},
    };
    const std::array placements{NodePlacement{
        .node = take_or_abort(logical.node_id(0), "graph node id"),
        .domain = ExecutionDomainId{0},
        .strength = PlacementStrength::kRequired,
    }};
    const std::array resources{
        ExternalResourceContract{.value = input,
                                 .domain = ExecutionDomainId{1},
                                 .resource = device_resource_.ref()},
        ExternalResourceContract{.value = output,
                                 .domain = ExecutionDomainId{1},
                                 .resource = device_resource_.ref()},
    };
    auto compiled = take_or_abort(
        hyle::graph::compile(
            logical, CompileEnvironment{.domains = domains},
            CompileOptions{.node_placements = placements,
                           .external_resources = resources}),
        "compile cross-domain graph");
    if (compiled.report.step_count() != 3U ||
        compiled.report.segment_count() != 3U ||
        compiled.report.copy_bytes() != 2U * description_.required_bytes().value())
      fail_setup("unexpected cross-domain plan");

    GraphRoute route;
    route.input = make_tensor(device_resource_.ref(), description_);
    route.output = make_tensor(device_resource_.ref(), description_);
    route.cpu_queue =
        take_or_abort(cpu_context_.create_queue(), "Graph CPU queue");
    route.cuda_queue =
        take_or_abort(cuda_context_.create_queue(), "Graph CUDA queue");
    route.invocation = take_or_abort(compiled.executable.create_invocation(),
                                     "Graph invocation");
    require_ok(route.invocation.bind(input, route.input.view.as_const()),
               "bind Graph input");
    require_ok(route.invocation.bind(output, route.output.view),
               "bind Graph output");
    return route;
  }

  void prepare_graph_replay() {
    const std::array queues{
        QueueBinding{.domain = ExecutionDomainId{0},
                     .queue = &graph_->cpu_queue},
        QueueBinding{.domain = ExecutionDomainId{1},
                     .queue = &graph_->cuda_queue},
    };
    require_ok(graph_->invocation.prepare_replay(
                   queues, {.mode = ReplayMode::kCommands}),
               "prepare Graph command replay");
    if (graph_->invocation.replay_route() != ReplayRoute::kCommands)
      fail_setup("Graph did not select command replay");
  }

  DirectRoute make_direct() {
    DirectRoute route;
    route.input = make_tensor(device_resource_.ref(), description_);
    route.host_input = make_tensor(pinned_resource_.ref(), description_);
    route.host_output = make_tensor(pinned_resource_.ref(), description_);
    route.output = make_tensor(device_resource_.ref(), description_);
    route.cpu_queue =
        take_or_abort(cpu_context_.create_queue(), "direct CPU queue");
    route.cuda_queue =
        take_or_abort(cuda_context_.create_queue(), "direct CUDA queue");
    require_ok(route.cuda_queue.reserve_completions(1),
               "reserve direct CUDA completion");
    auto prepared = take_or_abort(
        hyle::compute::prepare(cpu_context_, Unary{UnaryKind::kNeg},
                               description_),
        "prepare direct Unary");
    route.unary =
        take_or_abort(prepared.create_invocation(), "direct Unary invocation");
    return route;
  }

  static void initialize(OwnedTensor &tensor) {
    std::vector<float> values(kElements);
    for (std::size_t index = 0; index < values.size(); ++index)
      values[index] = static_cast<float>(static_cast<int>(index % 257U) - 128) /
                      128.0F;
    require_ok(tensor.buffer.view().write(std::as_bytes(std::span(values))),
               "device input initialization");
  }

  static void validate(const OwnedTensor &tensor) {
    std::vector<float> values(kElements);
    require_ok(tensor.buffer.view().read(
                   std::as_writable_bytes(std::span(values))),
               "device output readback");
    for (std::size_t index = 0; index < values.size(); ++index) {
      const float expected =
          -static_cast<float>(static_cast<int>(index % 257U) - 128) / 128.0F;
      if (values[index] != expected)
        fail_setup("cross-domain output mismatch");
    }
  }

  TensorDesc description_{};
  Context cpu_context_{};
  Context cuda_context_{};
  hyle::compute::cuda::MemoryResource pinned_resource_{};
  hyle::compute::cuda::MemoryResource device_resource_{};
  std::optional<GraphRoute> graph_{};
  std::optional<DirectRoute> direct_{};
};

} // namespace

int main() {
  const auto affinity =
      hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  const bool peer_allocation_first =
      environment_present("HYLE_BENCH_PEER_ALLOCATION_FIRST");
  std::fprintf(
      stderr,
      "# HYLE_GRAPH_CUDA_PIPELINE_RUN execution_order=%s allocation_order=%s "
      "affinity_topology=%s caller_cpu=%s affinity_ok=%s run_nonce=%s\n",
      peer_first ? "peer-first" : "candidate-first",
      peer_allocation_first ? "peer-first" : "candidate-first",
      affinity.topology, affinity.cpu, affinity.ok,
      hyle::bench::perf::run_nonce());

  Fixture fixture(peer_allocation_first);
  nb::Bench bench;
  bench.title("graph CUDA-CPU-CUDA Unary f32 4 MiB")
      .relative(true)
      .minEpochTime(std::chrono::milliseconds(20));
  run_pair(
      bench, peer_first, "HYLE Graph CUDA-CPU-CUDA Unary 4 MiB",
      [&] {
        const auto ok = fixture.run_graph();
        nb::doNotOptimizeAway(ok);
      },
      "HYLE direct CUDA-CPU-CUDA Unary 4 MiB",
      [&] {
        const auto ok = fixture.run_direct();
        nb::doNotOptimizeAway(ok);
      });
}
