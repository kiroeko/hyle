// Same-service CPU Graph comparison: both arms own the complete public
// Graph/Invocation/Completion journey. Bare leaf-call pairs were retired in R7.
#include <nanobench.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/providers/native.h"
#include "hyle/compute/providers/onednn.h"
#include "hyle/graph/cpu.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/providers/onednn.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"

#include "../common/perf_process.h"
#include "executor.h"

namespace {
namespace nb = ankerl::nanobench;

using hyle::Status;
using hyle::compute::ConstTensorView;
using hyle::compute::Context;
using hyle::compute::DataType;
using hyle::compute::GatedActivation;
using hyle::compute::GatedActivationKind;
using hyle::compute::GatedInputKind;
using hyle::compute::MatMul;
using hyle::compute::ParallelExecutorRef;
using hyle::compute::Provider;
using hyle::compute::Queue;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::graph::CompileOptions;
using hyle::graph::ConstantBinding;
using hyle::graph::GraphBuilder;
using hyle::graph::Invocation;
using hyle::graph::ValueId;
using hyle::mem::Buffer;

constexpr std::size_t kAlignment = 64;

[[noreturn]] void fail(std::string_view operation,
                       std::string_view detail = {}) {
  std::fprintf(stderr, "graph oneDNN region benchmark %.*s failed",
               static_cast<int>(operation.size()), operation.data());
  if (!detail.empty()) {
    std::fprintf(stderr, ": %.*s", static_cast<int>(detail.size()),
                 detail.data());
  }
  std::fputc('\n', stderr);
  std::abort();
}

template <class T> T take(hyle::Result<T> result, std::string_view operation) {
  if (!result)
    fail(operation, result.error().message());
  return std::move(*result);
}

void require(Status status, std::string_view operation) {
  if (!status.ok())
    fail(operation, status.message());
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

enum class AaRoute { kOff, kSelected, kPeer };

AaRoute aa_route() {
#if defined(_WIN32)
  char *storage = nullptr;
  std::size_t size = 0;
  if (_dupenv_s(&storage, &size, "HYLE_BENCH_AA_ROUTE") != 0)
    fail("read A/A route");
  const std::string value = storage == nullptr ? "" : storage;
  std::free(storage);
#else
  const char *storage = std::getenv("HYLE_BENCH_AA_ROUTE");
  const std::string value = storage == nullptr ? "" : storage;
#endif
  if (value.empty())
    return AaRoute::kOff;
  if (value == "selected")
    return AaRoute::kSelected;
  if (value == "peer")
    return AaRoute::kPeer;
  fail("select A/A route", value);
}

TensorDesc matrix(std::size_t rows, std::size_t columns) {
  const std::array dimensions{static_cast<std::int64_t>(rows),
                              static_cast<std::int64_t>(columns)};
  return take(TensorDesc::dense(DataType::kFloat32, dimensions),
              "create matrix description");
}

struct OwnedTensor final {
  Buffer buffer{};
  TensorView view{};
};

OwnedTensor make_tensor(TensorDesc description) {
  OwnedTensor tensor;
  tensor.buffer = take(
      Buffer::create(hyle::mem::system_resource(),
                     take(description.required_bytes(), "query tensor bytes"),
                     kAlignment),
      "allocate tensor");
  tensor.view = take(TensorView::create(description, tensor.buffer.view()),
                     "create tensor view");
  return tensor;
}

void write_tensor(OwnedTensor &tensor, std::span<const float> values) {
  require(tensor.buffer.view().write(std::as_bytes(values)),
          "initialize tensor");
}

std::vector<float> read_tensor(const OwnedTensor &tensor,
                               std::size_t elements) {
  std::vector<float> values(elements);
  require(tensor.buffer.view().read(
              std::as_writable_bytes(std::span<float>(values))),
          "read tensor");
  return values;
}

struct Problem final {
  std::size_t rows = 0;
  std::size_t inner = 0;
  std::size_t hidden = 0;
  TensorDesc activation_desc{};
  TensorDesc gate_weight_desc{};
  TensorDesc up_weight_desc{};
  TensorDesc down_weight_desc{};
  TensorDesc projection_desc{};
  TensorDesc output_desc{};
  OwnedTensor activation{};
  OwnedTensor gate_weight{};
  OwnedTensor up_weight{};
  OwnedTensor down_weight{};

  Problem(std::size_t rows_value, std::size_t inner_value,
          std::size_t hidden_value)
      : rows(rows_value), inner(inner_value), hidden(hidden_value),
        activation_desc(matrix(rows, inner)),
        gate_weight_desc(matrix(inner, hidden)),
        up_weight_desc(matrix(inner, hidden)),
        down_weight_desc(matrix(hidden, inner)),
        projection_desc(matrix(rows, hidden)), output_desc(matrix(rows, inner)),
        activation(make_tensor(activation_desc)),
        gate_weight(make_tensor(gate_weight_desc)),
        up_weight(make_tensor(up_weight_desc)),
        down_weight(make_tensor(down_weight_desc)) {
    std::vector<float> activation_values(rows * inner);
    std::vector<float> gate_values(inner * hidden);
    std::vector<float> up_values(inner * hidden);
    std::vector<float> down_values(hidden * inner);
    for (std::size_t index = 0; index < activation_values.size(); ++index) {
      activation_values[index] =
          static_cast<float>(static_cast<int>(index % 23) - 11) * 0.015625F;
    }
    for (std::size_t index = 0; index < gate_values.size(); ++index) {
      gate_values[index] =
          static_cast<float>(static_cast<int>(index % 17) - 8) * 0.00390625F;
    }
    for (std::size_t index = 0; index < up_values.size(); ++index) {
      up_values[index] =
          static_cast<float>(static_cast<int>(index % 19) - 9) * 0.00390625F;
    }
    for (std::size_t index = 0; index < down_values.size(); ++index) {
      down_values[index] =
          static_cast<float>(static_cast<int>(index % 13) - 6) * 0.00390625F;
    }
    write_tensor(activation, activation_values);
    write_tensor(gate_weight, gate_values);
    write_tensor(up_weight, up_values);
    write_tensor(down_weight, down_values);
  }
};

struct LogicalMlp final {
  hyle::graph::Graph graph{};
  ValueId activation{};
  ValueId gate_weight{};
  ValueId up_weight{};
  ValueId down_weight{};
  ValueId output{};
};

LogicalMlp build_graph(const Problem &problem) {
  GraphBuilder builder = take(GraphBuilder::create(), "create graph builder");
  LogicalMlp logical;
  logical.activation =
      take(builder.input(problem.activation_desc), "add activation input");
  logical.gate_weight =
      take(builder.constant(problem.gate_weight_desc), "add gate constant");
  logical.up_weight =
      take(builder.constant(problem.up_weight_desc), "add up constant");
  logical.down_weight =
      take(builder.constant(problem.down_weight_desc), "add down constant");
  const ValueId gate =
      take(builder.add(MatMul{}, logical.activation, logical.gate_weight),
           "add gate projection");
  const ValueId up =
      take(builder.add(MatMul{}, logical.activation, logical.up_weight),
           "add up projection");
  const ValueId gated =
      take(builder.add(GatedActivation{.activation = GatedActivationKind::kSilu,
                                       .inputs = GatedInputKind::kSeparate},
                       gate, up),
           "add gated activation");
  logical.output = take(builder.add(MatMul{}, gated, logical.down_weight),
                        "add down projection");
  require(builder.publish(logical.output), "publish graph output");
  logical.graph = take(std::move(builder).finish(), "finish graph");
  return logical;
}

class GraphRoute {
public:
  explicit GraphRoute(Problem &problem, bool use_default = true,
                      ParallelExecutorRef executor = {})
      : output_(make_tensor(problem.output_desc)) {
    const LogicalMlp logical = build_graph(problem);
    context_ =
        use_default
            ? take(hyle::graph::cpu::create_context({.executor = executor}),
                   "create default Graph context")
            : take(Context::create(
                       hyle::compute::cpu::device(),
                       hyle::compute::native::provider({.executor = executor})),
                   "create explicit Native context");
    queue_ = take(context_.create_queue(), "create Graph queue");
    const std::array constants{
        ConstantBinding{logical.gate_weight,
                        problem.gate_weight.view.as_const()},
        ConstantBinding{logical.up_weight, problem.up_weight.view.as_const()},
        ConstantBinding{logical.down_weight,
                        problem.down_weight.view.as_const()},
    };
    auto compiled =
        take(hyle::graph::compile(
                 logical.graph, context_,
                 CompileOptions{
                     .constants = constants,
                     .provider_replay =
                         hyle::graph::ProviderReplayPolicy::kRequirePrepared}),
             "compile Graph region");
    if (compiled.report.region_count() != (use_default ? 1U : 0U) ||
        compiled.report.provider().id !=
            (use_default ? hyle::graph::onednn::kProviderId
                         : hyle::compute::native::kProviderId)) {
      fail("select expected default/Native Graph plan");
    }
    invocation_ = take(compiled.executable.create_invocation(),
                       "create Graph invocation");
    require(invocation_.bind(logical.activation,
                             problem.activation.view.as_const()),
            "bind Graph activation");
    require(invocation_.bind(logical.output, output_.view),
            "bind Graph output");
    const std::array queues{hyle::graph::QueueBinding{
        .lane = hyle::graph::ExecutionLaneId{0}, .queue = &queue_}};
    require(invocation_.prepare_replay(queues), "prepare fixed Graph replay");
  }

  std::uint64_t run() noexcept {
    auto completion = invocation_.submit(queue_);
    if (!completion)
      return 0;
    const Status terminal = completion->wait();
    if (!terminal.ok())
      return 0;
    nb::doNotOptimizeAway(output_.buffer.native_handle());
    return 1;
  }

  std::vector<float> output(std::size_t count) const {
    return read_tensor(output_, count);
  }

private:
  Context context_{};
  Queue queue_{};
  OwnedTensor output_{};
  Invocation invocation_{};
};

class DefaultRoute final : public GraphRoute {
public:
  explicit DefaultRoute(Problem &problem, ParallelExecutorRef executor = {})
      : GraphRoute(problem, true, executor) {}
};

class NativeRoute final : public GraphRoute {
public:
  explicit NativeRoute(Problem &problem, ParallelExecutorRef executor = {})
      : GraphRoute(problem, false, executor) {}
};

template <class Peer, class Candidate = DefaultRoute> class PairFixture final {
public:
  PairFixture(std::size_t rows, std::size_t inner, std::size_t hidden,
              bool peer_allocation_first, ParallelExecutorRef executor = {})
      : problem_(rows, inner, hidden) {
    if (peer_allocation_first) {
      peer_ = std::make_unique<Peer>(problem_, executor);
      graph_ = std::make_unique<Candidate>(problem_, executor);
    } else {
      graph_ = std::make_unique<Candidate>(problem_, executor);
      peer_ = std::make_unique<Peer>(problem_, executor);
    }
    if (graph_->run() == 0 || peer_->run() == 0)
      fail("warm routes");
    const auto candidate = graph_->output(rows * inner);
    const auto peer = peer_->output(rows * inner);
    if (candidate.size() != peer.size())
      fail("compare output sizes");
    for (std::size_t index = 0; index < candidate.size(); ++index) {
      const float tolerance = std::abs(peer[index]) * 8.0e-5F + 8.0e-6F;
      if (std::abs(candidate[index] - peer[index]) > tolerance)
        fail("compare route outputs");
    }
  }

  Candidate &graph() noexcept { return *graph_; }
  Peer &peer() noexcept { return *peer_; }

private:
  Problem problem_;
  std::unique_ptr<Candidate> graph_;
  std::unique_ptr<Peer> peer_;
};

template <class Candidate, class Peer>
void run_pair(nb::Bench &bench, bool peer_first,
              std::string_view candidate_name, Candidate &&candidate,
              std::string_view peer_name, Peer &&peer) {
  if (peer_first) {
    (void)bench.compare(peer_name, std::forward<Peer>(peer), candidate_name,
                        std::forward<Candidate>(candidate));
  } else {
    (void)bench.compare(candidate_name, std::forward<Candidate>(candidate),
                        peer_name, std::forward<Peer>(peer));
  }
}

template <class Peer, class Candidate>
void measure(PairFixture<Peer, Candidate> &fixture, bool peer_first,
             std::string_view title, std::string_view candidate_name,
             std::string_view peer_name) {
  if (environment_present("HYLE_BENCH_VERIFY_ONLY"))
    return;
  nb::Bench bench;
  bench.title(std::string(title))
      .relative(true)
      .minEpochTime(std::chrono::milliseconds(8));
  run_pair(
      bench, peer_first, candidate_name,
      [&] {
        const auto ok = fixture.graph().run();
        nb::doNotOptimizeAway(ok);
      },
      peer_name,
      [&] {
        const auto ok = fixture.peer().run();
        nb::doNotOptimizeAway(ok);
      });
}

// A/A changes only the two route types. Allocation order, fixture lifetimes,
// completion boundaries and the normal measured compare call stay identical.
template <AaRoute Mode, class Peer>
using ModeFixture = PairFixture<
    std::conditional_t<Mode == AaRoute::kSelected, DefaultRoute, Peer>,
    std::conditional_t<Mode == AaRoute::kPeer, Peer, DefaultRoute>>;

template <AaRoute Mode>
void run_shape(std::size_t rows, std::size_t inner, std::size_t hidden,
               bool peer_first, bool peer_allocation_first,
               std::string_view shape_name, ParallelExecutorRef executor = {}) {
  ModeFixture<Mode, NativeRoute> native(rows, inner, hidden, peer_allocation_first,
                                  executor);
  const std::string graph_name =
      "HYLE Graph oneDNN region " + std::string(shape_name);
  const std::string native_title =
      "graph CPU default versus Native " + std::string(shape_name);
  const std::string native_name =
      "HYLE Graph explicit Native " + std::string(shape_name);
  measure(native, peer_first, native_title, graph_name, native_name);
}

template <AaRoute Mode>
void run_shapes(bool peer_first, bool peer_allocation_first,
                hyle::bench::cpu_route::Executor &executor) {
  run_shape<Mode>(1, 256, 512, peer_first, peer_allocation_first,
            "decode f32 M=1 K=256 H=512");
  run_shape<Mode>(16, 512, 1024, peer_first, peer_allocation_first,
            "short-prefill f32 M=16 K=512 H=1024");
  run_shape<Mode>(17, 129, 257, peer_first, peer_allocation_first,
            "holdout f32 M=17 K=129 H=257");
  if (executor.enabled()) {
    executor.start();
    run_shape<Mode>(17, 129, 257, peer_first, peer_allocation_first,
              "two-worker f32 M=17 K=129 H=257", executor.ref());
  }
}

} // namespace

int main(int argc, char **argv) {
  const auto qos =
      hyle::bench::perf::configure_thread_qos("HYLE_BENCH_APPLE_QOS");
  const auto affinity =
      hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  const bool peer_allocation_first =
      environment_present("HYLE_BENCH_PEER_ALLOCATION_FIRST");
  hyle::bench::cpu_route::Executor executor;

  const auto mode = aa_route();
  if (mode == AaRoute::kOff && environment_present("HYLE_PERF_RUN_NONCE")) {
    if (argc < 1)
      fail("locate workload A/A result");
    const auto status_path = std::filesystem::path(argv[0]).parent_path() /
                             "hyle-graph-workload-aa.status";
    std::ifstream status(status_path);
    std::string verdict;
    if (!(status >> verdict) || verdict != "PASS")
      fail("workload A/A preflight must pass before product sampling");
  }
  switch (mode) {
  case AaRoute::kOff:
    run_shapes<AaRoute::kOff>(peer_first, peer_allocation_first, executor);
    break;
  case AaRoute::kSelected:
    run_shapes<AaRoute::kSelected>(peer_first, peer_allocation_first, executor);
    break;
  case AaRoute::kPeer:
    run_shapes<AaRoute::kPeer>(peer_first, peer_allocation_first, executor);
    break;
  }
  std::fprintf(stderr,
               "# HYLE_GRAPH_CPU_AA aa_route=%s paired_mode=nanobench-compare\n",
               mode == AaRoute::kOff ? "off"
               : mode == AaRoute::kSelected ? "selected" : "peer");
  executor.print_telemetry(peer_first, peer_allocation_first, qos, affinity);
}
