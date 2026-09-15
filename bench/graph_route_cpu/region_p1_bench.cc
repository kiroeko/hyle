// Architecture qualification for the first real Graph region.  The first
// pair compares the public Graph journey with the identical fused BRGEMM/SIMD
// leaf sequence.  The second pair compares it with the natural unfused
// sequence, including each operation's independent mapping boundary.  No
// benchmark-only product operation or provider is introduced.
#include <nanobench.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "brgemm_internal.h"
#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/providers/native.h"
#include "hyle/compute/providers/onednn.h"
#include "hyle/graph/cpu.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/providers/onednn.h"
#include "hyle/mem/allocator.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"
#include "hyle/simd/math.h"

#include "../common/perf_process.h"
#include "executor.h"

namespace {
namespace nb = ankerl::nanobench;
namespace brgemm = hyle::compute::onednn::internal;

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
using hyle::mem::HostAccess;
using hyle::mem::HostMapping;

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

class NativeRoute final : public GraphRoute {
public:
  explicit NativeRoute(Problem &problem, ParallelExecutorRef executor = {})
      : GraphRoute(problem, false, executor) {}
};

class PlanOwner final {
public:
  PlanOwner(brgemm::BrgemmShape shape, std::size_t task_count) {
    require(brgemm::create_brgemm_plan(shape, task_count,
                                       hyle::mem::system_allocator(), &plan_),
            "create direct BRGEMM plan");
  }
  PlanOwner(const PlanOwner &) = delete;
  PlanOwner &operator=(const PlanOwner &) = delete;
  ~PlanOwner() {
    brgemm::destroy_brgemm_plan(plan_, hyle::mem::system_allocator());
  }
  const brgemm::BrgemmPlan *get() const noexcept { return plan_; }

private:
  brgemm::BrgemmPlan *plan_ = nullptr;
};

class DirectBase {
public:
  explicit DirectBase(Problem &problem, ParallelExecutorRef executor = {})
      : problem_(problem), output_(make_tensor(problem.output_desc)),
        gate_(take(Buffer::create(hyle::mem::system_resource(),
                                  problem.rows * problem.hidden * sizeof(float),
                                  kAlignment),
                   "allocate direct gate workspace")),
        up_(take(Buffer::create(hyle::mem::system_resource(),
                                problem.rows * problem.hidden * sizeof(float),
                                kAlignment),
                 "allocate direct up workspace")),
        gate_up_plan_({problem.rows, problem.inner, problem.hidden},
                      brgemm::recommended_brgemm_task_count(
                          problem.hidden, executor.concurrency())),
        down_plan_({problem.rows, problem.hidden, problem.inner},
                   brgemm::recommended_brgemm_task_count(
                       problem.inner, executor.concurrency())),
        executor_(executor) {
    const std::size_t scratch_size =
        (std::max)(brgemm::brgemm_scratchpad_size(gate_up_plan_.get()),
                   brgemm::brgemm_scratchpad_size(down_plan_.get()));
    scratch_ = take(
        Buffer::create(hyle::mem::system_resource(), scratch_size, kAlignment),
        "allocate direct scratch");
    scratch_mapping_ = take(scratch_.view().map_host(HostAccess::kReadWrite),
                            "map direct scratch");
    auto bytes =
        take(scratch_mapping_.writable_bytes(), "resolve direct scratch");
    scratch_data_ = bytes.data();
  }

  DirectBase(const DirectBase &) = delete;
  DirectBase &operator=(const DirectBase &) = delete;

  std::vector<float> output(std::size_t count) const {
    return read_tensor(output_, count);
  }

protected:
  static void activate_tile(void *context_value, std::size_t rows,
                            std::size_t leading_columns,
                            brgemm::BrgemmTaskRange range) noexcept {
    auto &context = *static_cast<std::array<float *, 2> *>(context_value);
    for (std::size_t row = 0; row < rows; ++row) {
      const std::size_t offset = row * leading_columns + range.first_column;
      hyle::simd::swiglu_f32_batch(context[0] + offset, context[1] + offset,
                                   context[0] + offset, range.column_count);
    }
  }

  Problem &problem_;
  OwnedTensor output_{};
  Buffer gate_{};
  Buffer up_{};
  Buffer scratch_{};
  HostMapping scratch_mapping_{};
  std::byte *scratch_data_ = nullptr;
  PlanOwner gate_up_plan_;
  PlanOwner down_plan_;
  ParallelExecutorRef executor_{};
};

class DirectFusedRoute final : public DirectBase {
public:
  explicit DirectFusedRoute(Problem &problem, ParallelExecutorRef executor = {})
      : DirectBase(problem, executor) {
    gate_mapping_ = take(gate_.view().map_host(HostAccess::kReadWrite),
                         "map fused gate workspace");
    up_mapping_ = take(up_.view().map_host(HostAccess::kReadWrite),
                       "map fused up workspace");
    gate_data_ = reinterpret_cast<float *>(
        take(gate_mapping_.writable_bytes(), "resolve fused gate").data());
    up_data_ = reinterpret_cast<float *>(
        take(up_mapping_.writable_bytes(), "resolve fused up").data());
  }

  std::uint64_t run() noexcept {
    auto activation = problem_.activation.buffer.view().map_host();
    auto gate_weight = problem_.gate_weight.buffer.view().map_host();
    auto up_weight = problem_.up_weight.buffer.view().map_host();
    auto down_weight = problem_.down_weight.buffer.view().map_host();
    auto output = output_.buffer.view().map_host(HostAccess::kWrite);
    if (!activation || !gate_weight || !up_weight || !down_weight || !output)
      fail("map fused route boundaries");
    auto output_bytes = output->writable_bytes();
    if (!output_bytes)
      fail("resolve fused route output", output_bytes.error().message());
    const auto *activation_data =
        reinterpret_cast<const float *>(activation->bytes().data());
    const auto *gate_weight_data =
        reinterpret_cast<const float *>(gate_weight->bytes().data());
    const auto *up_weight_data =
        reinterpret_cast<const float *>(up_weight->bytes().data());
    const auto *down_weight_data =
        reinterpret_cast<const float *>(down_weight->bytes().data());
    auto *output_data = reinterpret_cast<float *>(output_bytes->data());
    std::array activation_context{gate_data_, up_data_};
    require(brgemm::execute_brgemm_pair(
                gate_up_plan_.get(), executor_, activation_data,
                gate_weight_data, up_weight_data, gate_data_, up_data_,
                scratch_data_, &activate_tile, &activation_context),
            "execute fused gate/up");
    require(brgemm::execute_brgemm(down_plan_.get(), executor_, gate_data_,
                                   down_weight_data, output_data,
                                   scratch_data_),
            "execute fused down");
    require(output->close(), "close fused output mapping");
    require(down_weight->close(), "close fused down-weight mapping");
    require(up_weight->close(), "close fused up-weight mapping");
    require(gate_weight->close(), "close fused gate-weight mapping");
    require(activation->close(), "close fused activation mapping");
    nb::doNotOptimizeAway(output_.buffer.native_handle());
    return 1;
  }

private:
  HostMapping gate_mapping_{};
  HostMapping up_mapping_{};
  float *gate_data_ = nullptr;
  float *up_data_ = nullptr;
};

class DirectSequentialRoute final : public DirectBase {
public:
  explicit DirectSequentialRoute(Problem &problem,
                                 ParallelExecutorRef executor = {})
      : DirectBase(problem, executor) {}

  std::uint64_t run() noexcept {
    project(problem_.gate_weight, gate_, gate_up_plan_.get(),
            "execute sequential gate");
    project(problem_.up_weight, up_, gate_up_plan_.get(),
            "execute sequential up");
    activate();
    project_down();
    nb::doNotOptimizeAway(output_.buffer.native_handle());
    return 1;
  }

private:
  void project(OwnedTensor &weight, Buffer &destination,
               const brgemm::BrgemmPlan *plan, std::string_view operation) {
    auto activation = problem_.activation.buffer.view().map_host();
    auto weight_mapping = weight.buffer.view().map_host();
    auto destination_mapping = destination.view().map_host(HostAccess::kWrite);
    if (!activation || !weight_mapping || !destination_mapping)
      fail("map sequential projection");
    auto destination_bytes = destination_mapping->writable_bytes();
    if (!destination_bytes)
      fail("resolve sequential projection output");
    require(brgemm::execute_brgemm(
                plan, executor_,
                reinterpret_cast<const float *>(activation->bytes().data()),
                reinterpret_cast<const float *>(weight_mapping->bytes().data()),
                reinterpret_cast<float *>(destination_bytes->data()),
                scratch_data_),
            operation);
    require(destination_mapping->close(), "close projection output mapping");
    require(weight_mapping->close(), "close projection weight mapping");
    require(activation->close(), "close projection input mapping");
  }

  void activate() {
    auto gate_mapping = gate_.view().map_host(HostAccess::kReadWrite);
    auto up_mapping = up_.view().map_host();
    if (!gate_mapping || !up_mapping)
      fail("map sequential activation");
    auto gate_bytes = gate_mapping->writable_bytes();
    if (!gate_bytes)
      fail("resolve sequential activation output");
    auto *gate_data = reinterpret_cast<float *>(gate_bytes->data());
    const auto *up_data =
        reinterpret_cast<const float *>(up_mapping->bytes().data());
    for (std::size_t row = 0; row < problem_.rows; ++row) {
      const std::size_t offset = row * problem_.hidden;
      hyle::simd::swiglu_f32_batch(gate_data + offset, up_data + offset,
                                   gate_data + offset, problem_.hidden);
    }
    require(up_mapping->close(), "close activation up mapping");
    require(gate_mapping->close(), "close activation gate mapping");
  }

  void project_down() {
    auto gate_mapping = gate_.view().map_host();
    auto weight_mapping = problem_.down_weight.buffer.view().map_host();
    auto output_mapping = output_.buffer.view().map_host(HostAccess::kWrite);
    if (!gate_mapping || !weight_mapping || !output_mapping)
      fail("map sequential down projection");
    auto output_bytes = output_mapping->writable_bytes();
    if (!output_bytes)
      fail("resolve sequential down output");
    require(brgemm::execute_brgemm(
                down_plan_.get(), executor_,
                reinterpret_cast<const float *>(gate_mapping->bytes().data()),
                reinterpret_cast<const float *>(weight_mapping->bytes().data()),
                reinterpret_cast<float *>(output_bytes->data()), scratch_data_),
            "execute sequential down");
    require(output_mapping->close(), "close sequential output mapping");
    require(weight_mapping->close(), "close sequential down-weight mapping");
    require(gate_mapping->close(), "close sequential gate mapping");
  }
};

template <class Peer> class PairFixture final {
public:
  PairFixture(std::size_t rows, std::size_t inner, std::size_t hidden,
              bool peer_allocation_first, ParallelExecutorRef executor = {})
      : problem_(rows, inner, hidden) {
    if (peer_allocation_first) {
      peer_ = std::make_unique<Peer>(problem_, executor);
      graph_ = std::make_unique<GraphRoute>(problem_, true, executor);
    } else {
      graph_ = std::make_unique<GraphRoute>(problem_, true, executor);
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

  GraphRoute &graph() noexcept { return *graph_; }
  Peer &peer() noexcept { return *peer_; }

private:
  Problem problem_;
  std::unique_ptr<GraphRoute> graph_;
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

template <class Peer>
void measure(PairFixture<Peer> &fixture, bool peer_first,
             std::string_view title, std::string_view candidate_name,
             std::string_view peer_name) {
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

void run_shape(std::size_t rows, std::size_t inner, std::size_t hidden,
               bool peer_first, bool peer_allocation_first,
               std::string_view shape_name, ParallelExecutorRef executor = {}) {
  PairFixture<DirectFusedRoute> tax(rows, inner, hidden, peer_allocation_first,
                                    executor);
  const std::string tax_title =
      "graph oneDNN MLP abstraction " + std::string(shape_name);
  const std::string graph_name =
      "HYLE Graph oneDNN region " + std::string(shape_name);
  const std::string direct_name =
      "direct fused oneDNN region " + std::string(shape_name);
  measure(tax, peer_first, tax_title, graph_name, direct_name);

  PairFixture<DirectSequentialRoute> benefit(rows, inner, hidden,
                                             peer_allocation_first, executor);
  const std::string benefit_title =
      "graph oneDNN MLP region benefit " + std::string(shape_name);
  const std::string sequential_name =
      "direct sequential oneDNN operations " + std::string(shape_name);
  measure(benefit, peer_first, benefit_title, graph_name, sequential_name);

  PairFixture<NativeRoute> native(rows, inner, hidden, peer_allocation_first,
                                  executor);
  const std::string native_title =
      "graph CPU default versus Native " + std::string(shape_name);
  const std::string native_name =
      "HYLE Graph explicit Native " + std::string(shape_name);
  measure(native, peer_first, native_title, graph_name, native_name);
}

} // namespace

int main() {
  const auto qos =
      hyle::bench::perf::configure_thread_qos("HYLE_BENCH_APPLE_QOS");
  const auto affinity =
      hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  const bool peer_allocation_first =
      environment_present("HYLE_BENCH_PEER_ALLOCATION_FIRST");
  hyle::bench::cpu_route::Executor executor;

  run_shape(1, 256, 512, peer_first, peer_allocation_first,
            "decode f32 M=1 K=256 H=512");
  run_shape(16, 512, 1024, peer_first, peer_allocation_first,
            "short-prefill f32 M=16 K=512 H=1024");
  run_shape(17, 129, 257, peer_first, peer_allocation_first,
            "holdout f32 M=17 K=129 H=257");
  if (executor.enabled()) {
    executor.start();
    run_shape(17, 129, 257, peer_first, peer_allocation_first,
              "two-worker f32 M=17 K=129 H=257", executor.ref());
  }
  executor.print_telemetry(peer_first, peer_allocation_first, qos, affinity);
}
