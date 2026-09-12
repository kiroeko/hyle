// Same-contract steady execution of a three-operation transformer fragment.
// Graph and the direct sequence use the same Context, Queue, public
// PreparedOperation contract, tensor descriptions, arithmetic route and caller
// inputs. Construction, prepare, binding and allocation are outside timing.
#include <nanobench.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hyle/compute/device.h"
#include "hyle/compute/operation.h"
#include "hyle/graph/executable.h"
#include "hyle/mem/buffer.h"

#include "../common/perf_process.h"

namespace {
namespace nb = ankerl::nanobench;
using hyle::Status;
using hyle::compute::ConstTensorView;
using hyle::compute::Context;
using hyle::compute::DataType;
using hyle::compute::MatMul;
using hyle::compute::OperationInvocation;
using hyle::compute::Queue;
using hyle::compute::RmsNorm;
using hyle::compute::SiluAndMul;
using hyle::compute::TensorDesc;
using hyle::compute::TensorView;
using hyle::graph::GraphBuilder;
using hyle::graph::Invocation;

[[noreturn]] void fail_setup(std::string_view message) {
  std::fprintf(stderr, "graph multi-op benchmark setup failed: %.*s\n",
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

TensorDesc matrix(std::size_t rows, std::size_t columns) {
  const std::array dimensions{static_cast<std::int64_t>(rows),
                              static_cast<std::int64_t>(columns)};
  return take_or_abort(TensorDesc::dense(DataType::kFloat32, dimensions),
                       "matrix description");
}

TensorDesc row(std::size_t elements) {
  const std::array dimensions{static_cast<std::int64_t>(elements)};
  return take_or_abort(TensorDesc::dense(DataType::kFloat32, dimensions),
                       "row description");
}

struct OwnedTensor final {
  hyle::mem::Buffer buffer{};
  TensorView view{};
};

OwnedTensor make_tensor(TensorDesc description) {
  OwnedTensor tensor;
  const std::size_t bytes =
      take_or_abort(description.required_bytes(), "tensor byte size");
  tensor.buffer = take_or_abort(
      hyle::mem::Buffer::create(hyle::mem::system_resource(), bytes, 64),
      "tensor allocation");
  tensor.view = take_or_abort(
      TensorView::create(description, tensor.buffer.view()), "tensor view");
  return tensor;
}

void write_tensor(OwnedTensor &tensor, std::span<const float> values) {
  require_ok(tensor.buffer.view().write(std::as_bytes(values)),
             "tensor initialization");
}

std::vector<float> read_tensor(const OwnedTensor &tensor,
                               std::size_t elements) {
  std::vector<float> values(elements);
  require_ok(tensor.buffer.view().read(std::as_writable_bytes(
                 std::span<float>(values.data(), values.size()))),
             "tensor readback");
  return values;
}

struct Descriptions final {
  TensorDesc activation{};
  TensorDesc norm_weight{};
  TensorDesc stored_weight{};
  TensorDesc logical_weight{};
  TensorDesc normalized{};
  TensorDesc projected{};
  TensorDesc output{};
};

Descriptions make_descriptions(std::size_t rows, std::size_t hidden,
                               std::size_t ffn) {
  Descriptions result;
  result.activation = matrix(rows, hidden);
  result.norm_weight = row(hidden);
  result.stored_weight = matrix(2 * ffn, hidden);
  result.logical_weight = take_or_abort(result.stored_weight.transposed(),
                                        "logical weight transpose");
  result.normalized = result.activation;
  result.projected = matrix(rows, 2 * ffn);
  result.output = matrix(rows, ffn);
  return result;
}

struct DirectRoute final {
  OwnedTensor normalized{};
  OwnedTensor projected{};
  OwnedTensor output{};
  OperationInvocation rms{};
  OperationInvocation matmul{};
  OperationInvocation swiglu{};
};

struct GraphRoute final {
  OwnedTensor output{};
  Invocation invocation{};
};

class Fixture final {
public:
  Fixture(std::size_t rows, std::size_t hidden, std::size_t ffn,
          bool peer_allocation_first)
      : rows_(rows), hidden_(hidden), ffn_(ffn),
        descriptions_(make_descriptions(rows, hidden, ffn)),
        context_(take_or_abort(Context::create(hyle::compute::Device::cpu()),
                               "CPU context")),
        queue_(take_or_abort(context_.create_queue(), "CPU queue")),
        input_(make_tensor(descriptions_.activation)),
        norm_weight_(make_tensor(descriptions_.norm_weight)),
        matrix_weight_(make_tensor(descriptions_.logical_weight)) {
    initialize_inputs();
    if (peer_allocation_first) {
      direct_.emplace(make_direct());
      graph_.emplace(make_graph());
    } else {
      graph_.emplace(make_graph());
      direct_.emplace(make_direct());
    }
    if (run_graph() == 0 || run_direct() == 0)
      fail_setup("warm execution");
    validate_equal_outputs();
  }

  std::uint64_t run_graph() noexcept {
    auto completion = graph_->invocation.submit(queue_);
    if (!completion)
      return 0;
    return completion->wait().ok() ? 1U : 0U;
  }

  std::uint64_t run_direct() noexcept {
    auto rms = direct_->rms.submit(queue_, input_.view.as_const(),
                                   norm_weight_.view.as_const(),
                                   direct_->normalized.view);
    if (!rms || !rms->wait().ok())
      return 0;
    auto matmul = direct_->matmul.submit(
        queue_, direct_->normalized.view.as_const(),
        matrix_weight_.view.as_const(), direct_->projected.view);
    if (!matmul || !matmul->wait().ok())
      return 0;
    auto swiglu = direct_->swiglu.submit(
        queue_, direct_->projected.view.as_const(), direct_->output.view);
    if (!swiglu || !swiglu->wait().ok())
      return 0;
    return 1U;
  }

private:
  void initialize_inputs() {
    std::vector<float> input_values(rows_ * hidden_);
    for (std::size_t index = 0; index < input_values.size(); ++index) {
      input_values[index] =
          static_cast<float>(static_cast<int>(index % 23U) - 11) * 0.03125F;
    }
    std::vector<float> norm_values(hidden_);
    for (std::size_t index = 0; index < norm_values.size(); ++index) {
      norm_values[index] = 0.75F + static_cast<float>(index % 7U) * 0.03125F;
    }
    std::vector<float> weight_values(2 * ffn_ * hidden_);
    for (std::size_t index = 0; index < weight_values.size(); ++index) {
      weight_values[index] =
          static_cast<float>(static_cast<int>(index % 17U) - 8) * 0.001953125F;
    }
    write_tensor(input_, input_values);
    write_tensor(norm_weight_, norm_values);
    write_tensor(matrix_weight_, weight_values);
  }

  DirectRoute make_direct() {
    DirectRoute route;
    route.normalized = make_tensor(descriptions_.normalized);
    route.projected = make_tensor(descriptions_.projected);
    route.output = make_tensor(descriptions_.output);

    auto rms = take_or_abort(
        hyle::compute::prepare(context_, RmsNorm{}, descriptions_.activation,
                               descriptions_.norm_weight,
                               descriptions_.normalized),
        "direct RMSNorm prepare");
    auto matmul = take_or_abort(
        hyle::compute::prepare(context_, MatMul{}, descriptions_.normalized,
                               descriptions_.logical_weight,
                               descriptions_.projected),
        "direct MatMul prepare");
    auto swiglu = take_or_abort(
        hyle::compute::prepare(context_, SiluAndMul{},
                               descriptions_.projected, descriptions_.output),
        "direct SiluAndMul prepare");
    route.rms = take_or_abort(rms.create_invocation(),
                              "direct RMSNorm invocation");
    route.matmul = take_or_abort(matmul.create_invocation(),
                                 "direct MatMul invocation");
    route.swiglu = take_or_abort(swiglu.create_invocation(),
                                 "direct SiluAndMul invocation");
    return route;
  }

  GraphRoute make_graph() {
    GraphBuilder builder = take_or_abort(GraphBuilder::create(),
                                         "graph builder");
    const auto input = take_or_abort(builder.input(descriptions_.activation),
                                     "graph activation input");
    const auto norm_weight = take_or_abort(
        builder.input(descriptions_.norm_weight), "graph norm input");
    const auto matrix_weight = take_or_abort(
        builder.input(descriptions_.logical_weight), "graph matrix input");
    const auto normalized = take_or_abort(
        builder.add(RmsNorm{}, input, norm_weight), "graph RMSNorm node");
    const auto projected = take_or_abort(
        builder.add(MatMul{}, normalized, matrix_weight), "graph MatMul node");
    const auto output = take_or_abort(
        builder.add(SiluAndMul{}, projected), "graph SiluAndMul node");
    require_ok(builder.output(output), "graph published output");
    auto graph = take_or_abort(std::move(builder).finish(), "logical graph");
    auto compiled = take_or_abort(hyle::graph::compile(graph, context_),
                                  "graph compile");
    if (compiled.report.node_count != 3 ||
        compiled.report.provider != hyle::compute::ProviderKind::kHyleNative ||
        compiled.report.copy_bytes != 0 ||
        compiled.report.synchronization_count != 0) {
      fail_setup("unexpected compiled route");
    }

    GraphRoute route;
    route.output = make_tensor(descriptions_.output);
    route.invocation = take_or_abort(compiled.executable.create_invocation(),
                                     "graph invocation");
    require_ok(route.invocation.bind(input, input_.view.as_const()),
               "graph activation binding");
    require_ok(route.invocation.bind(norm_weight,
                                     norm_weight_.view.as_const()),
               "graph norm binding");
    require_ok(route.invocation.bind(matrix_weight,
                                     matrix_weight_.view.as_const()),
               "graph matrix binding");
    require_ok(route.invocation.bind(output, route.output.view),
               "graph output binding");
    return route;
  }

  void validate_equal_outputs() const {
    const auto graph_values = read_tensor(graph_->output, rows_ * ffn_);
    const auto direct_values = read_tensor(direct_->output, rows_ * ffn_);
    for (std::size_t index = 0; index < graph_values.size(); ++index) {
      const float tolerance = std::abs(direct_values[index]) * 1.0e-6F +
                              1.0e-7F;
      if (std::abs(graph_values[index] - direct_values[index]) > tolerance)
        fail_setup("Graph and direct outputs differ");
    }
  }

  std::size_t rows_ = 0;
  std::size_t hidden_ = 0;
  std::size_t ffn_ = 0;
  Descriptions descriptions_{};
  Context context_{};
  Queue queue_{};
  OwnedTensor input_{};
  OwnedTensor norm_weight_{};
  OwnedTensor matrix_weight_{};
  std::optional<DirectRoute> direct_{};
  std::optional<GraphRoute> graph_{};
};

void run_workload(Fixture &fixture, bool peer_first, std::string_view title,
                  std::string_view candidate_name,
                  std::string_view peer_name) {
  nb::Bench bench;
  bench.title(std::string(title))
      .relative(true)
      .minEpochTime(std::chrono::milliseconds(8));
  run_pair(
      bench, peer_first, candidate_name,
      [&] {
        const auto ok = fixture.run_graph();
        nb::doNotOptimizeAway(ok);
      },
      peer_name, [&] {
        const auto ok = fixture.run_direct();
        nb::doNotOptimizeAway(ok);
      });
}

} // namespace

int main() {
  const auto qos = hyle::bench::perf::configure_thread_qos(
      "HYLE_BENCH_APPLE_QOS");
  const auto affinity = hyle::bench::perf::configure_thread_affinity(
      "HYLE_PERF_CALLER_CPU");
  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  const bool peer_allocation_first =
      environment_present("HYLE_BENCH_PEER_ALLOCATION_FIRST");
  std::fprintf(
      stderr,
      "# HYLE_GRAPH_MULTI_OP_RUN execution_order=%s allocation_order=%s "
      "apple_qos=%s qos_ok=%s affinity_topology=%s caller_cpu=%s "
      "affinity_ok=%s run_nonce=%s\n",
      peer_first ? "peer-first" : "candidate-first",
      peer_allocation_first ? "peer-first" : "candidate-first", qos.name,
      qos.ok, affinity.topology, affinity.cpu, affinity.ok,
      hyle::bench::perf::run_nonce());

  Fixture decode(1, 256, 512, peer_allocation_first);
  run_workload(decode, peer_first,
               "graph multi-op decode f32 rows=1 hidden=256 ffn=512",
               "HYLE Graph decode 1x256x512",
               "HYLE direct decode 1x256x512");

  Fixture small_batch(4, 64, 128, peer_allocation_first);
  run_workload(small_batch, peer_first,
               "graph multi-op small batch f32 rows=4 hidden=64 ffn=128",
               "HYLE Graph small batch 4x64x128",
               "HYLE direct small batch 4x64x128");
}
