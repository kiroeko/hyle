// One-op abstraction-cost ladder over one normal inference row. The direct,
// Compute and Graph routes share the exact Mem storage, SIMD arithmetic,
// precision, queue/completion boundary and warmed lifetime. Only the safety
// and composition contracts added by each public layer differ.
#include <nanobench.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "hyle/compute/device.h"
#include "hyle/compute/operation.h"
#include "hyle/graph/executable.h"
#include "hyle/mem/buffer.h"
#include "hyle/simd/math.h"

#include "../common/perf_process.h"

namespace {
namespace nb = ankerl::nanobench;

constexpr std::size_t kElements = 4096;
constexpr float kEpsilon = 1.0e-5F;

[[noreturn]] void fail_setup(std::string_view message) {
  std::fprintf(stderr, "graph one-op benchmark setup failed: %.*s\n",
               static_cast<int>(message.size()), message.data());
  std::abort();
}

template <class T>
T take_or_abort(hyle::Result<T> result, std::string_view message) {
  if (!result) fail_setup(message);
  return std::move(*result);
}

void require_ok(hyle::Status status, std::string_view message) {
  if (!status.ok()) fail_setup(message);
}

bool environment_present(const char* name) {
#if defined(_WIN32)
  char* value = nullptr;
  std::size_t size = 0;
  const bool present = _dupenv_s(&value, &size, name) == 0 && value != nullptr;
  std::free(value);
  return present;
#else
  return std::getenv(name) != nullptr;
#endif
}

template <class Candidate, class Peer>
void run_pair(nb::Bench& bench, bool peer_first,
              std::string_view candidate_name, Candidate&& candidate,
              std::string_view peer_name, Peer&& peer) {
  if (peer_first) {
    bench.run(peer_name, std::forward<Peer>(peer));
    bench.run(candidate_name, std::forward<Candidate>(candidate));
  } else {
    bench.run(candidate_name, std::forward<Candidate>(candidate));
    bench.run(peer_name, std::forward<Peer>(peer));
  }
}

hyle::Status run_direct(hyle::mem::ConstBufferView input,
                        hyle::mem::BufferView output) noexcept {
  auto input_mapping = input.map_host();
  if (!input_mapping) return std::move(input_mapping).error();
  auto output_mapping = output.map_host(hyle::mem::HostAccess::kWrite);
  if (!output_mapping) {
    input_mapping->close().IgnoreError();
    return std::move(output_mapping).error();
  }
  auto writable = output_mapping->writable_bytes();
  if (!writable) {
    output_mapping->close().IgnoreError();
    input_mapping->close().IgnoreError();
    return std::move(writable).error();
  }

  const auto* input_data =
      reinterpret_cast<const float*>(input_mapping->bytes().data());
  auto* output_data = reinterpret_cast<float*>(writable->data());
  hyle::simd::rms_norm_f32(input_data, output_data, kElements, kEpsilon);

  hyle::Status terminal = output_mapping->close();
  hyle::Status input_close = input_mapping->close();
  if (terminal.ok() && !input_close.ok()) terminal = std::move(input_close);
  return terminal;
}

std::uint64_t run_prepared(const hyle::compute::PreparedOperation& prepared,
                           hyle::compute::Queue& queue,
                           const hyle::compute::ConstTensorView& input,
                           const hyle::compute::TensorView& output) noexcept {
  auto completion = prepared.submit(queue, input, output);
  if (!completion) return 0;
  return completion->wait().ok() ? 1U : 0U;
}

std::uint64_t run_graph(const hyle::graph::Invocation& invocation,
                        hyle::compute::Queue& queue) noexcept {
  auto completion = invocation.submit(queue);
  if (!completion) return 0;
  return completion->wait().ok() ? 1U : 0U;
}

}  // namespace

int main() {
  using namespace hyle::compute;
  using namespace hyle::graph;

  const auto qos = hyle::bench::perf::configure_thread_qos(
      "HYLE_BENCH_APPLE_QOS");
  const auto affinity = hyle::bench::perf::configure_thread_affinity(
      "HYLE_PERF_CALLER_CPU");
  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  std::fprintf(
      stderr,
      "# HYLE_COMPUTE_GRAPH_ONE_OP_RUN execution_order=%s apple_qos=%s "
      "qos_ok=%s affinity_topology=%s caller_cpu=%s affinity_ok=%s "
      "run_nonce=%s\n",
      peer_first ? "peer-first" : "candidate-first", qos.name, qos.ok,
      affinity.topology, affinity.cpu, affinity.ok,
      hyle::bench::perf::run_nonce());

  constexpr std::array<std::int64_t, 1> shape{
      static_cast<std::int64_t>(kElements)};
  TensorDesc description = take_or_abort(
      TensorDesc::dense(DataType::kFloat32, shape), "tensor description");
  Context context =
      take_or_abort(Context::create(Device::cpu()), "CPU context");
  Queue queue = take_or_abort(context.create_queue(), "CPU queue");

  constexpr std::size_t kBytes = kElements * sizeof(float);
  hyle::mem::Buffer input_buffer = take_or_abort(
      hyle::mem::Buffer::create(hyle::mem::system_resource(), kBytes,
                                alignof(float)),
      "input buffer");
  hyle::mem::Buffer output_buffer = take_or_abort(
      hyle::mem::Buffer::create(hyle::mem::system_resource(), kBytes,
                                alignof(float)),
      "output buffer");
  std::vector<float> values(kElements);
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = static_cast<float>((index % 127U) + 1U) / 127.0F;
  }
  require_ok(input_buffer.view().write(std::as_bytes(std::span(values))),
             "input initialization");

  TensorView mutable_input = take_or_abort(
      TensorView::create(description, input_buffer.view()), "input view");
  ConstTensorView input = mutable_input.as_const();
  TensorView output = take_or_abort(
      TensorView::create(description, output_buffer.view()), "output view");
  PreparedOperation prepared = take_or_abort(
      prepare(context, RmsNorm{.epsilon = kEpsilon}, description,
              description),
      "prepared RMSNorm");

  GraphBuilder builder;
  const ValueId input_id =
      take_or_abort(builder.input(description), "graph input");
  const ValueId output_id = take_or_abort(
      builder.add(RmsNorm{.epsilon = kEpsilon}, input_id), "graph RMSNorm");
  require_ok(builder.output(output_id), "graph output");
  Graph logical = take_or_abort(std::move(builder).finish(), "logical graph");
  CompiledGraph compiled =
      take_or_abort(compile(logical, context), "compiled graph");
  Invocation invocation = take_or_abort(
      compiled.executable.create_invocation(), "graph invocation");
  require_ok(invocation.bind(input_id, input), "graph input binding");
  require_ok(invocation.bind(output_id, output), "graph output binding");

  require_ok(run_direct(input.storage(), output.storage()), "direct warmup");
  if (run_prepared(prepared, queue, input, output) == 0U ||
      run_graph(invocation, queue) == 0U) {
    fail_setup("Compute/Graph warmup");
  }

  nb::Bench compute_bench;
  compute_bench.title("compute prepared RMSNorm f32 N=4096")
      .relative(true)
      .minEpochTime(std::chrono::milliseconds(8));
  run_pair(
      compute_bench, peer_first, "HYLE Compute PreparedOperation N=4096",
      [&] {
        const auto ok = run_prepared(prepared, queue, input, output);
        nb::doNotOptimizeAway(ok);
      },
      "HYLE direct Mem+SIMD N=4096", [&] {
        const bool ok = run_direct(input.storage(), output.storage()).ok();
        nb::doNotOptimizeAway(ok);
      });

  nb::Bench graph_bench;
  graph_bench.title("graph invocation RMSNorm f32 N=4096")
      .relative(true)
      .minEpochTime(std::chrono::milliseconds(8));
  run_pair(
      graph_bench, peer_first, "HYLE Graph Invocation N=4096",
      [&] {
        const auto ok = run_graph(invocation, queue);
        nb::doNotOptimizeAway(ok);
      },
      "HYLE Compute PreparedOperation N=4096", [&] {
        const auto ok = run_prepared(prepared, queue, input, output);
        nb::doNotOptimizeAway(ok);
      });
}
