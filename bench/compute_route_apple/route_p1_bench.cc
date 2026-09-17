#include <nanobench.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <mach-o/dyld.h>
#include <mlx/compile.h>
#include <mlx/memory.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>
#include <pthread.h>

#include "../common/perf_process.h"
#include "hyle/compute/apple.h"
#if HYLE_APPLE_GRAPH_BENCH
#include "hyle/graph/apple.h"
#endif

namespace {
namespace nb = ankerl::nanobench;
namespace c = hyle::compute;
namespace a = c::apple;
namespace mx = mlx::core;
constexpr int k = 4096, hidden = 11008, output = 4096, groups = k / 64;
constexpr int slots = 2;

[[noreturn]] void fail(std::string_view operation,
                       std::string_view detail = {}) {
  std::fprintf(stderr, "%.*s: %.*s\n", int(operation.size()), operation.data(),
               int(detail.size()), detail.data());
  std::exit(EXIT_FAILURE);
}
void require(hyle::Status status) {
  if (!status.ok())
    fail("Apple benchmark", status.message());
}
template <class T> T take(hyle::Result<T> result) {
  if (!result)
    fail("Apple benchmark", result.error().message());
  return std::move(*result);
}
std::string env(const char *name) {
  const char *value = std::getenv(name);
  return value ? value : "";
}

// Cold deterministic fixture data. Mix whole indices so large matrix rows do
// not repeat a tiny period and hide a broken result behind cancellation.
std::uint32_t sample_bits(std::uint32_t index) {
  index ^= index >> 16;
  index *= 0x7feb352dU;
  index ^= index >> 15;
  index *= 0x846ca68bU;
  return index ^ (index >> 16);
}

struct Data final {
  int rows;
  std::vector<mx::array> weights;
  std::array<std::vector<float>, slots> host_inputs;
  std::array<std::vector<std::uint32_t>, 2> packed;
  std::array<std::vector<float>, 2> scales, biases;
  std::vector<float> down;

  explicit Data(int m) : rows(m), down(hidden * output) {
    for (int p = 0; p < 2; ++p) {
      packed[p].resize(hidden * k / 8);
      scales[p].resize(hidden * groups);
      biases[p].resize(hidden * groups);
      for (int h = 0; h < hidden; ++h) {
        for (int g = 0; g < groups; ++g) {
          const auto index = h * groups + g;
          scales[p][index] = 0.001f + 0.0001f * float(
              sample_bits(std::uint32_t(index) ^ (0x8da6b343U * (p + 1))) % 5);
          biases[p][index] = -7.5f * scales[p][index];
        }
        for (int i = 0; i < k; ++i) {
          const auto q = sample_bits(std::uint32_t(h * k + i) ^
                                     (0xd8163841U * (p + 1))) & 15U;
          packed[p][(h * k + i) / 8] |= q << (4 * (i % 8));
        }
      }
      weights.emplace_back(packed[p].data(), mx::Shape{hidden, k / 8},
                           mx::uint32);
      weights.emplace_back(scales[p].data(), mx::Shape{hidden, groups},
                           mx::float32);
      weights.emplace_back(biases[p].data(), mx::Shape{hidden, groups},
                           mx::float32);
    }
    for (int i = 0; i < hidden * output; ++i)
      down[i] = float(int(sample_bits(std::uint32_t(i) ^ 0xcb1ab31fU) % 17) - 8) *
                0.001f;
    weights.emplace_back(down.data(), mx::Shape{hidden, output}, mx::float32);
    for (int slot = 0; slot < slots; ++slot) {
      auto &x = host_inputs[slot];
      x.resize(rows * k);
      for (int i = 0; i < rows * k; ++i)
        x[i] = float(int(sample_bits(std::uint32_t(i) ^
                                    (0x9e3779b9U * (slot + 1))) % 23) - 11) *
               0.09f;
    }
  }

  std::vector<float> oracle(int slot) const {
    std::vector<float> result(rows * output);
    for (int row = 0; row < rows; ++row) {
      std::vector<double> product(hidden);
      for (int h = 0; h < hidden; ++h) {
        double sums[2]{};
        for (int i = 0; i < k; ++i) {
          for (int p = 0; p < 2; ++p) {
            const auto q = (packed[p][(h * k + i) / 8] >> (4 * (i % 8))) & 15;
            const auto group = h * groups + i / 64;
            const double w = double(q) * scales[p][group] + biases[p][group];
            sums[p] += double(host_inputs[slot][row * k + i]) * w;
          }
        }
        product[h] = sums[0] / (1.0 + std::exp(-sums[0])) * sums[1];
      }
      // Traverse the independent CPU oracle's dense weights contiguously.
      // Each output retains the same increasing-h double accumulation order.
      std::vector<double> sums(output, 0);
      for (int h = 0; h < hidden; ++h)
        for (int n = 0; n < output; ++n)
          sums[n] += product[h] * down[h * output + n];
      for (int n = 0; n < output; ++n)
        result[row * output + n] = float(sums[n]);
    }
    return result;
  }
};

std::array<std::vector<mx::array>, slots> resident_inputs(const Data &data) {
  std::array<std::vector<mx::array>, slots> inputs;
  for (int slot = 0; slot < slots; ++slot)
    inputs[slot].emplace_back(data.host_inputs[slot].data(),
                              mx::Shape{data.rows, k}, mx::float32);
  return inputs;
}

// Independent expression of the pinned mature region, with no typed lowering
// or Graph calls. It preserves the same primitive sequence and fusion scope.
std::vector<mx::array> native_region(const std::vector<mx::array> &inputs,
                                     const std::vector<mx::array> &w,
                                     mx::Stream stream) {
  auto gate = mx::quantized_matmul(inputs[0], w[0], w[1], w[2], true, 64, 4,
                                   "affine", stream);
  auto up = mx::quantized_matmul(inputs[0], w[3], w[4], w[5], true, 64, 4,
                                 "affine", stream);
  auto gated = mx::multiply(
      mx::multiply(gate, mx::sigmoid(gate, stream), stream), up, stream);
  return {mx::matmul(gated, w[6], stream)};
}

hyle::Result<a::Program> typed_program(const a::Context &context,
                                       const Data &data) {
#if HYLE_APPLE_GRAPH_BENCH
  namespace g = hyle::graph;
  auto builder = take(g::GraphBuilder::create());
  const std::array<std::int64_t, 2> xs{data.rows, k}, ss{hidden, groups},
      ds{hidden, output};
  auto x = take(
      builder.input(take(c::TensorDesc::dense(c::DataType::kFloat32, xs))));
  const auto q = take(a::affine_q4_description(hidden, k));
  const auto s = take(c::TensorDesc::dense(c::DataType::kFloat32, ss));
  std::array<g::ValueId, 6> w;
  for (int i = 0; i < 6; ++i)
    w[i] = take(builder.constant(i % 3 == 0 ? q : s));
  auto down = take(
      builder.constant(take(c::TensorDesc::dense(c::DataType::kFloat32, ds))));
  const c::QuantizedMatMul qmm{.mode =
                                   c::QuantizedWeightMode::kUnsignedAffine4};
  auto gate = take(builder.add(qmm, x, w[0], w[1], w[2]));
  auto up = take(builder.add(qmm, x, w[3], w[4], w[5]));
  auto product = take(builder.add(
      c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}, gate, up));
  require(builder.publish(take(builder.add(c::MatMul{}, product, down))));
  auto graph = take(std::move(builder).finish());
  return g::apple::compile(context, graph, data.weights);
#else
  const std::array specs{a::InputSpec{{data.rows, k}, mx::float32}};
  return context.compile(
      [w = data.weights](const auto &inputs, mx::Stream stream)
          -> hyle::Result<std::vector<mx::array>> {
        const c::Operation qmm{c::QuantizedMatMul{
            .mode = c::QuantizedWeightMode::kUnsignedAffine4}};
        const std::array gi{inputs[0], w[0], w[1], w[2]},
            ui{inputs[0], w[3], w[4], w[5]};
        HYLE_ASSIGN_OR_RETURN(auto gate, a::lower(qmm, gi, stream));
        HYLE_ASSIGN_OR_RETURN(auto up, a::lower(qmm, ui, stream));
        const std::array gated_inputs{gate[0], up[0]};
        HYLE_ASSIGN_OR_RETURN(
            auto gated, a::lower(c::Operation(c::GatedActivation{
                                     .inputs = c::GatedInputKind::kSeparate}),
                                 gated_inputs, stream));
        const std::array down_inputs{gated[0], w[6]};
        return a::lower(c::Operation(c::MatMul{}), down_inputs, stream);
      },
      specs);
#endif
}

class Harness final {
public:
  Harness(const Data &data, bool reference)
      : data_(data), context_(take(a::Context::create())) {
    inputs_ = resident_inputs(data);
    // Cold-only observation of the actual owner, not a runtime instrumentation
    // hook or a different execution policy for candidate versus reference.
    const std::array tiny{a::InputSpec{{1}, mx::float32}};
    auto observe = take(context_.compile(
        [](const auto &x, mx::Stream stream) {
          std::fprintf(stderr, "# APPLE_OWNER_QOS value=%u\n",
                       unsigned(qos_class_self()));
          return std::vector{mx::negative(x[0], stream)};
        },
        tiny));
    (void)observe;
    const auto facts = context_.environment();
    if (!facts)
      fail("environment", facts.error().message());
    std::fprintf(stderr,
                 "# APPLE_ENV device=%s architecture=%s os=%s mlx=%s sdk=%u "
                 "deployment=%u metal=%s\n",
                 facts->device_name.c_str(), facts->architecture.c_str(),
                 facts->os_runtime.c_str(), facts->mlx_revision.c_str(),
                 facts->build_sdk, facts->deployment_target,
                 facts->metal_compiler.c_str());
    if (reference) {
      const std::array specs{a::InputSpec{{data.rows, k}, mx::float32}};
      program_ = take(context_.compile(
          [w = data.weights](const auto &x, mx::Stream stream) {
            return native_region(x, w, stream);
          },
          specs));
    } else
      program_ = take(typed_program(context_, data));
    for (auto &slot : invocations_)
      slot = take(program_.create_invocation());
  }
  ~Harness() { require(context_.close()); }
  void run() { execute(next_++ % slots, nullptr); }
  std::vector<float> output_at(int slot) {
    std::vector<float> result;
    execute(slot, &result);
    return result;
  }

private:
  void execute(int slot, std::vector<float> *host) {
    auto &invocation = invocations_[slot];
    require(invocation.bind(inputs_[slot]));
    auto work = take(invocation.submit());
    auto lease = take(work.wait());
    const auto &out = lease->arrays()[0];
    if (host)
      host->assign(out.data<float>(), out.data<float>() + data_.rows * output);
    nb::doNotOptimizeAway(out);
  }
  const Data &data_;
  a::Context context_;
  a::Program program_;
  std::array<std::vector<mx::array>, slots> inputs_;
  std::array<a::Invocation, slots> invocations_;
  int next_ = 0;
};

using Peer = Harness;

void verify(std::span<const float> actual, std::span<const float> expected,
            bool oracle) {
  if (actual.size() != expected.size())
    fail("output size");
  double worst = 0, largest_expected = 0;
  std::size_t zero_rejections = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double error = std::abs(double(actual[i]) - expected[i]);
    const double magnitude = std::abs(double(expected[i]));
    const double limit = 0.00001 + (oracle ? 0.0001 : 0.00001) * magnitude;
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
        error > limit) {
      std::fprintf(stderr,
                   "element=%zu actual=%.9g expected=%.9g tolerance=%.9g\n", i,
                   actual[i], expected[i], limit);
      fail("full affine Q4 FFN correctness");
    }
    worst = std::max(worst, error);
    largest_expected = std::max(largest_expected, magnitude);
    zero_rejections += magnitude > limit;
  }
  if (oracle && zero_rejections * 2 < expected.size())
    fail("degenerate fixture: at least half of oracle outputs must reject zero");
  std::fprintf(
      stderr, "# APPLE_CORRECTNESS elements=%zu oracle=%d max_abs_error=%.9g "
              "max_abs_expected=%.9g zero_rejections=%zu\n",
      actual.size(), oracle, worst, largest_expected, zero_rejections);
}
template <class T> struct Runner {
  T *harness;
  void operator()() const { harness->run(); }
};

void measure(int rows, bool peer_first, bool allocation_first,
             const std::string &aa, bool verify_only) {
  const auto data_begin = std::chrono::steady_clock::now();
  Data data(rows);
  const auto data_end = std::chrono::steady_clock::now();
  std::fprintf(
      stderr, "# APPLE_DATA rows=%d prepare_ms=%.3f native_active=%zu\n", rows,
      std::chrono::duration<double, std::milli>(data_end - data_begin).count(),
      mx::get_active_memory());
  std::unique_ptr<Harness> candidate;
  std::unique_ptr<Peer> peer;
  const auto begin = std::chrono::steady_clock::now();
  if (allocation_first) {
    peer = std::make_unique<Peer>(data, aa != "selected");
    candidate = std::make_unique<Harness>(data, aa == "native");
  } else {
    candidate = std::make_unique<Harness>(data, aa == "native");
    peer = std::make_unique<Peer>(data, aa != "selected");
  }
  const auto end = std::chrono::steady_clock::now();
  std::fprintf(
      stderr,
      "# APPLE_COLD rows=%d pair_prepare_ms=%.3f native_active=%zu "
      "native_peak=%zu native_cache=%zu\n",
      rows, std::chrono::duration<double, std::milli>(end - begin).count(),
      mx::get_active_memory(), mx::get_peak_memory(), mx::get_cache_memory());
  std::vector<float> previous;
  for (int slot = 0; slot < slots; ++slot) {
    auto actual = candidate->output_at(slot), reference = peer->output_at(slot);
    verify(actual, reference, false);
    if (slot && actual == previous)
      fail("different inputs must change output");
    previous = actual;
    if (verify_only) {
      const auto expected = data.oracle(slot);
      verify(actual, expected, true);
      verify(reference, expected, true);
    }
  }
  if (verify_only)
    return;
  for (int i = 0; i < 8; ++i) {
    if (i % 2) {
      peer->run();
      candidate->run();
    } else {
      candidate->run();
      peer->run();
    }
  }
  const std::string kind = HYLE_APPLE_GRAPH_BENCH ? "Graph FFN" : "Direct FFN";
  const std::string title =
      kind + " M=" + std::to_string(rows) + " K=4096 H=11008 N=4096";
  const auto c_name = "HYLE Apple " + title;
  const auto p_name = std::string("MLX lifecycle ") + title;
  nb::Bench b;
  b.title(title)
      .batch(1)
      .epochs(104)
      .minEpochIterations(2)
      .minEpochTime(std::chrono::milliseconds(4))
      .performanceCounters(false);
  if (peer_first)
    (void)b.compare(p_name, Runner{peer.get()}, c_name,
                    Runner{candidate.get()});
  else
    (void)b.compare(c_name, Runner{candidate.get()}, p_name,
                    Runner{peer.get()});
}

void require_aa(bool aa, bool verify) {
  if (aa || verify || env("HYLE_PERF_RUN_NONCE").empty())
    return;
  std::uint32_t size = 0;
  (void)_NSGetExecutablePath(nullptr, &size);
  std::string buffer(size, '\0');
  if (_NSGetExecutablePath(buffer.data(), &size) != 0)
    fail("locate A/A receipt");
  const auto path = std::filesystem::path(buffer.c_str()).parent_path() /
                    (HYLE_APPLE_GRAPH_BENCH ? "hyle_apple_graph_aa.ready"
                                            : "hyle_apple_compute_aa.ready");
  std::string status;
  std::ifstream file(path);
  file >> status;
  if (status != "PASS")
    fail("A/A has not passed for this artifact");
}
} // namespace

int main() {
  try {
    const auto qos = hyle::bench::perf::configure_thread_qos();
    const bool peer_first = !env("HYLE_BENCH_RIVAL_FIRST").empty();
    const bool allocation_first =
        !env("HYLE_BENCH_PEER_ALLOCATION_FIRST").empty();
    const bool verify_only = !env("HYLE_BENCH_VERIFY_ONLY").empty();
    const auto aa = env("HYLE_BENCH_AA_ROUTE");
    if (!aa.empty() && aa != "native" && aa != "selected")
      fail("invalid A/A route");
    require_aa(!aa.empty(), verify_only);
    std::fprintf(
        stderr,
        "# HYLE_APPLE_FFN_RUN mode=%s execution_order=%s allocation_order=%s "
        "aa_route=%s paired_mode=nanobench-compare apple_qos=%s qos_ok=%s "
        "run_nonce=%s verify_only=%d\n",
        HYLE_APPLE_GRAPH_BENCH ? "graph" : "direct",
        peer_first ? "peer-first" : "candidate-first",
        allocation_first ? "peer-first" : "candidate-first",
        aa.empty() ? "none" : aa.c_str(), qos.name, qos.ok,
        hyle::bench::perf::run_nonce(), verify_only);
    measure(1, peer_first, allocation_first, aa, verify_only);
    measure(16, peer_first, allocation_first, aa, verify_only);
  } catch (const std::exception &e) {
    fail("native MLX exception", e.what());
  }
}
