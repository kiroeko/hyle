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
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <mach-o/dyld.h>
#include <mlx/fast.h>
#include <mlx/memory.h>
#include <mlx/ops.h>
#include <pthread.h>

#include "../common/perf_process.h"
#include "hyle/graph/apple.h"

namespace {
namespace nb = ankerl::nanobench;
namespace c = hyle::compute;
namespace a = c::apple;
namespace g = hyle::graph;
namespace ga = g::apple;
namespace mx = mlx::core;

constexpr int kStateRows = 37;
constexpr int kRows = 17;
constexpr int kWidth = 2048;
constexpr int kSlots = 2;

[[noreturn]] void fail(std::string_view operation,
                       std::string_view detail = {}) {
  std::fprintf(stderr, "%.*s: %.*s\n", int(operation.size()), operation.data(),
               int(detail.size()), detail.data());
  std::exit(EXIT_FAILURE);
}

void require(hyle::Status status) {
  if (!status.ok())
    fail("Apple B1 benchmark", status.message());
}

template <class T> T take(hyle::Result<T> result) {
  if (!result)
    fail("Apple B1 benchmark", result.error().message());
  return std::move(*result);
}

std::string env(const char *name) {
  const char *value = std::getenv(name);
  return value ? value : "";
}

std::uint32_t sample_bits(std::uint32_t index) {
  index ^= index >> 16;
  index *= 0x7feb352dU;
  index ^= index >> 15;
  index *= 0x846ca68bU;
  return index ^ (index >> 16);
}

struct Data final {
  std::array<std::vector<float>, kSlots> states;
  std::array<std::vector<float>, kSlots> updates;
  std::array<std::vector<std::int64_t>, kSlots> ids;
  std::vector<float> weights;
  mx::array native_weights;

  Data() : weights(kWidth), native_weights(mx::array(0.0F)) {
    for (int column = 0; column < kWidth; ++column)
      weights[column] = 0.95F +
                        0.0001F * float(sample_bits(column ^ 0x91e10da5U) % 997);
    native_weights = mx::array(weights.data(), {kWidth}, mx::float32);
    mx::eval(native_weights);
    native_weights.wait();
    for (int slot = 0; slot < kSlots; ++slot) {
      states[slot].resize(kStateRows * kWidth);
      updates[slot].resize(kRows * kWidth);
      ids[slot].resize(kRows);
      for (std::size_t i = 0; i < states[slot].size(); ++i)
        states[slot][i] =
            float(int(sample_bits(std::uint32_t(i) ^
                                  (0x9e3779b9U * std::uint32_t(slot + 1))) %
                      257) -
                  128) *
            0.00390625F;
      for (std::size_t i = 0; i < updates[slot].size(); ++i)
        updates[slot][i] =
            float(int(sample_bits(std::uint32_t(i) ^
                                  (0x85ebca6bU * std::uint32_t(slot + 3))) %
                      129) -
                  64) *
            0.001953125F;
      for (int row = 0; row < kRows; ++row)
        ids[slot][row] = (row * 2 + slot) % kStateRows;
    }
  }
};

std::array<std::vector<mx::array>, kSlots>
resident_inputs(const Data &data) {
  std::array<std::vector<mx::array>, kSlots> result;
  for (int slot = 0; slot < kSlots; ++slot) {
    auto state = mx::astype(
        mx::array(data.states[slot].data(), {kStateRows, kWidth}, mx::float32),
        mx::bfloat16);
    auto update = mx::astype(
        mx::array(data.updates[slot].data(), {kRows, kWidth}, mx::float32),
        mx::bfloat16);
    auto ids = mx::array(data.ids[slot].data(), {kRows}, mx::int64);
    mx::eval({state, ids, update});
    state.wait();
    ids.wait();
    update.wait();
    result[slot] = {std::move(state), std::move(ids), std::move(update)};
  }
  return result;
}

mx::array silu(const mx::array &input, mx::Stream stream) {
  auto result = mx::multiply(input, mx::sigmoid(input, stream), stream);
  return mx::where(
      mx::equal(input,
                mx::array(-std::numeric_limits<float>::infinity()), stream),
      mx::array(0.0F), result, stream);
}

// Independent direct expression of the original MLX primitive region. It
// includes both required checked-index status reductions and makes the status
// a terminal output, so the peer pays the same validation and synchronization
// boundary as the HYLE managed program.
std::vector<mx::array> native_region(const std::vector<mx::array> &inputs,
                                     const mx::array &weights,
                                     mx::Stream stream) {
  auto indices = mx::reshape(inputs[1], {kRows}, stream);
  auto invalid = mx::logical_or(
      mx::less(indices, mx::array(0, indices.dtype()), stream),
      mx::greater_equal(indices, mx::array(kStateRows, indices.dtype()), stream),
      stream);
  auto safe = mx::where(invalid, mx::array(0, indices.dtype()), indices, stream);
  auto scatter_status = mx::where(mx::any(invalid, stream), mx::array(1),
                                  mx::array(0), stream);
  auto native_updates = mx::expand_dims(
      mx::reshape(inputs[2], {kRows, kWidth}, stream), 1, stream);
  auto changed = mx::astype(
      mx::scatter_add(mx::astype(inputs[0], mx::float32, stream), safe,
                      mx::astype(native_updates, mx::float32, stream), 0,
                      stream),
      mx::bfloat16, stream);
  changed = mx::depends({changed}, {scatter_status}).front();

  invalid = mx::logical_or(
      mx::less(indices, mx::array(0, indices.dtype()), stream),
      mx::greater_equal(indices, mx::array(kStateRows, indices.dtype()), stream),
      stream);
  safe = mx::where(invalid, mx::array(0, indices.dtype()), indices, stream);
  auto gather_status = mx::where(mx::any(invalid, stream), mx::array(1),
                                 mx::array(0), stream);
  auto embedded = mx::take(changed, safe, 0, stream);
  embedded = mx::depends({embedded}, {gather_status}).front();

  auto low = mx::astype(mx::astype(embedded, mx::float32, stream),
                        mx::bfloat16, stream);
  auto norm = mx::astype(
      mx::fast::rms_norm(mx::astype(low, mx::float32, stream), weights, 1e-5F,
                         stream),
      mx::bfloat16, stream);
  auto gate = mx::astype(
      mx::multiply(silu(mx::astype(norm, mx::float32, stream), stream),
                   mx::astype(low, mx::float32, stream), stream),
      mx::bfloat16, stream);
  auto residual = mx::astype(
      mx::add(mx::astype(gate, mx::float32, stream),
              mx::astype(low, mx::float32, stream), stream),
      mx::bfloat16, stream);
  auto status = mx::where(mx::equal(scatter_status, mx::array(0), stream),
                          gather_status, scatter_status, stream);
  return {std::move(changed), std::move(residual), std::move(status)};
}

hyle::Result<a::Program> graph_program(const a::Context &context,
                                       const Data &data) {
  auto builder = take(g::GraphBuilder::create());
  const auto desc = [](c::DataType type, auto shape) {
    return take(c::TensorDesc::dense(type, shape));
  };
  auto state = take(builder.input(desc(
      c::DataType::kBFloat16,
      std::array<std::int64_t, 2>{kStateRows, kWidth})));
  auto ids = take(builder.input(desc(c::DataType::kInt64,
                                     std::array<std::int64_t, 1>{kRows})));
  auto updates = take(builder.input(desc(
      c::DataType::kBFloat16, std::array<std::int64_t, 2>{kRows, kWidth})));
  auto weight = take(builder.constant(desc(
      c::DataType::kFloat32, std::array<std::int64_t, 1>{kWidth})));
  auto changed = take(builder.add(
      c::Scatter{.kind = c::ScatterKind::kAdd}, state, ids, updates));
  auto embedded = take(builder.add(c::Gather{}, changed, ids));
  auto wide = take(builder.add(c::Cast{c::DataType::kFloat32}, embedded));
  auto low = take(builder.add(c::Cast{c::DataType::kBFloat16}, wide));
  auto norm = take(builder.add(c::RmsNorm{}, low, weight));
  auto gate = take(builder.add(
      c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}, norm, low));
  auto residual = take(builder.add(c::Binary{}, gate, low));
  require(builder.publish(changed));
  require(builder.publish(residual));
  auto graph = take(std::move(builder).finish());
  return ga::compile(context, graph, std::array{data.native_weights});
}

class Harness final {
public:
  Harness(const Data &data, bool reference)
      : reference_(reference), context_(take(a::Context::create())),
        inputs_(resident_inputs(data)) {
    const std::array tiny{a::InputSpec{{1}, mx::float32}};
    (void)take(context_.compile(
        [](const auto &input, mx::Stream stream) {
          std::fprintf(stderr, "# APPLE_OWNER_QOS value=%u\n",
                       unsigned(qos_class_self()));
          return std::vector{mx::negative(input[0], stream)};
        },
        tiny));
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
    if (reference_) {
      const std::array specs{
          a::InputSpec{{kStateRows, kWidth}, mx::bfloat16},
          a::InputSpec{{kRows}, mx::int64},
          a::InputSpec{{kRows, kWidth}, mx::bfloat16}};
      program_ = take(context_.compile(
          [weights = data.native_weights](const auto &input,
                                          mx::Stream stream) {
            return native_region(input, weights, stream);
          },
          specs));
    } else {
      program_ = take(graph_program(context_, data));
    }
    for (auto &slot : invocations_)
      slot = take(program_.create_invocation());
  }

  ~Harness() { require(context_.close()); }

  void run() { execute(next_++ % kSlots, nullptr); }

  std::vector<std::uint16_t> output_at(int slot) {
    std::vector<std::uint16_t> bits;
    execute(slot, &bits);
    return bits;
  }

private:
  void execute(int slot, std::vector<std::uint16_t> *host) {
    auto &invocation = invocations_[slot];
    require(invocation.bind(inputs_[slot]));
    auto work = take(invocation.submit());
    auto lease = take(work.wait());
    const auto arrays = lease->arrays();
    const std::size_t expected = reference_ ? 3 : 2;
    if (arrays.size() != expected)
      fail("Apple B1 output count");
    if (reference_ && arrays[2].data<std::int32_t>()[0] != 0)
      fail("Apple B1 native status");
    if (host) {
      host->reserve(kStateRows * kWidth + kRows * kWidth);
      for (std::size_t output = 0; output < 2; ++output) {
        auto bits = mx::view(arrays[output], mx::uint16);
        mx::eval(bits);
        bits.wait();
        const auto count = output == 0 ? kStateRows * kWidth : kRows * kWidth;
        host->insert(host->end(), bits.data<std::uint16_t>(),
                     bits.data<std::uint16_t>() + count);
      }
    }
    nb::doNotOptimizeAway(arrays[0]);
    nb::doNotOptimizeAway(arrays[1]);
  }

  bool reference_;
  a::Context context_;
  a::Program program_;
  std::array<std::vector<mx::array>, kSlots> inputs_;
  std::array<a::Invocation, kSlots> invocations_;
  int next_ = 0;
};

template <class T> struct Runner {
  T *harness;
  void operator()() const { harness->run(); }
};

void measure(bool peer_first, bool allocation_first, const std::string &aa,
             bool verify_only) {
  Data data;
  std::unique_ptr<Harness> candidate;
  std::unique_ptr<Harness> peer;
  const auto begin = std::chrono::steady_clock::now();
  if (allocation_first) {
    peer = std::make_unique<Harness>(data, aa != "selected");
    candidate = std::make_unique<Harness>(data, aa == "native");
  } else {
    candidate = std::make_unique<Harness>(data, aa == "native");
    peer = std::make_unique<Harness>(data, aa != "selected");
  }
  const auto end = std::chrono::steady_clock::now();
  std::fprintf(stderr,
               "# APPLE_B1_COLD pair_prepare_ms=%.3f native_active=%zu "
               "native_peak=%zu native_cache=%zu\n",
               std::chrono::duration<double, std::milli>(end - begin).count(),
               mx::get_active_memory(), mx::get_peak_memory(),
               mx::get_cache_memory());
  std::vector<std::uint16_t> previous;
  for (int slot = 0; slot < kSlots; ++slot) {
    auto actual = candidate->output_at(slot);
    auto expected = peer->output_at(slot);
    if (actual != expected)
      fail("Apple B1 complete-region correctness");
    if (slot && actual == previous)
      fail("Apple B1 changed input must change output");
    previous = std::move(actual);
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
  const std::string title =
      "Graph B1 BF16 Scatter Norm Gate R=17 W=2048";
  const auto candidate_name = "HYLE Apple " + title;
  const auto peer_name = "MLX lifecycle " + title;
  nb::Bench bench;
  bench.title(title)
      .batch(1)
      .epochs(104)
      .minEpochIterations(2)
      .minEpochTime(std::chrono::milliseconds(4))
      .performanceCounters(false);
  if (peer_first)
    (void)bench.compare(peer_name, Runner{peer.get()}, candidate_name,
                        Runner{candidate.get()});
  else
    (void)bench.compare(candidate_name, Runner{candidate.get()}, peer_name,
                        Runner{peer.get()});
}

void require_aa(bool aa, bool verify) {
  if (aa || verify || env("HYLE_PERF_RUN_NONCE").empty())
    return;
  std::uint32_t size = 0;
  (void)_NSGetExecutablePath(nullptr, &size);
  std::string buffer(size, '\0');
  if (_NSGetExecutablePath(buffer.data(), &size) != 0)
    fail("locate Apple B1 A/A receipt");
  const auto path = std::filesystem::path(buffer.c_str()).parent_path() /
                    "hyle_apple_b1_aa.ready";
  std::string status;
  std::ifstream file(path);
  file >> status;
  if (status != "PASS")
    fail("Apple B1 A/A has not passed for this artifact");
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
      fail("invalid Apple B1 A/A route");
    require_aa(!aa.empty(), verify_only);
    std::fprintf(
        stderr,
        "# HYLE_APPLE_B1_RUN mode=graph execution_order=%s "
        "allocation_order=%s aa_route=%s paired_mode=nanobench-compare "
        "apple_qos=%s qos_ok=%s run_nonce=%s verify_only=%d\n",
        peer_first ? "peer-first" : "candidate-first",
        allocation_first ? "peer-first" : "candidate-first",
        aa.empty() ? "none" : aa.c_str(), qos.name, qos.ok,
        hyle::bench::perf::run_nonce(), verify_only);
    measure(peer_first, allocation_first, aa, verify_only);
  } catch (const std::exception &error) {
    fail("native MLX exception", error.what());
  }
}
