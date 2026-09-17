#include <nanobench.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "../common/perf_process.h"
#include "../compute_route_cuda/host_policy.h"
#include "reference.h"
#if HYLE_VULKAN_GRAPH_BENCH
#include "hyle/graph/vulkan.h"
#endif
#if HYLE_VULKAN_PLACEMENT
extern "C" int hyle_vulkan_placement_prefix(int);
#endif

namespace {
namespace nb = ankerl::nanobench;
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace reference = hyle::bench::vulkan;
constexpr int inner = 4096, hidden = 11008, output = 4096, slots = 2;

[[noreturn]] void fail(const char *message) {
  std::fprintf(stderr, "Vulkan FFN: %s\n", message);
  std::exit(1);
}
void require(hyle::Status status) {
  if (!status.ok()) {
    std::fprintf(stderr, "%.*s\n", int(status.message().size()),
                 status.message().data());
    fail("operation failed");
  }
}
template <class T> T take(hyle::Result<T> result) {
  if (!result)
    require(result.error());
  return std::move(*result);
}
std::string env(const char *name) {
  const auto *value = std::getenv(name);
  return value ? value : "";
}
c::TensorDesc dense(int rows, int columns) {
  return take(c::TensorDesc::dense(c::DataType::kFloat32,
                                   std::array<std::int64_t, 2>{rows, columns}));
}
std::uint32_t mix(std::uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}

struct Data {
  int rows;
  std::array<std::vector<float>, 3> weights;
  std::array<std::vector<float>, slots> inputs;
  explicit Data(int m) : rows(m) {
    for (int p = 0; p < 3; ++p) {
      weights[p].resize(std::size_t(p == 2 ? hidden : inner) *
                        (p == 2 ? output : hidden));
      for (std::size_t i = 0; i < weights[p].size(); ++i)
        weights[p][i] =
            float(int(mix(std::uint32_t(i) ^ (0x8da6b343U * (p + 1))) % 33) -
                  16) *
            (p == 2 ? 0.002f : 0.003f);
    }
    for (int slot = 0; slot < slots; ++slot) {
      inputs[slot].resize(std::size_t(rows) * inner);
      for (std::size_t i = 0; i < inputs[slot].size(); ++i)
        inputs[slot][i] =
            float(int(mix(std::uint32_t(i) ^ (0x9e3779b9U * (slot + 1))) % 23) -
                  11) *
            0.09f;
    }
  }
  std::vector<float> oracle(int slot) const {
    std::vector<float> result(std::size_t(rows) * output);
    for (int row = 0; row < rows; ++row) {
      std::array<std::vector<double>, 2> accum{std::vector<double>(hidden),
                                               std::vector<double>(hidden)};
      for (int k = 0; k < inner; ++k) {
        const double x = inputs[slot][row * inner + k];
        for (int h = 0; h < hidden; ++h) {
          accum[0][h] += x * weights[0][k * hidden + h];
          accum[1][h] += x * weights[1][k * hidden + h];
        }
      }
      std::vector<double> down(output);
      for (int h = 0; h < hidden; ++h) {
        // Each public operation publishes F32. Independent double math starts
        // again from that node boundary, not from an unrounded fused oracle.
        const auto gate = float(accum[0][h]), up = float(accum[1][h]);
        const float product =
            float(double(gate) / (1.0 + std::exp(-double(gate))) * up);
        for (int n = 0; n < output; ++n)
          down[n] += double(product) * weights[2][h * output + n];
      }
      for (int n = 0; n < output; ++n)
        result[row * output + n] = float(down[n]);
    }
    return result;
  }
};

std::array<v::Tensor, 3> upload_weights(const v::Context &context,
                                        const Data &data) {
  std::array<v::Tensor, 3> weights;
  for (int i = 0; i < 3; ++i)
    weights[i] = take(
        context.upload(dense(i == 2 ? hidden : inner, i == 2 ? output : hidden),
                       std::as_bytes(std::span(data.weights[i]))));
  return weights;
}

#if HYLE_VULKAN_GRAPH_BENCH
v::Program graph_program(const v::Context &context, const Data &data) {
  namespace g = hyle::graph;
  auto weights = upload_weights(context, data);
  auto builder = take(g::GraphBuilder::create());
  const auto x = take(builder.input(dense(data.rows, inner)));
  const auto w0 = take(builder.constant(dense(inner, hidden)));
  const auto w1 = take(builder.constant(dense(inner, hidden)));
  const auto w2 = take(builder.constant(dense(hidden, output)));
  const auto gate = take(builder.add(c::MatMul{}, x, w0));
  const auto up = take(builder.add(c::MatMul{}, x, w1));
  const auto product = take(builder.add(
      c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}, gate, up));
  require(builder.publish(take(builder.add(c::MatMul{}, product, w2))));
  auto graph = take(std::move(builder).finish());
  return take(g::vulkan::compile(context, graph, weights));
}
#endif

class Harness {
public:
  Harness(const Data &data, bool peer)
      : data_(data), quota_(hyle::mem::system_allocator()),
        context_(take(v::Context::create({
#if defined(HYLE_VULKAN_BENCH_LOADER)
            .loader_path = HYLE_VULKAN_BENCH_LOADER,
#endif
            .memory_quota = &quota_}))) {
    const auto &f = context_.environment();
    std::fprintf(
        stderr,
        "# VULKAN_ENV device=%s vendor=%u device_id=%u driver_id=%u driver=%u "
        "driver_name=%s driver_info=%s loader=%u api=%u headers=%u subgroup=%u "
        "shared_bytes=%u compiler=%s shaders=%s\n",
        f.device_name.c_str(), f.vendor_id, f.device_id, f.driver_id,
        f.driver_version, f.driver_name.c_str(), f.driver_info.c_str(),
        f.loader_version, f.api_version, f.header_version, f.subgroup_size,
        f.shared_memory_bytes, f.shader_compiler.c_str(),
        f.shader_revision.c_str());
    for (int slot = 0; slot < slots; ++slot)
      inputs_[slot] =
          take(context_.upload(dense(data.rows, inner),
                               std::as_bytes(std::span(data.inputs[slot]))));
    if (peer)
      reference_ = std::make_unique<reference::Reference>(
          context_, data.rows, inner, hidden, output, data.weights);
#if HYLE_VULKAN_GRAPH_BENCH
    program_ = peer ? reference_->owning_program(context_)
                    : graph_program(context_, data);
    for (auto &slot : invocations_)
      slot = take(program_.create_invocation());
#else
    if (peer) {
      for (int slot = 0; slot < slots; ++slot) {
        values_[slot] = reference_->values(context_, inputs_[slot]);
        recorded_[slot] = reference_->record(context_, values_[slot]);
      }
    } else {
      auto weights = upload_weights(context_, data);
      const auto input = dense(data.rows, inner),
                 middle = dense(data.rows, hidden);
      const std::array prepared{
          take(v::prepare(context_, c::MatMul{}, input, weights[0])),
          take(v::prepare(context_, c::MatMul{}, input, weights[1])),
          take(v::prepare(
              context_,
              c::GatedActivation{.inputs = c::GatedInputKind::kSeparate},
              middle, middle)),
          take(v::prepare(context_, c::MatMul{}, middle, weights[2]))};
      for (int slot = 0; slot < slots; ++slot) {
        auto &values = values_[slot];
        values[0] = inputs_[slot];
        for (int i = 4; i < 7; ++i)
          values[i] = take(context_.allocate(middle));
        values[7] = take(context_.allocate(dense(data.rows, output)));
        auto recorder = take(v::Recorder::create(context_));
        require(recorder.append(prepared[0], std::array{values[0]},
                                std::array{values[4]}));
        require(recorder.append(prepared[1], std::array{values[0]},
                                std::array{values[5]}));
        require(recorder.append(prepared[2], std::array{values[4], values[5]},
                                std::array{values[6]}));
        require(recorder.append(prepared[3], std::array{values[6]},
                                std::array{values[7]}));
        recorded_[slot] = take(recorder.finish());
      }
    }
#endif
    resident_bytes_ = quota_.used();
    std::fprintf(stderr,
                 "# VULKAN_RESOURCES rows=%d peer=%d resident_bytes=%zu "
                 "peak_bytes=%zu\n",
                 data.rows, peer, quota_.used(), quota_.peak());
  }
  ~Harness() { require(context_.close()); }
  void run() { execute(next_++ % slots, nullptr); }
  std::vector<float> output_at(int slot) {
    std::vector<float> result(std::size_t(data_.rows) * output);
    execute(slot, &result);
    if (quota_.used() != resident_bytes_)
      fail("replay changed resident Vulkan memory");
    return result;
  }

private:
  void execute(int slot, std::vector<float> *host) {
#if HYLE_VULKAN_GRAPH_BENCH
    auto &invocation = invocations_[slot];
    require(invocation.bind(inputs_[slot]));
    auto work = take(invocation.submit());
    auto lease = take(work.wait());
    read(lease->tensors()[0], host);
#else
    auto work = take(recorded_[slot].submit());
    require(work.wait());
    read(values_[slot][7], host);
#endif
  }
  void read(const v::Tensor &tensor, std::vector<float> *host) {
    if (host)
      require(
          context_.download(tensor, std::as_writable_bytes(std::span(*host))));
    nb::doNotOptimizeAway(tensor);
  }
  const Data &data_;
  hyle::mem::QuotaAllocator quota_;
  v::Context context_;
  std::unique_ptr<reference::Reference> reference_;
  v::Program program_;
  std::array<v::Tensor, slots> inputs_;
  std::array<std::array<v::Tensor, 8>, slots> values_;
  std::array<v::Invocation, slots> invocations_;
  std::array<v::RecordedProgram, slots> recorded_;
  std::size_t resident_bytes_ = 0;
  std::uint64_t next_ = 0;
};

void verify(std::span<const float> actual, std::span<const float> expected,
            bool oracle) {
  if (actual.size() != expected.size())
    fail("output shape");
  std::size_t zero_rejections = 0;
  double worst = 0, largest = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const auto error = std::abs(double(actual[i]) - expected[i]);
    const auto limit =
        1e-5 + (oracle ? 1e-4 : 1e-5) * std::abs(double(expected[i]));
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
        error > limit) {
      std::fprintf(stderr, "element=%zu actual=%.9g expected=%.9g limit=%.9g\n",
                   i, actual[i], expected[i], limit);
      fail("complete F32 FFN correctness");
    }
    worst = std::max(worst, error);
    largest = std::max(largest, std::abs(double(expected[i])));
    zero_rejections += std::abs(double(expected[i])) > limit;
  }
  if (oracle && zero_rejections * 2 < expected.size())
    fail("degenerate oracle fixture");
  std::fprintf(stderr,
               "# VULKAN_CORRECTNESS oracle=%d elements=%zu max_error=%.9g "
               "max_expected=%.9g zero_rejections=%zu\n",
               oracle, actual.size(), worst, largest, zero_rejections);
}
struct Runner {
  Harness *harness;
  void operator()() const { harness->run(); }
};

void measure(int rows, bool peer_first, bool allocation_first,
             const std::string &aa, bool verify_only) {
  Data data(rows);
  std::unique_ptr<Harness> candidate, peer;
  const auto begin = std::chrono::steady_clock::now();
  if (allocation_first) {
    peer = std::make_unique<Harness>(data, aa != "selected");
    candidate = std::make_unique<Harness>(data, aa == "native");
  } else {
    candidate = std::make_unique<Harness>(data, aa == "native");
    peer = std::make_unique<Harness>(data, aa != "selected");
  }
  std::fprintf(stderr, "# VULKAN_COLD rows=%d pair_prepare_ms=%.3f\n", rows,
               std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - begin)
                   .count());
  std::vector<float> previous;
  for (int slot = 0; slot < slots; ++slot) {
    const auto actual = candidate->output_at(slot),
               reference_output = peer->output_at(slot);
    verify(actual, reference_output, false);
    if (slot && actual == previous)
      fail("changing inputs did not change output");
    previous = actual;
    if (verify_only) {
      const auto expected = data.oracle(slot);
      verify(actual, expected, true);
      verify(reference_output, expected, true);
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
  const std::string kind = HYLE_VULKAN_GRAPH_BENCH ? "Graph FFN" : "Direct FFN";
  const auto title =
      kind + " M=" + std::to_string(rows) + " K=4096 H=11008 N=4096";
  const auto candidate_name = "HYLE Vulkan " + title;
  const auto peer_name = std::string("ggml shader lifecycle ") + title;
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
} // namespace

int main(int argc, char **argv) {
  try {
    (void)argc;
    const auto qos = hyle::bench::perf::configure_thread_qos();
    const auto affinity =
        hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
    const hyle::bench::cuda::HostPolicy host;
#if HYLE_VULKAN_PLACEMENT
    nb::doNotOptimizeAway(hyle_vulkan_placement_prefix(11));
#endif
    const bool peer_first = !env("HYLE_BENCH_RIVAL_FIRST").empty();
    const bool allocation_first =
        !env("HYLE_BENCH_PEER_ALLOCATION_FIRST").empty();
    const bool verify_only = !env("HYLE_BENCH_VERIFY_ONLY").empty();
    const auto aa = env("HYLE_BENCH_AA_ROUTE");
    if (!aa.empty() && aa != "native" && aa != "selected")
      fail("invalid A/A route");
    if (!verify_only && aa.empty() && !env("HYLE_PERF_RUN_NONCE").empty()) {
      std::string status;
      const auto ready =
          std::filesystem::absolute(argv[0]).parent_path() /
          (HYLE_VULKAN_GRAPH_BENCH ? "hyle_vulkan_graph_aa.ready"
                                   : "hyle_vulkan_compute_aa.ready");
      std::ifstream(ready) >> status;
      if (status != "PASS")
        fail("A/A has not passed for this build");
    }
    std::fprintf(
        stderr,
        "# HYLE_VULKAN_FFN_RUN mode=%s execution_order=%s allocation_order=%s "
        "aa_route=%s paired_mode=nanobench-compare apple_qos=%s qos_ok=%s "
        "affinity_topology=%s caller_cpu=%s affinity_ok=%s parent_mask=%s "
        "priority=%s "
        "run_nonce=%s verify_only=%d placement=%d\n",
        HYLE_VULKAN_GRAPH_BENCH ? "graph" : "direct",
        peer_first ? "peer-first" : "candidate-first",
        allocation_first ? "peer-first" : "candidate-first",
        aa.empty() ? "none" : aa.c_str(), qos.name, qos.ok, affinity.topology,
        affinity.cpu, affinity.ok, host.mask.c_str(), host.priority,
        hyle::bench::perf::run_nonce(), verify_only, HYLE_VULKAN_PLACEMENT);
    measure(1, peer_first, allocation_first, aa, verify_only);
    measure(16, peer_first, allocation_first, aa, verify_only);
  } catch (const std::exception &error) {
    fail(error.what());
  }
}
