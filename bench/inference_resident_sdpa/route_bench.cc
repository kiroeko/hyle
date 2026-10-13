#include "../common/perf_process.h"
#include "hyle/graph/vulkan.h"
#include "reference.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <nanobench.h>
#include <span>
#include <string>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace g = hyle::graph;
namespace nb = ankerl::nanobench;
[[noreturn]] void fail(const char *message) {
  std::fprintf(stderr, "resident SDPA: %s\n", message);
  std::exit(2);
}
void check(hyle::Status status) {
  if (!status.ok()) {
    std::fprintf(stderr, "%.*s\n", int(status.message().size()),
                 status.message().data());
    fail("operation failed");
  }
}
template <class T> T take(hyle::Result<T> result) {
  if (!result)
    check(result.error());
  return std::move(*result);
}
std::string env(const char *name) {
  const auto *value = std::getenv(name);
  return value ? value : "";
}
std::uint32_t mix(std::uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}
struct Data {
  static constexpr int heads = 8, kv_heads = 2;
  int rows, keys, depth, capacity;
  c::TensorDesc query, physical, cache;
  std::array<std::vector<float>, 3> input;
  Data(int q, int k, int d) : rows(q), keys(k), depth(d), capacity(k + 13) {
    query = take(c::TensorDesc::dense(
        c::DataType::kFloat32,
        std::array<std::int64_t, 4>{1, rows, heads, depth}));
    physical = take(c::TensorDesc::dense(
        c::DataType::kFloat32,
        std::array<std::int64_t, 4>{1, capacity, kv_heads, depth}));
    cache = take(c::TensorDesc::strided_bytes(
        c::DataType::kFloat32,
        std::array<std::int64_t, 4>{1, keys, kv_heads, depth},
        physical.byte_strides()));
    for (int port = 0; port < 3; ++port) {
      input[port].resize(*(port == 0 ? query : physical).element_count());
      for (std::size_t i = 0; i < input[port].size(); ++i)
        input[port][i] =
            float(int(mix(std::uint32_t(i) ^
                          (0x9e3779b9U * std::uint32_t(port + 1))) %
                      31) -
                  15) *
            .015625F;
      if (port)
        for (std::size_t i = std::size_t(keys) * kv_heads * depth;
             i < input[port].size(); ++i)
          input[port][i] = 128.F;
    }
  }
  std::vector<float> oracle(bool causal) const {
    std::vector<float> result(input[0].size());
    std::vector<double> probability(keys);
    for (int row = 0; row < rows; ++row)
      for (int head = 0; head < heads; ++head) {
        double sum = 0;
        for (int key = 0; key < keys; ++key) {
          double dot = 0;
          for (int z = 0; z < depth; ++z)
            dot += double(input[0][(row * heads + head) * depth + z]) *
                   input[1][(key * kv_heads + head / 4) * depth + z];
          probability[key] = causal && key > keys - rows + row
                                 ? 0.
                                 : std::exp(dot / std::sqrt(double(depth)));
          sum += probability[key];
        }
        for (int z = 0; z < depth; ++z) {
          double value = 0;
          for (int key = 0; key < keys; ++key)
            value += probability[key] / sum *
                     input[2][(key * kv_heads + head / 4) * depth + z];
          result[(row * heads + head) * depth + z] = float(value);
        }
      }
    return result;
  }
};
struct Harness {
  hyle::mem::QuotaAllocator quota{hyle::mem::system_allocator(), 256u << 20};
  v::Context context;
  std::array<v::Tensor, 3> inputs;
  v::Program program;
  std::array<v::Invocation, 2> slots;
  std::size_t resident_bytes = 0;
  unsigned next = 0;
  Harness(const Data &data, bool causal, bool native) {
    v::ContextOptions options;
    options.memory_quota = &quota;
#ifdef HYLE_RESIDENT_VULKAN_LOADER
    options.loader_path = HYLE_RESIDENT_VULKAN_LOADER;
#endif
    context = take(v::Context::create(options));
    inputs[0] = take(
        context.upload(data.query, std::as_bytes(std::span(data.input[0]))));
    for (int port : {1, 2}) {
      auto physical = take(context.upload(
          data.physical, std::as_bytes(std::span(data.input[port]))));
      inputs[port] = take(physical.view(take(c::View::slice(1, 0, data.keys))));
    }
    if (native) {
      program = hyle::bench::resident::reference_program(context, data.query,
                                                         data.cache, causal);
    } else {
      auto builder = take(g::GraphBuilder::create());
      const auto query = take(builder.input(data.query));
      const auto key = take(builder.input(data.cache));
      const auto value = take(builder.input(data.cache));
      const auto output = take(builder.add(
          c::ScaledDotProductAttention{
              .layout = c::AttentionLayout::kBatchSequenceHeadDepth,
              .scale = 1.F / std::sqrt(float(data.depth)),
              .causal = causal},
          query, key, value));
      check(builder.publish(output));
      program =
          take(g::vulkan::compile(context, take(std::move(builder).finish())));
    }
    for (auto &slot : slots)
      slot = take(program.create_invocation());
    resident_bytes = quota.used();
    std::fprintf(stderr,
                 "# RESIDENT_RESOURCES native=%d rows=%d keys=%d depth=%d "
                 "bytes=%zu peak=%zu\n",
                 native, data.rows, data.keys, data.depth, quota.used(),
                 quota.peak());
  }
  ~Harness() { check(context.close()); }
  void execute(std::vector<float> *host = nullptr) {
    auto &slot = slots[next++ % slots.size()];
    check(slot.bind(inputs));
    auto work = take(slot.submit());
    auto result = take(work.wait());
    const auto output = result->tensors()[0];
    if (host)
      check(context.download(output, std::as_writable_bytes(std::span(*host))));
    nb::doNotOptimizeAway(output);
    // Both Tensor and ResultLease are destroyed before the next invocation.
  }
};
void verify(Harness &harness, const Data &data, bool causal) {
  const auto expected = data.oracle(causal);
  std::vector<float> actual(expected.size());
  harness.execute(&actual);
  double worst = 0;
  std::size_t worst_index = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const auto bound = 6e-6 * std::abs(double(expected[i])) + 2e-6;
    const auto ratio = std::abs(double(actual[i]) - expected[i]) / bound;
    if (ratio > worst) {
      worst = ratio;
      worst_index = i;
    }
    if (!std::isfinite(actual[i]))
      fail("nonfinite original oracle result");
  }
  if (worst > 1.) {
    std::fprintf(stderr,
                 "# RESIDENT_ORACLE_FAIL index=%zu actual=%.9g expected=%.9g "
                 "ratio=%.9g\n",
                 worst_index, actual[worst_index], expected[worst_index],
                 worst);
    fail("original attention oracle tolerance exceeded");
  }
  std::fprintf(stderr, "# RESIDENT_CORRECTNESS max_ratio=%.9g elements=%zu\n",
               worst, actual.size());
}
void measure(int rows, int keys, int depth, bool causal, bool verify_only) {
  Data data(rows, keys, depth);
  const auto aa = env("HYLE_BENCH_AA_ROUTE");
  if (!aa.empty() && aa != "native" && aa != "selected")
    fail("unknown A/A route");
  std::unique_ptr<Harness> candidate, peer;
  auto make_candidate = [&] {
    candidate = std::make_unique<Harness>(data, causal, aa == "native");
  };
  auto make_peer = [&] {
    peer = std::make_unique<Harness>(data, causal, aa != "selected");
  };
  if (env("HYLE_BENCH_PEER_ALLOCATION_FIRST").empty()) {
    make_candidate();
    make_peer();
  } else {
    make_peer();
    make_candidate();
  }
  verify(*candidate, data, causal);
  verify(*peer, data, causal);
  for (int i = 0; i < 8; ++i) {
    candidate->execute();
    peer->execute();
  }
  std::fprintf(stderr,
               "# RESIDENT_FIRST_USE candidate_before=%zu candidate_after=%zu "
               "peer_before=%zu peer_after=%zu\n",
               candidate->resident_bytes, candidate->quota.used(),
               peer->resident_bytes, peer->quota.used());
  candidate->resident_bytes = candidate->quota.used();
  peer->resident_bytes = peer->quota.used();
  for (int i = 0; i < 4; ++i) {
    candidate->execute();
    peer->execute();
  }
  if (candidate->quota.used() != candidate->resident_bytes ||
      peer->quota.used() != peer->resident_bytes)
    fail("steady replay changed resident storage");
  if (verify_only)
    return;
  const auto title = "Resident SDPA Q=" + std::to_string(rows) +
                     " KV=" + std::to_string(keys) +
                     " D=" + std::to_string(depth) +
                     " causal=" + std::to_string(causal);
  nb::Bench bench;
  bench.title(title)
      .batch(1)
      .epochs(104)
      .minEpochIterations(2)
      .minEpochTime(std::chrono::milliseconds(4))
      .performanceCounters(false);
  const auto run_candidate = [&] { candidate->execute(); };
  const auto run_peer = [&] { peer->execute(); };
  if (env("HYLE_BENCH_RIVAL_FIRST").empty())
    (void)bench.compare("HYLE resident", run_candidate,
                        "Original flash lifecycle", run_peer);
  else
    (void)bench.compare("Original flash lifecycle", run_peer, "HYLE resident",
                        run_candidate);
}
} // namespace
int main() {
  const auto qos = hyle::bench::perf::configure_thread_qos();
  const auto affinity =
      hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
  std::string parent_mask = "not-applicable", priority = "not-applicable";
#if defined(_WIN32)
  DWORD_PTR mask = 0, system_mask = 0;
  if (!GetProcessAffinityMask(GetCurrentProcess(), &mask, &system_mask))
    fail("process affinity observation failed");
  char text[40]{};
  std::snprintf(text, sizeof(text), "0x%llx",
                static_cast<unsigned long long>(mask));
  parent_mask = text;
  const auto actual_priority = GetPriorityClass(GetCurrentProcess());
  if (actual_priority == 0)
    fail("process priority observation failed");
  priority = std::to_string(actual_priority);
#endif
  const bool verify_only = !env("HYLE_BENCH_VERIFY_ONLY").empty();
  std::fprintf(stderr,
               "# HYLE_RESIDENT_RUN run_nonce=%s apple_qos=%s qos_ok=%s "
               "affinity_topology=%s caller_cpu=%s affinity_ok=%s "
               "parent_mask=%s priority=%s verify_only=%d\n",
               hyle::bench::perf::run_nonce(), qos.name, qos.ok,
               affinity.topology, affinity.cpu, affinity.ok,
               parent_mask.c_str(), priority.c_str(), verify_only);
  measure(1, 1024, 256, true, verify_only);
  measure(128, 1024, 256, true, verify_only);
  measure(17, 257, 64, false, verify_only);
}
