#include "../common/perf_process.h"
#include "cpu_reference.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/mem/buffer.h"

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <nanobench.h>
#include <string>
#include <vector>

namespace {
#ifndef HYLE_RESIDENT_PLACEMENT
#define HYLE_RESIDENT_PLACEMENT 0
#endif
namespace c = hyle::compute;
namespace m = hyle::mem;
namespace nb = ankerl::nanobench;
[[noreturn]] void fail(const char *message) {
  std::fprintf(stderr, "CPU resident: %s\n", message);
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
float decode(std::uint16_t value) {
  return std::bit_cast<float>(std::uint32_t(value) << 16);
}
std::uint16_t encode(float value) {
  return std::uint16_t(std::bit_cast<std::uint32_t>(value) >> 16);
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
  int rows, keys, capacity, depth;
  bool causal;
  float scale;
  c::TensorDesc query, physical, cache;
  std::array<std::vector<std::uint16_t>, 3> input;
  explicit Data(int q, int k = 1024, int d = 256, bool masked = true)
      : rows(q), keys(k), capacity(k + 13), depth(d), causal(masked),
        scale(1.F / std::sqrt(float(d))) {
    query = take(c::TensorDesc::dense(
        c::DataType::kBFloat16,
        std::array<std::int64_t, 4>{1, heads, rows, depth}));
    physical = take(c::TensorDesc::dense(
        c::DataType::kBFloat16,
        std::array<std::int64_t, 4>{1, kv_heads, capacity, depth}));
    cache = take(c::TensorDesc::strided_bytes(
        c::DataType::kBFloat16,
        std::array<std::int64_t, 4>{1, kv_heads, keys, depth},
        physical.byte_strides()));
    for (int port = 0; port < 3; ++port) {
      input[port].resize(*(port == 0 ? query : physical).element_count());
      for (std::size_t i = 0; i < input[port].size(); ++i)
        input[port][i] =
            encode(float(int(mix(std::uint32_t(i) ^
                                 (0x9e3779b9U * std::uint32_t(port + 1))) %
                             31) -
                         15) *
                   .015625F);
      if (port)
        for (int head = 0; head < kv_heads; ++head)
          for (int row = keys; row < capacity; ++row)
            for (int z = 0; z < depth; ++z)
              input[port][(head * capacity + row) * depth + z] = encode(128.F);
    }
  }
  std::vector<double> oracle() const {
    std::vector<double> result(input[0].size()), weights(keys);
    for (int head = 0; head < heads; ++head)
      for (int row = 0; row < rows; ++row) {
        double sum = 0;
        for (int key = 0; key < keys; ++key) {
          double dot = 0;
          for (int z = 0; z < depth; ++z)
            dot += double(decode(input[0][(head * rows + row) * depth + z])) *
                   decode(input[1][((head / 4) * capacity + key) * depth + z]);
          weights[key] = causal && key > keys - rows + row
                             ? 0. : std::exp(dot * scale);
          sum += weights[key];
        }
        for (int z = 0; z < depth; ++z)
          for (int key = 0; key < keys; ++key)
            result[(head * rows + row) * depth + z] +=
                weights[key] / sum *
                decode(input[2][((head / 4) * capacity + key) * depth + z]);
      }
    return result;
  }
};
struct Harness {
  c::Context context;
  c::Queue queue;
  c::PreparedOperation prepared;
  std::array<c::OperationInvocation, 2> slots;
  std::array<m::Buffer, 3> buffers;
  m::Buffer output;
  std::array<c::ConstTensorView, 3> inputs;
  c::TensorView result;
  unsigned next = 0;
  explicit Harness(const Data &data, bool native) {
    context = native
                  ? take(c::Context::create(
                        c::cpu::device(),
                        take(hyle::bench::resident::cpu_reference_provider())))
                  : take(c::cpu::create_context({.native_threads = 1}));
    queue = take(context.create_queue());
    prepared = take(
        c::prepare(context,
                   c::ScaledDotProductAttention{
                       .layout = c::AttentionLayout::kBatchHeadSequenceDepth,
                       .scale = data.scale,
                       .causal = data.causal},
                   data.query, data.cache, data.cache));
    if (!native && prepared.provider().id != c::ggml_cpu::kProviderId)
      fail("ordinary CPU did not select mature ggml");
    for (auto &slot : slots)
      slot = take(prepared.create_invocation());
    for (int port = 0; port < 3; ++port) {
      buffers[port] = take(m::Buffer::create(m::system_resource(),
                                             data.input[port].size() * 2, 64));
      check(buffers[port].view().write(
          std::as_bytes(std::span(data.input[port]))));
      inputs[port] = take(c::ConstTensorView::create(
          port == 0 ? data.query : data.cache, buffers[port].view()));
    }
    output = take(m::Buffer::create(m::system_resource(),
                                    *data.query.required_bytes(), 64));
    result = take(c::TensorView::create(data.query, output.view()));
    std::fprintf(stderr,
                 "# CPU_RESIDENT_PROVIDER native=%d provider=%.*s version=%.*s "
                 "private_bound_known=%d\n",
                 native, int(prepared.provider().name.size()),
                 prepared.provider().name.data(),
                 int(prepared.provider().version.size()),
                 prepared.provider().version.data(),
                 prepared.invocation_private_bytes().has_value());
  }
  ~Harness() { check(context.close()); }
  void execute() {
    auto completion = take(slots[next++ % slots.size()].submit(
        queue, inputs, std::span(&result, 1)));
    check(completion.wait());
    nb::doNotOptimizeAway(result);
  }
  void verify(const Data &data, std::span<const double> expected) {
    execute();
    std::vector<std::uint16_t> actual(expected.size());
    check(output.view().read(std::as_writable_bytes(std::span(actual))));
    double worst = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
      if (!std::isfinite(decode(actual[i])))
        fail("nonfinite BF16 reference output");
      worst =
          std::max(worst, std::abs(double(decode(actual[i])) - expected[i]));
    }
    if (worst > 8e-3)
      fail("original BF16 oracle tolerance exceeded");
    for (int port : {1, 2}) {
      std::vector<std::uint16_t> after(data.input[port].size());
      check(
          buffers[port].view().read(std::as_writable_bytes(std::span(after))));
      if (after != data.input[port])
        fail("readonly resident cache changed");
    }
    std::fprintf(stderr,
                 "# CPU_RESIDENT_CORRECTNESS max_abs_error=%.9g elements=%zu\n",
                 worst, actual.size());
  }
};
void measure(int rows, int keys, int depth, bool causal, bool verify_only) {
  Data data(rows, keys, depth, causal);
  const auto aa = env("HYLE_BENCH_AA_ROUTE");
  if (!aa.empty() && aa != "native" && aa != "selected")
    fail("unknown CPU A/A route");
  std::unique_ptr<Harness> candidate, peer;
  auto make_candidate = [&] {
    candidate = std::make_unique<Harness>(data, aa == "native");
  };
  auto make_peer = [&] {
    peer = std::make_unique<Harness>(data, aa != "selected");
  };
  if (env("HYLE_BENCH_PEER_ALLOCATION_FIRST").empty()) {
    make_candidate();
    make_peer();
  } else {
    make_peer();
    make_candidate();
  }
  const auto expected = data.oracle();
  candidate->verify(data, expected);
  peer->verify(data, expected);
  for (int i = 0; i < 8; ++i) {
    candidate->execute();
    peer->execute();
  }
  if (verify_only)
    return;
  nb::Bench bench;
  bench.title("CPU resident BF16 Q=" + std::to_string(rows) +
               " KV=" + std::to_string(keys) + " D=" + std::to_string(depth) +
               " causal=" + std::to_string(causal))
      .batch(1)
      .epochs(16)
      .minEpochIterations(2)
      .minEpochTime(std::chrono::milliseconds(4))
      .performanceCounters(false);
  const auto run_candidate = [&] { candidate->execute(); };
  const auto run_peer = [&] { peer->execute(); };
  if (env("HYLE_BENCH_RIVAL_FIRST").empty())
    (void)bench.compare("HYLE CPU resident", run_candidate,
                        "Original CPU flash lifecycle", run_peer);
  else
    (void)bench.compare("Original CPU flash lifecycle", run_peer,
                        "HYLE CPU resident", run_candidate);
}
} // namespace
int main(int, char **argv) {
  const auto qos = hyle::bench::perf::configure_thread_qos();
  const auto affinity =
      hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
  const bool verify_only = !env("HYLE_BENCH_VERIFY_ONLY").empty();
  if (!verify_only && env("HYLE_BENCH_AA_ROUTE").empty() &&
      !env("HYLE_PERF_RUN_NONCE").empty()) {
    std::string ready;
    std::ifstream(std::filesystem::absolute(argv[0]).parent_path() /
                  "g01_cpu_resident_aa.ready") >> ready;
    if (ready != "PASS")
      fail("finite CPU workload A/A has not passed for this build");
  }
  std::string parent_mask = "not-applicable", priority = "not-applicable";
#if defined(_WIN32)
  DWORD_PTR mask = 0, system_mask = 0;
  if (!GetProcessAffinityMask(GetCurrentProcess(), &mask, &system_mask))
    fail("process affinity observation failed");
  char mask_text[40]{};
  std::snprintf(mask_text, sizeof(mask_text), "0x%llx",
                static_cast<unsigned long long>(mask));
  parent_mask = mask_text;
  const auto actual_priority = GetPriorityClass(GetCurrentProcess());
  if (actual_priority == 0)
    fail("process priority observation failed");
  priority = std::to_string(actual_priority);
#endif
  std::fprintf(
      stderr,
      "# HYLE_CPU_RESIDENT_RUN run_nonce=%s apple_qos=%s qos_ok=%s "
      "affinity_topology=%s caller_cpu=%s affinity_ok=%s verify_only=%d "
      "parent_mask=%s priority=%s placement=%d execution_order=%s "
      "allocation_order=%s aa_route=%s paired_mode=nanobench-compare paired_rounds=16\n",
      hyle::bench::perf::run_nonce(), qos.name, qos.ok, affinity.topology,
      affinity.cpu, affinity.ok, verify_only, parent_mask.c_str(),
      priority.c_str(), HYLE_RESIDENT_PLACEMENT,
      env("HYLE_BENCH_RIVAL_FIRST").empty() ? "candidate-first" : "peer-first",
      env("HYLE_BENCH_PEER_ALLOCATION_FIRST").empty() ? "candidate-first" : "peer-first",
      env("HYLE_BENCH_AA_ROUTE").empty() ? "none" : env("HYLE_BENCH_AA_ROUTE").c_str());
  measure(1, 1024, 256, true, verify_only);
  measure(128, 1024, 256, true, verify_only);
  measure(17, 257, 64, false, verify_only);
}
