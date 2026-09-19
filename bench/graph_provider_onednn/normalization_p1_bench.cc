// C1-AA: the same complete public boundary on both sides, including checks,
// mappings, vendor allocations, Graph intermediates and terminal completion.
#include <nanobench.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
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
#include <utility>
#include <vector>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/compute/providers/onednn.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/providers/onednn.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"

#include "../common/perf_process.h"

#if HYLE_NORM_PLACEMENT
extern "C" int hyle_normalization_placement_prefix(int);
#endif

namespace {
namespace nb = ankerl::nanobench;
namespace compute = hyle::compute;
namespace graph = hyle::graph;
using hyle::mem::Buffer;

[[noreturn]] void fail(std::string_view action, std::string_view detail = {}) {
  std::fprintf(stderr, "normalization benchmark %.*s failed: %.*s\n",
               static_cast<int>(action.size()), action.data(),
               static_cast<int>(detail.size()), detail.data());
  std::exit(2);
}

template <class T> T take(hyle::Result<T> result, std::string_view action) {
  if (!result)
    fail(action, result.error().message());
  return std::move(*result);
}

void require(hyle::Status status, std::string_view action) {
  if (!status.ok())
    fail(action, status.message());
}

std::string environment(const char *name) {
#if defined(_WIN32)
  char *value = nullptr;
  std::size_t size = 0;
  if (_dupenv_s(&value, &size, name) != 0)
    fail("read environment", name);
  const std::string result = value == nullptr ? "" : value;
  std::free(value);
  return result;
#else
  const char *value = std::getenv(name);
  return value == nullptr ? "" : value;
#endif
}

compute::TensorDesc description(std::span<const std::int64_t> shape) {
  return take(compute::TensorDesc::dense(compute::DataType::kFloat32, shape),
              "describe tensor");
}

struct Tensor final {
  Buffer buffer;
  compute::TensorView view;

  explicit Tensor(compute::TensorDesc desc)
      : buffer(take(Buffer::create(hyle::mem::system_resource(),
                                   take(desc.required_bytes(), "tensor bytes"),
                                   64),
                    "allocate tensor")),
        view(take(compute::TensorView::create(desc, buffer.view()),
                  "create view")) {}

  void write(std::span<const float> values) {
    require(buffer.view().write(std::as_bytes(values)), "initialize tensor");
  }

  std::vector<float> read(std::size_t count) const {
    std::vector<float> result(count);
    require(buffer.view().read(std::as_writable_bytes(std::span(result))),
            "read result");
    return result;
  }
};

struct Problem final {
  std::size_t rows, width;
  bool rms;
  compute::TensorDesc data_desc, parameter_desc;
  Tensor input, scale, bias, post_scale;
  std::vector<float> values, scales, biases, post_scales;

  Problem(std::size_t r, std::size_t w, bool is_rms)
      : rows(r), width(w), rms(is_rms),
        data_desc(description(std::array{static_cast<std::int64_t>(r),
                                         static_cast<std::int64_t>(w)})),
        parameter_desc(description(std::array{static_cast<std::int64_t>(w)})),
        input(data_desc), scale(parameter_desc), bias(parameter_desc),
        post_scale(parameter_desc), values(r * w), scales(w), biases(w),
        post_scales(w) {
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = static_cast<float>(static_cast<int>(i % 43) - 21) / 32.F;
    for (std::size_t i = 0; i < width; ++i) {
      scales[i] = 0.75F + static_cast<float>(i % 7) / 32.F;
      biases[i] = static_cast<float>(static_cast<int>(i % 5) - 2) / 64.F;
      post_scales[i] = 0.5F + static_cast<float>(i % 3) / 8.F;
    }
    input.write(values);
    scale.write(scales);
    bias.write(biases);
    post_scale.write(post_scales);
  }

  compute::Operation operation() const {
    return rms ? compute::Operation(compute::RmsNorm{.epsilon = 1.e-5F})
               : compute::Operation(compute::LayerNorm{.epsilon = 1.e-5F});
  }

  void verify(std::span<const float> actual, bool fused) const {
    if (actual.size() != values.size())
      fail("oracle result size");
    for (std::size_t row = 0; row < rows; ++row) {
      double mean = 0;
      if (!rms) {
        for (std::size_t col = 0; col < width; ++col)
          mean += values[row * width + col];
        mean /= static_cast<double>(width);
      }
      double squares = 0;
      for (std::size_t col = 0; col < width; ++col) {
        const double centered = values[row * width + col] - mean;
        squares += centered * centered;
      }
      const double inverse =
          1.0 / std::sqrt(squares / static_cast<double>(width) + 1.e-5F);
      for (std::size_t col = 0; col < width; ++col) {
        const std::size_t i = row * width + col;
        double expected = (values[i] - mean) * inverse * scales[col];
        if (!rms)
          expected += biases[col];
        if (fused)
          expected = expected * post_scales[col] + values[i];
        if (!std::isfinite(actual[i]) ||
            std::abs(actual[i] - expected) > 3.e-5 + std::abs(expected) * 3.e-5)
          fail("independent FP64 normalization oracle");
      }
    }
  }
};

class DirectRoute final {
public:
  DirectRoute(Problem &problem, bool vendor)
      : output_(problem.data_desc),
        context_(
            take(compute::Context::create(compute::cpu::device(),
                                          vendor ? compute::onednn::provider()
                                                 : compute::native::provider()),
                 "create Direct context")),
        queue_(take(context_.create_queue(), "create Direct queue")),
        inputs_{problem.input.view.as_const(), problem.scale.view.as_const(),
                problem.bias.view.as_const()},
        input_count_(problem.rms ? 2U : 3U) {
    const std::array outputs{output_.view};
    prepared_ =
        take(compute::prepare(context_, problem.operation(),
                              std::span(inputs_).first(input_count_), outputs),
             "prepare Direct operation");
    if (prepared_.provider().id !=
        (vendor ? compute::onednn::kProviderId : compute::native::kProviderId))
      fail("unexpected Direct provider");
    const auto requirement = prepared_.workspace_requirement();
    if (requirement.size != 0)
      workspace_ = take(Buffer::create(hyle::mem::system_resource(),
                                       requirement.size, requirement.alignment),
                        "allocate Direct workspace");
    invocation_ = take(prepared_.create_invocation(workspace_.view()),
                       "create Direct invocation");
  }

  void run() {
    auto completion =
        take(invocation_.submit(queue_, std::span(inputs_).first(input_count_),
                                output_.view),
             "submit Direct");
    require(completion.wait(), "complete Direct");
    nb::doNotOptimizeAway(output_.buffer.native_handle());
  }
  void verify(const Problem &problem) const {
    problem.verify(output_.read(problem.rows * problem.width), false);
  }

  void observe(const char *workload, const char *role, bool,
               const Problem &problem) const {
    std::fprintf(
        stderr,
        "# HYLE_NORMALIZATION_INSTANCE workload=%s role=%s "
        "provider=%s object=%p input_handle=%p output_handle=%p "
        "workspace_handle=%p workspace_bytes=%zu\n",
        workload, role,
        prepared_.provider().id == compute::onednn::kProviderId ? "onednn"
                                                                : "native",
        static_cast<const void *>(this), problem.input.buffer.native_handle(),
        output_.buffer.native_handle(), workspace_.native_handle(),
        workspace_.size());
  }

private:
  Tensor output_;
  compute::Context context_;
  compute::Queue queue_;
  std::array<compute::ConstTensorView, 3> inputs_;
  std::size_t input_count_;
  compute::PreparedOperation prepared_;
  Buffer workspace_;
  compute::OperationInvocation invocation_;
};

class GraphRoute final {
public:
  GraphRoute(Problem &problem, bool vendor) : output_(problem.data_desc) {
    const std::array providers{graph::onednn::provider(),
                               compute::onednn::provider(),
                               compute::native::provider()};
    context_ = take(compute::Context::create(
                        compute::cpu::device(),
                        {.providers = vendor ? std::span(providers)
                                             : std::span(providers).last(1)}),
                    "create Graph context");
    queue_ = take(context_.create_queue(), "create Graph queue");
    auto builder = take(graph::GraphBuilder::create(), "create Graph builder");
    const auto input = take(builder.input(problem.data_desc), "add input");
    const auto scale =
        take(builder.constant(problem.parameter_desc), "add scale");
    const auto bias =
        problem.rms
            ? graph::ValueId{}
            : take(builder.constant(problem.parameter_desc), "add bias");
    const auto post_scale =
        take(builder.constant(problem.parameter_desc), "add post scale");
    const std::array norm_inputs{input, scale, bias};
    const auto norm_node = take(
        builder.add_node(problem.operation(),
                         std::span(norm_inputs).first(problem.rms ? 2U : 3U)),
        "add normalization");
    const auto normalized = take(builder.result(norm_node), "get norm result");
    const auto multiplied =
        take(builder.add(compute::Binary{compute::BinaryKind::kMultiply},
                         normalized, post_scale),
             "add multiplication");
    const auto result =
        take(builder.add(compute::Binary{compute::BinaryKind::kAdd}, multiplied,
                         input),
             "add residual");
    require(builder.publish(result), "publish Graph result");
    const auto logical = take(std::move(builder).finish(), "finish Graph");
    const std::array constants{
        graph::ConstantBinding{scale, problem.scale.view.as_const()},
        graph::ConstantBinding{post_scale, problem.post_scale.view.as_const()},
        graph::ConstantBinding{bias, problem.bias.view.as_const()}};
    auto compiled = take(
        graph::compile(
            logical, context_,
            {.constants = std::span(constants).first(problem.rms ? 2U : 3U)}),
        "compile Graph");
    const auto first = take(compiled.report.step(0), "inspect Graph prefix");
    const bool split_residual = vendor && problem.rows > 1;
    if (compiled.report.region_count() != (vendor ? 1U : 0U) ||
        compiled.report.step_count() != (vendor ? (split_residual ? 2U : 1U) : 3U) ||
        first.node_count != (vendor ? (split_residual ? 2U : 3U) : 1U) ||
        first.provider.id != (vendor ? graph::onednn::kProviderId
                                     : compute::native::kProviderId) ||
        compiled.report.mixed_providers() != split_residual)
      fail("unexpected Graph provider or fusion");
    invocation_ = take(compiled.executable.create_invocation(),
                       "create Graph invocation");
    require(invocation_.bind(input, problem.input.view.as_const()),
            "bind input");
    require(invocation_.bind(result, output_.view), "bind result");
    const std::array queues{graph::QueueBinding{
        .lane = graph::ExecutionLaneId{0}, .queue = &queue_}};
    require(invocation_.prepare_replay(queues), "prepare Graph replay");
  }

  void run() {
    auto completion = take(invocation_.submit(queue_), "submit Graph");
    require(completion.wait(), "complete Graph");
    nb::doNotOptimizeAway(output_.buffer.native_handle());
  }
  void verify(const Problem &problem) const {
    problem.verify(output_.read(problem.rows * problem.width), true);
  }

  void observe(const char *workload, const char *role, bool vendor,
               const Problem &problem) const {
    // The constructor checked the actual provider/region/step topology.
    // Private Graph backing is deliberately not exposed for instrumentation.
    std::fprintf(stderr,
                 "# HYLE_NORMALIZATION_INSTANCE workload=%s role=%s "
                 "provider=%s object=%p input_handle=%p output_handle=%p "
                 "workspace_handle=opaque workspace_bytes=opaque\n",
                 workload, role, vendor ? "onednn-prefix" : "native",
                 static_cast<const void *>(this),
                 problem.input.buffer.native_handle(),
                 output_.buffer.native_handle());
  }

private:
  Tensor output_;
  compute::Context context_;
  compute::Queue queue_;
  graph::Invocation invocation_;
};

struct ColdTimes {
  std::chrono::steady_clock::time_point wall;
  std::uint64_t thread_cpu_100ns = 0;
  unsigned long cpu = 0;
};

ColdTimes cold_times() {
#if defined(_WIN32)
  FILETIME created{}, exited{}, kernel{}, user{};
  if (!::GetThreadTimes(::GetCurrentThread(), &created, &exited, &kernel,
                        &user))
    fail("observe thread times");
  const auto ticks = [](FILETIME value) {
    return (std::uint64_t{value.dwHighDateTime} << 32) | value.dwLowDateTime;
  };
  return {std::chrono::steady_clock::now(), ticks(kernel) + ticks(user),
          ::GetCurrentProcessorNumber()};
#else
  fail("instance observations require W1-WIN");
#endif
}

void observe_code(const char *workload) {
#if defined(_WIN32)
  std::uintptr_t address = 0;
  std::size_t ranges = 0;
  MEMORY_BASIC_INFORMATION region{};
  while (::VirtualQuery(reinterpret_cast<const void *>(address), &region,
                        sizeof(region)) == sizeof(region)) {
    if (region.State == MEM_COMMIT && region.Type == MEM_PRIVATE &&
        (region.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ |
                           PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)) &&
        !(region.Protect & PAGE_GUARD)) {
      if (++ranges > 256)
        fail("private executable observation budget");
      std::fprintf(stderr,
                   "# HYLE_NORMALIZATION_CODE workload=%s base=%p "
                   "allocation=%p bytes=%zu protection=%lu\n",
                   workload, region.BaseAddress, region.AllocationBase,
                   region.RegionSize, region.Protect);
    }
    const auto base = reinterpret_cast<std::uintptr_t>(region.BaseAddress);
    if (region.RegionSize >
            (std::numeric_limits<std::uintptr_t>::max)() - base ||
        base + region.RegionSize <= address)
      break;
    address = base + region.RegionSize;
  }
  std::fprintf(stderr,
               "# HYLE_NORMALIZATION_PROCESS workload=%s pid=%lu tid=%lu "
               "cpu=%lu executable_base=%p private_executable_ranges=%zu\n",
               workload, ::GetCurrentProcessId(), ::GetCurrentThreadId(),
               ::GetCurrentProcessorNumber(),
               static_cast<void *>(::GetModuleHandleW(nullptr)), ranges);
#else
  (void)workload;
  fail("code observations require W1-WIN");
#endif
}

template <class Route>
void measure(Problem &problem, std::string_view journey, std::string_view aa,
             bool peer_first, bool peer_allocation_first) {
  std::unique_ptr<Route> candidate, peer;
  if (peer_allocation_first) {
    peer = std::make_unique<Route>(problem, aa == "selected");
    candidate = std::make_unique<Route>(problem, aa != "peer");
  } else {
    candidate = std::make_unique<Route>(problem, aa != "peer");
    peer = std::make_unique<Route>(problem, aa == "selected");
  }
  candidate->run();
  peer->run();
  candidate->verify(problem);
  peer->verify(problem);
  const bool observe = !environment("HYLE_BENCH_OBSERVE_INSTANCES").empty();
  if (observe && aa.empty())
    fail("instance observations are A/A-only");
  std::string observed_workload;
  if (observe) {
    observed_workload =
        std::string(journey) + "-" + (problem.rms ? "rms-" : "layer-") +
        std::to_string(problem.rows) + "x" + std::to_string(problem.width);
    candidate->observe(observed_workload.c_str(), "candidate", aa != "peer",
                       problem);
    peer->observe(observed_workload.c_str(), "peer", aa == "selected", problem);
    observe_code(observed_workload.c_str());
  }
  if (!environment("HYLE_BENCH_VERIFY_ONLY").empty())
    return;
  const std::string shape = std::string(journey) + " " +
                            (problem.rms ? "rms" : "layer") +
                            " f32 R=" + std::to_string(problem.rows) +
                            " W=" + std::to_string(problem.width);
  const std::string candidate_name = "HYLE oneDNN " + shape;
  const std::string peer_name = "HYLE Native " + shape;
  nb::Bench bench;
  bench.title("normalization " + shape)
      .relative(true)
      .minEpochTime(std::chrono::milliseconds(8));
  const auto before = observe ? cold_times() : ColdTimes{};
  if (peer_first)
    (void)bench.compare(
        peer_name, [&] { peer->run(); }, candidate_name,
        [&] { candidate->run(); });
  else
    (void)bench.compare(
        candidate_name, [&] { candidate->run(); }, peer_name,
        [&] { peer->run(); });
  if (observe) {
    const auto after = cold_times();
    const auto wall = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          after.wall - before.wall)
                          .count();
    std::fprintf(stderr,
                 "# HYLE_NORMALIZATION_INTERVAL workload=%s wall_ns=%lld "
                 "thread_cpu_100ns=%llu cpu_before=%lu cpu_after=%lu\n",
                 observed_workload.c_str(), static_cast<long long>(wall),
                 static_cast<unsigned long long>(after.thread_cpu_100ns -
                                                 before.thread_cpu_100ns),
                 before.cpu, after.cpu);
  }
}
} // namespace

int main(int argc, char **argv) {
#if HYLE_NORM_PLACEMENT
  nb::doNotOptimizeAway(hyle_normalization_placement_prefix(argc));
#endif
  const auto qos =
      hyle::bench::perf::configure_thread_qos("HYLE_BENCH_APPLE_QOS");
  const auto affinity =
      hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
  const auto aa = environment("HYLE_BENCH_AA_ROUTE");
  if (!aa.empty() && aa != "selected" && aa != "peer")
    fail("invalid A/A route", aa);
  if (environment("HYLE_BENCH_OBSERVE_INSTANCES").empty() &&
      (!environment("ONEDNN_JIT_DUMP").empty() ||
       !environment("DNNL_JIT_DUMP").empty()))
    fail("vendor JIT dump is confined to A/A observations");
  if (aa.empty() && !environment("HYLE_PERF_RUN_NONCE").empty()) {
    if (argc < 1)
      fail("locate workload A/A receipt");
    std::ifstream status(std::filesystem::path(argv[0]).parent_path() /
                         "hyle-normalization-workload-aa.status");
    std::string verdict;
    if (!(status >> verdict) || verdict != "PASS")
      fail("workload A/A must pass before sampling");
  }
  const bool peer_first = !environment("HYLE_BENCH_RIVAL_FIRST").empty();
  const bool peer_allocation_first =
      !environment("HYLE_BENCH_PEER_ALLOCATION_FIRST").empty();
  for (bool rms : {true, false})
    for (const auto shape : {std::array<std::size_t, 2>{1, 2048},
                             {17, 2048},
                             {128, 2048},
                             {17, 257}}) {
      Problem problem(shape[0], shape[1], rms);
      measure<DirectRoute>(problem, "direct", aa, peer_first,
                           peer_allocation_first);
      measure<GraphRoute>(problem, "graph", aa, peer_first,
                          peer_allocation_first);
    }
  std::fprintf(
      stderr,
      "# HYLE_NORMALIZATION_RUN execution_order=%s allocation_order=%s "
      "aa_route=%s paired_mode=nanobench-compare tasks=1 apple_qos=%s "
      "qos_ok=%s affinity_topology=%s caller_cpu=%s affinity_ok=%s "
      "run_nonce=%s\n",
      peer_first ? "peer-first" : "candidate-first",
      peer_allocation_first ? "peer-first" : "candidate-first",
      aa.empty() ? "off" : aa.c_str(), qos.name, qos.ok, affinity.topology,
      affinity.cpu, affinity.ok, hyle::bench::perf::run_nonce());
}
