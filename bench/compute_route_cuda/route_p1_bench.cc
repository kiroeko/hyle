#include <nanobench.h>

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

#include <cuda_fp16.h>

#include "../common/perf_process.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/cuda.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cublas.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/mem/buffer.h"
#include "reference/reference_provider.h"
#if HYLE_CUDA_GRAPH_BENCH
#include "hyle/graph/cuda.h"
#endif

// See cuda-r4-performance-contract.md. A pair has identical public lifecycle
// services and complete math; only its packed precise SwiGLU kernel differs.
namespace {
namespace nb = ankerl::nanobench;
namespace compute = hyle::compute;
namespace cuda = compute::cuda;
namespace mem = hyle::mem;
#if HYLE_CUDA_GRAPH_BENCH
namespace graph = hyle::graph;
#endif

constexpr std::size_t kInner = 4096;
constexpr std::size_t kHidden = 11008;
constexpr std::size_t kOutput = 4096;
constexpr std::size_t kGroups = kInner / 128;
constexpr std::size_t kSlots = 2;

[[noreturn]] void fail(const char *operation, std::string_view detail = {}) {
  std::fprintf(stderr, "%s failed: %.*s\n", operation,
               static_cast<int>(detail.size()), detail.data());
  std::exit(EXIT_FAILURE);
}

template <class T> T take(hyle::Result<T> value, const char *operation) {
  if (!value)
    fail(operation, value.error().message());
  return std::move(value).value();
}

void require(hyle::Status status, const char *operation) {
  if (!status.ok())
    fail(operation, status.message());
}

std::string environment(const char *name) {
#if defined(_WIN32)
  char *value = nullptr;
  std::size_t size = 0;
  if (_dupenv_s(&value, &size, name) != 0)
    fail("read benchmark environment");
  const std::string result = value == nullptr ? "" : value;
  std::free(value);
  return result;
#else
  const char *value = std::getenv(name);
  return value == nullptr ? "" : value;
#endif
}

float half_round(float value) { return __half2float(__float2half_rn(value)); }

// Integer mixing prevents the fixture from becoming a repeated low-rank
// matrix, while preserving byte-identical data across order and placement.
std::uint32_t mix(std::uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}

compute::TensorDesc matrix(std::size_t rows, std::size_t columns) {
  return take(compute::TensorDesc::dense(
                  compute::DataType::kFloat16,
                  std::array{static_cast<std::int64_t>(rows),
                             static_cast<std::int64_t>(columns)}),
              "describe FP16 matrix");
}

struct Tensor final {
  mem::Buffer storage;
  compute::TensorView view;

  void create(mem::MemoryResourceRef resource, std::size_t rows,
              std::size_t columns) {
    const auto desc = matrix(rows, columns);
    storage =
        take(mem::Buffer::create(
                 resource, take(desc.required_bytes(), "tensor bytes"), 256),
             "allocate CUDA tensor");
    view = take(compute::TensorView::create(desc, storage.view()),
                "create CUDA tensor view");
  }

  void write(std::span<const __half> values) {
    require(storage.view().write(std::as_bytes(values)), "upload CUDA input");
  }

  std::vector<float> read() {
    std::vector<__half> values(
        take(view.description().element_count(), "output element count"));
    require(storage.view().read(std::as_writable_bytes(std::span(values))),
            "download CUDA result");
    std::vector<float> output(values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
      output[i] = __half2float(values[i]);
    return output;
  }
};

struct DataSet final {
  compute::Context context;
  cuda::MemoryResource resource;
  cuda::QuantizedWeights quantized;
  Tensor down;
  std::vector<std::int8_t> q;
  std::vector<float> scales;
  std::vector<__half> down_values;
  std::array<std::vector<__half>, kSlots> inputs;
  std::size_t rows;

  explicit DataSet(std::size_t row_count) : rows(row_count) {
    context = take(cuda::create_context(), "create constant owner");
    resource =
        take(cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice),
             "create constant resource");
    q.resize(2 * kHidden * kInner);
    scales.resize(2 * kHidden * kGroups);
    for (std::size_t i = 0; i < q.size(); ++i)
      q[i] = static_cast<std::int8_t>(
          static_cast<int>(mix(static_cast<std::uint32_t>(i) + 17U) % 16U) - 8);
    for (std::size_t i = 0; i < scales.size(); ++i)
      scales[i] = static_cast<float>(
                      1U + mix(static_cast<std::uint32_t>(i) + 23U) % 4U) /
                  256.0F;
    quantized =
        take(cuda::QuantizedWeights::pack(context, resource.ref(), 2 * kHidden,
                                          kInner, q, scales),
             "pack resident Q4 constants");
    if (quantized.resident_bytes() != q.size() / 2 + scales.size() * 2)
      fail("Q4 resident format");
    down.create(resource.ref(), kHidden, kOutput);
    down_values.resize(kHidden * kOutput);
    for (std::size_t i = 0; i < down_values.size(); ++i)
      down_values[i] = __float2half_rn(
          static_cast<float>(
              static_cast<int>(mix(static_cast<std::uint32_t>(i) + 31U) % 17U) -
              8) /
          1024.0F);
    down.write(down_values);
    for (std::size_t slot = 0; slot < kSlots; ++slot) {
      inputs[slot].resize(rows * kInner);
      for (std::size_t i = 0; i < inputs[slot].size(); ++i)
        inputs[slot][i] = __float2half_rn(
            static_cast<float>(
                static_cast<int>(
                    mix(static_cast<std::uint32_t>(i + slot * 100003) + 47U) %
                    33U) -
                16) /
            32.0F);
    }
  }

  // Independent full-chain oracle, only in the explicit qualification pass.
  // It uses mathematical unpacked weights, double GEMM accumulation and FP16
  // stage boundaries; neither GPU implementation supplies reference values.
  std::vector<float> oracle(std::size_t slot) const {
    std::vector<float> projected(rows * 2 * kHidden);
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t col = 0; col < 2 * kHidden; ++col) {
        double sum = 0;
        for (std::size_t k = 0; k < kInner; ++k) {
          const float weight =
              half_round(static_cast<float>(q[col * kInner + k]) *
                         half_round(scales[col * kGroups + k / 128]));
          sum += static_cast<double>(
                     __half2float(inputs[slot][row * kInner + k])) *
                 weight;
        }
        projected[row * 2 * kHidden + col] =
            half_round(static_cast<float>(sum));
      }
    std::vector<float> activated(rows * kHidden);
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t k = 0; k < kHidden; ++k) {
        const float gate = projected[row * 2 * kHidden + k];
        const float up = projected[row * 2 * kHidden + kHidden + k];
        activated[row * kHidden + k] =
            half_round(gate / (1.0F + std::exp(-gate)) * up);
      }
    std::vector<float> output(rows * kOutput);
    // Each block's inner loop reads contiguous down weights, avoiding a huge
    // stride in the cold oracle without changing its double accumulation.
    constexpr std::size_t kBlock = 64;
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t col = 0; col < kOutput; col += kBlock) {
        std::array<double, kBlock> sums{};
        for (std::size_t k = 0; k < kHidden; ++k)
          for (std::size_t j = 0; j < kBlock; ++j)
            sums[j] += static_cast<double>(activated[row * kHidden + k]) *
                       __half2float(down_values[k * kOutput + col + j]);
        for (std::size_t j = 0; j < kBlock; ++j)
          output[row * kOutput + col + j] =
              half_round(static_cast<float>(sums[j]));
      }
    return output;
  }
};

class Harness final {
public:
  Harness(DataSet &data, bool reference) : data_(&data) {
    if (reference) {
      const std::array providers{compute::cublas::provider(),
                                 compute::marlin::provider(),
                                 compute::sglang_reference::provider()};
      context_ = take(
          compute::Context::create(take(cuda::device(0), "select CUDA device"),
                                   {.providers = providers}),
          "create reference composition");
    } else {
      context_ = take(cuda::create_context(), "create ordinary CUDA route");
    }
    // One borrowed device resource describes Graph's explicit storage domain.
    // The two contexts still have independent buffers, queues and run states.
    resource_ = data.resource.ref();
    queue_ = take(context_.create_queue(), "create CUDA queue");
    require(queue_.reserve_completions(3), "reserve real Completions");
    for (std::size_t i = 0; i < kSlots; ++i) {
      slots_[i].input.create(resource_, data.rows, kInner);
      slots_[i].input.write(data.inputs[i]);
      slots_[i].output.create(resource_, data.rows, kOutput);
    }
#if HYLE_CUDA_GRAPH_BENCH
    auto builder = take(graph::GraphBuilder::create(), "create FFN builder");
    const auto x =
        take(builder.input(matrix(data.rows, kInner)), "graph input");
    const auto w =
        take(builder.constant(data.quantized.weights().description()),
             "graph Q4 constant");
    const auto s = take(builder.constant(data.quantized.scales().description()),
                        "graph scale constant");
    const auto d = take(builder.constant(data.down.view.description()),
                        "graph down constant");
    const auto gate_up =
        take(builder.add(cuda::QuantizedWeights::operation(), x, w, s),
             "graph gate/up");
    const auto activated = take(
        builder.add(compute::GatedActivation{}, gate_up), "graph activation");
    const auto y =
        take(builder.add(compute::MatMul{}, activated, d), "graph down");
    require(builder.publish(y), "publish FFN output");
    auto logical = take(std::move(builder).finish(), "finish FFN graph");
    const std::array constants{
        graph::ConstantBinding{w, data.quantized.weights()},
        graph::ConstantBinding{s, data.quantized.scales()},
        graph::ConstantBinding{d, data.down.view.as_const()}};
    compiled_ = take(graph::cuda::compile(logical, context_, resource_,
                                          {.constants = constants}),
                     "compile whole FFN");
    for (std::size_t i = 0; i < kSlots; ++i) {
      graph_slots_[i] =
          take(compiled_.executable.create_invocation(), "create graph slot");
      require(graph_slots_[i].bind(x, slots_[i].input.view.as_const()),
              "bind graph input");
      require(graph_slots_[i].bind(y, slots_[i].output.view),
              "bind graph output");
      require(graph::cuda::prepare_replay(
                  graph_slots_[i], queue_,
                  {.mode = graph::ReplayMode::kRequireCapture,
                   .host_wait = graph::HostWaitPolicy::kReject}),
              "capture complete FFN");
      if (graph_slots_[i].replay_route() != graph::ReplayRoute::kCaptured)
        fail("require actual native captured route");
    }
#else
    for (auto &slot : slots_) {
      slot.projected.create(resource_, data.rows, 2 * kHidden);
      slot.activated.create(resource_, data.rows, kHidden);
    }
    const std::array first_inputs{slots_[0].input.view.as_const(),
                                  data.quantized.weights(),
                                  data.quantized.scales()};
    prepared_[0] = take(
        compute::prepare(
            context_, compute::Operation(cuda::QuantizedWeights::operation()),
            first_inputs, std::array{slots_[0].projected.view}),
        "prepare Q4 gate/up");
    prepared_[1] = take(compute::prepare(context_, compute::GatedActivation{},
                                         slots_[0].projected.view.description(),
                                         resource_),
                        "prepare precise activation");
    // Shared constants have their own resource owner. Bind the actual views so
    // the prepared resource contract describes both owners exactly.
    prepared_[2] =
        take(compute::prepare(context_, compute::Operation(compute::MatMul{}),
                              std::array{slots_[0].activated.view.as_const(),
                                         data.down.view.as_const()},
                              std::array{slots_[0].output.view}),
             "prepare dense down");
    if (prepared_[0].provider().id != compute::marlin::kProviderId ||
        prepared_[1].provider().id !=
            (reference ? compute::sglang_reference::kProviderId
                       : compute::cuda_native::kProviderId) ||
        prepared_[2].provider().id != compute::cublas::kProviderId)
      fail("cold Provider binding");
    for (auto &slot : slots_)
      for (std::size_t i = 0; i < prepared_.size(); ++i) {
        const auto needed = prepared_[i].workspace_requirement();
        slot.workspace[i] =
            take(mem::Buffer::create(resource_, needed.size, needed.alignment),
                 "allocate operation workspace");
        slot.invocations[i] =
            take(prepared_[i].create_invocation(slot.workspace[i].view()),
                 "create operation slot");
      }
#endif
  }

  void run_slot(std::size_t index) {
#if HYLE_CUDA_GRAPH_BENCH
    auto done = take(graph_slots_[index].submit(queue_), "submit captured FFN");
    require(done.wait(), "complete captured FFN");
#else
    auto &slot = slots_[index];
    const std::array inputs{slot.input.view.as_const(),
                            data_->quantized.weights(),
                            data_->quantized.scales()};
    auto first = take(slot.invocations[0].submit(
                          queue_, inputs, std::array{slot.projected.view}),
                      "submit gate/up");
    auto middle =
        take(slot.invocations[1].submit(queue_, slot.projected.view.as_const(),
                                        slot.activated.view),
             "submit activation");
    auto last = take(slot.invocations[2].submit(
                         queue_, slot.activated.view.as_const(),
                         data_->down.view.as_const(), slot.output.view),
                     "submit down");
    require(last.wait(), "complete down");
    require(middle.wait(), "complete activation");
    require(first.wait(), "complete gate/up");
#endif
  }

  void run() { run_slot((next_slot_++) % kSlots); }
  std::vector<float> output(std::size_t slot) {
    return slots_[slot].output.read();
  }

private:
  struct Slot final {
    Tensor input, output;
#if !HYLE_CUDA_GRAPH_BENCH
    Tensor projected, activated;
    std::array<mem::Buffer, 3> workspace;
    std::array<compute::OperationInvocation, 3> invocations;
#endif
  };
  DataSet *data_;
  compute::Context context_;
  mem::MemoryResourceRef resource_;
  compute::Queue queue_;
  std::array<Slot, kSlots> slots_;
#if HYLE_CUDA_GRAPH_BENCH
  graph::CompiledGraph compiled_;
  std::array<graph::Invocation, kSlots> graph_slots_;
#else
  std::array<compute::PreparedOperation, 3> prepared_;
#endif
  std::size_t next_slot_ = 0;
};

void verify(std::span<const float> actual, std::span<const float> expected,
            bool oracle) {
  if (actual.size() != expected.size())
    fail("compare complete FFN output sizes");
  float max_error = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float error = std::abs(actual[i] - expected[i]);
    const float tolerance = oracle ? 0.002F + 0.004F * std::abs(expected[i])
                                   : 0.0005F + 0.001F * std::abs(expected[i]);
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
        error > tolerance) {
      std::fprintf(stderr,
                   "index=%zu actual=%.9g expected=%.9g tolerance=%.9g\n", i,
                   actual[i], expected[i], tolerance);
      fail(oracle ? "independent double FFN oracle"
                  : "paired complete FFN correctness");
    }
    if (error > max_error)
      max_error = error;
  }
  std::fprintf(stderr,
               "# FFN_CORRECTNESS elements=%zu oracle=%d max_abs_error=%.9g\n",
               actual.size(), oracle ? 1 : 0, max_error);
}

struct Runner final {
  Harness *harness;
  void operator()() const { harness->run(); }
};

void measure(std::size_t rows, bool peer_first, bool allocation_first,
             std::string_view aa, bool verify_only) {
  DataSet data(rows);
  std::unique_ptr<Harness> candidate, peer;
  const bool candidate_reference = aa == "native";
  const bool peer_reference = aa != "selected";
  const auto cold_start = std::chrono::steady_clock::now();
  if (allocation_first) {
    peer = std::make_unique<Harness>(data, peer_reference);
    candidate = std::make_unique<Harness>(data, candidate_reference);
  } else {
    candidate = std::make_unique<Harness>(data, candidate_reference);
    peer = std::make_unique<Harness>(data, peer_reference);
  }
  const auto cold_end = std::chrono::steady_clock::now();
  std::fprintf(
      stderr,
      "# FFN_COLD rows=%zu pair_prepare_ms=%.3f q4_resident_bytes=%zu "
      "slots=2\n",
      rows,
      std::chrono::duration<double, std::milli>(cold_end - cold_start).count(),
      data.quantized.resident_bytes());
  std::vector<float> previous;
  for (std::size_t i = 0; i < kSlots; ++i) {
    candidate->run_slot(i);
    peer->run_slot(i);
    auto result = candidate->output(i);
    verify(result, peer->output(i), false);
    if (i != 0 && result == previous)
      fail("different resident inputs must change FFN output");
    previous = result;
    if (verify_only) {
      const auto expected = data.oracle(i);
      verify(result, expected, true);
      verify(peer->output(i), expected, true);
    }
  }
  if (verify_only)
    return;
  for (std::size_t i = 0; i < 8; ++i) {
    if ((i & 1U) == 0) {
      candidate->run();
      peer->run();
    } else {
      peer->run();
      candidate->run();
    }
  }
  const std::string shape =
      " M=" + std::to_string(rows) + " K=4096 H=11008 N=4096";
  const std::string kind =
      HYLE_CUDA_GRAPH_BENCH ? "Graph captured FFN" : "Direct FFN";
  const std::string candidate_name = "HYLE CUDA " + kind + shape;
  const std::string peer_name = "SGLang precise " + kind + shape;
  nb::Bench bench;
  bench.title(kind + shape)
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

int main() {
  const auto affinity =
      hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
  const bool peer_first = !environment("HYLE_BENCH_RIVAL_FIRST").empty();
  const bool allocation_first =
      !environment("HYLE_BENCH_PEER_ALLOCATION_FIRST").empty();
  const bool verify_only = !environment("HYLE_BENCH_VERIFY_ONLY").empty();
  const auto aa = environment("HYLE_BENCH_AA_ROUTE");
  if (!aa.empty() && aa != "selected" && aa != "native")
    fail("HYLE_BENCH_AA_ROUTE must be selected or native");
  int driver = 0, runtime = 0;
  require(cuda::status(cudaDriverGetVersion(&driver)),
          "read CUDA driver API version");
  require(cuda::status(cudaRuntimeGetVersion(&runtime)),
          "read CUDA runtime version");
  cudaDeviceProp properties{};
  require(cuda::status(cudaGetDeviceProperties(&properties, 0)),
          "read CUDA device facts");
  std::fprintf(
      stderr,
      "# HYLE_CUDA_FFN_RUN mode=%s execution_order=%s allocation_order=%s "
      "aa_route=%s paired_mode=nanobench-compare gpu=%s sm=%d%d "
      "driver_api=%d runtime=%d affinity_topology=%s caller_cpu=%s "
      "affinity_ok=%s "
      "run_nonce=%s verify_only=%d\n",
      HYLE_CUDA_GRAPH_BENCH ? "captured" : "direct",
      peer_first ? "peer-first" : "candidate-first",
      allocation_first ? "peer-first" : "candidate-first",
      aa.empty() ? "none" : aa.c_str(), properties.name, properties.major,
      properties.minor, driver, runtime, affinity.topology, affinity.cpu,
      affinity.ok, hyle::bench::perf::run_nonce(), verify_only ? 1 : 0);
  measure(1, peer_first, allocation_first, aa, verify_only);
  measure(16, peer_first, allocation_first, aa, verify_only);
}
