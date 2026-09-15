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
#include <string_view>
#include <utility>
#include <vector>

#include "../common/perf_process.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/compute/providers/onednn.h"
#include "hyle/compute/tensor.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"

namespace {
namespace nb = ankerl::nanobench;

constexpr std::size_t kAlignment = 64;

struct MatrixShape final {
  std::size_t rows;
  std::size_t inner;
  std::size_t columns;
  bool transposed_weight = false;
};

[[noreturn]] void fail(const char *operation, std::string_view detail = {}) {
  std::fprintf(stderr, "%s failed", operation);
  if (!detail.empty()) {
    std::fprintf(stderr, ": %.*s", static_cast<int>(detail.size()),
                 detail.data());
  }
  std::fputc('\n', stderr);
  std::abort();
}

template <class T> T take(hyle::Result<T> result, const char *operation) {
  if (!result)
    fail(operation, result.error().message());
  return std::move(result).value();
}

void require(hyle::Status status, const char *operation) {
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

enum class AaRoute { kNone, kSelected, kNative };

AaRoute aa_route_requested() {
  const char *value = nullptr;
#if defined(_WIN32)
  char *storage = nullptr;
  std::size_t size = 0;
  if (_dupenv_s(&storage, &size, "HYLE_BENCH_AA_ROUTE") == 0)
    value = storage;
  const auto release = [&] { std::free(storage); };
#else
  value = std::getenv("HYLE_BENCH_AA_ROUTE");
  const auto release = [] {};
#endif
  if (value == nullptr) {
    release();
    return AaRoute::kNone;
  }
  const std::string_view requested(value);
  const AaRoute route =
      requested == "selected"
          ? AaRoute::kSelected
          : (requested == "native" ? AaRoute::kNative : AaRoute::kNone);
  if (route == AaRoute::kNone)
    fail("HYLE_BENCH_AA_ROUTE must be selected or native");
  release();
  return route;
}

hyle::compute::TensorDesc matrix(std::size_t rows, std::size_t columns) {
  const std::array dimensions{static_cast<std::int64_t>(rows),
                              static_cast<std::int64_t>(columns)};
  return take(hyle::compute::TensorDesc::dense(
                  hyle::compute::DataType::kFloat32, dimensions),
              "create matrix description");
}

class MatrixInputs final {
public:
  explicit MatrixInputs(MatrixShape shape) {
    left_ = take(hyle::mem::Buffer::create(
                     hyle::mem::system_resource(),
                     shape.rows * shape.inner * sizeof(float), kAlignment),
                 "allocate left matrix");
    right_ = take(hyle::mem::Buffer::create(
                      hyle::mem::system_resource(),
                      shape.inner * shape.columns * sizeof(float), kAlignment),
                  "allocate right matrix");
    left_mapping_ =
        take(left_.view().map_host(hyle::mem::HostAccess::kReadWrite),
             "map left matrix");
    right_mapping_ =
        take(right_.view().map_host(hyle::mem::HostAccess::kReadWrite),
             "map right matrix");

    auto left_bytes = take(left_mapping_.writable_bytes(), "write left matrix");
    auto right_bytes =
        take(right_mapping_.writable_bytes(), "write right matrix");
    auto *left_data = reinterpret_cast<float *>(left_bytes.data());
    auto *right_data = reinterpret_cast<float *>(right_bytes.data());
    for (std::size_t index = 0; index < shape.rows * shape.inner; ++index) {
      left_data[index] =
          static_cast<float>(static_cast<int>(index % 23) - 11) / 64.0F;
    }
    for (std::size_t index = 0; index < shape.inner * shape.columns; ++index) {
      right_data[index] =
          static_cast<float>(static_cast<int>(index % 29) - 14) / 64.0F;
    }
  }

  hyle::mem::BufferView left_view() noexcept { return left_.view(); }
  hyle::mem::BufferView right_view() noexcept { return right_.view(); }

private:
  hyle::mem::Buffer left_;
  hyle::mem::Buffer right_;
  hyle::mem::HostMapping left_mapping_;
  hyle::mem::HostMapping right_mapping_;
};

class MatrixOutput final {
public:
  explicit MatrixOutput(MatrixShape shape) : size_(shape.rows * shape.columns) {
    output_ = take(hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                             size_ * sizeof(float), kAlignment),
                   "allocate output matrix");
    output_mapping_ =
        take(output_.view().map_host(hyle::mem::HostAccess::kReadWrite),
             "map output matrix");
    auto output_bytes =
        take(output_mapping_.writable_bytes(), "write output matrix");
    output_data_ = reinterpret_cast<float *>(output_bytes.data());
    std::fill_n(output_data_, size_, 0.0F);
  }

  hyle::mem::BufferView view() noexcept { return output_.view(); }
  float *data() noexcept { return output_data_; }
  std::span<const float> values() const noexcept {
    return {output_data_, size_};
  }

private:
  std::size_t size_ = 0;
  hyle::mem::Buffer output_;
  hyle::mem::HostMapping output_mapping_;
  float *output_data_ = nullptr;
};

class MatMulHarness final {
public:
  MatMulHarness(MatrixShape shape, bool selected_route, MatrixInputs &inputs,
                MatrixOutput &output_storage)
      : shape_(shape), output_storage_(&output_storage),
        left_description_(matrix(shape.rows, shape.inner)),
        stored_right_description_(
            matrix(shape.transposed_weight ? shape.columns : shape.inner,
                   shape.transposed_weight ? shape.inner : shape.columns)),
        right_description_(shape.transposed_weight
                               ? take(stored_right_description_.transposed(),
                                      "transpose weight description")
                               : stored_right_description_),
        output_description_(matrix(shape.rows, shape.columns)) {
    if (selected_route) {
      context_ = take(hyle::compute::cpu::create_context(),
                      "create default Compute context");
    } else {
      context_ = take(
          hyle::compute::Context::create(hyle::compute::cpu::device(),
                                         hyle::compute::native::provider()),
          "create explicit Native context");
    }
    queue_ = take(context_.create_queue(), "create Compute queue");
    prepared_ = take(hyle::compute::prepare(
                         context_, hyle::compute::MatMul{}, left_description_,
                         right_description_, output_description_),
                     "prepare Compute MatMul");

    const bool expect_onednn =
        selected_route && !shape.transposed_weight && shape.rows >= 16;
    const auto expected = expect_onednn ? hyle::compute::onednn::kProviderId
                                        : hyle::compute::native::kProviderId;
    if (prepared_.provider().id != expected)
      fail("select expected Compute route");

    const auto requirement = prepared_.workspace_requirement();
    if (requirement.size != 0) {
      workspace_ = take(hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                                  requirement.size,
                                                  requirement.alignment),
                        "allocate Compute workspace");
    }
    invocation_ = take(prepared_.create_invocation(workspace_.view()),
                       "create Compute invocation");
    left_ = take(hyle::compute::ConstTensorView::create(left_description_,
                                                        inputs.left_view()),
                 "create left tensor view");
    right_ = take(hyle::compute::ConstTensorView::create(right_description_,
                                                         inputs.right_view()),
                  "create right tensor view");
    output_ = take(hyle::compute::TensorView::create(output_description_,
                                                     output_storage_->view()),
                   "create output tensor view");
  }

  void run() noexcept {
    auto submitted = invocation_.submit(queue_, left_, right_, output_);
    if (!submitted)
      fail("submit Compute MatMul", submitted.error().message());
    require(submitted->wait(), "wait for Compute MatMul");
    nb::doNotOptimizeAway(
        output_storage_->data()[shape_.columns * (shape_.rows - 1) + 17]);
  }

private:
  MatrixShape shape_;
  MatrixOutput *output_storage_ = nullptr;
  hyle::compute::TensorDesc left_description_;
  hyle::compute::TensorDesc stored_right_description_;
  hyle::compute::TensorDesc right_description_;
  hyle::compute::TensorDesc output_description_;
  hyle::compute::Context context_;
  hyle::compute::Queue queue_;
  hyle::compute::PreparedOperation prepared_;
  hyle::mem::Buffer workspace_;
  hyle::compute::OperationInvocation invocation_;
  hyle::compute::ConstTensorView left_;
  hyle::compute::ConstTensorView right_;
  hyle::compute::TensorView output_;
};

struct HarnessRunner final {
  MatMulHarness *harness = nullptr;

  void operator()() const noexcept { harness->run(); }
};

struct MatMulPair final {
  MatrixInputs inputs;
  MatrixOutput output;
  std::unique_ptr<MatMulHarness> candidate;
  std::unique_ptr<MatMulHarness> peer;

  MatMulPair(MatrixShape shape, bool candidate_selected, bool peer_selected,
             bool peer_allocation_first)
      : inputs(shape), output(shape) {
    if (peer_allocation_first) {
      peer =
          std::make_unique<MatMulHarness>(shape, peer_selected, inputs, output);
      candidate = std::make_unique<MatMulHarness>(shape, candidate_selected,
                                                  inputs, output);
    } else {
      candidate = std::make_unique<MatMulHarness>(shape, candidate_selected,
                                                  inputs, output);
      peer =
          std::make_unique<MatMulHarness>(shape, peer_selected, inputs, output);
    }
  }
};

void verify_equal(std::span<const float> candidate,
                  std::span<const float> peer) {
  if (candidate.size() != peer.size())
    fail("compare Compute output sizes");
  for (std::size_t index = 0; index < candidate.size(); ++index) {
    const float tolerance = std::abs(peer[index]) * 5.0e-4F + 5.0e-3F;
    if (std::abs(candidate[index] - peer[index]) > tolerance)
      fail("compare Compute outputs");
  }
}

template <class Candidate, class Peer>
void run_pair(nb::Bench &bench, bool peer_first, const char *candidate_name,
              const char *peer_name, Candidate &&candidate, Peer &&peer) {
  if (peer_first) {
    (void)bench.compare(peer_name, std::forward<Peer>(peer), candidate_name,
                        std::forward<Candidate>(candidate));
  } else {
    (void)bench.compare(candidate_name, std::forward<Candidate>(candidate),
                        peer_name, std::forward<Peer>(peer));
  }
}

void measure_matmul(MatrixShape shape, const char *title,
                    const char *candidate_name, const char *peer_name,
                    bool peer_first, bool peer_allocation_first,
                    AaRoute aa_route) {
  const bool candidate_selected = aa_route != AaRoute::kNative;
  const bool peer_selected = aa_route == AaRoute::kSelected;
  MatMulPair pair(shape, candidate_selected, peer_selected,
                  peer_allocation_first);

  pair.candidate->run();
  const std::vector<float> candidate_output(pair.output.values().begin(),
                                            pair.output.values().end());
  pair.peer->run();
  verify_equal(candidate_output, pair.output.values());

  // Alternate complete warm operations so neither row inherits a one-sided
  // code/data/cache phase before the paired measurement starts.
  for (std::size_t warmup = 0; warmup < 8; ++warmup) {
    if ((warmup & 1U) == 0) {
      pair.candidate->run();
      pair.peer->run();
    } else {
      pair.peer->run();
      pair.candidate->run();
    }
  }

  nb::Bench bench;
  bench.title(title)
      .batch(1)
      .epochs(104)
      .minEpochIterations(1)
      .minEpochTime(std::chrono::milliseconds(4))
      .performanceCounters(false);
  run_pair(bench, peer_first, candidate_name, peer_name,
           HarnessRunner{pair.candidate.get()}, HarnessRunner{pair.peer.get()});
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
  const AaRoute aa_route = aa_route_requested();
  std::fprintf(
      stderr,
      "# HYLE_COMPUTE_ROUTE_RUN execution_order=%s allocation_order=%s "
      "aa_route=%s paired_mode=nanobench-compare provider_version=%.*s "
      "apple_qos=%s qos_ok=%s affinity_topology=%s caller_cpu=%s "
      "affinity_ok=%s run_nonce=%s\n",
      peer_first ? "peer-first" : "candidate-first",
      peer_allocation_first ? "peer-first" : "candidate-first",
      aa_route == AaRoute::kSelected
          ? "selected"
          : (aa_route == AaRoute::kNative ? "native" : "none"),
      static_cast<int>(hyle::compute::onednn::version().size()),
      hyle::compute::onednn::version().data(), qos.name, qos.ok,
      affinity.topology, affinity.cpu, affinity.ok,
      hyle::bench::perf::run_nonce());

  measure_matmul({1, 1024, 1024, false},
                 "compute route row-major MatMul M=1 K=1024 N=1024",
                 "HYLE selected MatMul M=1 K=1024 N=1024",
                 "HYLE native MatMul M=1 K=1024 N=1024", peer_first,
                 peer_allocation_first, aa_route);
  measure_matmul({4, 1024, 1024, false},
                 "compute route row-major MatMul M=4 K=1024 N=1024",
                 "HYLE selected MatMul M=4 K=1024 N=1024",
                 "HYLE native MatMul M=4 K=1024 N=1024", peer_first,
                 peer_allocation_first, aa_route);
  measure_matmul({32, 512, 1024, false},
                 "compute route row-major MatMul M=32 K=512 N=1024",
                 "HYLE selected MatMul M=32 K=512 N=1024",
                 "HYLE native MatMul M=32 K=512 N=1024", peer_first,
                 peer_allocation_first, aa_route);
  measure_matmul({1, 1024, 1024, true},
                 "compute route transposed-weight MatMul M=1 K=1024 N=1024",
                 "HYLE selected transposed MatMul M=1 K=1024 N=1024",
                 "HYLE native transposed MatMul M=1 K=1024 N=1024", peer_first,
                 peer_allocation_first, aa_route);
}
