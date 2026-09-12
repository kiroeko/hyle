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

#include "oneapi/dnnl/dnnl_ukernel.h"

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/onednn.h"
#include "hyle/compute/tensor.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"

#include "../common/perf_process.h"

#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
#error "The Compute oneDNN P1 peer requires the locked BRGEMM ukernel API"
#endif

namespace {
namespace nb = ankerl::nanobench;

constexpr std::size_t kRows = 32;
constexpr std::size_t kInner = 4096;
constexpr std::size_t kColumns = 4096;
constexpr std::size_t kAlignment = 64;
constexpr char kTitle[] = "oneDNN BRGEMM provider wrapper M=32 K=4096 N=4096";
constexpr char kCandidateName[] = "HYLE oneDNN provider M=32 K=4096 N=4096";
constexpr char kPeerName[] = "direct oneDNN BRGEMM M=32 K=4096 N=4096";

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

void require_dnnl(dnnl_status_t status, const char *operation) {
  if (status != dnnl_success) {
    std::fprintf(stderr, "%s failed: oneDNN status=%d\n", operation,
                 static_cast<int>(status));
    std::abort();
  }
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

hyle::compute::TensorDesc matrix(std::size_t rows, std::size_t columns) {
  const std::array dimensions{static_cast<std::int64_t>(rows),
                              static_cast<std::int64_t>(columns)};
  return take(hyle::compute::TensorDesc::dense(
                  hyle::compute::DataType::kFloat32, dimensions),
              "create matrix description");
}

class MatrixStorage final {
public:
  MatrixStorage()
      : left_(take(hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                             kRows * kInner * sizeof(float),
                                             kAlignment),
                   "allocate left matrix")),
        right_(take(hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                              kInner * kColumns * sizeof(float),
                                              kAlignment),
                    "allocate right matrix")),
        output_(take(hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                               kRows * kColumns * sizeof(float),
                                               kAlignment),
                     "allocate output matrix")),
        left_mapping_(
            take(left_.view().map_host(hyle::mem::HostAccess::kReadWrite),
                 "map left matrix")),
        right_mapping_(
            take(right_.view().map_host(hyle::mem::HostAccess::kReadWrite),
                 "map right matrix")),
        output_mapping_(
            take(output_.view().map_host(hyle::mem::HostAccess::kReadWrite),
                 "map output matrix")) {
    auto left_bytes = take(left_mapping_.writable_bytes(), "write left matrix");
    auto right_bytes =
        take(right_mapping_.writable_bytes(), "write right matrix");
    auto output_bytes =
        take(output_mapping_.writable_bytes(), "write output matrix");
    left_data_ = reinterpret_cast<float *>(left_bytes.data());
    right_data_ = reinterpret_cast<float *>(right_bytes.data());
    output_data_ = reinterpret_cast<float *>(output_bytes.data());

    for (std::size_t index = 0; index < kRows * kInner; ++index) {
      left_data_[index] =
          static_cast<float>(static_cast<int>(index % 29) - 14) / 29.0F;
    }
    for (std::size_t index = 0; index < kInner * kColumns; ++index) {
      right_data_[index] =
          static_cast<float>(static_cast<int>(index % 31) - 15) / 31.0F;
    }
    std::fill_n(output_data_, kRows * kColumns, 0.0F);
  }

  MatrixStorage(const MatrixStorage &) = delete;
  MatrixStorage &operator=(const MatrixStorage &) = delete;

  hyle::mem::BufferView left_view() noexcept { return left_.view(); }
  hyle::mem::BufferView right_view() noexcept { return right_.view(); }
  hyle::mem::BufferView output_view() noexcept { return output_.view(); }
  const float *left_data() const noexcept { return left_data_; }
  const float *right_data() const noexcept { return right_data_; }
  float *output_data() noexcept { return output_data_; }
  std::span<const float> output() const noexcept {
    return {output_data_, kRows * kColumns};
  }

private:
  // Mappings are declared after their Buffer owners so reverse destruction
  // closes every mapping before releasing its allocation.
  hyle::mem::Buffer left_;
  hyle::mem::Buffer right_;
  hyle::mem::Buffer output_;
  hyle::mem::HostMapping left_mapping_;
  hyle::mem::HostMapping right_mapping_;
  hyle::mem::HostMapping output_mapping_;
  float *left_data_ = nullptr;
  float *right_data_ = nullptr;
  float *output_data_ = nullptr;
};

class HyleHarness final {
public:
  HyleHarness()
      : left_description_(matrix(kRows, kInner)),
        right_description_(matrix(kInner, kColumns)),
        output_description_(matrix(kRows, kColumns)) {
    context_ =
        take(hyle::compute::Context::create(hyle::compute::cpu::device(),
                                            hyle::compute::onednn::provider()),
             "create HYLE Compute context");
    queue_ = take(context_.create_queue(), "create HYLE Compute queue");
    prepared_ = take(hyle::compute::prepare(
                         context_, hyle::compute::MatMul{}, left_description_,
                         right_description_, output_description_),
                     "prepare HYLE oneDNN MatMul");
    if (prepared_.provider().id != hyle::compute::onednn::kProviderId) {
      fail("select HYLE oneDNN MatMul route");
    }
    const auto requirement = prepared_.workspace_requirement();
    if (requirement.size == 0 || requirement.alignment != kAlignment)
      fail("observe HYLE oneDNN workspace contract");
    workspace_ =
        take(hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                       requirement.size, requirement.alignment),
             "allocate HYLE oneDNN workspace");
    invocation_ = take(prepared_.create_invocation(workspace_.view()),
                       "create HYLE oneDNN invocation");
    left_ = take(hyle::compute::ConstTensorView::create(left_description_,
                                                        storage_.left_view()),
                 "create HYLE left view");
    right_ = take(hyle::compute::ConstTensorView::create(right_description_,
                                                         storage_.right_view()),
                  "create HYLE right view");
    output_ = take(hyle::compute::TensorView::create(output_description_,
                                                     storage_.output_view()),
                   "create HYLE output view");
  }

  void run() noexcept {
    auto submitted = invocation_.submit(queue_, left_, right_, output_);
    if (!submitted)
      fail("submit HYLE oneDNN MatMul", submitted.error().message());
    require(submitted->wait(), "wait for HYLE oneDNN MatMul");
    nb::doNotOptimizeAway(storage_.output_data()[kColumns + 17]);
  }

  std::span<const float> output() const noexcept { return storage_.output(); }

private:
  MatrixStorage storage_;
  hyle::compute::TensorDesc left_description_;
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

class DirectOneDnnHarness final {
public:
  DirectOneDnnHarness() {
    require_dnnl(dnnl_brgemm_create(&kernel_, static_cast<dnnl_dim_t>(kRows),
                                    static_cast<dnnl_dim_t>(kColumns),
                                    static_cast<dnnl_dim_t>(kInner), 1,
                                    static_cast<dnnl_dim_t>(kInner),
                                    static_cast<dnnl_dim_t>(kColumns),
                                    static_cast<dnnl_dim_t>(kColumns), dnnl_f32,
                                    dnnl_f32, dnnl_f32),
                 "create direct oneDNN BRGEMM");
    require_dnnl(dnnl_brgemm_set_add_C(kernel_, 0),
                 "set direct oneDNN overwrite semantics");
    require_dnnl(dnnl_brgemm_finalize(kernel_),
                 "finalize direct oneDNN BRGEMM");
    require_dnnl(dnnl_brgemm_generate(kernel_),
                 "generate direct oneDNN BRGEMM");
    std::size_t scratch_size = 0;
    require_dnnl(dnnl_brgemm_get_scratchpad_size(kernel_, &scratch_size),
                 "query direct oneDNN scratch size");
    scratch_ = take(hyle::mem::Buffer::create(
                        hyle::mem::system_resource(),
                        (std::max)(scratch_size, std::size_t{1}), kAlignment),
                    "allocate direct oneDNN scratch");
    scratch_mapping_ =
        take(scratch_.view().map_host(hyle::mem::HostAccess::kReadWrite),
             "map direct oneDNN scratch");
    auto bytes =
        take(scratch_mapping_.writable_bytes(), "write direct oneDNN scratch");
    scratch_data_ = bytes.data();
  }

  DirectOneDnnHarness(const DirectOneDnnHarness &) = delete;
  DirectOneDnnHarness &operator=(const DirectOneDnnHarness &) = delete;

  ~DirectOneDnnHarness() {
    if (kernel_ != nullptr)
      require_dnnl(dnnl_brgemm_destroy(kernel_),
                   "destroy direct oneDNN BRGEMM");
  }

  void run() noexcept {
    const std::array<dnnl_dim_t, 2> offsets{0, 0};
    require_dnnl(dnnl_brgemm_set_hw_context(kernel_),
                 "set direct oneDNN hardware context");
    const dnnl_status_t executed = dnnl_brgemm_execute(
        kernel_, storage_.left_data(), storage_.right_data(), offsets.data(),
        storage_.output_data(), scratch_data_);
    const dnnl_status_t released = dnnl_brgemm_release_hw_context();
    require_dnnl(executed, "execute direct oneDNN BRGEMM");
    require_dnnl(released, "release direct oneDNN hardware context");
    nb::doNotOptimizeAway(storage_.output_data()[kColumns + 17]);
  }

  std::span<const float> output() const noexcept { return storage_.output(); }

private:
  MatrixStorage storage_;
  dnnl_brgemm_t kernel_ = nullptr;
  hyle::mem::Buffer scratch_;
  hyle::mem::HostMapping scratch_mapping_;
  std::byte *scratch_data_ = nullptr;
};

struct HarnessPair final {
  std::unique_ptr<HyleHarness> hyle;
  std::unique_ptr<DirectOneDnnHarness> peer;

  explicit HarnessPair(bool peer_allocation_first) {
    if (peer_allocation_first) {
      peer = std::make_unique<DirectOneDnnHarness>();
      hyle = std::make_unique<HyleHarness>();
    } else {
      hyle = std::make_unique<HyleHarness>();
      peer = std::make_unique<DirectOneDnnHarness>();
    }
  }
};

void verify_equal(std::span<const float> candidate,
                  std::span<const float> peer) {
  if (candidate.size() != peer.size())
    fail("compare HYLE and direct oneDNN output sizes");
  for (std::size_t index = 0; index < candidate.size(); ++index) {
    const float tolerance = std::abs(peer[index]) * 1.0e-5F + 1.0e-4F;
    if (std::abs(candidate[index] - peer[index]) > tolerance)
      fail("compare HYLE and direct oneDNN outputs");
  }
}

template <class Candidate, class Peer>
void run_pair(nb::Bench &bench, bool peer_first, Candidate &&candidate,
              Peer &&peer) {
  if (peer_first) {
    bench.run(kPeerName, std::forward<Peer>(peer));
    bench.run(kCandidateName, std::forward<Candidate>(candidate));
  } else {
    bench.run(kCandidateName, std::forward<Candidate>(candidate));
    bench.run(kPeerName, std::forward<Peer>(peer));
  }
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
  std::fprintf(
      stderr,
      "# HYLE_COMPUTE_ONEDNN_RUN execution_order=%s allocation_order=%s "
      "shape=32x4096x4096 provider_version=%.*s tasks=1 apple_qos=%s "
      "qos_ok=%s affinity_topology=%s caller_cpu=%s affinity_ok=%s "
      "run_nonce=%s\n",
      peer_first ? "peer-first" : "candidate-first",
      peer_allocation_first ? "peer-first" : "candidate-first",
      static_cast<int>(hyle::compute::onednn::version().size()),
      hyle::compute::onednn::version().data(), qos.name, qos.ok,
      affinity.topology, affinity.cpu, affinity.ok,
      hyle::bench::perf::run_nonce());

  HarnessPair harnesses(peer_allocation_first);
  harnesses.hyle->run();
  harnesses.peer->run();
  verify_equal(harnesses.hyle->output(), harnesses.peer->output());

  nb::Bench bench;
  bench.title(kTitle)
      .relative(true)
      .batch(1)
      .minEpochIterations(1)
      .minEpochTime(std::chrono::milliseconds(8));
  run_pair(
      bench, peer_first, [&] { harnesses.hyle->run(); },
      [&] { harnesses.peer->run(); });
}
