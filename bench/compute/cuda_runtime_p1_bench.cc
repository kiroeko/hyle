// Same-contract CUDA queue-prefix route: one preallocated device-to-device
// copy followed by terminal event completion. Allocation, first touch and
// event/stream construction remain outside timing on both sides.
#include <nanobench.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/transfer.h"
#include "hyle/mem/buffer.h"

#include "../common/perf_process.h"

namespace {
namespace nb = ankerl::nanobench;

constexpr std::size_t kBytes = 1U << 20U;

[[noreturn]] void fail_setup(std::string_view message) {
  std::fprintf(stderr, "compute CUDA runtime benchmark setup failed: %.*s\n",
               static_cast<int>(message.size()), message.data());
  std::abort();
}

template <class T>
T take_or_abort(hyle::Result<T> result, std::string_view message) {
  if (!result)
    fail_setup(message);
  return std::move(*result);
}

void require_cuda(cudaError_t status, std::string_view message) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%.*s: %s\n", static_cast<int>(message.size()),
                 message.data(), cudaGetErrorString(status));
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

template <class Candidate, class Peer>
void run_pair(nb::Bench &bench, bool peer_first,
              std::string_view candidate_name, Candidate &&candidate,
              std::string_view peer_name, Peer &&peer) {
  if (peer_first) {
    bench.run(peer_name, std::forward<Peer>(peer));
    bench.run(candidate_name, std::forward<Candidate>(candidate));
  } else {
    bench.run(candidate_name, std::forward<Candidate>(candidate));
    bench.run(peer_name, std::forward<Peer>(peer));
  }
}

struct BufferPair final {
  hyle::mem::Buffer source{};
  hyle::mem::Buffer destination{};
};

class Fixture final {
public:
  explicit Fixture(bool peer_allocation_first)
      : device_(take_or_abort(hyle::compute::cuda::device(0), "CUDA device")),
        context_(take_or_abort(hyle::compute::Context::create(device_),
                               "CUDA context")),
        resource_(take_or_abort(hyle::compute::cuda::MemoryResource::create(
                                    context_, hyle::compute::cuda::MemoryKind::kDevice),
                                "CUDA device resource")),
        queue_(take_or_abort(context_.create_queue(), "HYLE CUDA queue")) {
    const int ordinal = take_or_abort(
        hyle::compute::cuda::device_ordinal(device_), "CUDA ordinal");
    require_cuda(cudaSetDevice(ordinal), "select CUDA device");
    require_cuda(cudaStreamCreateWithFlags(&peer_stream_, cudaStreamNonBlocking),
                 "raw CUDA stream");
    require_cuda(cudaEventCreateWithFlags(&peer_event_, cudaEventDisableTiming),
                 "raw CUDA event");
    if (peer_allocation_first) {
      peer_ = make_pair();
      candidate_ = make_pair();
    } else {
      candidate_ = make_pair();
      peer_ = make_pair();
    }
    initialize(candidate_);
    initialize(peer_);
    if (!queue_.reserve_completions(1).ok())
      fail_setup("reserve HYLE CUDA completion");
    // Keep the GPU outside its idle clock transition before either crossed
    // route is timed. This is setup, not an uncharged candidate-only warmup.
    for (int warmup = 0; warmup < 2048; ++warmup) {
      if (run_candidate() == 0 || run_peer() == 0)
        fail_setup("warm CUDA copy");
    }
    validate(candidate_);
    validate(peer_);
  }

  Fixture(const Fixture &) = delete;
  Fixture &operator=(const Fixture &) = delete;

  ~Fixture() {
    if (peer_event_ != nullptr)
      (void)cudaEventDestroy(peer_event_);
    if (peer_stream_ != nullptr)
      (void)cudaStreamDestroy(peer_stream_);
  }

  std::uint64_t run_candidate() noexcept {
    auto completion = hyle::compute::copy_async(
        queue_, candidate_.destination.view(), candidate_.source.view());
    if (!completion) {
      std::fprintf(stderr, "HYLE CUDA copy submit failed: %.*s\n",
                   static_cast<int>(completion.error().message().size()),
                   completion.error().message().data());
      std::abort();
    }
    const hyle::Status terminal = completion->wait();
    if (!terminal.ok()) {
      std::fprintf(stderr, "HYLE CUDA copy wait failed: %.*s\n",
                   static_cast<int>(terminal.message().size()),
                   terminal.message().data());
      std::abort();
    }
    return 1U;
  }

  std::uint64_t run_peer() noexcept {
    auto source = hyle::compute::cuda::pointer<std::byte>(peer_.source.view());
    auto destination =
        hyle::compute::cuda::pointer<std::byte>(peer_.destination.view());
    if (!source) {
      std::fprintf(stderr, "raw CUDA source pointer failed: %.*s\n",
                   static_cast<int>(source.error().message().size()),
                   source.error().message().data());
      std::abort();
    }
    if (!destination) {
      std::fprintf(stderr, "raw CUDA destination pointer failed: %.*s\n",
                   static_cast<int>(destination.error().message().size()),
                   destination.error().message().data());
      std::abort();
    }
    require_cuda(cudaMemcpyAsync(*destination, *source, kBytes,
                                 cudaMemcpyDeviceToDevice, peer_stream_),
                 "raw CUDA copy submit failed");
    require_cuda(cudaEventRecord(peer_event_, peer_stream_),
                 "raw CUDA event record failed");
    require_cuda(cudaEventSynchronize(peer_event_),
                 "raw CUDA event wait failed");
    return 1U;
  }

private:
  BufferPair make_pair() {
    return {take_or_abort(hyle::mem::Buffer::create(resource_.ref(), kBytes, 256),
                          "source device buffer"),
            take_or_abort(hyle::mem::Buffer::create(resource_.ref(), kBytes, 256),
                          "destination device buffer")};
  }

  static void initialize(BufferPair &pair) {
    std::vector<std::byte> values(kBytes);
    for (std::size_t index = 0; index < values.size(); ++index)
      values[index] = static_cast<std::byte>(index % 251U);
    if (!pair.source.view().write(values).ok())
      fail_setup("device source initialization");
  }

  static void validate(const BufferPair &pair) {
    std::vector<std::byte> values(kBytes);
    if (!pair.destination.view().read(values).ok())
      fail_setup("device destination readback");
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (values[index] != static_cast<std::byte>(index % 251U))
        fail_setup("device copy result mismatch");
    }
  }

  hyle::compute::Device device_{};
  hyle::compute::Context context_{};
  hyle::compute::cuda::MemoryResource resource_{};
  hyle::compute::Queue queue_{};
  BufferPair candidate_{};
  BufferPair peer_{};
  cudaStream_t peer_stream_ = nullptr;
  cudaEvent_t peer_event_ = nullptr;
};

} // namespace

int main() {
  const auto affinity =
      hyle::bench::perf::configure_thread_affinity("HYLE_PERF_CALLER_CPU");
  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  const bool peer_allocation_first =
      environment_present("HYLE_BENCH_PEER_ALLOCATION_FIRST");
  std::fprintf(
      stderr,
      "# HYLE_COMPUTE_CUDA_RUNTIME_RUN execution_order=%s "
      "allocation_order=%s affinity_topology=%s caller_cpu=%s affinity_ok=%s "
      "run_nonce=%s\n",
      peer_first ? "peer-first" : "candidate-first",
      peer_allocation_first ? "peer-first" : "candidate-first",
      affinity.topology, affinity.cpu, affinity.ok,
      hyle::bench::perf::run_nonce());

  Fixture fixture(peer_allocation_first);
  nb::Bench bench;
  bench.title("compute CUDA D2D 1 MiB queue-prefix completion")
      .relative(true)
      .minEpochTime(std::chrono::milliseconds(20));
  run_pair(
      bench, peer_first, "HYLE CUDA copy+Completion 1 MiB",
      [&] {
        const auto ok = fixture.run_candidate();
        nb::doNotOptimizeAway(ok);
      },
      "CUDA raw copy+event 1 MiB",
      [&] {
        const auto ok = fixture.run_peer();
        nb::doNotOptimizeAway(ok);
      });
}
