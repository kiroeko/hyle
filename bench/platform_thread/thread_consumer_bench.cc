#include <nanobench.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string_view>
#include <thread>
#include <utility>

// oneTBB's concurrent_monitor is the production consumer of this private
// binary semaphore. Qualification pins and links the exact oneTBB checkout;
// using its implementation directly avoids substituting a differently shaped
// condition variable or standard-library policy.
#include "semaphore.h"

#include "hyle/platform/wait.h"

#include "../common/perf_process.h"

namespace {
namespace nb = ankerl::nanobench;

constexpr std::size_t kBatch = 64;

bool environment_present(const char* name) {
#if defined(_WIN32)
  char* value = nullptr;
  std::size_t size = 0;
  const bool present =
      _dupenv_s(&value, &size, name) == 0 && value != nullptr;
  std::free(value);
  return present;
#else
  return std::getenv(name) != nullptr;
#endif
}

template <class Candidate, class Peer>
void run_pair(nb::Bench& bench, bool peer_first,
              std::string_view candidate_name, Candidate&& candidate,
              std::string_view peer_name, Peer&& peer) {
  if (peer_first) {
    bench.run(peer_name, std::forward<Peer>(peer));
    bench.run(candidate_name, std::forward<Candidate>(candidate));
  } else {
    bench.run(candidate_name, std::forward<Candidate>(candidate));
    bench.run(peer_name, std::forward<Peer>(peer));
  }
}

class HyleHandoff {
 public:
  HyleHandoff() : worker_([this] { worker_loop(); }) {
    while (!ready_.load(std::memory_order_acquire)) std::this_thread::yield();
  }
  ~HyleHandoff() {
    stopping_.store(true, std::memory_order_release);
    request_.unpark();
    worker_.join();
  }

  void run() noexcept {
    for (std::size_t index = 0; index < kBatch; ++index) {
      request_.unpark();
      response_.park();
    }
  }

 private:
  void worker_loop() noexcept {
    (void)hyle::bench::perf::configure_thread_affinity(
        "HYLE_PERF_WORKER_CPU");
    ready_.store(true, std::memory_order_release);
    for (;;) {
      request_.park();
      if (stopping_.load(std::memory_order_acquire)) return;
      response_.unpark();
    }
  }

  std::atomic<bool> stopping_{false};
  std::atomic<bool> ready_{false};
  hyle::platform::Parker request_;
  hyle::platform::Parker response_;
  std::thread worker_;
};

class TbbHandoff {
 public:
  TbbHandoff() : worker_([this] { worker_loop(); }) {
    while (!ready_.load(std::memory_order_acquire)) std::this_thread::yield();
  }
  ~TbbHandoff() {
    stopping_.store(true, std::memory_order_release);
    request_.V();
    worker_.join();
  }

  void run() noexcept {
    for (std::size_t index = 0; index < kBatch; ++index) {
      request_.V();
      response_.P();
    }
  }

 private:
  void worker_loop() noexcept {
    (void)hyle::bench::perf::configure_thread_affinity(
        "HYLE_PERF_WORKER_CPU");
    ready_.store(true, std::memory_order_release);
    for (;;) {
      request_.P();
      if (stopping_.load(std::memory_order_acquire)) return;
      response_.V();
    }
  }

  std::atomic<bool> stopping_{false};
  std::atomic<bool> ready_{false};
  tbb::detail::r1::binary_semaphore request_;
  tbb::detail::r1::binary_semaphore response_;
  std::thread worker_;
};

struct HandoffPair {
  std::unique_ptr<HyleHandoff> hyle;
  std::unique_ptr<TbbHandoff> tbb;

  explicit HandoffPair(bool peer_allocation_first) {
    if (peer_allocation_first) {
      tbb = std::make_unique<TbbHandoff>();
      hyle = std::make_unique<HyleHandoff>();
    } else {
      hyle = std::make_unique<HyleHandoff>();
      tbb = std::make_unique<TbbHandoff>();
    }
  }
};

nb::Bench make_bench(const char* title) {
  nb::Bench bench;
  bench.title(title)
      .relative(true)
      .batch(kBatch)
      .minEpochTime(std::chrono::milliseconds(8));
  return bench;
}

}  // namespace

int main() {
  const auto qos = hyle::bench::perf::configure_thread_qos(
      "HYLE_BENCH_APPLE_QOS");
  const auto affinity = hyle::bench::perf::configure_thread_affinity(
      "HYLE_PERF_CALLER_CPU");
  const char* worker_cpu = std::getenv("HYLE_PERF_WORKER_CPU");
  const bool peer_first = environment_present("HYLE_BENCH_RIVAL_FIRST");
  const bool peer_allocation_first =
      environment_present("HYLE_BENCH_PEER_ALLOCATION_FIRST");
  std::fprintf(
      stderr,
      "# HYLE_PLATFORM_THREAD_RUN execution_order=%s allocation_order=%s "
      "apple_qos=%s qos_ok=%s affinity_topology=%s caller_cpu=%s "
      "worker_cpu=%s affinity_ok=%s run_nonce=%s\n",
      peer_first ? "peer-first" : "candidate-first",
      peer_allocation_first ? "peer-first" : "candidate-first", qos.name,
      qos.ok, affinity.topology, affinity.cpu,
      worker_cpu != nullptr ? worker_cpu : "external",
      worker_cpu != nullptr ? affinity.ok : "not-applicable",
      hyle::bench::perf::run_nonce());

  HandoffPair handoffs(peer_allocation_first);
  handoffs.hyle->run();
  handoffs.tbb->run();
  nb::Bench handoff = make_bench("scheduler worker idle handoff aggregate N=64");
  run_pair(handoff, peer_first, "HYLE Parker handoff N=64",
           [&] { handoffs.hyle->run(); }, "oneTBB binary semaphore handoff N=64",
           [&] { handoffs.tbb->run(); });

}
