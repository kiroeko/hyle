#include <nanobench.h>

#include <marl/pool.h>

#include <algorithm>
#include <array>
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

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

#include "hyle/mem/object_pool.h"

#include "../common/perf_process.h"

namespace {
namespace nb = ankerl::nanobench;

constexpr std::size_t kBatch = 64;
constexpr std::size_t kCapacity = 256;
constexpr std::size_t kTailSamples = 1024;

struct TailLatency {
  std::uint64_t p50_ns = 0;
  std::uint64_t p95_ns = 0;
  std::uint64_t p99_ns = 0;
  std::uint64_t max_ns = 0;
};

template <class Function>
std::uint64_t elapsed_ns(Function&& function) {
  const auto begin = std::chrono::steady_clock::now();
  function();
  const auto end = std::chrono::steady_clock::now();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count());
}

TailLatency summarize_tail(
    std::array<std::uint64_t, kTailSamples> samples) {
  std::sort(samples.begin(), samples.end());
  const auto percentile = [&](std::size_t numerator) {
    const std::size_t index =
        (samples.size() * numerator + 99u) / 100u - 1u;
    return samples[index];
  };
  return {percentile(50), percentile(95), percentile(99), samples.back()};
}

struct Slot {
  std::uint64_t sequence = 0;
  std::uint64_t payload = 0;
};

using HylePool = hyle::mem::ObjectPool<Slot>;
using HyleLoan = hyle::mem::PoolLoan<Slot, HylePool>;
using MarlPool = marl::BoundedPool<Slot, static_cast<int>(kCapacity)>;
using MarlLoan = MarlPool::Loan;

bool environment_present(const char* name) {
  char* value = nullptr;
  std::size_t size = 0;
  const bool present =
      _dupenv_s(&value, &size, name) == 0 && value != nullptr;
  std::free(value);
  return present;
}

inline void spin_pause() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
  _mm_pause();
#else
  std::this_thread::yield();
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

class HandoffReplay {
 public:
  explicit HandoffReplay(bool peer_allocation_first) {
    if (peer_allocation_first) {
      marl_ = std::make_unique<MarlPool>();
      hyle_ = std::make_unique<HylePool>(hyle::mem::system_allocator(),
                                         static_cast<std::uint32_t>(kCapacity));
    } else {
      hyle_ = std::make_unique<HylePool>(hyle::mem::system_allocator(),
                                         static_cast<std::uint32_t>(kCapacity));
      marl_ = std::make_unique<MarlPool>();
    }
    if (!hyle_->valid()) std::abort();
    worker_ = std::thread([this] { worker_loop(); });
    while (!ready_.load(std::memory_order_acquire)) spin_pause();
  }

  ~HandoffReplay() {
    command_.store(Command::kStop, std::memory_order_relaxed);
    const std::uint64_t generation =
        issued_.fetch_add(1, std::memory_order_release) + 1;
    while (completed_.load(std::memory_order_acquire) != generation) {
      spin_pause();
    }
    worker_.join();
  }

  HandoffReplay(const HandoffReplay&) = delete;
  HandoffReplay& operator=(const HandoffReplay&) = delete;

  void run_hyle() {
    const std::uint64_t base = sequence_;
    for (std::size_t index = 0; index < kBatch; ++index) {
      HyleLoan loan = hyle_->try_loan();
      if (!loan) std::abort();
      loan->sequence = base + index;
      loan->payload = (base + index) ^ 0x9e3779b97f4a7c15ull;
      hyle_loans_[index] = std::move(loan);
    }
    sequence_ += kBatch;
    dispatch(Command::kHyle);
  }

  void run_marl() {
    const std::uint64_t base = sequence_;
    for (std::size_t index = 0; index < kBatch; ++index) {
      auto [borrowed, acquired] = marl_->tryBorrow();
      if (!acquired || borrowed.get() == nullptr) std::abort();
      borrowed->sequence = base + index;
      borrowed->payload =
          (base + index) ^ 0x9e3779b97f4a7c15ull;
      marl_loans_[index] = std::move(borrowed);
    }
    sequence_ += kBatch;
    dispatch(Command::kMarl);
  }

  void verify_capacity() {
    std::array<HyleLoan, kCapacity> hyle_all{};
    for (HyleLoan& loan : hyle_all) {
      loan = hyle_->try_loan();
      if (!loan) std::abort();
    }
    if (hyle_->try_loan()) std::abort();
    for (HyleLoan& loan : hyle_all) loan.reset();

    std::array<MarlLoan, kCapacity> marl_all{};
    for (MarlLoan& loan : marl_all) {
      auto [borrowed, acquired] = marl_->tryBorrow();
      if (!acquired || borrowed.get() == nullptr) std::abort();
      loan = std::move(borrowed);
    }
    auto [marl_exhausted, marl_acquired] = marl_->tryBorrow();
    if (marl_acquired || marl_exhausted.get() != nullptr) std::abort();
    for (MarlLoan& loan : marl_all) loan.reset();
  }

 private:
  enum class Command : unsigned char { kNone, kHyle, kMarl, kStop };

  void dispatch(Command command) {
    command_.store(command, std::memory_order_relaxed);
    const std::uint64_t generation =
        issued_.fetch_add(1, std::memory_order_release) + 1;
    while (completed_.load(std::memory_order_acquire) != generation) {
      spin_pause();
    }
    nb::doNotOptimizeAway(last_checksum_);
  }

  void worker_loop() noexcept {
    (void)hyle::bench::perf::configure_thread_affinity(
        "HYLE_PERF_WORKER_CPU");
    ready_.store(true, std::memory_order_release);
    std::uint64_t observed = 0;
    for (;;) {
      std::uint64_t generation = issued_.load(std::memory_order_acquire);
      while (generation == observed) {
        spin_pause();
        generation = issued_.load(std::memory_order_acquire);
      }
      observed = generation;
      const Command command = command_.load(std::memory_order_relaxed);
      if (command == Command::kStop) {
        completed_.store(observed, std::memory_order_release);
        return;
      }

      std::uint64_t checksum = 0;
      if (command == Command::kHyle) {
        for (HyleLoan& loan : hyle_loans_) {
          if (!loan) std::abort();
          checksum += loan->sequence ^ loan->payload;
          // Destruction on the consumer thread is the remote-return boundary.
          loan.reset();
        }
      } else if (command == Command::kMarl) {
        for (MarlLoan& loan : marl_loans_) {
          if (loan.get() == nullptr) std::abort();
          checksum += loan->sequence ^ loan->payload;
          loan.reset();
        }
      } else {
        std::abort();
      }
      if (checksum == 0) std::abort();
      last_checksum_ = checksum;
      completed_.store(observed, std::memory_order_release);
    }
  }

  std::unique_ptr<HylePool> hyle_;
  std::unique_ptr<MarlPool> marl_;
  std::array<HyleLoan, kBatch> hyle_loans_{};
  std::array<MarlLoan, kBatch> marl_loans_{};
  std::thread worker_;
  alignas(64) std::atomic<std::uint64_t> issued_{0};
  alignas(64) std::atomic<std::uint64_t> completed_{0};
  std::atomic<Command> command_{Command::kNone};
  std::atomic<bool> ready_{false};
  std::uint64_t sequence_ = 1;
  std::uint64_t last_checksum_ = 0;
};

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
      "# HYLE_MEM_HANDOFF_CONSUMER_RUN execution_order=%s "
      "allocation_order=%s apple_qos=%s qos_ok=%s "
      "affinity_topology=%s caller_cpu=%s worker_cpu=%s affinity_ok=%s "
      "run_nonce=%s\n",
      peer_first ? "peer-first" : "candidate-first",
      peer_allocation_first ? "peer-first" : "candidate-first", qos.name,
      qos.ok, affinity.topology, affinity.cpu,
      worker_cpu != nullptr ? worker_cpu : "external",
      worker_cpu != nullptr ? affinity.ok : "not-applicable",
      hyle::bench::perf::run_nonce());

  HandoffReplay replay(peer_allocation_first);
  replay.verify_capacity();
  for (int index = 0; index < 32; ++index) {
    replay.run_hyle();
    replay.run_marl();
  }

  nb::Bench bench;
  bench.title("fixed object slot cross-core handoff aggregate N=64")
      .relative(true)
      .batch(1)
      .minEpochTime(std::chrono::milliseconds(8));
  run_pair(bench, peer_first, "HYLE hard ObjectPool handoff N=64",
           [&] { replay.run_hyle(); }, "Marl BoundedPool handoff N=64",
           [&] { replay.run_marl(); });

  std::array<std::uint64_t, kTailSamples> hyle_tail{};
  std::array<std::uint64_t, kTailSamples> marl_tail{};
  for (std::size_t index = 0; index < kTailSamples; ++index) {
    if (peer_first) {
      marl_tail[index] = elapsed_ns([&] { replay.run_marl(); });
      hyle_tail[index] = elapsed_ns([&] { replay.run_hyle(); });
    } else {
      hyle_tail[index] = elapsed_ns([&] { replay.run_hyle(); });
      marl_tail[index] = elapsed_ns([&] { replay.run_marl(); });
    }
  }
  const TailLatency hyle_latency = summarize_tail(hyle_tail);
  const TailLatency marl_latency = summarize_tail(marl_tail);
  std::fprintf(
      stderr,
      "# HYLE_MEM_HANDOFF_RESOURCE capacity=%zu batch=%zu "
      "hyle_p50_ns=%llu hyle_p95_ns=%llu hyle_p99_ns=%llu hyle_max_ns=%llu "
      "marl_p50_ns=%llu marl_p95_ns=%llu marl_p99_ns=%llu marl_max_ns=%llu\n",
      kCapacity, kBatch,
      static_cast<unsigned long long>(hyle_latency.p50_ns),
      static_cast<unsigned long long>(hyle_latency.p95_ns),
      static_cast<unsigned long long>(hyle_latency.p99_ns),
      static_cast<unsigned long long>(hyle_latency.max_ns),
      static_cast<unsigned long long>(marl_latency.p50_ns),
      static_cast<unsigned long long>(marl_latency.p95_ns),
      static_cast<unsigned long long>(marl_latency.p99_ns),
      static_cast<unsigned long long>(marl_latency.max_ns));
}
