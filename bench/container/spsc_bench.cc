#include <nanobench.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <thread>
#include <utility>

#include "hyle/container/spsc_queue.h"
#include "hyle/platform/convenience.h"
#include "hyle/platform/thread.h"

namespace {
namespace nb = ankerl::nanobench;
using hyle::container::SpscQueue;

constexpr std::size_t kCapacity = 1024;
constexpr std::uint64_t kItems = 1u << 18;

// Same bounded SPSC protocol as folly::ProducerConsumerQueue: one reserved
// sentinel slot, producer/consumer indices on separate cache lines, and one
// acquire of the opposite index per operation. Kept local because a raw Folly
// source checkout has generated-config dependencies; this peer deliberately
// contains no HYLE optimization (remote-index cache or monotonic mask).
template <class T>
class FollyShapedSpsc {
 public:
  explicit FollyShapedSpsc(std::size_t capacity)
      : size_(capacity + 1), records_(new Storage[size_]) {}

  ~FollyShapedSpsc() {
    std::size_t read = read_.load(std::memory_order_relaxed);
    const std::size_t write = write_.load(std::memory_order_relaxed);
    while (read != write) {
      std::destroy_at(pointer(read));
      if (++read == size_) read = 0;
    }
  }

  bool try_push(const T& value) {
    const std::size_t write = write_.load(std::memory_order_relaxed);
    std::size_t next = write + 1;
    if (next == size_) next = 0;
    if (next == read_.load(std::memory_order_acquire)) return false;
    std::construct_at(raw_pointer(write), value);
    write_.store(next, std::memory_order_release);
    return true;
  }

  bool try_pop(T& out) {
    const std::size_t read = read_.load(std::memory_order_relaxed);
    if (read == write_.load(std::memory_order_acquire)) return false;
    T* value = pointer(read);
    out = std::move(*value);
    std::destroy_at(value);
    std::size_t next = read + 1;
    if (next == size_) next = 0;
    read_.store(next, std::memory_order_release);
    return true;
  }

 private:
  struct Storage {
    alignas(T) std::byte value[sizeof(T)];
  };
  T* raw_pointer(std::size_t index) noexcept {
    return reinterpret_cast<T*>(records_[index].value);
  }
  T* pointer(std::size_t index) noexcept {
    return std::launder(raw_pointer(index));
  }

  const std::size_t size_;
  std::unique_ptr<Storage[]> records_;
  alignas(64) std::atomic<std::size_t> read_{0};
  alignas(64) std::atomic<std::size_t> write_{0};
};

template <class Queue>
class PersistentTransfer {
 public:
  PersistentTransfer(Queue& queue, unsigned producer_cpu,
                     unsigned consumer_cpu)
      : queue_(queue), producer_([this, producer_cpu] {
          pin(producer_cpu);
          worker_produce();
        }), consumer_([this, consumer_cpu] {
          pin(consumer_cpu);
          worker_consume();
        }) {}

  ~PersistentTransfer() {
    stop_.store(true, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_release);
    generation_.notify_all();
    producer_.join();
    consumer_.join();
  }

  std::uint64_t run() {
    completed_.store(0, std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_release);
    generation_.notify_all();
    while (completed_.load(std::memory_order_acquire) != 2) {
      const unsigned current = completed_.load(std::memory_order_relaxed);
      completed_.wait(current, std::memory_order_relaxed);
    }
    return sum_;
  }

 private:
  static void pin(unsigned cpu) {
    hyle::platform::this_thread::set_affinity(cpu).IgnoreError();
  }

  bool wait_for_round(std::uint64_t& observed) {
    std::uint64_t next = generation_.load(std::memory_order_acquire);
    while (next == observed) {
      generation_.wait(next, std::memory_order_relaxed);
      next = generation_.load(std::memory_order_acquire);
    }
    observed = next;
    return !stop_.load(std::memory_order_acquire);
  }

  void finish_round() {
    completed_.fetch_add(1, std::memory_order_release);
    completed_.notify_one();
  }

  void worker_produce() {
    std::uint64_t round = 0;
    while (wait_for_round(round)) {
      for (std::uint64_t value = 1; value <= kItems; ++value)
        while (!queue_.try_push(value)) {}
      finish_round();
    }
  }

  void worker_consume() {
    std::uint64_t round = 0;
    while (wait_for_round(round)) {
      std::uint64_t local = 0;
      for (std::uint64_t index = 0; index < kItems; ++index) {
        std::uint64_t value = 0;
        while (!queue_.try_pop(value)) {}
        local += value;
      }
      sum_ = local;
      finish_round();
    }
  }

  Queue& queue_;
  std::atomic<std::uint64_t> generation_{0};
  std::atomic<unsigned> completed_{0};
  std::atomic<bool> stop_{false};
  std::uint64_t sum_ = 0;
  std::thread producer_;
  std::thread consumer_;
};

}  // namespace

int main() {
  unsigned producer_cpu = 0;
  unsigned consumer_cpu = 0;
  auto available = hyle::platform::available_cpu_ids(
      hyle::platform::CpuOrder::kPerformanceFirst);
  if (available && !available->empty()) {
    producer_cpu = (*available)[0];
    consumer_cpu = (*available)[available->size() / 2];
  }
  SpscQueue<std::uint64_t> hyle(kCapacity);
  FollyShapedSpsc<std::uint64_t> folly(kCapacity);
  // Keep only one worker pair alive at a time. On heterogeneous macOS hosts
  // hard affinity is unavailable; retaining both pairs lets thread creation
  // order systematically place one rival on P cores and the other on E cores.
  {
    PersistentTransfer runner(hyle, producer_cpu, consumer_cpu);
    nb::Bench bench;
    bench.title("bounded SPSC 1P/1C stream, 1024 slots")
        .relative(true)
        .batch(kItems)
        .minEpochIterations(15)
        .minEpochTime(std::chrono::milliseconds(20));
    bench.run("HYLE SpscQueue", [&] {
      const std::uint64_t sum = runner.run();
      nb::doNotOptimizeAway(sum);
    });
  }
  {
    PersistentTransfer runner(folly, producer_cpu, consumer_cpu);
    nb::Bench bench;
    bench.title("bounded SPSC 1P/1C stream, 1024 slots")
        .relative(true)
        .batch(kItems)
        .minEpochIterations(15)
        .minEpochTime(std::chrono::milliseconds(20));
    bench.run("portable Folly ProducerConsumerQueue shape", [&] {
      const std::uint64_t sum = runner.run();
      nb::doNotOptimizeAway(sum);
    });
  }
}
