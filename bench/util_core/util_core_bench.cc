#include <nanobench.h>

#include <algorithm>
#include <array>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "hyle/util/random.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/spin_lock.h"

#if defined(HYLE_BENCH_TBB)
#  include <oneapi/tbb/spin_mutex.h>
#endif

namespace {
namespace nb = ankerl::nanobench;

struct IntrusiveWidget : hyle::util::RefCounted<IntrusiveWidget> {
  std::uint64_t value = 0;
};

struct SharedWidget {
  std::uint64_t value = 0;
};

template <class Lock>
std::uint64_t contended_increment(std::size_t thread_count,
                                  std::size_t iterations) {
  Lock lock;
  std::uint64_t value = 0;
  std::barrier start{static_cast<std::ptrdiff_t>(thread_count)};
  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  for (std::size_t thread = 0; thread < thread_count; ++thread) {
    threads.emplace_back([&] {
      start.arrive_and_wait();
      for (std::size_t i = 0; i < iterations; ++i) {
        std::lock_guard<Lock> guard(lock);
        ++value;
      }
    });
  }
  for (auto& thread : threads) thread.join();
  return value;
}

template <class Lock>
void run_uncontended(nb::Bench& bench, const char* name) {
  bench.run(name, [] {
    static Lock lock;
    lock.lock();
    nb::doNotOptimizeAway(&lock);
    lock.unlock();
  });
}

template <class Lock>
void run_contended(nb::Bench& bench, const char* name,
                   std::size_t thread_count, std::size_t iterations) {
  bench.run(name, [=] {
    nb::doNotOptimizeAway(
        contended_increment<Lock>(thread_count, iterations));
  });
}

}  // namespace

int main() {
  std::cout << "layout SpinLock=" << sizeof(hyle::util::SpinLock)
            << " Ref=" << sizeof(hyle::util::Ref<IntrusiveWidget>)
            << " Rng=" << sizeof(hyle::util::Rng) << '\n';

  {
    nb::Bench bench;
    bench.title("util-core PRNG next_u64")
        .relative(true)
        .batch(1)
        .minEpochIterations(2'000'000);
    bench.run("std::mt19937_64", [] {
      static std::mt19937_64 rng{12345};
      nb::doNotOptimizeAway(rng());
    });
    bench.run("HYLE Rng xoshiro256++", [] {
      static hyle::util::Rng rng{12345};
      nb::doNotOptimizeAway(rng());
    });
  }

  {
    std::array<std::uint64_t, 256> bounds{};
    for (std::size_t i = 0; i < bounds.size(); ++i) {
      bounds[i] = 17 + i * 7919;
    }
    std::size_t index = 0;
    nb::Bench bench;
    bench.title("util-core PRNG varying bounded u64")
        .relative(true)
        .batch(1)
        .minEpochIterations(1'000'000);
    bench.run("std::uniform_int_distribution", [&] {
      static std::mt19937_64 rng{7};
      static std::uniform_int_distribution<std::uint64_t> distribution;
      const std::uint64_t hi = bounds[index++ & 255];
      nb::doNotOptimizeAway(
          distribution(rng, decltype(distribution)::param_type{0, hi}));
    });
    index = 0;
    bench.run("HYLE Rng::range_u64", [&] {
      static hyle::util::Rng rng{7};
      const std::uint64_t hi = bounds[index++ & 255];
      nb::doNotOptimizeAway(rng.range_u64(0, hi));
    });
  }

  // Same HYLE generator and Fisher-Yates contract: measures whether the new
  // standard-URBG shape costs anything versus the native callback convenience.
  {
    constexpr std::size_t kCount = 1024;
    std::array<std::uint32_t, kCount> values{};
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] = static_cast<std::uint32_t>(i);
    }
    nb::Bench bench;
    bench.title("util-core shuffle via HYLE Rng")
        .relative(true)
        .batch(kCount)
        .minEpochIterations(2'000);
    bench.run("std::shuffle(Rng URBG)", [&] {
      static hyle::util::Rng rng{9};
      std::shuffle(values.begin(), values.end(), rng);
      nb::doNotOptimizeAway(values);
    });
    bench.run("Rng::shuffle(iterators)", [&] {
      static hyle::util::Rng rng{9};
      rng.shuffle(values.begin(), values.end());
      nb::doNotOptimizeAway(values);
    });
  }

  {
    nb::Bench bench;
    bench.title("util-core uncontended lock+unlock")
        .relative(true)
        .batch(1)
        .minEpochIterations(2'000'000);
    run_uncontended<std::mutex>(bench, "std::mutex");
#if defined(HYLE_BENCH_TBB)
    run_uncontended<oneapi::tbb::spin_mutex>(bench, "oneTBB spin_mutex");
#endif
    run_uncontended<hyle::util::SpinLock>(bench, "HYLE SpinLock");
  }

  constexpr std::size_t kIterations = 100'000;
  for (std::size_t thread_count : {std::size_t{2}, std::size_t{8}}) {
    nb::Bench bench;
    bench.title("util-core contended increment " +
                std::to_string(thread_count) + " threads")
        .relative(true)
        .batch(thread_count * kIterations)
        .minEpochIterations(1)
        .epochs(7);
    run_contended<std::mutex>(bench, "std::mutex", thread_count, kIterations);
#if defined(HYLE_BENCH_TBB)
    run_contended<oneapi::tbb::spin_mutex>(
        bench, "oneTBB spin_mutex", thread_count, kIterations);
#endif
    run_contended<hyle::util::SpinLock>(
        bench, "HYLE SpinLock", thread_count, kIterations);
  }

#if defined(HYLE_BENCH_TBB)
  // Repeat the mature-peer comparison in reverse execution order. Sub-2%
  // lock gaps are particularly vulnerable to frequency/warmup ordering; a
  // release certificate must not promote those into a win or loss.
  {
    nb::Bench bench;
    bench.title("util-core uncontended lock+unlock reverse order")
        .relative(true)
        .batch(1)
        .minEpochIterations(2'000'000);
    run_uncontended<hyle::util::SpinLock>(bench, "HYLE SpinLock");
    run_uncontended<oneapi::tbb::spin_mutex>(bench, "oneTBB spin_mutex");
  }
  for (std::size_t thread_count : {std::size_t{2}, std::size_t{8}}) {
    nb::Bench bench;
    bench.title("util-core contended increment reverse order " +
                std::to_string(thread_count) + " threads")
        .relative(true)
        .batch(thread_count * kIterations)
        .minEpochIterations(1)
        .epochs(7);
    run_contended<hyle::util::SpinLock>(
        bench, "HYLE SpinLock", thread_count, kIterations);
    run_contended<oneapi::tbb::spin_mutex>(
        bench, "oneTBB spin_mutex", thread_count, kIterations);
  }
#endif

  {
    constexpr std::size_t kFanout = 64;
    auto intrusive = hyle::util::make_ref<IntrusiveWidget>();
    auto shared = std::make_shared<SharedWidget>();
    std::vector<hyle::util::Ref<IntrusiveWidget>> intrusive_refs;
    std::vector<std::shared_ptr<SharedWidget>> shared_refs;
    intrusive_refs.reserve(kFanout);
    shared_refs.reserve(kFanout);
    nb::Bench bench;
    bench.title("util-core smart pointer fanout 64")
        .relative(true)
        .batch(kFanout)
        .minEpochIterations(20'000);
    bench.run("std::shared_ptr", [&] {
      for (std::size_t i = 0; i < kFanout; ++i) shared_refs.push_back(shared);
      nb::doNotOptimizeAway(shared_refs.data());
      shared_refs.clear();
    });
    bench.run("HYLE intrusive Ref", [&] {
      for (std::size_t i = 0; i < kFanout; ++i) {
        intrusive_refs.push_back(intrusive);
      }
      nb::doNotOptimizeAway(intrusive_refs.data());
      intrusive_refs.clear();
    });
  }
}
