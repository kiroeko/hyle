#include <nanobench.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include "hyle/container/concurrent_hash_map.h"
#include "hyle/container/concurrent_node_hash_map.h"

#if defined(HYLE_BENCH_TBB)
#include "oneapi/tbb/concurrent_hash_map.h"
#endif

namespace {
namespace nb = ankerl::nanobench;
using FlatMap =
    hyle::container::ConcurrentFlatHashMap<std::uint64_t, std::uint64_t>;
using Map =
    hyle::container::ConcurrentNodeHashMap<std::uint64_t, std::uint64_t>;

constexpr std::size_t kKeys = 1u << 15;
constexpr std::size_t kThreads = 8;
constexpr std::size_t kReadOpsPerThread = 1u << 15;
constexpr std::size_t kMixedOpsPerThread = 1u << 14;

std::uint64_t key_at(std::size_t index) noexcept {
  return static_cast<std::uint64_t>(index & (kKeys - 1));
}

template <class Fn> void run_threads(Fn &&fn) {
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (std::size_t thread = 0; thread < kThreads; ++thread)
    threads.emplace_back(fn, thread);
  for (std::thread &thread : threads)
    thread.join();
}

void fill_hyle(Map &map) {
  Map::Accessor accessor;
  for (std::size_t i = 0; i < kKeys; ++i) {
    map.emplace(accessor, static_cast<std::uint64_t>(i),
                static_cast<std::uint64_t>(i));
    accessor.release();
  }
}

#if defined(HYLE_BENCH_TBB)
struct TbbHashCompare {
  static std::size_t hash(std::uint64_t value) noexcept {
    return static_cast<std::size_t>(
        hyle::container::Hash<std::uint64_t>{}(value));
  }
  static bool equal(std::uint64_t left, std::uint64_t right) noexcept {
    return left == right;
  }
};
using TbbMap = oneapi::tbb::concurrent_hash_map<std::uint64_t, std::uint64_t,
                                                TbbHashCompare>;

void fill_tbb(TbbMap &map) {
  for (std::size_t i = 0; i < kKeys; ++i) {
    TbbMap::accessor accessor;
    map.insert(accessor, static_cast<std::uint64_t>(i));
    accessor->second = i;
  }
}
#endif

} // namespace

int main() {
  Map hyle(kKeys * 2);
  hyle.reserve(kKeys);
  fill_hyle(hyle);
  FlatMap hyle_flat(256);
  hyle_flat.reserve(kKeys);
  {
    FlatMap::Accessor accessor;
    for (std::size_t i = 0; i < kKeys; ++i) {
      hyle_flat.emplace(accessor, static_cast<std::uint64_t>(i),
                        static_cast<std::uint64_t>(i));
      accessor.release();
    }
  }
#if defined(HYLE_BENCH_TBB)
  TbbMap tbb;
  tbb.rehash(kKeys * 2);
  fill_tbb(tbb);
#endif

  {
    nb::Bench bench;
    bench.title("shared registry: single-thread protected hit")
        .relative(true)
        .batch(kKeys)
        .minEpochIterations(100)
        .minEpochTime(std::chrono::milliseconds(20));
    bench.run("HYLE ConcurrentNodeHashMap const accessor", [&] {
      std::uint64_t checksum = 0;
      Map::ConstAccessor accessor;
      for (std::size_t i = 0; i < kKeys; ++i) {
        hyle.find(accessor, key_at(i * 17));
        checksum += *accessor;
        accessor.release();
      }
      nb::doNotOptimizeAway(checksum);
    });
    bench.run("HYLE ConcurrentFlatHashMap const accessor", [&] {
      std::uint64_t checksum = 0;
      FlatMap::ConstAccessor accessor;
      for (std::size_t i = 0; i < kKeys; ++i) {
        hyle_flat.find(accessor, key_at(i * 17));
        checksum += *accessor;
        accessor.release();
      }
      nb::doNotOptimizeAway(checksum);
    });
#if defined(HYLE_BENCH_TBB)
    bench.run("oneTBB concurrent_hash_map const_accessor", [&] {
      std::uint64_t checksum = 0;
      TbbMap::const_accessor accessor;
      for (std::size_t i = 0; i < kKeys; ++i) {
        tbb.find(accessor, key_at(i * 17));
        checksum += accessor->second;
        accessor.release();
      }
      nb::doNotOptimizeAway(checksum);
    });
#endif
  }

  {
    nb::Bench bench;
    bench.title("shared registry: 8 readers / existing keys")
        .relative(true)
        .batch(kThreads * kReadOpsPerThread)
        .minEpochIterations(20)
        .minEpochTime(std::chrono::milliseconds(20));
    bench.run("HYLE ConcurrentNodeHashMap 8R", [&] {
      run_threads([&](std::size_t thread) {
        std::uint64_t checksum = 0;
        Map::ConstAccessor accessor;
        for (std::size_t i = 0; i < kReadOpsPerThread; ++i) {
          hyle.find(accessor, key_at(i * 17 + thread * 101));
          checksum += *accessor;
          accessor.release();
        }
        nb::doNotOptimizeAway(checksum);
      });
    });
    bench.run("HYLE ConcurrentFlatHashMap 8R", [&] {
      run_threads([&](std::size_t thread) {
        std::uint64_t checksum = 0;
        FlatMap::ConstAccessor accessor;
        for (std::size_t i = 0; i < kReadOpsPerThread; ++i) {
          hyle_flat.find(accessor, key_at(i * 17 + thread * 101));
          checksum += *accessor;
          accessor.release();
        }
        nb::doNotOptimizeAway(checksum);
      });
    });
#if defined(HYLE_BENCH_TBB)
    bench.run("oneTBB concurrent_hash_map 8R", [&] {
      run_threads([&](std::size_t thread) {
        std::uint64_t checksum = 0;
        TbbMap::const_accessor accessor;
        for (std::size_t i = 0; i < kReadOpsPerThread; ++i) {
          tbb.find(accessor, key_at(i * 17 + thread * 101));
          checksum += accessor->second;
          accessor.release();
        }
        nb::doNotOptimizeAway(checksum);
      });
    });
#endif
  }

  {
    nb::Bench bench;
    bench.title("shared state: 8 threads, 90% read / 10% update")
        .relative(true)
        .batch(kThreads * kMixedOpsPerThread)
        .minEpochIterations(20)
        .minEpochTime(std::chrono::milliseconds(20));
    bench.run("HYLE ConcurrentNodeHashMap mixed", [&] {
      run_threads([&](std::size_t thread) {
        std::uint64_t checksum = 0;
        Map::ConstAccessor reader;
        Map::Accessor writer;
        for (std::size_t i = 0; i < kMixedOpsPerThread; ++i) {
          const std::uint64_t key = key_at(i * 17 + thread * 101);
          if (i % 10 == 0) {
            hyle.find(writer, key);
            ++*writer;
            writer.release();
          } else {
            hyle.find(reader, key);
            checksum += *reader;
            reader.release();
          }
        }
        nb::doNotOptimizeAway(checksum);
      });
    });
    bench.run("HYLE ConcurrentFlatHashMap mixed", [&] {
      run_threads([&](std::size_t thread) {
        std::uint64_t checksum = 0;
        FlatMap::ConstAccessor reader;
        FlatMap::Accessor writer;
        for (std::size_t i = 0; i < kMixedOpsPerThread; ++i) {
          const std::uint64_t key = key_at(i * 17 + thread * 101);
          if (i % 10 == 0) {
            hyle_flat.find(writer, key);
            ++*writer;
            writer.release();
          } else {
            hyle_flat.find(reader, key);
            checksum += *reader;
            reader.release();
          }
        }
        nb::doNotOptimizeAway(checksum);
      });
    });
#if defined(HYLE_BENCH_TBB)
    bench.run("oneTBB concurrent_hash_map mixed", [&] {
      run_threads([&](std::size_t thread) {
        std::uint64_t checksum = 0;
        for (std::size_t i = 0; i < kMixedOpsPerThread; ++i) {
          const std::uint64_t key = key_at(i * 17 + thread * 101);
          if (i % 10 == 0) {
            TbbMap::accessor writer;
            tbb.find(writer, key);
            ++writer->second;
          } else {
            TbbMap::const_accessor reader;
            tbb.find(reader, key);
            checksum += reader->second;
          }
        }
        nb::doNotOptimizeAway(checksum);
      });
    });
#endif
  }

  {
    nb::Bench bench;
    bench.title("registry rebuild: clear + 8 disjoint writers")
        .relative(true)
        .batch(kKeys)
        .minEpochIterations(20)
        .minEpochTime(std::chrono::milliseconds(20));
    bench.run("HYLE ConcurrentNodeHashMap 8W rebuild", [&] {
      hyle.clear();
      run_threads([&](std::size_t thread) {
        Map::Accessor accessor;
        for (std::size_t i = thread; i < kKeys; i += kThreads) {
          hyle.emplace(accessor, static_cast<std::uint64_t>(i),
                       static_cast<std::uint64_t>(i));
          accessor.release();
        }
      });
      nb::doNotOptimizeAway(hyle.size());
    });
    bench.run("HYLE ConcurrentFlatHashMap 8W rebuild", [&] {
      hyle_flat.clear();
      run_threads([&](std::size_t thread) {
        FlatMap::Accessor accessor;
        for (std::size_t i = thread; i < kKeys; i += kThreads) {
          hyle_flat.emplace(accessor, static_cast<std::uint64_t>(i),
                            static_cast<std::uint64_t>(i));
          accessor.release();
        }
      });
      nb::doNotOptimizeAway(hyle_flat.size());
    });
#if defined(HYLE_BENCH_TBB)
    bench.run("oneTBB concurrent_hash_map 8W rebuild", [&] {
      tbb.clear();
      run_threads([&](std::size_t thread) {
        for (std::size_t i = thread; i < kKeys; i += kThreads) {
          TbbMap::accessor accessor;
          tbb.insert(accessor, static_cast<std::uint64_t>(i));
          accessor->second = i;
        }
      });
      nb::doNotOptimizeAway(tbb.size());
    });
#endif
  }
}
