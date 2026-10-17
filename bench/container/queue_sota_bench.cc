// Queue mechanism comparisons. Optional peers come from the caller-selected
// local moodycamel, rigtorp and oneTBB checkouts; record their actual identities
// when measuring. No historical result establishes a current ranking.
// BoundedQueue and rigtorp use one fixed-capacity MPMC ring. ShardedQueue has
// per-shard bounds/order and total capacity P*kCap here; moodycamel's kCap is
// initial storage, its enqueue may grow, and order is per producer. Those rows
// do not share a full/backpressure contract. Owner-only work-stealing paths
// also have different responsibilities from general MPMC; crossbeam's owner
// comparison lives separately in bench/rust.

#include <nanobench.h>

#include <atomic>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <string_view>
#include <thread>
#include <vector>

#include "hyle/container/bounded_queue.h"
#include "hyle/container/concurrent_vector.h"
#include "hyle/container/sharded_queue.h"
#include "hyle/container/work_stealing_deque.h"
#include "hyle/container/ws_fifo_queue.h"
#include "hyle/mem/allocator.h"

#if defined(HYLE_BENCH_MOODYCAMEL)
#include "concurrentqueue.h"
#endif
#if defined(HYLE_BENCH_TBB)
#include "oneapi/tbb/concurrent_queue.h"
#include "oneapi/tbb/concurrent_vector.h"
#endif
#if defined(HYLE_BENCH_RIGTORP)
#include "rigtorp/MPMCQueue.h"
#endif

using hyle::container::BoundedQueue;
using hyle::container::ConcurrentVector;
using hyle::container::ShardedQueue;
using hyle::container::WorkStealingDeque;
using hyle::container::WsFifoQueue;
using hyle::mem::system_allocator;

namespace {

constexpr std::size_t kCap = 1024;   // ring capacity (power of two)
constexpr int kBatch = 512;          // items per fill/drain wave (< kCap)

struct ConsumerResult {
  std::uint64_t count = 0;
  std::uint64_t sum = 0;
  std::uint64_t xor_sum = 0;
};

template <std::size_t N>
void require_conservation(const std::array<ConsumerResult, N>& consumers,
                          std::uint64_t expected_count) {
  ConsumerResult total;
  for (const auto& consumer : consumers) {
    total.count += consumer.count;
    total.sum += consumer.sum;
    total.xor_sum ^= consumer.xor_sum;
  }
  const std::uint64_t expected_xor =
      expected_count % 4 == 0 ? expected_count :
      expected_count % 4 == 1 ? 1 :
      expected_count % 4 == 2 ? expected_count + 1 : 0;
  if (total.count != expected_count ||
      total.sum != expected_count * (expected_count + 1) / 2 ||
      total.xor_sum != expected_xor)
    std::abort();
}

// ---- 1. single-thread enqueue+dequeue throughput ----------------------------
// Purest queue overhead: no contention, no thread scheduling. Fill `kBatch`
// then drain `kBatch`; one op == one push or one pop. This is the floor every
// MPMC algorithm pays even when uncontended (per-slot seq CAS for Vyukov).
template <class Push, class Pop>
void checked_single_thread_wave(Push push, Pop pop) {
  for (int i = 0; i < kBatch; ++i) {
    if (!push(static_cast<std::uint64_t>(i))) std::abort();
  }
  std::uint64_t out = 0;
  for (int i = 0; i < kBatch; ++i) {
    if (!pop(out) || out != static_cast<std::uint64_t>(i)) std::abort();
  }
  if (pop(out)) std::abort();
  ankerl::nanobench::doNotOptimizeAway(out);
}

void bench_single_thread() {
  BoundedQueue<std::uint64_t> bounded(kCap);
  if (!bounded.valid()) std::abort();
  auto bounded_wave = [&] {
    checked_single_thread_wave(
        [&](std::uint64_t value) { return bounded.try_push(value); },
        [&](std::uint64_t& value) { return bounded.try_pop(value); });
  };
#if defined(HYLE_BENCH_MOODYCAMEL)
  moodycamel::ConcurrentQueue<std::uint64_t> moody(kCap);
  auto moody_wave = [&] {
    checked_single_thread_wave(
        [&](std::uint64_t value) { return moody.enqueue(value); },
        [&](std::uint64_t& value) { return moody.try_dequeue(value); });
  };
#endif
#if defined(HYLE_BENCH_TBB)
  tbb::concurrent_bounded_queue<std::uint64_t> tbb_queue;
  tbb_queue.set_capacity(kCap);
  auto tbb_wave = [&] {
    checked_single_thread_wave(
        [&](std::uint64_t value) { return tbb_queue.try_push(value); },
        [&](std::uint64_t& value) { return tbb_queue.try_pop(value); });
  };
#endif
#if defined(HYLE_BENCH_RIGTORP)
  rigtorp::mpmc::Queue<std::uint64_t> rigtorp_queue(kCap);
  auto rigtorp_wave = [&] {
    checked_single_thread_wave(
        [&](std::uint64_t value) { return rigtorp_queue.try_push(value); },
        [&](std::uint64_t& value) { return rigtorp_queue.try_pop(value); });
  };
#endif
  // Construction, implicit producer registration and first-touch are cold work.
  // Destruction is also outside each timed callback for every enabled peer.
  for (unsigned wave = 0; wave != 32; ++wave) {
    bounded_wave();
#if defined(HYLE_BENCH_MOODYCAMEL)
    moody_wave();
#endif
#if defined(HYLE_BENCH_TBB)
    tbb_wave();
#endif
#if defined(HYLE_BENCH_RIGTORP)
    rigtorp_wave();
#endif
  }

  ankerl::nanobench::Bench b;
  b.title("queue warm single-thread fill+drain (512 checked items)")
      .relative(true)
      .batch(static_cast<std::uint64_t>(kBatch) * 2)  // push + pop; empty check extra
      .minEpochIterations(2000);
  b.run("HYLE BoundedQueue", bounded_wave);
#if defined(HYLE_BENCH_MOODYCAMEL)
  b.run("moodycamel ConcurrentQueue", moody_wave);
#endif
#if defined(HYLE_BENCH_TBB)
  b.run("tbb concurrent_bounded_queue", tbb_wave);
#endif
#if defined(HYLE_BENCH_RIGTORP)
  b.run("rigtorp MPMCQueue (bounded ring)", rigtorp_wave);
#endif
}

// ---- 2. MPMC throughput (P producers / C consumers, end-to-end) -------------
// Each producer pushes kPerThread disjoint ids; consumers drain until every
// producer has finished AND the queue reads empty. Spin-retry on a full bounded
// queue (the bounded queues only). batch == kTotal so the number is ns per item
// moved through the queue.
//
// Each consumer accumulates payload checks locally. After acquiring all producer
// completions it retries pop before accepting an empty queue as fully drained.
// Count, sum and xor are checked after join; no per-item shared counter is added.
template <int kProducers, int kConsumers>
void bench_mpmc(bool bounded_only = false) {
  constexpr int kPerThread = 50000;
  constexpr int kTotal = kProducers * kPerThread;

  ankerl::nanobench::Bench b;
  char title[96];
  std::snprintf(title, sizeof(title), "queue MPMC %dP/%dC (%d items)",
                kProducers, kConsumers, kTotal);
  // Every episode includes queue allocation/destruction and P+C thread
  // construction/join. This is not a measurement of warm queue operations alone.
  b.title(title).relative(true).batch(kTotal).minEpochIterations(50);
  if (bounded_only) b.epochs(11).epochIterations(2).warmup(1);
  if (std::getenv("HYLE_BENCH_SMOKE") != nullptr)
    b.epochs(1).epochIterations(1).warmup(0).output(nullptr);

  auto bounded_episode = [&] {
    BoundedQueue<std::uint64_t> q(kCap);
    if (!q.valid()) std::abort();
    std::atomic<int> producers_done{0};
    std::array<ConsumerResult, kConsumers> consumers{};
    std::vector<std::thread> ts;
    for (int p = 0; p < kProducers; ++p)
      ts.emplace_back([&q, &producers_done, p] {
        std::uint64_t base = 1 + static_cast<std::uint64_t>(p) * kPerThread;
        for (int k = 0; k < kPerThread; ++k)
          while (!q.try_push(base + k)) std::this_thread::yield();
        producers_done.fetch_add(1, std::memory_order_release);
      });
    for (int c = 0; c < kConsumers; ++c)
      ts.emplace_back([&q, &producers_done, &consumers, c] {
        std::uint64_t out = 0, local = 0, sum = 0, xor_sum = 0;
        bool draining = false;
        for (;;) {
          if (q.try_pop(out)) {
            ++local;
            sum += out;
            xor_sum ^= out;
          } else if (draining) {
            break;
          } else if (producers_done.load(std::memory_order_acquire) == kProducers) {
            draining = true;  // Retry empty only after acquiring all publications.
          } else {
            std::this_thread::yield();
          }
        }
        consumers[static_cast<std::size_t>(c)] = {local, sum, xor_sum};
      });
    for (auto& t : ts) t.join();
    require_conservation(consumers, static_cast<std::uint64_t>(kTotal));
  };
  if (!bounded_only) b.run("HYLE BoundedQueue", bounded_episode);

  // One shard per producer: total capacity is P*kCap, and only per-shard order
  // remains. This is a different bounded-admission policy from the single ring.
  if (!bounded_only) b.run("HYLE ShardedQueue (1 shard/producer)", [&] {
    ShardedQueue<std::uint64_t> q(kProducers, kCap);
    if (!q.valid()) std::abort();
    std::atomic<int> producers_done{0};
    std::array<ConsumerResult, kConsumers> consumers{};
    std::vector<std::thread> ts;
    for (int p = 0; p < kProducers; ++p)
      ts.emplace_back([&q, &producers_done, p] {
        std::uint64_t base = 1 + static_cast<std::uint64_t>(p) * kPerThread;
        for (int k = 0; k < kPerThread; ++k)
          while (!q.try_push(static_cast<std::size_t>(p), base + k))
            std::this_thread::yield();
        producers_done.fetch_add(1, std::memory_order_release);
      });
    for (int c = 0; c < kConsumers; ++c)
      ts.emplace_back([&q, &producers_done, &consumers, c] {
        std::uint64_t out = 0, local = 0, sum = 0, xor_sum = 0;
        bool draining = false;
        std::size_t cursor = static_cast<std::size_t>(c);  // spread across shards
        for (;;) {
          if (q.try_pop(cursor, out)) {
            ++local;
            sum += out;
            xor_sum ^= out;
          } else if (draining) {
            break;
          } else if (producers_done.load(std::memory_order_acquire) == kProducers) {
            draining = true;
          } else {
            std::this_thread::yield();
          }
        }
        consumers[static_cast<std::size_t>(c)] = {local, sum, xor_sum};
      });
    for (auto& t : ts) t.join();
    require_conservation(consumers, static_cast<std::uint64_t>(kTotal));
  });

#if defined(HYLE_BENCH_MOODYCAMEL)
  if (!bounded_only) b.run("moodycamel ConcurrentQueue", [&] {
    moodycamel::ConcurrentQueue<std::uint64_t> q(kCap);
    std::atomic<int> producers_done{0};
    std::array<ConsumerResult, kConsumers> consumers{};
    std::vector<std::thread> ts;
    for (int p = 0; p < kProducers; ++p)
      ts.emplace_back([&q, &producers_done, p] {
        std::uint64_t base = 1 + static_cast<std::uint64_t>(p) * kPerThread;
        for (int k = 0; k < kPerThread; ++k)
          if (!q.enqueue(base + k)) std::abort();
        producers_done.fetch_add(1, std::memory_order_release);
      });
    for (int c = 0; c < kConsumers; ++c)
      ts.emplace_back([&q, &producers_done, &consumers, c] {
        std::uint64_t out = 0, local = 0, sum = 0, xor_sum = 0;
        bool draining = false;
        for (;;) {
          if (q.try_dequeue(out)) {
            ++local;
            sum += out;
            xor_sum ^= out;
          } else if (draining) {
            break;
          } else if (producers_done.load(std::memory_order_acquire) == kProducers) {
            draining = true;
          } else {
            std::this_thread::yield();
          }
        }
        consumers[static_cast<std::size_t>(c)] = {local, sum, xor_sum};
      });
    for (auto& t : ts) t.join();
    require_conservation(consumers, static_cast<std::uint64_t>(kTotal));
  });
#endif
#if defined(HYLE_BENCH_TBB)
  if (!bounded_only) b.run("tbb concurrent_bounded_queue", [&] {
    tbb::concurrent_bounded_queue<std::uint64_t> q;
    q.set_capacity(kCap);
    std::atomic<int> producers_done{0};
    std::array<ConsumerResult, kConsumers> consumers{};
    std::vector<std::thread> ts;
    for (int p = 0; p < kProducers; ++p)
      ts.emplace_back([&q, &producers_done, p] {
        std::uint64_t base = 1 + static_cast<std::uint64_t>(p) * kPerThread;
        for (int k = 0; k < kPerThread; ++k)
          while (!q.try_push(base + k)) std::this_thread::yield();
        producers_done.fetch_add(1, std::memory_order_release);
      });
    for (int c = 0; c < kConsumers; ++c)
      ts.emplace_back([&q, &producers_done, &consumers, c] {
        std::uint64_t out = 0, local = 0, sum = 0, xor_sum = 0;
        bool draining = false;
        for (;;) {
          if (q.try_pop(out)) {
            ++local;
            sum += out;
            xor_sum ^= out;
          } else if (draining) {
            break;
          } else if (producers_done.load(std::memory_order_acquire) == kProducers) {
            draining = true;
          } else {
            std::this_thread::yield();
          }
        }
        consumers[static_cast<std::size_t>(c)] = {local, sum, xor_sum};
      });
    for (auto& t : ts) t.join();
    require_conservation(consumers, static_cast<std::uint64_t>(kTotal));
  });
#endif
#if defined(HYLE_BENCH_RIGTORP)
  // Fixed-capacity single-ring peer; the same outer retry policy is charged.
  auto rigtorp_episode = [&] {
    rigtorp::mpmc::Queue<std::uint64_t> q(kCap);
    std::atomic<int> producers_done{0};
    std::array<ConsumerResult, kConsumers> consumers{};
    std::vector<std::thread> ts;
    for (int p = 0; p < kProducers; ++p)
      ts.emplace_back([&q, &producers_done, p] {
        std::uint64_t base = 1 + static_cast<std::uint64_t>(p) * kPerThread;
        for (int k = 0; k < kPerThread; ++k)
          while (!q.try_push(base + k)) std::this_thread::yield();
        producers_done.fetch_add(1, std::memory_order_release);
      });
    for (int c = 0; c < kConsumers; ++c)
      ts.emplace_back([&q, &producers_done, &consumers, c] {
        std::uint64_t out = 0, local = 0, sum = 0, xor_sum = 0;
        bool draining = false;
        for (;;) {
          if (q.try_pop(out)) {
            ++local;
            sum += out;
            xor_sum ^= out;
          } else if (draining) {
            break;
          } else if (producers_done.load(std::memory_order_acquire) == kProducers) {
            draining = true;
          } else {
            std::this_thread::yield();
          }
        }
        consumers[static_cast<std::size_t>(c)] = {local, sum, xor_sum};
      });
    for (auto& t : ts) t.join();
    require_conservation(consumers, static_cast<std::uint64_t>(kTotal));
  };
  if (bounded_only) {
    const char* aa = std::getenv("HYLE_BENCH_AA_ROUTE");
    const bool same_bounded = aa != nullptr && std::string_view(aa) == "candidate";
    const bool same_rigtorp = aa != nullptr && std::string_view(aa) == "peer";
    if (aa != nullptr && !same_bounded && !same_rigtorp) std::abort();
    auto run_bounded = [&] {
      if (same_rigtorp) b.run("HYLE BoundedQueue", rigtorp_episode);
      else b.run("HYLE BoundedQueue", bounded_episode);
    };
    auto run_rigtorp = [&] {
      if (same_bounded) b.run("rigtorp MPMCQueue (bounded ring)", bounded_episode);
      else b.run("rigtorp MPMCQueue (bounded ring)", rigtorp_episode);
    };
    if (std::getenv("HYLE_BENCH_RIVAL_FIRST") != nullptr) {
      run_rigtorp();
      run_bounded();
    } else {
      run_bounded();
      run_rigtorp();
    }
  } else {
    b.run("rigtorp MPMCQueue (bounded ring)", rigtorp_episode);
  }
#endif
  const char* json = bounded_only ? std::getenv("HYLE_BENCH_JSON") : nullptr;
  if (json != nullptr) {
    std::ofstream output(json);
    if (!output) std::abort();
    b.render(ankerl::nanobench::templates::json(), output);
    if (!output) std::abort();
  }
}

// ---- 3. work-stealing OWNER (single-thread fill+drain) ---------------------
// FIFO and LIFO are separate policies. Queue construction/destruction is outside
// timed callbacks; each arm gets 200 explicit checked warm waves. Crossbeam's
// native Worker can still shrink/grow during each wave, unlike these fixed rings.
template <class Push, class Pop>
void checked_ws_owner_wave(Push push, Pop pop, bool lifo) {
  for (int i = 0; i < kBatch; ++i)
    if (!push(static_cast<std::uint64_t>(i))) std::abort();
  std::uint64_t value = 0;
  for (int i = 0; i < kBatch; ++i) {
    const auto expected = static_cast<std::uint64_t>(lifo ? kBatch - 1 - i : i);
    if (!pop(value) || value != expected) std::abort();
  }
  if (pop(value)) std::abort();
  ankerl::nanobench::doNotOptimizeAway(value);
}

struct CheckedStealBatch {
  std::array<bool, kBatch> seen{};
  ConsumerResult result;

  void add(std::uint64_t value) {
    if (value >= kBatch || seen[static_cast<std::size_t>(value)]) std::abort();
    seen[static_cast<std::size_t>(value)] = true;
    ++result.count;
    result.sum += value;
    result.xor_sum ^= value;
  }

  void finish() const {
    if (result.count != kBatch || result.sum != kBatch * (kBatch - 1) / 2 ||
        result.xor_sum != 0)
      std::abort();
    ankerl::nanobench::doNotOptimizeAway(result);
  }
};

void bench_ws_owner() {
  WsFifoQueue<std::uint64_t> fifo(kCap);
  WorkStealingDeque<std::uint64_t> lifo(kCap);
  if (!fifo.valid() || !lifo.valid()) std::abort();
  auto fifo_wave = [&] {
    checked_ws_owner_wave(
        [&](std::uint64_t value) { return fifo.try_push_back(value); },
        [&](std::uint64_t& value) { return fifo.try_pop_front(value); }, false);
  };
  auto lifo_wave = [&] {
    checked_ws_owner_wave(
        [&](std::uint64_t value) { return lifo.try_push_bottom(value); },
        [&](std::uint64_t& value) { return lifo.try_pop_bottom(value); }, true);
  };
#if defined(HYLE_BENCH_MOODYCAMEL)
  // General MPMC reference only; not a same-family owner comparison.
  moodycamel::ConcurrentQueue<std::uint64_t> moody(kCap);
  auto moody_wave = [&] {
    checked_ws_owner_wave(
        [&](std::uint64_t value) { return moody.enqueue(value); },
        [&](std::uint64_t& value) { return moody.try_dequeue(value); }, false);
  };
#endif
  for (unsigned wave = 0; wave != 200; ++wave) {
    fifo_wave();
    lifo_wave();
#if defined(HYLE_BENCH_MOODYCAMEL)
    moody_wave();
#endif
  }
  // Functional smoke never enters nanobench's timing path.
  if (std::getenv("HYLE_BENCH_SMOKE") != nullptr) return;
  ankerl::nanobench::Bench b;
  b.title("work-stealing owner single-thread checked fill+drain (512 items)")
      .relative(true)
      .batch(static_cast<std::uint64_t>(kBatch) * 2)
      .minEpochIterations(2000);
  b.run("HYLE WsFifoQueue (FIFO owner)", fifo_wave);
  b.run("HYLE WorkStealingDeque (LIFO owner)", lifo_wave);
#if defined(HYLE_BENCH_MOODYCAMEL)
  b.run("moodycamel (general MPMC ref)", moody_wave);
#endif
}

// ---- 4. work-stealing APIs, single-thread fill+steal+drain -------------------
// There is no concurrent owner/thief here. Owner fill is inside every callback;
// the denominator is 512 transferred items, not isolated steal operations.
// Hyle steals ceil(remaining/2); Crossbeam's default batch cap is 32. Compare
// conservation, not batch boundaries or the order of each directly returned item.
void bench_ws_steal() {
  WsFifoQueue<std::uint64_t> victim(kCap);
  WsFifoQueue<std::uint64_t> thief(kCap);
  WorkStealingDeque<std::uint64_t> deque(kCap);
  if (!victim.valid() || !thief.valid() || !deque.valid()) std::abort();
  auto batch_wave = [&] {
    for (int i = 0; i < kBatch; ++i)
      if (!victim.try_push_back(static_cast<std::uint64_t>(i))) std::abort();
    CheckedStealBatch checked;
    std::uint64_t value = 0;
    while (victim.try_steal_into(thief, value)) {
      checked.add(value);
      while (thief.try_pop_front(value)) checked.add(value);
    }
    checked.finish();
    if (victim.try_pop_front(value) || thief.try_pop_front(value)) std::abort();
  };
  auto single_wave = [&] {
    for (int i = 0; i < kBatch; ++i)
      if (!deque.try_push_bottom(static_cast<std::uint64_t>(i))) std::abort();
    std::uint64_t count = 0, value = 0;
    for (;;) {
      const auto status = deque.try_steal_top(value);
      if (status == hyle::container::StealStatus::kRetry) continue;
      if (status == hyle::container::StealStatus::kEmpty) break;
      if (count >= kBatch || value != count++) std::abort();
    }
    if (count != kBatch || deque.try_pop_bottom(value)) std::abort();
    ankerl::nanobench::doNotOptimizeAway(count);
  };
  for (unsigned wave = 0; wave != 200; ++wave) {
    batch_wave();
    single_wave();
  }
  if (std::getenv("HYLE_BENCH_SMOKE") != nullptr) return;
  ankerl::nanobench::Bench b;
  b.title("work-stealing API single-thread checked fill+steal+drain (512 items)")
      .relative(true)
      .batch(kBatch)
      .minEpochIterations(1000);
  b.run("HYLE WsFifoQueue steal-half + local drain", batch_wave);
  b.run("HYLE WorkStealingDeque single-item steal", single_wave);
}

// ---- 5. ConcurrentVector: push_back + indexed read (vs tbb) ------------------
void bench_concurrent_vector() {
  constexpr int kN = 100000;
  const char* variant = std::getenv("HYLE_QUEUE_BENCH_VARIANT");
  const bool run_hyle =
      variant == nullptr || std::string_view(variant) == "hyle";
  const bool run_tbb =
      variant == nullptr || std::string_view(variant) == "tbb";
  const char* selected_case = std::getenv("HYLE_QUEUE_BENCH_CV_CASE");
  const auto run_case = [selected_case](std::string_view name) {
    return selected_case == nullptr || std::string_view(selected_case) == name;
  };
  if (run_case("push")) {
    ankerl::nanobench::Bench b;
    b.title("ConcurrentVector single-thread push_back (100k)")
        .relative(true)
        .batch(kN)
        .minEpochIterations(20);
    if (run_hyle) b.run("HYLE ConcurrentVector", [&] {
      ConcurrentVector<std::uint64_t> v;
      for (int i = 0; i < kN; ++i) v.push_back(static_cast<std::uint64_t>(i));
      ankerl::nanobench::doNotOptimizeAway(v.size());
    });
#if defined(HYLE_BENCH_TBB)
    if (run_tbb) b.run("tbb concurrent_vector", [&] {
      tbb::concurrent_vector<std::uint64_t> v;
      for (int i = 0; i < kN; ++i) v.push_back(static_cast<std::uint64_t>(i));
      ankerl::nanobench::doNotOptimizeAway(v.size());
    });
#endif
  }
  if (run_case("writers")) {
    constexpr int kThreads = 8;
    constexpr int kPerThread = 25000;
    constexpr int kTotal = kThreads * kPerThread;
    ankerl::nanobench::Bench b;
    b.title("ConcurrentVector 8-writer reserved append (200k)")
        .relative(true)
        .batch(kTotal)
        .minEpochIterations(10);
    if (run_hyle) b.run("HYLE ConcurrentVector", [&] {
      ConcurrentVector<std::uint64_t> v;
      v.reserve(kTotal);
      std::atomic<bool> go{false};
      std::vector<std::thread> writers;
      writers.reserve(kThreads);
      for (int t = 0; t < kThreads; ++t) {
        writers.emplace_back([&, t] {
          while (!go.load(std::memory_order_acquire)) {}
          for (int i = 0; i < kPerThread; ++i)
            v.emplace_back(static_cast<std::uint64_t>(t) * kPerThread + i);
        });
      }
      go.store(true, std::memory_order_release);
      for (auto& writer : writers) writer.join();
      ankerl::nanobench::doNotOptimizeAway(v.size());
    });
#if defined(HYLE_BENCH_TBB)
    if (run_tbb) b.run("tbb concurrent_vector", [&] {
      tbb::concurrent_vector<std::uint64_t> v;
      v.reserve(kTotal);
      std::atomic<bool> go{false};
      std::vector<std::thread> writers;
      writers.reserve(kThreads);
      for (int t = 0; t < kThreads; ++t) {
        writers.emplace_back([&, t] {
          while (!go.load(std::memory_order_acquire)) {}
          for (int i = 0; i < kPerThread; ++i)
            v.emplace_back(static_cast<std::uint64_t>(t) * kPerThread + i);
        });
      }
      go.store(true, std::memory_order_release);
      for (auto& writer : writers) writer.join();
      ankerl::nanobench::doNotOptimizeAway(v.size());
    });
#endif
  }
  if (run_case("index")) {
    // Lock-free read hot path: sequential index sum over a prebuilt vector.
    ConcurrentVector<std::uint64_t> hv;
    for (int i = 0; i < kN; ++i) hv.push_back(static_cast<std::uint64_t>(i));
    ankerl::nanobench::Bench b;
    b.title("ConcurrentVector sequential read sum (100k)")
        .relative(true)
        .batch(kN)
        .minEpochIterations(50);
    if (run_hyle) b.run("HYLE ConcurrentVector", [&] {
      std::uint64_t sum = 0;
      for (int i = 0; i < kN; ++i) sum += hv[static_cast<std::size_t>(i)];
      ankerl::nanobench::doNotOptimizeAway(sum);
    });
#if defined(HYLE_BENCH_TBB)
    static tbb::concurrent_vector<std::uint64_t> tv;
    if (tv.empty())
      for (int i = 0; i < kN; ++i) tv.push_back(static_cast<std::uint64_t>(i));
    if (run_tbb) b.run("tbb concurrent_vector", [&] {
      std::uint64_t sum = 0;
      for (int i = 0; i < kN; ++i) sum += tv[static_cast<std::size_t>(i)];
      ankerl::nanobench::doNotOptimizeAway(sum);
    });
#endif
  }
  if (run_case("iterate")) {
    ConcurrentVector<std::uint64_t> hv;
    for (int i = 0; i < kN; ++i) hv.push_back(static_cast<std::uint64_t>(i));
    ankerl::nanobench::Bench b;
    b.title("ConcurrentVector range iteration sum (100k)")
        .relative(true)
        .batch(kN)
        .minEpochIterations(50);
    if (run_hyle) b.run("HYLE ConcurrentVector", [&] {
      std::uint64_t sum = 0;
      for (std::uint64_t value : hv) sum += value;
      ankerl::nanobench::doNotOptimizeAway(sum);
    });
#if defined(HYLE_BENCH_TBB)
    static tbb::concurrent_vector<std::uint64_t> tv;
    if (tv.empty())
      for (int i = 0; i < kN; ++i) tv.push_back(static_cast<std::uint64_t>(i));
    if (run_tbb) b.run("tbb concurrent_vector", [&] {
      std::uint64_t sum = 0;
      for (std::uint64_t value : tv) sum += value;
      ankerl::nanobench::doNotOptimizeAway(sum);
    });
#endif
  }

  // ---- CONCURRENT lock-free read WHILE a writer appends (the headline) -------
  // The single-thread reads above don't exercise ConcurrentVector's actual claim:
  // "insert once, read many, FROM MANY THREADS" with LOCK-FREE readers. Here R
  // reader threads sum a FULLY-BUILT prefix [0,kPre) in a loop while ONE writer
  // appends another kPre items concurrently (forcing segment growth). Readers only
  // touch the pre-built prefix, so neither HYLE nor tbb can read a half-constructed
  // tail (fair). This stresses stable segmented indexing while the reservation
  // counter and cold growth path are active on another thread. Win or lose,
  // printed. batch =
  // R*kPre*kRounds (total reads); the number is ns per lock-free read under writer
  // contention. minEpochIterations low (each epoch spawns+joins R+1 threads).
  if (run_case("read-write")) {
    constexpr int kPre = 100000;     // pre-built, fully-constructed prefix
    constexpr int kReaders = 4;      // concurrent lock-free readers
    constexpr int kRounds = 8;       // full-prefix sweeps per reader thread
    ankerl::nanobench::Bench b;
    b.title("ConcurrentVector 4 readers (lock-free) + 1 writer append")
        .relative(true)
        .batch(static_cast<std::uint64_t>(kReaders) * kPre * kRounds)
        .minEpochIterations(10);

    if (run_hyle) b.run("HYLE ConcurrentVector", [&] {
      ConcurrentVector<std::uint64_t> v;
      for (int i = 0; i < kPre; ++i) v.push_back(static_cast<std::uint64_t>(i));
      std::atomic<bool> go{false};
      std::vector<std::thread> ts;
      std::atomic<std::uint64_t> sink{0};
      for (int r = 0; r < kReaders; ++r)
        ts.emplace_back([&] {
          while (!go.load(std::memory_order_acquire)) {}
          std::uint64_t s = 0;
          for (int round = 0; round < kRounds; ++round)
            for (int i = 0; i < kPre; ++i) s += v[static_cast<std::size_t>(i)];
          sink.fetch_add(s, std::memory_order_relaxed);
        });
      std::thread writer([&] {
        while (!go.load(std::memory_order_acquire)) {}
        for (int i = 0; i < kPre; ++i)
          v.push_back(static_cast<std::uint64_t>(kPre + i));  // forces growth
      });
      go.store(true, std::memory_order_release);
      for (auto& t : ts) t.join();
      writer.join();
      ankerl::nanobench::doNotOptimizeAway(sink.load());
    });
#if defined(HYLE_BENCH_TBB)
    if (run_tbb) b.run("tbb concurrent_vector", [&] {
      tbb::concurrent_vector<std::uint64_t> v;
      for (int i = 0; i < kPre; ++i) v.push_back(static_cast<std::uint64_t>(i));
      std::atomic<bool> go{false};
      std::vector<std::thread> ts;
      std::atomic<std::uint64_t> sink{0};
      for (int r = 0; r < kReaders; ++r)
        ts.emplace_back([&] {
          while (!go.load(std::memory_order_acquire)) {}
          std::uint64_t s = 0;
          for (int round = 0; round < kRounds; ++round)
            for (int i = 0; i < kPre; ++i) s += v[static_cast<std::size_t>(i)];
          sink.fetch_add(s, std::memory_order_relaxed);
        });
      std::thread writer([&] {
        while (!go.load(std::memory_order_acquire)) {}
        for (int i = 0; i < kPre; ++i)
          v.push_back(static_cast<std::uint64_t>(kPre + i));
      });
      go.store(true, std::memory_order_release);
      for (auto& t : ts) t.join();
      writer.join();
      ankerl::nanobench::doNotOptimizeAway(sink.load());
    });
#endif
  }
}

}  // namespace

int main() {
  const char* filter = std::getenv("HYLE_QUEUE_BENCH_FILTER");
  if (filter != nullptr) {
    if (std::string_view(filter) == "ws-owner") {
      bench_ws_owner();
      return 0;
    }
    if (std::string_view(filter) == "ws-steal") {
      bench_ws_steal();
      return 0;
    }
    if (std::string_view(filter) == "concurrent-vector") {
      bench_concurrent_vector();
      return 0;
    }
#if defined(HYLE_BENCH_RIGTORP)
    if (std::string_view(filter) == "bounded-mpmc") {
      bench_mpmc<4, 4>(true);
      return 0;
    }
#endif
    if (std::string_view(filter) == "mpmc") {
      bench_mpmc<4, 4>();
      return 0;
    }
    return 2;
  }
  bench_single_thread();
  bench_mpmc<2, 2>();
  bench_mpmc<4, 4>();
  bench_ws_owner();
  bench_ws_steal();
  bench_concurrent_vector();
  return 0;
}
