// StringBuf interning vs the naive std way (std::unordered_map<std::string,id>
// + a vector to keep the strings). Measures interning a stream with heavy
// duplication (the real interning workload: many repeats collapse to one
// store).

#include <nanobench.h>

#include <array>
#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(HYLE_BENCH_ABSL)
#  include <absl/container/flat_hash_set.h>
#endif

#include "hyle/container/string_buf.h"
#include "hyle/container/string_ref.h"
#include "hyle/mem/allocator.h"

using hyle::container::string_ref;
using hyle::container::StringBuf;
using hyle::mem::system_allocator;

namespace {
constexpr int kStream = 100000; // intern calls
constexpr int kUnique = 2000;   // distinct strings (so ~50x dup)

std::vector<std::string> gen_stream() {
  std::mt19937_64 rng(7);
  std::vector<std::string> pool;
  pool.reserve(kUnique);
  for (int i = 0; i < kUnique; ++i) pool.push_back("symbol_" + std::to_string(rng() % 1000000));
  std::vector<std::string> stream;
  stream.reserve(kStream);
  for (int i = 0; i < kStream; ++i) stream.push_back(pool[rng() % kUnique]);
  return stream;
}

template <class Set> class ShardedPeer {
public:
  ShardedPeer() {
    const unsigned workers = std::thread::hardware_concurrency();
    shard_count_ = workers > 32 ? 64u : workers > 8 ? 32u : 16u;
  }

  void intern(const std::string& text) {
    const std::size_t shard_index = std::hash<std::string>{}(text) & (shard_count_ - 1);
    Shard& shard = shards_[shard_index];
    std::lock_guard<std::mutex> lock(shard.mu);
    shard.values.emplace(text);
  }

  std::size_t size() const {
    std::size_t result = 0;
    for (std::size_t i = 0; i < shard_count_; ++i) result += shards_[i].values.size();
    return result;
  }

private:
  struct alignas(64) Shard {
    std::mutex mu;
    Set values;
  };
  std::array<Shard, 64> shards_;
  std::size_t shard_count_ = 16;
};

std::vector<std::vector<std::string>> gen_thread_streams(int thread_count, int per_thread) {
  std::vector<std::vector<std::string>> streams(thread_count);
  for (int thread = 0; thread < thread_count; ++thread) {
    auto& stream = streams[thread];
    stream.reserve(per_thread);
    for (int i = 0; i < per_thread; ++i)
      stream.push_back("thread_" + std::to_string(thread) + "_symbol_" + std::to_string(i & 4095));
  }
  return streams;
}
} // namespace

int bench_string_buf() {
  const std::vector<std::string> stream = gen_stream();
  std::vector<string_ref> stream_views;
  stream_views.reserve(stream.size());
  for (const auto& s : stream)
    stream_views.emplace_back(s.data(), static_cast<std::uint32_t>(s.size()));

  ankerl::nanobench::Bench b;
  b.title("intern 100k-stream / 2k-unique").relative(true).batch(kStream);

  // Floor reference: a NON-thread-safe std interner (no lock). NOT StringBuf's
  // apples-to-apples peer -- StringBuf is thread-safe (takes a mutex per
  // intern), so racing it against an unlocked map measures "thread-safe vs
  // not", not the interner cores. Kept as the single-thread floor.
  b.run("std::unordered_map interner (NO lock, floor)", [&] {
    std::unordered_map<std::string, std::uint32_t> pool;
    const char* last = nullptr;
    for (const auto& s : stream) {
      auto [it, ins] = pool.try_emplace(s, 0);
      last = it->first.c_str();
    }
    ankerl::nanobench::doNotOptimizeAway(last);
    ankerl::nanobench::doNotOptimizeAway(pool.size());
  });

  // Apples-to-apples peer: a THREAD-SAFE std interner (unordered_map +
  // std::mutex), same thread-safety contract as StringBuf. Both pay one
  // uncontended lock/op; the honest question is whose interner CORE is faster
  // (StringBuf uses SwissTable, which beats unordered_map ~2.4x on find).
  b.run("std::unordered_map + mutex interner (thread-safe peer)", [&] {
    std::unordered_map<std::string, std::uint32_t> pool;
    std::mutex mu;
    const char* last = nullptr;
    for (const auto& s : stream) {
      std::lock_guard<std::mutex> lk(mu);
      auto [it, ins] = pool.try_emplace(s, 0);
      last = it->first.c_str();
    }
    ankerl::nanobench::doNotOptimizeAway(last);
    ankerl::nanobench::doNotOptimizeAway(pool.size());
  });

  b.run("HYLE StringBuf (thread-safe, SwissTable core)", [&] {
    StringBuf buf(system_allocator());
    const char* last = nullptr;
    for (const auto& s : stream) {
      string_ref r = buf.intern(string_ref{s.data(), static_cast<std::uint32_t>(s.size())});
      last = r.data;
    }
    ankerl::nanobench::doNotOptimizeAway(last);
    ankerl::nanobench::doNotOptimizeAway(buf.size());
  });

  b.run("HYLE StringBuf intern_many (one lock set / batch)", [&] {
    StringBuf buf(system_allocator());
    const std::size_t done = buf.intern_many(stream_views.data(), nullptr, stream_views.size());
    ankerl::nanobench::doNotOptimizeAway(done);
    ankerl::nanobench::doNotOptimizeAway(buf.size());
  });

  // Long unique strings are the complexity adversary: an uncached hash scans
  // every resident key again on each table growth. StringBuf stores the hash in
  // the key, so each input is byte-hashed once and growth only moves fixed-size
  // slots. This benchmark keeps the bytes large enough to make regressions
  // loud.
  {
    constexpr int kLongCount = 4096;
    constexpr int kLongBytes = 1024;
    std::vector<std::string> long_strings;
    std::vector<string_ref> long_views;
    long_strings.reserve(kLongCount);
    long_views.reserve(kLongCount);
    for (int i = 0; i < kLongCount; ++i) {
      long_strings.push_back(std::string(kLongBytes, static_cast<char>('a' + i % 23)) +
                             std::to_string(i));
      const auto& text = long_strings.back();
      long_views.emplace_back(text.data(), static_cast<std::uint32_t>(text.size()));
    }
    ankerl::nanobench::Bench growth;
    growth.title("long-string growth: 4096 unique x 1KiB")
        .relative(true)
        .batch(kLongCount)
        .minEpochIterations(10);
    growth.run("std::unordered_set (unlocked floor)", [&] {
      std::unordered_set<std::string> values;
      for (const auto& text : long_strings) values.emplace(text);
      ankerl::nanobench::doNotOptimizeAway(values.size());
    });
    growth.run("HYLE StringBuf intern_many (cached hash)", [&] {
      StringBuf buf(system_allocator());
      const std::size_t done = buf.intern_many(long_views.data(), nullptr, long_views.size());
      ankerl::nanobench::doNotOptimizeAway(done);
      ankerl::nanobench::doNotOptimizeAway(buf.size());
    });
#if defined(HYLE_BENCH_ABSL)
    growth.run("absl::flat_hash_set (unlocked SOTA floor)", [&] {
      absl::flat_hash_set<std::string> values;
      for (const auto& text : long_strings) values.emplace(text);
      ankerl::nanobench::doNotOptimizeAway(values.size());
    });
#endif
  }

  // Same-contract contention test: both sides select the same adaptive
  // 16/32/64 independently locked shard count and own stable strings. Thread
  // creation is amortized over 400k calls; this measures contention scaling.
  {
    constexpr int kThreads = 8;
    constexpr int kPerThread = 50000;
    const auto streams = gen_thread_streams(kThreads, kPerThread);
    ankerl::nanobench::Bench concurrent;
    concurrent.title("8-thread sharded intern / 400k calls")
        .relative(true)
        .batch(static_cast<std::uint64_t>(kThreads) * kPerThread)
        .minEpochIterations(1);
    concurrent.run("adaptive-shard std::unordered_set + mutex peer", [&] {
      ShardedPeer<std::unordered_set<std::string>> peer;
      std::array<std::thread, kThreads> threads;
      for (int t = 0; t < kThreads; ++t)
        threads[t] = std::thread([&, t] {
          for (const auto& text : streams[t]) peer.intern(text);
        });
      for (auto& thread : threads) thread.join();
      ankerl::nanobench::doNotOptimizeAway(peer.size());
    });
    concurrent.run("HYLE StringBuf adaptive shards", [&] {
      StringBuf buf(system_allocator());
      std::array<std::thread, kThreads> threads;
      for (int t = 0; t < kThreads; ++t)
        threads[t] = std::thread([&, t] {
          for (const auto& text : streams[t]) buf.intern(text);
        });
      for (auto& thread : threads) thread.join();
      ankerl::nanobench::doNotOptimizeAway(buf.size());
    });
#if defined(HYLE_BENCH_ABSL)
    concurrent.run("adaptive-shard absl::flat_hash_set + mutex SOTA peer", [&] {
      ShardedPeer<absl::flat_hash_set<std::string>> peer;
      std::array<std::thread, kThreads> threads;
      for (int t = 0; t < kThreads; ++t)
        threads[t] = std::thread([&, t] {
          for (const auto& text : streams[t]) peer.intern(text);
        });
      for (auto& thread : threads) thread.join();
      ankerl::nanobench::doNotOptimizeAway(peer.size());
    });
#endif
  }

  // ---- add/retain/remove refcount churn (the lifecycle intern never hits)
  // ---- intern() above is steady-state insert (2000 uniques, never freed). The
  // refcount lifecycle -- add (intern + bump), retain (bump by hash), remove
  // (drop; on the last ref, SwissTable::erase + alloc.deallocate) -- exercises
  // a DIFFERENT path: the erase + free churn. Workload: add each unique, retain
  // it, then remove it twice (drives the count to zero -> storage reclaimed).
  // Rival: a std:: unordered_map<string,refcount> doing the same
  // add/retain/remove (erase on zero).
  {
    std::vector<std::string> uniques;
    uniques.reserve(kUnique);
    {
      std::mt19937_64 rng(11);
      for (int i = 0; i < kUnique; ++i) uniques.push_back("sym_" + std::to_string(rng() % 1000000));
    }
    ankerl::nanobench::Bench b2;
    b2.title("refcount churn: add+retain+remove*2 (2k uniques)")
        .relative(true)
        .batch(static_cast<std::uint64_t>(kUnique) * 4); // add+retain+2x remove
    std::vector<string_ref> lifecycle_views;
    std::vector<hyle::container::InternId> lifecycle_ids(uniques.size());
    lifecycle_views.reserve(uniques.size());
    for (const auto& text : uniques)
      lifecycle_views.emplace_back(text.data(), static_cast<std::uint32_t>(text.size()));

    b2.run("std::unordered_map<string,int> add/retain/remove", [&] {
      std::unordered_map<std::string, int> pool;
      for (const auto& s : uniques) {
        auto [it, ins] = pool.try_emplace(s, 0); // add
        ++it->second;
        it->second += 1;                       // retain
        it->second -= 1;                       // remove (1)
        if (--it->second == 0) pool.erase(it); // remove (2) -> reclaim
      }
      ankerl::nanobench::doNotOptimizeAway(pool.size());
    });

    // Same handle contract: add resolves bytes -> id, while retain/remove use
    // the id directly. The map owns the string exactly once and its node
    // address stays stable across unordered_map rehashes; the slot vector is
    // the std Same generation-tagged direct-index shape as StringBuf, so this
    // peer does not pay a second key hash merely because its public handle is
    // opaque.
    b2.run("std::unordered_map + direct-id slots + mutex (same contract)", [&] {
      struct Slot {
        const std::string* key;
        int refs;
      };
      std::unordered_map<std::string, std::uint32_t> pool;
      std::vector<Slot> slots;
      slots.reserve(uniques.size());
      std::mutex mu;
      for (const auto& s : uniques) {
        std::uint32_t id;
        {
          std::lock_guard<std::mutex> lk(mu);
          auto [it, inserted] = pool.try_emplace(s, static_cast<std::uint32_t>(slots.size()));
          if (inserted) slots.push_back(Slot{&it->first, 1});
          else ++slots[it->second].refs;
          id = it->second;
        }
        {
          std::lock_guard<std::mutex> lk(mu);
          ++slots[id].refs;
        }
        {
          std::lock_guard<std::mutex> lk(mu);
          --slots[id].refs;
        }
        {
          std::lock_guard<std::mutex> lk(mu);
          Slot& slot = slots[id];
          if (--slot.refs == 0) {
            pool.erase(*slot.key);
            slot.key = nullptr;
          }
        }
      }
      ankerl::nanobench::doNotOptimizeAway(pool.size());
    });

    b2.run("HYLE StringBuf add/retain/remove (direct ID)", [&] {
      StringBuf buf(system_allocator());
      for (const auto& s : uniques) {
        auto h = buf.add(string_ref{s.data(), static_cast<std::uint32_t>(s.size())});
        buf.retain(h);
        buf.remove(h); // drop the retain
        buf.remove(h); // drop the add -> count zero -> erase + deallocate
      }
      ankerl::nanobench::doNotOptimizeAway(buf.size());
    });

    b2.run("HYLE StringBuf batch lifecycle", [&] {
      StringBuf buf(system_allocator());
      const std::size_t added =
          buf.add_many(lifecycle_views.data(), lifecycle_ids.data(), lifecycle_ids.size());
      const std::size_t retained = buf.retain_many(lifecycle_ids.data(), lifecycle_ids.size());
      const std::size_t removed1 = buf.remove_many(lifecycle_ids.data(), lifecycle_ids.size());
      const std::size_t removed2 = buf.remove_many(lifecycle_ids.data(), lifecycle_ids.size());
      ankerl::nanobench::doNotOptimizeAway(added);
      ankerl::nanobench::doNotOptimizeAway(retained);
      ankerl::nanobench::doNotOptimizeAway(removed1);
      ankerl::nanobench::doNotOptimizeAway(removed2);
      ankerl::nanobench::doNotOptimizeAway(buf.size());
    });

    // Repeated churn is where a lifetime pool should differ from a node map:
    // StringBuf reuses its short-entry slabs and ID slots at the high-water
    // mark; the peer retains buckets/slot capacity but still allocates map
    // nodes.
    constexpr int kCycles = 16;
    ankerl::nanobench::Bench reuse;
    reuse.title("repeated lifecycle churn / 16 cycles")
        .relative(true)
        .batch(static_cast<std::uint64_t>(kCycles) * kUnique * 4);
    reuse.run("std direct-id + mutex (buckets retained)", [&] {
      struct Slot {
        const std::string* key;
        int refs;
      };
      std::unordered_map<std::string, std::uint32_t> pool;
      std::vector<Slot> slots;
      slots.reserve(uniques.size());
      std::mutex mu;
      for (int cycle = 0; cycle < kCycles; ++cycle) {
        slots.clear();
        for (const auto& text : uniques) {
          std::uint32_t id;
          {
            std::lock_guard<std::mutex> lock(mu);
            auto [it, inserted] = pool.try_emplace(text, static_cast<std::uint32_t>(slots.size()));
            if (inserted) slots.push_back(Slot{&it->first, 1});
            else ++slots[it->second].refs;
            id = it->second;
          }
          {
            std::lock_guard<std::mutex> lock(mu);
            ++slots[id].refs;
          }
          {
            std::lock_guard<std::mutex> lock(mu);
            --slots[id].refs;
          }
          {
            std::lock_guard<std::mutex> lock(mu);
            Slot& slot = slots[id];
            if (--slot.refs == 0) {
              pool.erase(*slot.key);
              slot.key = nullptr;
            }
          }
        }
      }
      ankerl::nanobench::doNotOptimizeAway(pool.size());
    });
    reuse.run("HYLE StringBuf (slab + slot reuse)", [&] {
      StringBuf buf(system_allocator());
      for (int cycle = 0; cycle < kCycles; ++cycle) {
        for (const auto& text : uniques) {
          const auto id = buf.add(text);
          buf.retain(id);
          buf.remove(id);
          buf.remove(id);
        }
      }
      ankerl::nanobench::doNotOptimizeAway(buf.size());
    });
  }

  return 0;
}
