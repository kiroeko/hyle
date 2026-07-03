// StringBuf interning vs the naive std way (std::unordered_map<std::string,id> +
// a vector to keep the strings). Measures interning a stream with heavy
// duplication (the real interning workload: many repeats collapse to one store).

#include <nanobench.h>

#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "hyle/container/string_buf.h"
#include "hyle/container/string_ref.h"
#include "hyle/mem/allocator.h"

using hyle::container::StringBuf;
using hyle::container::string_ref;
using hyle::mem::system_allocator;

namespace {
constexpr int kStream = 100000;  // intern calls
constexpr int kUnique = 2000;    // distinct strings (so ~50x dup)

std::vector<std::string> gen_stream() {
  std::mt19937_64 rng(7);
  std::vector<std::string> pool;
  pool.reserve(kUnique);
  for (int i = 0; i < kUnique; ++i)
    pool.push_back("symbol_" + std::to_string(rng() % 1000000));
  std::vector<std::string> stream;
  stream.reserve(kStream);
  for (int i = 0; i < kStream; ++i) stream.push_back(pool[rng() % kUnique]);
  return stream;
}
}  // namespace

int bench_string_buf() {
  const std::vector<std::string> stream = gen_stream();

  ankerl::nanobench::Bench b;
  b.title("intern 100k-stream / 2k-unique").relative(true).batch(kStream);

  // Floor reference: a NON-thread-safe std interner (no lock). NOT StringBuf's
  // apples-to-apples peer -- StringBuf is thread-safe (takes a mutex per intern),
  // so racing it against an unlocked map measures "thread-safe vs not", not the
  // interner cores. Kept as the single-thread floor.
  b.run("std::unordered_map interner (NO lock, floor)", [&] {
    std::unordered_map<std::string, std::uint32_t> pool;
    const char* last = nullptr;
    for (const auto& s : stream) {
      auto [it, ins] = pool.try_emplace(s, 0);
      ++it->second;
      last = it->first.c_str();
    }
    ankerl::nanobench::doNotOptimizeAway(last);
    ankerl::nanobench::doNotOptimizeAway(pool.size());
  });

  // Apples-to-apples peer: a THREAD-SAFE std interner (unordered_map + std::mutex),
  // same thread-safety contract as StringBuf. Both pay one uncontended lock/op; the
  // honest question is whose interner CORE is faster (StringBuf uses SwissTable,
  // which beats unordered_map ~2.4x on find).
  b.run("std::unordered_map + mutex interner (thread-safe peer)", [&] {
    std::unordered_map<std::string, std::uint32_t> pool;
    std::mutex mu;
    const char* last = nullptr;
    for (const auto& s : stream) {
      std::lock_guard<std::mutex> lk(mu);
      auto [it, ins] = pool.try_emplace(s, 0);
      ++it->second;
      last = it->first.c_str();
    }
    ankerl::nanobench::doNotOptimizeAway(last);
    ankerl::nanobench::doNotOptimizeAway(pool.size());
  });

  b.run("HYLE StringBuf (thread-safe, SwissTable core)", [&] {
    StringBuf buf(system_allocator());
    const char* last = nullptr;
    for (const auto& s : stream) {
      string_ref r = buf.intern(string_ref{s.data(),
                                            static_cast<std::uint32_t>(s.size())});
      last = r.data;
    }
    ankerl::nanobench::doNotOptimizeAway(last);
    ankerl::nanobench::doNotOptimizeAway(buf.size());
  });

  // ---- add/retain/remove refcount churn (the lifecycle intern never hits) ----
  // intern() above is steady-state insert (2000 uniques, never freed). The
  // refcount lifecycle -- add (intern + bump), retain (bump by hash), remove (drop;
  // on the last ref, SwissTable::erase + alloc.deallocate) -- exercises a DIFFERENT
  // path: the erase + free churn. Workload: add each unique, retain it, then remove
  // it twice (drives the count to zero -> storage reclaimed). Rival: a std::
  // unordered_map<string,refcount> doing the same add/retain/remove (erase on zero).
  {
    std::vector<std::string> uniques;
    uniques.reserve(kUnique);
    {
      std::mt19937_64 rng(11);
      for (int i = 0; i < kUnique; ++i)
        uniques.push_back("sym_" + std::to_string(rng() % 1000000));
    }
    ankerl::nanobench::Bench b2;
    b2.title("refcount churn: add+retain+remove*2 (2k uniques)")
        .relative(true)
        .batch(static_cast<std::uint64_t>(kUnique) * 4);  // add+retain+2x remove

    b2.run("std::unordered_map<string,int> add/retain/remove", [&] {
      std::unordered_map<std::string, int> pool;
      for (const auto& s : uniques) {
        auto [it, ins] = pool.try_emplace(s, 0);  // add
        ++it->second;
        it->second += 1;                           // retain
        it->second -= 1;                           // remove (1)
        if (--it->second == 0) pool.erase(it);     // remove (2) -> reclaim
      }
      ankerl::nanobench::doNotOptimizeAway(pool.size());
    });

    // Apples-to-apples peer: THREAD-SAFE (unordered_map + mutex) AND matching
    // StringBuf's HANDLE contract. StringBuf's handle is a 64-bit hash (stable,
    // cross-thread-passable), NOT an iterator -- so retain()/remove() re-find by
    // hash rather than reusing an iterator. To compare cores fairly (not reward
    // the no-lock rival for skipping both the lock AND the re-find), this peer
    // takes a lock per op and re-looks-up by key each retain/remove, exactly like
    // StringBuf. Now the honest question is whose map core is faster under the
    // SAME work (lock + hash + find + erase/free), which is SwissTable vs
    // unordered_map -- the naive no-lock/iterator-reuse floor above is NOT that.
    b2.run("std::unordered_map + mutex, re-find per op (thread-safe peer)", [&] {
      std::unordered_map<std::string, int> pool;
      std::mutex mu;
      for (const auto& s : uniques) {
        {  // add: intern + bump
          std::lock_guard<std::mutex> lk(mu);
          ++pool[s];
        }
        {  // retain: re-find by key + bump
          std::lock_guard<std::mutex> lk(mu);
          ++pool[s];
        }
        {  // remove (1): re-find + drop
          std::lock_guard<std::mutex> lk(mu);
          --pool[s];
        }
        {  // remove (2): re-find + drop -> zero -> erase (reclaim)
          std::lock_guard<std::mutex> lk(mu);
          auto it = pool.find(s);
          if (it != pool.end() && --it->second == 0) pool.erase(it);
        }
      }
      ankerl::nanobench::doNotOptimizeAway(pool.size());
    });

    b2.run("HYLE StringBuf add/retain/remove (SwissTable erase+free)", [&] {
      StringBuf buf(system_allocator());
      for (const auto& s : uniques) {
        auto h = buf.add(string_ref{s.data(),
                                    static_cast<std::uint32_t>(s.size())});
        buf.retain(h);
        buf.remove(h);  // drop the retain
        buf.remove(h);  // drop the add -> count zero -> erase + deallocate
      }
      ankerl::nanobench::doNotOptimizeAway(buf.size());
    });
  }

  return 0;
}
