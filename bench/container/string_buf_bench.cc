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

  return 0;
}
