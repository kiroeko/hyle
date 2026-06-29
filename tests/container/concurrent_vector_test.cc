#include "hyle/container/concurrent_vector.h"

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <thread>
#include <unordered_set>
#include <vector>

#include "hyle/mem/allocator.h"

using hyle::container::ConcurrentVector;
using hyle::mem::Allocator;
using hyle::mem::system_allocator;

TEST(ConcurrentVector, EmptyOnConstruction) {
  ConcurrentVector<int> v;
  EXPECT_EQ(v.size(), 0u);
  EXPECT_TRUE(v.empty());
}

TEST(ConcurrentVector, AppendAndIndex) {
  ConcurrentVector<int> v(4);
  for (int i = 0; i < 1000; ++i) {
    std::size_t idx = v.emplace_back(i * 10);
    EXPECT_EQ(idx, static_cast<std::size_t>(i));
  }
  EXPECT_EQ(v.size(), 1000u);
  for (int i = 0; i < 1000; ++i) EXPECT_EQ(v[i], i * 10);
}

// Growth crosses many segment boundaries; every element must remain readable
// and unchanged (segmented storage never moves an element).
TEST(ConcurrentVector, StableAcrossGrowth) {
  ConcurrentVector<int> v(1);  // tiny first seg -> many grows
  std::vector<int*> addrs;
  for (int i = 0; i < 500; ++i) {
    v.emplace_back(i);
    addrs.push_back(&v[i]);
  }
  // Addresses captured early must still point at the same value after growth.
  for (int i = 0; i < 500; ++i) {
    EXPECT_EQ(*addrs[i], i);
    EXPECT_EQ(&v[i], addrs[i]);  // element did NOT move
  }
}

// Move-only T: clay's copy-on-grow couldn't hold these; HYLE's no-copy growth can.
TEST(ConcurrentVector, MoveOnlyType) {
  ConcurrentVector<std::unique_ptr<int>> v(2);
  for (int i = 0; i < 100; ++i) v.emplace_back(std::make_unique<int>(i));
  EXPECT_EQ(v.size(), 100u);
  for (int i = 0; i < 100; ++i) EXPECT_EQ(*v[i], i);
}

// Non-trivial destructor must run exactly once per element on container teardown.
TEST(ConcurrentVector, DestructorRunsPerElement) {
  static std::atomic<int> live{0};
  struct Counted {
    Counted() { live.fetch_add(1); }
    Counted(const Counted&) { live.fetch_add(1); }
    Counted(Counted&&) { live.fetch_add(1); }
    ~Counted() { live.fetch_sub(1); }
  };
  {
    ConcurrentVector<Counted> v(4);
    for (int i = 0; i < 50; ++i) v.emplace_back();
    EXPECT_EQ(live.load(), 50);
  }
  EXPECT_EQ(live.load(), 0);  // all destroyed, none leaked/double-freed
}

// Many concurrent readers while one writer appends: readers never see a torn or
// uninitialized element. (TSan build is the real gate; this is the logic check.)
TEST(ConcurrentVector, ConcurrentReadersOneWriter) {
  ConcurrentVector<std::uint64_t, Allocator> v(system_allocator(), 8);
  constexpr int kN = 20000;
  std::atomic<bool> stop{false};

  auto reader = [&] {
    while (!stop.load(std::memory_order_relaxed)) {
      std::size_t n = v.size();
      for (std::size_t i = 0; i < n; ++i) {
        // Invariant established by the writer: element at index i equals i*3.
        ASSERT_EQ(v[i], static_cast<std::uint64_t>(i) * 3);
      }
    }
  };
  std::thread r1(reader), r2(reader), r3(reader);
  for (int i = 0; i < kN; ++i) {
    std::size_t idx = v.emplace_back(static_cast<std::uint64_t>(i) * 3);
    ASSERT_EQ(idx, static_cast<std::size_t>(i));
  }
  stop.store(true);
  r1.join(); r2.join(); r3.join();
  EXPECT_EQ(v.size(), static_cast<std::size_t>(kN));
}

// Concurrent writers: all indices returned are unique and cover [0, total).
TEST(ConcurrentVector, ConcurrentWritersUniqueIndices) {
  ConcurrentVector<int> v(8);
  constexpr int kPerThread = 5000;
  constexpr int kThreads = 4;
  std::vector<std::thread> ts;
  std::vector<std::vector<std::size_t>> got(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    ts.emplace_back([&, t] {
      for (int i = 0; i < kPerThread; ++i) got[t].push_back(v.emplace_back(t));
    });
  }
  for (auto& th : ts) th.join();

  EXPECT_EQ(v.size(), static_cast<std::size_t>(kThreads * kPerThread));
  std::unordered_set<std::size_t> all;
  for (auto& g : got)
    for (auto idx : g) all.insert(idx);
  EXPECT_EQ(all.size(), static_cast<std::size_t>(kThreads * kPerThread));  // no dups
  for (std::size_t i = 0; i < all.size(); ++i)
    EXPECT_TRUE(all.count(i));  // contiguous [0, total)
}
