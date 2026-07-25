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

namespace {
void* fail_alloc(void*, std::size_t, std::size_t) noexcept { return nullptr; }
void fail_free(void*, void*, std::size_t, std::size_t) noexcept {}
void* fail_realloc(void*, void*, std::size_t, std::size_t,
                   std::size_t) noexcept {
  return nullptr;
}
}  // namespace

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
    Counted() noexcept { live.fetch_add(1); }
    Counted(const Counted&) noexcept { live.fetch_add(1); }
    Counted(Counted&&) noexcept { live.fetch_add(1); }
    ~Counted() noexcept { live.fetch_sub(1); }
  };
  {
    ConcurrentVector<Counted> v(4);
    for (int i = 0; i < 50; ++i) v.emplace_back();
    EXPECT_EQ(live.load(), 50);
  }
  EXPECT_EQ(live.load(), 0);  // all destroyed, none leaked/double-freed
}

TEST(ConcurrentVector, LargeAppendAcrossSegments) {
  ConcurrentVector<std::uint64_t> v;
  constexpr std::size_t kCount = 100000;
  for (std::size_t i = 0; i < kCount; ++i) {
    ASSERT_EQ(v.emplace_back(i), i);
  }
  ASSERT_EQ(v.size(), kCount);
  EXPECT_EQ(v[0], 0u);
  EXPECT_EQ(v[kCount / 2], kCount / 2);
  EXPECT_EQ(v[kCount - 1], kCount - 1);
}

TEST(ConcurrentVector, AllocationFailureDoesNotReserveAHole) {
  Allocator failing{&fail_alloc, &fail_free, &fail_realloc, nullptr};
  using FailingVector = ConcurrentVector<int, Allocator>;
  FailingVector v(failing, 1);
  EXPECT_EQ(v.try_emplace_back(1), FailingVector::npos);
  EXPECT_EQ(v.size(), 0u);
  EXPECT_EQ(v.capacity(), 0u);
}

TEST(ConcurrentVector, ReturnedIndexIsUsableBeforeOtherWriterCompletes) {
  struct Delayed {
    int value;
    Delayed(int v, std::atomic<bool>* entered,
            std::atomic<bool>* release) noexcept
        : value(v) {
      if (entered != nullptr) entered->store(true, std::memory_order_release);
      while (release != nullptr &&
             !release->load(std::memory_order_acquire))
        std::this_thread::yield();
    }
  };

  ConcurrentVector<Delayed> v(2);
  std::atomic<bool> entered{false};
  std::atomic<bool> release{false};
  std::size_t first = ConcurrentVector<Delayed>::npos;
  std::size_t second = ConcurrentVector<Delayed>::npos;
  std::thread slow([&] { first = v.emplace_back(1, &entered, &release); });
  while (!entered.load(std::memory_order_acquire)) std::this_thread::yield();
  std::thread fast([&] { second = v.emplace_back(2, nullptr, nullptr); });
  fast.join();

  EXPECT_EQ(second, 1u);
  EXPECT_EQ(v[second].value, 2);
  EXPECT_EQ(v.size(), 2u);  // reservation count; slot 0 is still in flight.

  release.store(true, std::memory_order_release);
  slow.join();
  EXPECT_EQ(first, 0u);
  EXPECT_EQ(v.size(), 2u);
  EXPECT_EQ(v[0].value, 1);
  EXPECT_EQ(v[1].value, 2);
}

// Many concurrent readers while one writer appends: readers never see a torn or
// uninitialized element. (TSan build is the real gate; this is the logic check.)
TEST(ConcurrentVector, ConcurrentReadersOneWriter) {
  ConcurrentVector<std::uint64_t, Allocator> v(system_allocator(), 8);
  constexpr int kPrefix = 2000;
  constexpr int kN = 20000;
  std::atomic<bool> stop{false};

  for (int i = 0; i < kPrefix; ++i)
    v.emplace_back(static_cast<std::uint64_t>(i) * 3);

  auto reader = [&] {
    while (!stop.load(std::memory_order_relaxed)) {
      // Only the prefix completed before concurrent appends is readable without
      // per-element publication. size() includes in-flight reservations.
      for (std::size_t i = 0; i < kPrefix; ++i) {
        ASSERT_EQ(v[i], static_cast<std::uint64_t>(i) * 3);
      }
    }
  };
  std::thread r1(reader), r2(reader), r3(reader);
  for (int i = 0; i < kN; ++i) {
    const std::size_t expected = static_cast<std::size_t>(kPrefix + i);
    std::size_t idx = v.emplace_back(expected * 3);
    ASSERT_EQ(idx, expected);
  }
  stop.store(true);
  r1.join(); r2.join(); r3.join();
  EXPECT_EQ(v.size(), static_cast<std::size_t>(kPrefix + kN));
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
