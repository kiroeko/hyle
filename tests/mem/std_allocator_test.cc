#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "hyle/mem/allocator.h"
#include "hyle/mem/stats.h"
#include "hyle/mem/std_allocator.h"

namespace {

using hyle::mem::Allocator;
using hyle::mem::StatsAllocator;
using hyle::mem::StdAllocator;
using hyle::mem::system_allocator;

void* oom_allocate(void*, std::size_t, std::size_t) noexcept { return nullptr; }
void oom_deallocate(void*, void*, std::size_t, std::size_t) noexcept {}
void* oom_reallocate(void*, void*, std::size_t, std::size_t,
                     std::size_t) noexcept {
  return nullptr;
}

Allocator oom_allocator() {
  return Allocator(&oom_allocate, &oom_deallocate, &oom_reallocate, nullptr);
}

struct SharedValue {
  SharedValue(bool* destroyed, int value) noexcept
      : destroyed(destroyed), value(value) {}
  ~SharedValue() { *destroyed = true; }

  bool* destroyed;
  int value;
};

struct alignas(128) OverAligned {
  std::uint64_t words[16]{};
};

TEST(StdAllocator, DefaultUsesSystemAllocator) {
  StdAllocator<int> allocator;
  int* values = allocator.allocate(4);
  ASSERT_NE(values, nullptr);
  values[3] = 17;
  EXPECT_EQ(values[3], 17);
  allocator.deallocate(values, 4);
}

TEST(StdAllocator, VectorUsesInjectedBackingAndRespectsLifetimeOrder) {
  StatsAllocator stats = StatsAllocator::create(system_allocator()).value();
  {
    std::vector<int, StdAllocator<int>> values{StdAllocator<int>(stats.get())};
    for (int i = 0; i < 1024; ++i) values.push_back(i);
    EXPECT_EQ(values.front(), 0);
    EXPECT_EQ(values.back(), 1023);
    EXPECT_GT(stats.snapshot().live_bytes, 0u);
  }
  EXPECT_TRUE(stats.balanced());
}

TEST(StdAllocator, AllocateSharedRebindsControlBlockToBacking) {
  StatsAllocator stats = StatsAllocator::create(system_allocator()).value();
  bool destroyed = false;
  {
    auto value = std::allocate_shared<SharedValue>(
        StdAllocator<SharedValue>(stats.get()), &destroyed, 23);
    EXPECT_EQ(value->value, 23);
    EXPECT_FALSE(destroyed);
    EXPECT_GT(stats.snapshot().live_bytes, 0u);
  }
  EXPECT_TRUE(destroyed);
  EXPECT_TRUE(stats.balanced());
}

TEST(StdAllocator, PreservesOverAlignment) {
  StdAllocator<OverAligned> allocator;
  OverAligned* values = allocator.allocate(2);
  ASSERT_NE(values, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(values) % alignof(OverAligned), 0u);
  allocator.deallocate(values, 2);
}

TEST(StdAllocator, EqualityIsCompleteBackingIdentityAcrossRebind) {
  const Allocator system = system_allocator();
  StdAllocator<int> ints(system);
  StdAllocator<double> doubles(system);
  EXPECT_EQ(ints, doubles);

  StatsAllocator first = StatsAllocator::create(system).value();
  StatsAllocator second = StatsAllocator::create(system).value();
  StdAllocator<int> first_a(first.get());
  StdAllocator<int> first_copy(first_a);
  StdAllocator<int> second_a(second.get());
  EXPECT_EQ(first_a, first_copy);
  EXPECT_NE(first_a, second_a);
}

TEST(StdAllocator, OomAndCountOverflowUseStandardExceptions) {
  StdAllocator<int> allocator(oom_allocator());
  EXPECT_THROW((void)allocator.allocate(1), std::bad_alloc);
  EXPECT_THROW((void)allocator.allocate(
                   (std::numeric_limits<std::size_t>::max)() / sizeof(int) + 1),
               std::bad_array_new_length);

  std::vector<int, StdAllocator<int>> values(allocator);
  EXPECT_THROW(values.reserve(1), std::bad_alloc);
}

}  // namespace
