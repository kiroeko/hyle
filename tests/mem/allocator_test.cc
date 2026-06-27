#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "hyle/mem/allocator.h"

using hyle::mem::Allocator;
using hyle::mem::kDefaultAlignment;
using hyle::mem::system_allocator;

namespace {
bool is_aligned(const void* p, std::size_t a) {
  return (reinterpret_cast<std::uintptr_t>(p) & (a - 1)) == 0;
}
}  // namespace

TEST(Allocator, SystemAllocatorIsValid) {
  Allocator a = system_allocator();
  EXPECT_TRUE(a.valid());
  EXPECT_NE(a.fn(), nullptr);
}

TEST(Allocator, DefaultAllocatorIsInvalid) {
  Allocator a;
  EXPECT_FALSE(a.valid());
}

TEST(Allocator, AllocateWriteFree) {
  Allocator a = system_allocator();
  void* p = a.allocate(128);
  ASSERT_NE(p, nullptr);
  EXPECT_TRUE(is_aligned(p, kDefaultAlignment));
  std::memset(p, 0xAB, 128);  // must be fully writable
  a.deallocate(p, 128);
}

TEST(Allocator, AllocateZeroReturnsNull) {
  Allocator a = system_allocator();
  EXPECT_EQ(a.allocate(0), nullptr);  // new_size==0 collapses to the free path
}

// The headline of mem-01: alignment is honored end-to-end (clay dropped it).
TEST(Allocator, RespectsExplicitAlignment) {
  Allocator a = system_allocator();
  for (std::size_t align : {std::size_t{16}, std::size_t{32}, std::size_t{64},
                            std::size_t{128}, std::size_t{256},
                            std::size_t{4096}}) {
    void* p = a.allocate(align * 4, align);
    ASSERT_NE(p, nullptr) << "align=" << align;
    EXPECT_TRUE(is_aligned(p, align)) << "align=" << align;
    a.deallocate(p, align * 4, align);
  }
}

TEST(Allocator, SmallAlignmentBumpedToDefault) {
  Allocator a = system_allocator();
  void* p = a.allocate(8, 1);  // sub-default request still safe-aligned
  ASSERT_NE(p, nullptr);
  EXPECT_TRUE(is_aligned(p, kDefaultAlignment));
  a.deallocate(p, 8, 1);
}

TEST(Allocator, ReallocatePreservesData) {
  Allocator a = system_allocator();
  constexpr int n = 64;
  auto* p = static_cast<int*>(a.allocate(sizeof(int) * n, 64));
  ASSERT_NE(p, nullptr);
  for (int i = 0; i < n; ++i) p[i] = i * 7;

  auto* q = static_cast<int*>(
      a.reallocate(p, sizeof(int) * n, sizeof(int) * n * 2, 64));
  ASSERT_NE(q, nullptr);
  EXPECT_TRUE(is_aligned(q, 64));
  for (int i = 0; i < n; ++i) EXPECT_EQ(q[i], i * 7);  // old data intact
  a.deallocate(q, sizeof(int) * n * 2, 64);
}

TEST(Allocator, ReallocateFromNullIsAllocate) {
  Allocator a = system_allocator();
  void* p = a.reallocate(nullptr, 0, 256, 64);
  ASSERT_NE(p, nullptr);
  EXPECT_TRUE(is_aligned(p, 64));
  a.deallocate(p, 256, 64);
}

TEST(Allocator, ManyDistinctNonOverlapping) {
  Allocator a = system_allocator();
  constexpr int kN = 256;
  std::vector<void*> ps;
  ps.reserve(kN);
  for (int i = 0; i < kN; ++i) {
    void* p = a.allocate(64, 64);
    ASSERT_NE(p, nullptr);
    EXPECT_TRUE(is_aligned(p, 64));
    std::memset(p, i & 0xFF, 64);  // write each fully
    ps.push_back(p);
  }
  for (void* p : ps) a.deallocate(p, 64, 64);
}
