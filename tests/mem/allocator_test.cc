#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

#include "hyle/mem/allocator.h"

using hyle::mem::Allocator;
using hyle::mem::kDefaultAlignment;
using hyle::mem::system_allocator;
using hyle::mem::system_good_size;
using hyle::mem::system_usable_size;

namespace {
bool is_aligned(const void* p, std::size_t a) {
  return (reinterpret_cast<std::uintptr_t>(p) & (a - 1)) == 0;
}
}  // namespace

TEST(Allocator, SystemAllocatorIsValid) {
  Allocator a = system_allocator();
  EXPECT_TRUE(a.valid());
  EXPECT_NE(a.alloc_fn(), nullptr);
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

// ---- typed convenience layer (mem-01b, clay allocator.h parity) -------------
namespace {
std::atomic<int> g_typed_live{0};
struct TCounted {
  int v;
  explicit TCounted(int x) : v(x) { g_typed_live.fetch_add(1); }
  ~TCounted() { g_typed_live.fetch_sub(1); }
};
}  // namespace

TEST(AllocatorTyped, AllocateNAlignedAndWritable) {
  Allocator a = system_allocator();
  int* p = a.allocate_n<int>(5);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % alignof(int), 0u);
  for (int i = 0; i < 5; ++i) p[i] = i;  // writable
  EXPECT_EQ(p[4], 4);
  a.deallocate_n(p, 5);
}

TEST(AllocatorTyped, AllocateZeroedZeroes) {
  Allocator a = system_allocator();
  int* p = a.allocate_zeroed<int>(8);
  ASSERT_NE(p, nullptr);
  for (int i = 0; i < 8; ++i) EXPECT_EQ(p[i], 0);
  a.deallocate_n(p, 8);
}

TEST(AllocatorTyped, ConstructRunsCtorDestroyRunsDtor) {
  g_typed_live.store(0);
  Allocator a = system_allocator();
  TCounted* c = a.construct<TCounted>(7);
  ASSERT_NE(c, nullptr);
  EXPECT_EQ(c->v, 7);
  EXPECT_EQ(g_typed_live.load(), 1);
  a.destroy(c);  // ~T + free through THIS allocator (not global delete)
  EXPECT_EQ(g_typed_live.load(), 0);
}

TEST(AllocatorTyped, MakeUniqueAutoDestroys) {
  g_typed_live.store(0);
  Allocator a = system_allocator();
  {
    auto u = a.make_unique<TCounted>(99);
    ASSERT_NE(u.get(), nullptr);
    EXPECT_EQ(u->v, 99);
    EXPECT_EQ(g_typed_live.load(), 1);
  }  // unique_ptr deleter destroys through the allocator
  EXPECT_EQ(g_typed_live.load(), 0);
}

// ---- usable_size / good_size (SystemAllocator only) -------------------------

TEST(SystemSize, UsableSizeAtLeastRequested) {
  auto a = system_allocator();
  void* p = a.allocate(100);
  ASSERT_NE(p, nullptr);
  std::size_t u = system_usable_size(p);
  EXPECT_GE(u, 100u);  // usable is never less than requested
  // The whole usable range is writable (mimalloc rounded up to a size class).
  std::memset(p, 0xAB, u);
  EXPECT_EQ(static_cast<unsigned char*>(p)[u - 1], 0xAB);
  a.deallocate(p, 100);
  EXPECT_EQ(system_usable_size(nullptr), 0u);  // null -> 0
}

TEST(SystemSize, GoodSizePreviewMatchesUsable) {
  // good_size previews the class rounding WITHOUT allocating; it must agree with
  // the actual usable size of a real allocation of that request.
  std::size_t g = system_good_size(100);
  EXPECT_GE(g, 100u);
  auto a = system_allocator();
  void* p = a.allocate(100);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(g, system_usable_size(p));  // preview == reality
  a.deallocate(p, 100);
  EXPECT_EQ(system_good_size(0), 0u);   // 0 -> 0
}
