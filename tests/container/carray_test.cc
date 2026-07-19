#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <vector>

#include "hyle/container/carray.h"
#include "hyle/mem/allocator.h"

using hyle::container::CArray;
using hyle::container::carray_header;
using hyle::mem::system_allocator;

namespace {
void* fail_alloc(void*, std::size_t, std::size_t) noexcept { return nullptr; }
void fail_free(void*, void*, std::size_t, std::size_t) noexcept {}
void* fail_realloc(void*, void*, std::size_t, std::size_t,
                   std::size_t) noexcept {
  return nullptr;
}
}  // namespace

TEST(CArray, EmptyState) {
  CArray<int> a(system_allocator());
  EXPECT_EQ(a.size(), 0u);
  EXPECT_EQ(a.capacity(), 0u);
  EXPECT_TRUE(a.empty());
  EXPECT_EQ(a.data(), nullptr);
}

TEST(CArray, PushBackGrowsAndIndexes) {
  CArray<int> a(system_allocator());
  for (int i = 0; i < 100; ++i) a.push_back(i * 3);
  EXPECT_EQ(a.size(), 100u);
  EXPECT_GE(a.capacity(), 100u);
  for (int i = 0; i < 100; ++i) EXPECT_EQ(a[i], i * 3);  // a[i] zero-indirection
}

TEST(CArray, HeaderBeforeDataLayout) {
  CArray<int> a(system_allocator());
  a.push_back(7);
  // The element pointer IS the data; header sits immediately before it.
  EXPECT_EQ(a.data()[0], 7);
  EXPECT_EQ(carray_header(a.data())->size, 1u);
  // data can be passed straight to a C API expecting int* -- prove it reads.
  int* raw = a.data();
  EXPECT_EQ(raw[0], 7);
}

TEST(CArray, PopAndClear) {
  CArray<int> a(system_allocator());
  for (int i = 0; i < 5; ++i) a.push_back(i);
  a.pop_back();
  EXPECT_EQ(a.size(), 4u);
  EXPECT_EQ(a.back(), 3);
  a.clear();
  EXPECT_EQ(a.size(), 0u);
  EXPECT_GE(a.capacity(), 4u);  // capacity retained
}

TEST(CArray, ReserveDoesNotChangeSize) {
  CArray<int> a(system_allocator());
  a.reserve(64);
  EXPECT_GE(a.capacity(), 64u);
  EXPECT_EQ(a.size(), 0u);
}

TEST(CArray, DataSurvivesReallocation) {
  CArray<std::uint64_t> a(system_allocator());
  for (std::uint64_t i = 0; i < 1000; ++i) a.push_back(i * i);  // forces regrows
  for (std::uint64_t i = 0; i < 1000; ++i) EXPECT_EQ(a[i], i * i);  // memcpy kept data
}

TEST(CArray, OverAlignedElement) {
  struct alignas(64) Cacheline { char b[64]; };
  CArray<Cacheline> a(system_allocator());
  a.push_back(Cacheline{});
  a.push_back(Cacheline{});
  // header-before-data must still leave the elements 64-aligned (clay's fixed
  // max_align_t header would have broken this).
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a.data()) % 64, 0u);
}

TEST(CArray, InitListAndIterate) {
  CArray<int> a(system_allocator(), {10, 20, 30});
  EXPECT_EQ(a.size(), 3u);
  int sum = std::accumulate(a.begin(), a.end(), 0);
  EXPECT_EQ(sum, 60);
}

TEST(CArray, MoveTransfers) {
  CArray<int> a(system_allocator());
  a.push_back(1);
  a.push_back(2);
  int* p = a.data();
  CArray<int> b(std::move(a));
  EXPECT_EQ(a.data(), nullptr);
  EXPECT_EQ(b.data(), p);  // no realloc, just handle transfer
  EXPECT_EQ(b.size(), 2u);
}

TEST(CArray, PushBackMayAliasElementAcrossGrowth) {
  CArray<int> a(system_allocator());
  for (int i = 0; i < 4; ++i) a.push_back(10 + i);
  ASSERT_EQ(a.size(), a.capacity());
  a.push_back(a[1]);
  EXPECT_EQ(a.back(), 11);
}

TEST(CArray, ExistingStorageKeepsRecordedAlignmentOnTypeErasedGrow) {
  void* data = hyle::container::detail::carray_set_capacity_internal(
      nullptr, 4, sizeof(int), 64, system_allocator());
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(reinterpret_cast<std::uintptr_t>(data) % 64, 0u);
  data = hyle::container::detail::carray_set_capacity_internal(
      data, 16, sizeof(int), alignof(int), system_allocator());
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(data) % 64, 0u);
  EXPECT_EQ(carray_header(data)->align, 64u);
  hyle::container::detail::carray_free_internal(data, sizeof(int));
}

TEST(CArray, FallibleGrowthPreservesStateOnFailure) {
  hyle::mem::Allocator failing{&fail_alloc, &fail_free, &fail_realloc, nullptr};
  CArray<int> a(failing);
  EXPECT_FALSE(a.try_reserve(4));
  EXPECT_FALSE(a.try_push_back(7));
  EXPECT_EQ(a.data(), nullptr);
  EXPECT_EQ(a.size(), 0u);
  EXPECT_EQ(a.capacity(), 0u);
}

TEST(CArray, FallibleReserveRejectsAddressSpaceOverflow) {
  CArray<std::uint64_t> a(system_allocator());
  EXPECT_FALSE(a.try_reserve(UINT64_MAX));
  EXPECT_EQ(a.data(), nullptr);
}
