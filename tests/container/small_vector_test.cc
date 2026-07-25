#include <gtest/gtest.h>

#include <numeric>

#include "hyle/container/small_vector.h"
#include "hyle/mem/allocator.h"

using hyle::container::SmallVector;
using hyle::mem::Allocator;
using hyle::mem::system_allocator;

// ---- SmallVector ------------------------------------------------------------

TEST(SmallVector, StaysInlineUnderN) {
  SmallVector<int, 4> v;  // default = compile-time SystemAllocator (zero-overhead path)
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);
  v.push_back(4);
  EXPECT_EQ(v.size(), 4u);
  EXPECT_TRUE(v.inline_storage());  // no heap touched at capacity N
  for (int i = 0; i < 4; ++i) EXPECT_EQ(v[i], i + 1);
}

TEST(SmallVector, SpillsToHeapPastN) {
  // Runtime-injection path (opt-in mem::Allocator): exercises the vtable allocator + spill.
  SmallVector<int, 4, Allocator> v(system_allocator());
  for (int i = 0; i < 100; ++i) v.push_back(i);
  EXPECT_EQ(v.size(), 100u);
  EXPECT_FALSE(v.inline_storage());  // spilled
  for (int i = 0; i < 100; ++i) EXPECT_EQ(v[i], i);  // data preserved across spill
}

TEST(SmallVector, PopClearIterate) {
  SmallVector<int, 2> v({10, 20, 30});  // default alloc, forces spill
  EXPECT_EQ(v.size(), 3u);
  EXPECT_EQ(std::accumulate(v.begin(), v.end(), 0), 60);
  v.pop_back();
  EXPECT_EQ(v.back(), 20);
  v.clear();
  EXPECT_TRUE(v.empty());
}

TEST(SmallVector, MoveStealsHeap) {
  SmallVector<int, 2> a;
  for (int i = 0; i < 50; ++i) a.push_back(i);  // on heap
  int* p = a.data();
  SmallVector<int, 2> b(std::move(a));
  EXPECT_EQ(b.data(), p);  // heap buffer stolen, not copied
  EXPECT_EQ(b.size(), 50u);
  EXPECT_EQ(b[49], 49);
}

TEST(SmallVector, MoveInlineCopies) {
  SmallVector<int, 4> a;
  a.push_back(7);
  a.push_back(8);
  SmallVector<int, 4> b(std::move(a));  // inline -> element copy
  EXPECT_TRUE(b.inline_storage());
  EXPECT_EQ(b.size(), 2u);
  EXPECT_EQ(b[0], 7);
  EXPECT_EQ(b[1], 8);
}

TEST(SmallVector, MoveAssignmentHandlesInlineAndHeap) {
  SmallVector<int, 2> heap;
  heap.push_back(1);
  heap.push_back(2);
  heap.push_back(3);
  SmallVector<int, 2> inline_values;
  inline_values.push_back(9);

  inline_values = std::move(heap);
  ASSERT_EQ(inline_values.size(), 3u);
  EXPECT_EQ(inline_values[2], 3);
  EXPECT_TRUE(heap.empty());

  SmallVector<int, 2> other;
  other.push_back(7);
  inline_values = std::move(other);
  ASSERT_EQ(inline_values.size(), 1u);
  EXPECT_EQ(inline_values[0], 7);
}

TEST(SmallVector, PushBackMayAliasElementAcrossSpill) {
  SmallVector<int, 2> v;
  v.push_back(4);
  v.push_back(5);
  v.push_back(v[0]);
  ASSERT_EQ(v.size(), 3u);
  EXPECT_EQ(v[2], 4);
}
