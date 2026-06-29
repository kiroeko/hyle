#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <vector>

#include "hyle/container/intrusive_list.h"
#include "hyle/container/small_vector.h"
#include "hyle/mem/allocator.h"

using hyle::container::IntrusiveList;
using hyle::container::IntrusiveListHook;
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

// ---- IntrusiveList ----------------------------------------------------------

namespace {
struct Node {
  int v;
  IntrusiveListHook hook;
  explicit Node(int x) : v(x) {}
};
using NodeList = IntrusiveList<Node, &Node::hook>;
}  // namespace

TEST(IntrusiveList, PushBackPopFrontFifo) {
  NodeList list;
  EXPECT_TRUE(list.empty());
  Node a(1), b(2), c(3);
  list.push_back(&a);
  list.push_back(&b);
  list.push_back(&c);
  EXPECT_FALSE(list.empty());
  EXPECT_EQ(list.pop_front()->v, 1);
  EXPECT_EQ(list.pop_front()->v, 2);
  EXPECT_EQ(list.pop_front()->v, 3);
  EXPECT_TRUE(list.empty());
  EXPECT_EQ(list.pop_front(), nullptr);
}

TEST(IntrusiveList, PushFrontAndFrontBack) {
  NodeList list;
  Node a(1), b(2);
  list.push_front(&a);
  list.push_front(&b);  // b, a
  EXPECT_EQ(list.front()->v, 2);
  EXPECT_EQ(list.back()->v, 1);
}

TEST(IntrusiveList, O1RemoveFromMiddle) {
  NodeList list;
  Node a(1), b(2), c(3);
  list.push_back(&a);
  list.push_back(&b);
  list.push_back(&c);
  list.remove(&b);  // remove middle, O(1), no traversal
  EXPECT_FALSE(b.hook.linked());
  EXPECT_EQ(list.pop_front()->v, 1);
  EXPECT_EQ(list.pop_front()->v, 3);
  EXPECT_TRUE(list.empty());
}

TEST(IntrusiveList, NoAllocationRecoversNode) {
  // The element is recovered from the hook with correct offset even when hook
  // is not the first member.
  NodeList list;
  Node n(42);
  list.push_back(&n);
  EXPECT_EQ(list.front(), &n);
  EXPECT_EQ(list.front()->v, 42);
  list.remove(&n);
}
