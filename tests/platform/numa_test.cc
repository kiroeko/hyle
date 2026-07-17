#include <gtest/gtest.h>

#include <cstring>

#include "hyle/platform/numa.h"
#include "hyle/platform/virtual_memory.h"

using hyle::platform::kNumaAny;
using hyle::platform::numa_current_node;
using hyle::platform::numa_node_count;
using hyle::platform::page_size;
using hyle::platform::vm_bind_numa;
using hyle::platform::vm_commit;
using hyle::platform::vm_release;
using hyle::platform::vm_reserve;
using hyle::platform::vm_reserve_numa;
using hyle::platform::VmRegion;

TEST(Numa, NodeCountAtLeastOne) {
  unsigned n = numa_node_count();
  EXPECT_GE(n, 1u);
  EXPECT_LT(n, 256u);  // sanity
}

TEST(Numa, CurrentNodeInRange) {
  unsigned cur = numa_current_node();
  EXPECT_LT(cur, numa_node_count());
}

TEST(Numa, BindAnyIsNoOp) {
  std::size_t p = page_size();
  void* base = vm_reserve(p);
  ASSERT_NE(base, nullptr);
  ASSERT_TRUE(vm_commit(base, p));
  EXPECT_TRUE(vm_bind_numa(base, p, kNumaAny));
  EXPECT_TRUE(vm_bind_numa(nullptr, 0, 0));
  ASSERT_TRUE(vm_release(base, p));
}

TEST(Numa, BindNodeZero) {
  std::size_t p = page_size();
  void* base = vm_reserve(p * 4);
  ASSERT_NE(base, nullptr);
  ASSERT_TRUE(vm_commit(base, p * 4));
  // Soft preferred bind to node 0 — must not crash; may be no-op on Win.
  EXPECT_TRUE(vm_bind_numa(base, p * 4, 0));
  std::memset(base, 0x7E, p);
  EXPECT_EQ(static_cast<unsigned char*>(base)[0], 0x7E);
  ASSERT_TRUE(vm_release(base, p * 4));
}

TEST(Numa, BindInvalidNodeFails) {
  std::size_t p = page_size();
  void* base = vm_reserve(p);
  ASSERT_NE(base, nullptr);
  int bad = static_cast<int>(numa_node_count()) + 10;
  EXPECT_FALSE(vm_bind_numa(base, p, bad));
  ASSERT_TRUE(vm_release(base, p));
}

TEST(Numa, ReserveNumaWritable) {
  std::size_t p = page_size();
  void* base = vm_reserve_numa(p * 2, 0);
  ASSERT_NE(base, nullptr);
  ASSERT_TRUE(vm_commit(base, p * 2));
  std::memset(base, 0x11, p * 2);
  EXPECT_EQ(static_cast<unsigned char*>(base)[0], 0x11);
  ASSERT_TRUE(vm_release(base, p * 2));
}

TEST(Numa, ReserveNumaAnyEqualsPlain) {
  std::size_t p = page_size();
  void* a = vm_reserve_numa(p, kNumaAny);
  ASSERT_NE(a, nullptr);
  ASSERT_TRUE(vm_release(a, p));
}

TEST(Numa, VmRegionReserveNuma) {
  std::size_t p = page_size();
  VmRegion r = VmRegion::reserve_numa(p * 3, 0);
  ASSERT_TRUE(r.valid());
  ASSERT_TRUE(r.commit_usable());
  std::memset(r.data(), 0x22, r.size());
  EXPECT_EQ(static_cast<unsigned char*>(r.data())[0], 0x22);
}

TEST(Numa, CurrentNodeLocalReserve) {
  int home = static_cast<int>(numa_current_node());
  std::size_t p = page_size();
  void* base = vm_reserve_numa(p, home);
  ASSERT_NE(base, nullptr);
  ASSERT_TRUE(vm_commit(base, p));
  EXPECT_TRUE(vm_bind_numa(base, p, home));
  ASSERT_TRUE(vm_release(base, p));
}
