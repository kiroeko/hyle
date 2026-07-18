#include <gtest/gtest.h>

#include <cstring>

#include "hyle/platform/numa.h"
#include "hyle/platform/virtual_memory.h"

using hyle::platform::kNumaAny;
using hyle::platform::NumaPolicy;
using hyle::platform::numa_current_node;
using hyle::platform::numa_node_count;
using hyle::platform::numa_node_of_cpu;
using hyle::platform::page_size;
using hyle::platform::vm_bind_numa;
using hyle::platform::vm_bind_numa_interleave;
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

TEST(Numa, NodeOfCpuInRange) {
  const unsigned count = numa_node_count();
  EXPECT_LT(numa_node_of_cpu(0), count);         // CPU 0 always maps to a valid node
  EXPECT_LT(numa_node_of_cpu(1u << 20), count);  // out-of-range CPU: 0 fallback, no crash
  // On a single-node box every CPU is node 0.
  if (count == 1u) EXPECT_EQ(numa_node_of_cpu(0), 0u);
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

TEST(Numa, BindHardAndInterleave) {
  std::size_t p = page_size();
  void* base = vm_reserve(p * 4);
  ASSERT_NE(base, nullptr);
  ASSERT_TRUE(vm_commit(base, p * 4));
  // Honest contract: on a single-node box every per-node policy is a genuine
  // no-op -> true. On a real multi-node box, hard bind / interleave only work
  // where the OS exposes an API (Linux); elsewhere they must report false (not
  // pretend). Region stays usable regardless, so the memset below always works.
  const bool single_node = numa_node_count() <= 1u;
  bool hard = vm_bind_numa(base, p * 4, 0, NumaPolicy::kBind);
  bool inter = vm_bind_numa_interleave(base, p * 4);
  if (single_node) {
    EXPECT_TRUE(hard);
    EXPECT_TRUE(inter);
  }
#if defined(_WIN32)
  // Multi-node Windows: no post-hoc hard bind / interleave -> honest false.
  if (!single_node) {
    EXPECT_FALSE(hard);
    EXPECT_FALSE(inter);
  }
#endif
  std::memset(base, 0x5A, p);
  EXPECT_EQ(static_cast<unsigned char*>(base)[0], 0x5A);
  ASSERT_TRUE(vm_release(base, p * 4));
}
