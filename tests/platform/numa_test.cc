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
  if (count == 1u) {
    EXPECT_EQ(numa_node_of_cpu(0), 0u);
  }
}

TEST(Numa, BindAnyIsNoOp) {
  std::size_t p = page_size();
  void* base = vm_reserve(p).value();
  ASSERT_TRUE(vm_commit(base, p).ok());
  EXPECT_TRUE(vm_bind_numa(base, p, kNumaAny).has_value());
  EXPECT_TRUE(vm_bind_numa(nullptr, 0, 0).has_value());
  ASSERT_TRUE(vm_release(base, p).ok());
}

TEST(Numa, BindNodeZero) {
  std::size_t p = page_size();
  void* base = vm_reserve(p * 4).value();
  ASSERT_TRUE(vm_commit(base, p * 4).ok());
  // Soft preferred bind to node 0 — must not crash; may be no-op on Win.
  EXPECT_TRUE(vm_bind_numa(base, p * 4, 0).has_value());
  std::memset(base, 0x7E, p);
  EXPECT_EQ(static_cast<unsigned char*>(base)[0], 0x7E);
  ASSERT_TRUE(vm_release(base, p * 4).ok());
}

TEST(Numa, BindInvalidNodeFails) {
  std::size_t p = page_size();
  void* base = vm_reserve(p).value();
  int bad = static_cast<int>(numa_node_count()) + 10;
  auto result = vm_bind_numa(base, p, bad);
  ASSERT_FALSE(result);
  EXPECT_TRUE(result.error().IsOutOfRange());
  ASSERT_TRUE(vm_release(base, p).ok());
}

TEST(Numa, InvalidRangesAndReserveNodeFailHonestly) {
  auto bind = vm_bind_numa(nullptr, page_size(), 0);
  ASSERT_FALSE(bind);
  EXPECT_TRUE(bind.error().IsInvalidArgument());
  const int bad = static_cast<int>(numa_node_count()) + 10;
  auto reserve = vm_reserve_numa(page_size(), bad);
  ASSERT_FALSE(reserve);
  EXPECT_TRUE(reserve.error().IsOutOfRange());
}

TEST(Numa, ReserveNumaWritable) {
  std::size_t p = page_size();
  auto reservation = vm_reserve_numa(p * 2, 0).value();
  void* base = reservation.data;
  ASSERT_TRUE(vm_commit(base, p * 2).ok());
  std::memset(base, 0x11, p * 2);
  EXPECT_EQ(static_cast<unsigned char*>(base)[0], 0x11);
  ASSERT_TRUE(vm_release(base, p * 2).ok());
}

TEST(Numa, ReserveNumaAnyEqualsPlain) {
  std::size_t p = page_size();
  auto reservation = vm_reserve_numa(p, kNumaAny).value();
  ASSERT_NE(reservation.data, nullptr);
  EXPECT_EQ(reservation.placement,
            hyle::platform::NumaPlacement::kNotNeeded);
  ASSERT_TRUE(vm_release(reservation.data, p).ok());
}

TEST(Numa, VmRegionReserveNuma) {
  std::size_t p = page_size();
  VmRegion r = VmRegion::reserve_numa(p * 3, 0).value();
  ASSERT_TRUE(r.commit_usable().ok());
  std::memset(r.data(), 0x22, r.size());
  EXPECT_EQ(static_cast<unsigned char*>(r.data())[0], 0x22);
}

TEST(Numa, CurrentNodeLocalReserve) {
  int home = static_cast<int>(numa_current_node());
  std::size_t p = page_size();
  auto reservation = vm_reserve_numa(p, home).value();
  void* base = reservation.data;
  ASSERT_TRUE(vm_commit(base, p).ok());
  EXPECT_TRUE(vm_bind_numa(base, p, home).has_value());
  ASSERT_TRUE(vm_release(base, p).ok());
}

TEST(Numa, BindHardAndInterleave) {
  std::size_t p = page_size();
  void* base = vm_reserve(p * 4).value();
  ASSERT_TRUE(vm_commit(base, p * 4).ok());
  // Honest contract: on a single-node box every per-node policy is a genuine
  // no-op. On a real multi-node box, hard bind / interleave only work where the
  // OS exposes an API (Linux); elsewhere they return structured errors rather
  // than pretending. Region stays usable, so the memset below always works.
  const bool single_node = numa_node_count() <= 1u;
  auto hard = vm_bind_numa(base, p * 4, 0, NumaPolicy::kBind);
  auto inter = vm_bind_numa_interleave(base, p * 4);
  if (single_node) {
    ASSERT_TRUE(hard);
    ASSERT_TRUE(inter);
    EXPECT_EQ(*hard, hyle::platform::NumaPlacement::kNotNeeded);
    EXPECT_EQ(*inter, hyle::platform::NumaPlacement::kNotNeeded);
  }
#if defined(_WIN32)
  // Multi-node Windows: no post-hoc hard bind / interleave -> honest errors.
  if (!single_node) {
    EXPECT_FALSE(hard.has_value());
    EXPECT_FALSE(inter.has_value());
  }
#endif
  std::memset(base, 0x5A, p);
  EXPECT_EQ(static_cast<unsigned char*>(base)[0], 0x5A);
  ASSERT_TRUE(vm_release(base, p * 4).ok());
}
