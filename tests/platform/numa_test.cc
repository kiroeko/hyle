#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <utility>

#include "hyle/platform.h"
#include "hyle/platform/numa.h"
#include "hyle/platform/virtual_memory.h"

using hyle::platform::kNumaAny;
using hyle::platform::has_numa_capability;
using hyle::platform::NumaCapability;
using hyle::platform::NumaPolicy;
using hyle::platform::numa_capabilities;
using hyle::platform::numa_current_node;
using hyle::platform::numa_node_available;
using hyle::platform::numa_node_count;
using hyle::platform::numa_node_of_cpu;
using hyle::platform::numa_nodes;
using hyle::platform::page_size;
using hyle::platform::vm_allocate_numa_interleaved;
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
  EXPECT_EQ(n, numa_nodes().size());
  unsigned previous = 0;
  for (std::size_t i = 0; i < numa_nodes().size(); ++i) {
    EXPECT_TRUE(numa_node_available(numa_nodes()[i]));
    if (i != 0) {
      EXPECT_LT(previous, numa_nodes()[i]);
    }
    previous = numa_nodes()[i];
  }
}

TEST(Numa, CurrentNodeInRange) {
  unsigned cur = numa_current_node();
  EXPECT_TRUE(numa_node_available(cur));
}

TEST(Numa, NodeOfCpuInRange) {
  const unsigned count = numa_node_count();
  EXPECT_TRUE(numa_node_available(numa_node_of_cpu(0)));
  EXPECT_TRUE(numa_node_available(numa_node_of_cpu(1u << 20)));
  // On a single-node box every CPU falls back to the sole available node ID.
  if (count == 1u) {
    EXPECT_EQ(numa_node_of_cpu(0), numa_nodes().front());
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
  // Soft preferred bind to an available node must not crash; may be a no-op.
  EXPECT_TRUE(vm_bind_numa(base, p * 4,
                           static_cast<int>(numa_nodes().front())).has_value());
  std::memset(base, 0x7E, p);
  EXPECT_EQ(static_cast<unsigned char*>(base)[0], 0x7E);
  ASSERT_TRUE(vm_release(base, p * 4).ok());
}

TEST(Numa, BindInvalidNodeFails) {
  std::size_t p = page_size();
  void* base = vm_reserve(p).value();
  int bad = static_cast<int>(numa_nodes().back()) + 10;
  auto result = vm_bind_numa(base, p, bad);
  ASSERT_FALSE(result);
  EXPECT_TRUE(result.error().IsOutOfRange());
  ASSERT_TRUE(vm_release(base, p).ok());
}

TEST(Numa, InvalidRangesAndReserveNodeFailHonestly) {
  auto bind = vm_bind_numa(nullptr, page_size(), 0);
  ASSERT_FALSE(bind);
  EXPECT_TRUE(bind.error().IsInvalidArgument());
  const int bad = static_cast<int>(numa_nodes().back()) + 10;
  auto reserve = vm_reserve_numa(page_size(), bad);
  ASSERT_FALSE(reserve);
  EXPECT_TRUE(reserve.error().IsOutOfRange());
}

TEST(Numa, ReserveNumaWritable) {
  std::size_t p = page_size();
  auto reservation =
      vm_reserve_numa(p * 2, static_cast<int>(numa_nodes().front())).value();
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
  VmRegion r =
      VmRegion::reserve_numa(p * 3, static_cast<int>(numa_nodes().front()))
          .value();
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
  auto hard = vm_bind_numa(base, p * 4,
                           static_cast<int>(numa_nodes().front()),
                           NumaPolicy::kBind);
  auto inter = vm_bind_numa_interleave(base, p * 4);
  if (single_node) {
    ASSERT_TRUE(hard);
    ASSERT_TRUE(inter);
    EXPECT_EQ(*hard, hyle::platform::NumaPlacement::kNotNeeded);
    EXPECT_EQ(*inter, hyle::platform::NumaPlacement::kNotNeeded);
  }
#if defined(HYLE_OS_WINDOWS)
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

TEST(Numa, CapabilitiesDescribeSelectedInterleavePrimitive) {
  const auto capabilities = numa_capabilities();
  EXPECT_EQ(capabilities, numa_capabilities());
  if (numa_nodes().size() > 1u) {
    EXPECT_TRUE(has_numa_capability(
        capabilities, NumaCapability::kTopology));
  }
}

TEST(Numa, SelectedInterleaveAllocationIsWritableAndOrdinaryReleaseWorks) {
  const std::size_t p = page_size();
  const std::array<unsigned, 1> selected{numa_nodes().front()};
  auto allocation = vm_allocate_numa_interleaved(p * 8, selected, p * 2);
  if (numa_nodes().size() > 1u &&
      !has_numa_capability(numa_capabilities(),
                           NumaCapability::kSelectedInterleave)) {
    ASSERT_FALSE(allocation);
    EXPECT_TRUE(allocation.error().IsUnimplemented());
    return;
  }
  ASSERT_TRUE(allocation);
  ASSERT_NE(allocation->data, nullptr);
  EXPECT_EQ(allocation->size, p * 8);
  std::memset(allocation->data, 0x6C, allocation->size);
  EXPECT_EQ(static_cast<unsigned char*>(allocation->data)[p * 7], 0x6C);
  EXPECT_TRUE(vm_release(allocation->data, allocation->size).ok());
}

TEST(Numa, SelectedInterleaveManagedRegionUsesAllAvailableNodes) {
  const std::size_t p = page_size();
  auto result = VmRegion::allocate_numa_interleaved(p * 8, {}, p * 2);
  if (numa_nodes().size() > 1u &&
      !has_numa_capability(numa_capabilities(),
                           NumaCapability::kSelectedInterleave)) {
    ASSERT_FALSE(result);
    EXPECT_TRUE(result.error().IsUnimplemented());
    return;
  }
  ASSERT_TRUE(result);
  VmRegion region = *std::move(result);
  EXPECT_TRUE(region.numa_interleaved());
  std::memset(region.data(), 0x3D, region.size());
  EXPECT_EQ(static_cast<unsigned char*>(region.data())[region.size() - 1],
            0x3D);
  auto decommit = region.decommit(0, p);
  EXPECT_TRUE(decommit.IsFailedPrecondition());
}

TEST(Numa, SelectedInterleaveRejectsInvalidShape) {
  const std::size_t p = page_size();
  const unsigned node = numa_nodes().front();
  const std::array<unsigned, 2> duplicate{node, node};
  auto duplicate_result = vm_allocate_numa_interleaved(p, duplicate, p);
  ASSERT_FALSE(duplicate_result);
  EXPECT_TRUE(duplicate_result.error().IsInvalidArgument());

  const std::array<unsigned, 1> invalid{numa_nodes().back() + 10u};
  auto invalid_result = vm_allocate_numa_interleaved(p, invalid, p);
  ASSERT_FALSE(invalid_result);
  EXPECT_TRUE(invalid_result.error().IsOutOfRange());

  auto bad_chunk = vm_allocate_numa_interleaved(p, {}, p + 1u);
  ASSERT_FALSE(bad_chunk);
  EXPECT_TRUE(bad_chunk.error().IsInvalidArgument());
  EXPECT_TRUE(vm_allocate_numa_interleaved(0).error().IsInvalidArgument());
}
