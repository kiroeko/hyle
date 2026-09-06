#include <gtest/gtest.h>

#include <cstring>
#include <utility>

#include "hyle/mem/hugepage.h"
#include "hyle/core/environment.h"
#include "hyle/platform/numa.h"
#include "hyle/platform/virtual_memory.h"

using hyle::mem::HugeFlags;
using hyle::mem::HugeMapping;
using hyle::platform::kNumaAny;
using hyle::platform::numa_nodes;
using hyle::platform::page_size;

TEST(MemNuma, HugeMappingOnAvailableNode) {
  HugeMapping r = HugeMapping::allocate(page_size() * 8, HugeFlags::kThpHint,
                                      static_cast<int>(numa_nodes().front()))
                     .value();
  ASSERT_TRUE(r.valid());
  std::memset(r.data(), 0xAB, 128);
  EXPECT_EQ(static_cast<unsigned char*>(r.data())[0], 0xAB);
}

TEST(MemNuma, HugeMappingLocal) {
  HugeMapping r = HugeMapping::allocate_local(page_size() * 8,
                                             HugeFlags::kThpHint)
                     .value();
  ASSERT_TRUE(r.valid());
  EXPECT_GE(r.size(), page_size() * 8);
  std::memset(r.data(), 0xCD, 64);
  EXPECT_EQ(static_cast<unsigned char*>(r.data())[0], 0xCD);
}

TEST(MemNuma, HugeMappingAnyNode) {
  HugeMapping r = HugeMapping::allocate(page_size() * 4, HugeFlags::kNone,
                                      kNumaAny)
                     .value();
  ASSERT_TRUE(r.valid());
}

TEST(MemNuma, HugeMappingInterleaved) {
  HugeMapping r =
      HugeMapping::allocate_interleaved(page_size() * 16).value();
  ASSERT_TRUE(r.valid());
  std::memset(r.data(), 0x3C, 64);
  EXPECT_EQ(static_cast<unsigned char*>(r.data())[0], 0x3C);
}

TEST(MemNuma, HugeMappingHardBindFlag) {
  auto result =
      HugeMapping::allocate(page_size() * 8,
                           HugeFlags::kThpHint | HugeFlags::kNumaBind,
                           static_cast<int>(numa_nodes().front()));
#if HYLE_OS_WINDOWS
  if (numa_nodes().size() > 1u) {
    EXPECT_FALSE(result);
    EXPECT_TRUE(result.error().IsUnimplemented());
    return;
  }
#endif
  HugeMapping r = *std::move(result);
  ASSERT_TRUE(r.valid());
}
