#include <gtest/gtest.h>

#include <cstring>
#include <utility>

#include "hyle/mem/large_mapping.h"
#include "hyle/core/environment.h"
#include "hyle/platform/numa.h"
#include "hyle/platform/virtual_memory.h"

using hyle::mem::LargeMappingFlags;
using hyle::mem::LargeMapping;
using hyle::platform::kNumaAny;
using hyle::platform::numa_nodes;
using hyle::platform::page_size;

TEST(MemNuma, LargeMappingOnAvailableNode) {
  LargeMapping r = LargeMapping::allocate(page_size() * 8, LargeMappingFlags::kThpHint,
                                      static_cast<int>(numa_nodes().front()))
                     .value();
  ASSERT_TRUE(r.valid());
  std::memset(r.data(), 0xAB, 128);
  EXPECT_EQ(static_cast<unsigned char*>(r.data())[0], 0xAB);
}
TEST(MemNuma, LargeMappingLocal) {
  LargeMapping r = LargeMapping::allocate_local(page_size() * 8,
                                             LargeMappingFlags::kThpHint)
                     .value();
  ASSERT_TRUE(r.valid());
  EXPECT_GE(r.size(), page_size() * 8);
  std::memset(r.data(), 0xCD, 64);
  EXPECT_EQ(static_cast<unsigned char*>(r.data())[0], 0xCD);
}

TEST(MemNuma, LargeMappingAnyNode) {
  LargeMapping r = LargeMapping::allocate(page_size() * 4, LargeMappingFlags::kNone,
                                      kNumaAny)
                     .value();
  ASSERT_TRUE(r.valid());
}

TEST(MemNuma, LargeMappingInterleaved) {
  LargeMapping r =
      LargeMapping::allocate_interleaved(page_size() * 16).value();
  ASSERT_TRUE(r.valid());
  std::memset(r.data(), 0x3C, 64);
  EXPECT_EQ(static_cast<unsigned char*>(r.data())[0], 0x3C);
}

TEST(MemNuma, LargeMappingHardBindFlag) {
  auto result =
      LargeMapping::allocate(page_size() * 8,
                           LargeMappingFlags::kThpHint | LargeMappingFlags::kNumaBind,
                           static_cast<int>(numa_nodes().front()));
#if HYLE_OS_WINDOWS
  if (numa_nodes().size() > 1u) {
    EXPECT_FALSE(result);
    EXPECT_TRUE(result.error().IsUnimplemented());
    return;
  }
#endif
  LargeMapping r = *std::move(result);
  ASSERT_TRUE(r.valid());
}
