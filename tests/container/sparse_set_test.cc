#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "hyle/container/sparse_set.h"

using hyle::container::SparseSet;

TEST(SparseSet, ConstantTimeMembershipAndDuplicateInsert) {
  SparseSet<std::uint32_t> set(100);
  EXPECT_TRUE(set.insert(7));
  EXPECT_TRUE(set.insert(90));
  EXPECT_FALSE(set.insert(7));
  EXPECT_TRUE(set.contains(7));
  EXPECT_FALSE(set.contains(6));
  EXPECT_FALSE(set.contains(100));
  EXPECT_EQ(set.size(), 2u);
}

TEST(SparseSet, SwapEraseRepairsMovedIndex) {
  SparseSet<std::uint32_t> set(32);
  set.insert(3);
  set.insert(9);
  set.insert(17);
  EXPECT_TRUE(set.erase(9));
  EXPECT_FALSE(set.contains(9));
  EXPECT_TRUE(set.contains(3));
  EXPECT_TRUE(set.contains(17));
  EXPECT_FALSE(set.erase(9));
  EXPECT_EQ(set.size(), 2u);
}

TEST(SparseSet, ClearIsConstantTimeAndStaleSparseEntriesAreSafe) {
  SparseSet<std::uint32_t> set(256);
  for (std::uint32_t i = 0; i < 100; ++i) set.insert(i * 2);
  set.clear();
  EXPECT_TRUE(set.empty());
  for (std::uint32_t i = 0; i < 200; ++i) EXPECT_FALSE(set.contains(i));
  EXPECT_TRUE(set.insert(198));
  EXPECT_TRUE(set.insert(0));
  EXPECT_TRUE(set.contains(198));
  EXPECT_TRUE(set.contains(0));
}

TEST(SparseSet, DenseIterationAndUniverseReset) {
  SparseSet<std::uint16_t> set(1000);
  set.insert(31);
  set.insert(5);
  set.insert(700);
  std::vector<std::uint16_t> values(set.begin(), set.end());
  std::sort(values.begin(), values.end());
  EXPECT_EQ(values, (std::vector<std::uint16_t>{5, 31, 700}));
  EXPECT_TRUE(set.try_reset_universe(8));
  EXPECT_TRUE(set.empty());
  EXPECT_EQ(set.universe_size(), 8u);
  EXPECT_FALSE(set.contains(31));
}

TEST(SparseSet, ByteIdsCoverTheFullByteUniverse) {
  SparseSet<std::uint8_t> set(256);
  for (unsigned id = 0; id < 256; ++id)
    EXPECT_TRUE(set.insert(static_cast<std::uint8_t>(id)));
  EXPECT_EQ(set.size(), 256u);
  EXPECT_TRUE(set.contains(0));
  EXPECT_TRUE(set.contains(255));
  EXPECT_TRUE(set.erase(255));
  EXPECT_FALSE(set.contains(255));
}
