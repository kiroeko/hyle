#include <gtest/gtest.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "hyle/container/dynamic_bitset.h"

using hyle::container::DynamicBitSet;

TEST(DynamicBitSet, BoundaryBitsResizeAndPadding) {
  DynamicBitSet bits(130);
  EXPECT_EQ(bits.size(), 130u);
  EXPECT_TRUE(bits.none());
  bits.set(0).set(63).set(64).set(129);
  EXPECT_EQ(bits.count(), 4u);
  EXPECT_TRUE(bits.test(63));
  EXPECT_TRUE(bits.test(64));

  bits.resize(65);
  EXPECT_EQ(bits.count(), 3u);
  bits.resize(130, false);
  EXPECT_EQ(bits.count(), 3u);
  EXPECT_FALSE(bits.test(129));
  bits.resize(140, true);
  EXPECT_EQ(bits.count(), 13u);
  EXPECT_TRUE(bits.test(139));
  bits.flip();
  EXPECT_EQ(bits.count(), 127u);
}

TEST(DynamicBitSet, SetAlgebraUsesCanonicalTail) {
  DynamicBitSet left(4099);
  DynamicBitSet right(4099);
  for (std::size_t i = 0; i < left.size(); i += 3) left.set(i);
  for (std::size_t i = 0; i < right.size(); i += 5) right.set(i);

  DynamicBitSet intersection = left & right;
  EXPECT_EQ(intersection.count(), 274u);
  EXPECT_TRUE(left.intersects(right));
  EXPECT_TRUE(intersection.is_subset_of(left));
  EXPECT_TRUE(intersection.is_subset_of(right));

  DynamicBitSet union_set = left | right;
  EXPECT_EQ(union_set.count(), left.count() + right.count() - 274u);
  DynamicBitSet difference = left;
  difference.and_not(right);
  EXPECT_EQ(difference.count(), left.count() - 274u);
  DynamicBitSet xor_set = left ^ right;
  EXPECT_EQ(xor_set.count(), union_set.count() - intersection.count());
}

TEST(DynamicBitSet, SetBitViewAndFindSkipZeroWords) {
  DynamicBitSet bits(1024);
  bits.set(1).set(65).set(511).set(1000);
  std::vector<std::size_t> visited;
  for (std::size_t index : bits.set_bits()) visited.push_back(index);
  EXPECT_EQ(visited, (std::vector<std::size_t>{1, 65, 511, 1000}));
  EXPECT_EQ(bits.find_next(1000), DynamicBitSet::npos);
  EXPECT_EQ(bits.find_next(DynamicBitSet::npos), DynamicBitSet::npos);
}

TEST(DynamicBitSet, CopyMoveAndAllEmptySemantics) {
  DynamicBitSet empty;
  EXPECT_TRUE(empty.all());
  DynamicBitSet source(77, true);
  DynamicBitSet copy = source;
  EXPECT_TRUE(copy.all());
  copy.reset(76);
  EXPECT_FALSE(copy.all());
  EXPECT_TRUE(source.all());
  DynamicBitSet moved = std::move(copy);
  EXPECT_EQ(moved.count(), 76u);
  EXPECT_TRUE(copy.empty());
}

TEST(DynamicBitSet, ShiftAndAppendCrossWordBoundaries) {
  DynamicBitSet bits;
  for (std::size_t i = 0; i < 130; ++i) bits.push_back(false);
  bits.set(0).set(63).set(64).set(129);
  bits <<= 1;
  EXPECT_EQ(bits.count(), 3u);
  EXPECT_TRUE(bits.test(1));
  EXPECT_TRUE(bits.test(64));
  EXPECT_TRUE(bits.test(65));
  bits >>= 64;
  EXPECT_EQ(bits.count(), 2u);
  EXPECT_TRUE(bits.test(0));
  EXPECT_TRUE(bits.test(1));
  bits.pop_back();
  EXPECT_EQ(bits.size(), 129u);
  bits >>= bits.size();
  EXPECT_TRUE(bits.none());
}
