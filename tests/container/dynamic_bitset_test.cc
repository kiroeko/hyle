#include <gtest/gtest.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "hyle/container/dynamic_bitset.h"
#include "hyle/mem/allocator.h"

using hyle::container::DynamicBitSet;

namespace {

void* fail_allocate(void*, std::size_t, std::size_t) noexcept {
  return nullptr;
}

void fail_deallocate(void*, void*, std::size_t, std::size_t) noexcept {}

void* fail_reallocate(void*, void*, std::size_t, std::size_t,
                      std::size_t) noexcept {
  return nullptr;
}

}  // namespace

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

TEST(DynamicBitSet, CopyMoveAssignmentSwapAndSelfAssignmentPreserveValue) {
  DynamicBitSet source(130);
  source.set(0).set(64).set(129);

  DynamicBitSet copy_target(7, true);
  copy_target = source;
  EXPECT_EQ(copy_target, source);
  copy_target = copy_target;
  EXPECT_EQ(copy_target, source);

  DynamicBitSet move_target(3, true);
  move_target = std::move(copy_target);
  EXPECT_EQ(move_target, source);
  EXPECT_TRUE(copy_target.empty());

  DynamicBitSet other(5, true);
  swap(move_target, other);
  EXPECT_EQ(other, source);
  EXPECT_EQ(move_target.size(), 5u);
  EXPECT_TRUE(move_target.all());
}

TEST(DynamicBitSet, FallibleGrowthAndBothKernelSizesKeepStateAtomic) {
  hyle::mem::AllocatorRef failing(&fail_allocate, &fail_deallocate,
                                  &fail_reallocate, nullptr);
  DynamicBitSet unavailable(failing);
  EXPECT_FALSE(unavailable.try_reserve(64));
  EXPECT_FALSE(unavailable.try_resize(64, true));
  EXPECT_TRUE(unavailable.empty());

  DynamicBitSet left(65);
  DynamicBitSet right(65);
  left.set(0).set(64);
  right.set(0).set(1);
  EXPECT_EQ((left & right).count(), 1u);
  EXPECT_EQ((left | right).count(), 3u);
  EXPECT_EQ((left ^ right).count(), 2u);
  DynamicBitSet difference = left;
  difference.and_not(right);
  EXPECT_EQ(difference.count(), 1u);
  EXPECT_TRUE(difference.test(64));

  DynamicBitSet large(4099);
  large.set();
  EXPECT_TRUE(large.all());
  large.flip();
  EXPECT_TRUE(large.none());
}

TEST(DynamicBitSet, EqualityUsesLogicalValueNotReservedCapacity) {
  DynamicBitSet empty;
  DynamicBitSet reserved_empty;
  reserved_empty.reserve(4096);
  EXPECT_EQ(empty, reserved_empty);

  DynamicBitSet left(130);
  left.set(0).set(64).set(129);
  DynamicBitSet same = left;
  same.reserve(8192);
  EXPECT_EQ(left, same);

  same.flip(129);
  EXPECT_NE(left, same);

  DynamicBitSet different_size(131);
  different_size.set(0).set(64).set(129);
  EXPECT_NE(left, different_size);
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
