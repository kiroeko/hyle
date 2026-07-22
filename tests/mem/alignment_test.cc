#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "hyle/mem/alignment.h"

using hyle::mem::align_down;
using hyle::mem::align_down_ptr;
using hyle::mem::align_up;
using hyle::mem::align_up_ptr;
using hyle::mem::is_aligned;
using hyle::mem::is_aligned_ptr;
using hyle::mem::is_pow2;
using hyle::mem::kMaxSimdAlignment;

TEST(Alignment, IsPow2) {
  EXPECT_TRUE(is_pow2(1));
  EXPECT_TRUE(is_pow2(8));
  EXPECT_TRUE(is_pow2(64));
  EXPECT_FALSE(is_pow2(0));
  EXPECT_FALSE(is_pow2(3));
  EXPECT_FALSE(is_pow2(48));
}

TEST(Alignment, AlignUpSize) {
  EXPECT_EQ(align_up(0u, 8), 0u);
  EXPECT_EQ(align_up(1u, 8), 8u);
  EXPECT_EQ(align_up(8u, 8), 8u);    // already aligned -> unchanged
  EXPECT_EQ(align_up(9u, 8), 16u);
  EXPECT_EQ(align_up(63u, 64), 64u);
  EXPECT_EQ(align_up(64u, 64), 64u);
  EXPECT_EQ(align_up(65u, 64), 128u);
}

TEST(Alignment, AlignDownSize) {
  EXPECT_EQ(align_down(0u, 8), 0u);
  EXPECT_EQ(align_down(7u, 8), 0u);
  EXPECT_EQ(align_down(8u, 8), 8u);
  EXPECT_EQ(align_down(15u, 8), 8u);
  EXPECT_EQ(align_down(129u, 64), 128u);
}

TEST(Alignment, IsAlignedSize) {
  EXPECT_TRUE(is_aligned(0u, 16));
  EXPECT_TRUE(is_aligned(16u, 16));
  EXPECT_TRUE(is_aligned(32u, 16));
  EXPECT_FALSE(is_aligned(1u, 16));
  EXPECT_FALSE(is_aligned(17u, 16));
}

TEST(Alignment, UpDownConsistency) {
  // For any n, align_down(n) <= n <= align_up(n), and both are aligned.
  for (std::size_t n = 0; n < 500; ++n) {
    std::size_t up = align_up(n, 16);
    std::size_t down = align_down(n, 16);
    EXPECT_GE(up, n);
    EXPECT_LE(down, n);
    EXPECT_TRUE(is_aligned(up, 16));
    EXPECT_TRUE(is_aligned(down, 16));
    EXPECT_LE(up - down, 16u);
  }
}

TEST(Alignment, AddressAlignUpReportsOverflow) {
  std::uintptr_t out = 0;
  EXPECT_TRUE(hyle::mem::detail::try_align_up_address(65, 64, &out));
  EXPECT_EQ(out, 128u);

  out = 0xA5;
  EXPECT_FALSE(hyle::mem::detail::try_align_up_address(UINTPTR_MAX - 31, 64, &out));
  EXPECT_EQ(out, 0xA5u);  // failure does not publish a wrapped address
}

TEST(Alignment, PointerVariants) {
  alignas(64) char buf[256];
  char* base = buf;  // 64-aligned
  EXPECT_TRUE(is_aligned_ptr(base, 64));

  // A deliberately misaligned interior pointer.
  void* p = base + 5;
  EXPECT_FALSE(is_aligned_ptr(p, 8));
  void* up = align_up_ptr(p, 8);
  void* down = align_down_ptr(p, 8);
  EXPECT_TRUE(is_aligned_ptr(up, 8));
  EXPECT_TRUE(is_aligned_ptr(down, 8));
  EXPECT_EQ(up, base + 8);
  EXPECT_EQ(down, base + 0);
}

TEST(Alignment, ConstexprUsable) {
  static_assert(align_up(1u, 8) == 8u);
  static_assert(align_down(15u, 8) == 8u);
  static_assert(is_aligned(64u, 64));
  static_assert(kMaxSimdAlignment == 64);
  SUCCEED();
}
