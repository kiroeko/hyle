#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "hyle/container/probe.h"

using hyle::container::detail::BitMask;
using hyle::container::detail::ctrl_t;
using hyle::container::detail::GroupPortable;
using hyle::container::detail::h1;
using hyle::container::detail::h2;
using hyle::container::detail::kDeleted;
using hyle::container::detail::kEmpty;
using hyle::container::detail::kSentinel;

namespace {
// Collect set bit positions from a BitMask via its range-for interface.
std::vector<std::uint32_t> bits(BitMask m) {
  std::vector<std::uint32_t> out;
  for (std::uint32_t i : m) out.push_back(i);
  return out;
}
}  // namespace

TEST(Probe, H2IsLow7BitsTopClear) {
  ctrl_t v = h2(0xABCDEF12);
  EXPECT_GE(v, 0);                 // top bit clear => "full" marker
  EXPECT_EQ(v, static_cast<ctrl_t>(0x12 & 0x7F));
}

TEST(Probe, H1H2SplitDistinct) {
  std::uint64_t hash = 0x123456789ABCDEF0ull;
  EXPECT_EQ(h2(hash), static_cast<ctrl_t>(hash & 0x7F));
  EXPECT_EQ(h1(hash), static_cast<std::size_t>(hash >> 7));
}

// ---- GroupPortable (SWAR, always available) ---------------------------------

TEST(GroupPortable, MatchFindsEqualLanes) {
  ctrl_t c[8] = {5, 9, 5, kEmpty, 5, kDeleted, 9, 5};
  GroupPortable g(c);
  EXPECT_EQ(bits(g.match(5)), (std::vector<std::uint32_t>{0, 2, 4, 7}));
  EXPECT_EQ(bits(g.match(9)), (std::vector<std::uint32_t>{1, 6}));
}

TEST(GroupPortable, MatchEmptyOnlyEmptyNotDeleted) {
  ctrl_t c[8] = {1, kEmpty, kDeleted, kEmpty, 2, kSentinel, 3, 4};
  GroupPortable g(c);
  EXPECT_EQ(bits(g.match_empty()), (std::vector<std::uint32_t>{1, 3}));
}

TEST(GroupPortable, MatchEmptyOrDeletedExcludesFull) {
  ctrl_t c[8] = {1, kEmpty, kDeleted, 7, 2, kSentinel, 3, 4};
  GroupPortable g(c);
  // empty(1), deleted(2), sentinel(5) have top bit set; full(0,3,4,6,7) don't.
  EXPECT_EQ(bits(g.match_empty_or_deleted()),
            (std::vector<std::uint32_t>{1, 2, 5}));
}

TEST(GroupPortable, NoFalseMatchOnEmpty) {
  ctrl_t c[8] = {kEmpty, kEmpty, kEmpty, kEmpty, kEmpty, kEmpty, kEmpty, kEmpty};
  GroupPortable g(c);
  EXPECT_FALSE(static_cast<bool>(g.match(5)));  // 5 is full-range; no match
}

#if defined(HYLE_ARCH_X64)
#include "hyle/container/probe.h"
using hyle::container::detail::GroupSse2;

// On x64 the SSE2 group must agree with the portable one lane-for-lane (over
// the first 8 lanes they share). Proves the two backends are consistent.
TEST(GroupSse2, AgreesWithPortableOnSharedLanes) {
  ctrl_t c[16] = {5, 9, 5, kEmpty, 5, kDeleted, 9, 5,
                  2, 2, kEmpty, 7, kSentinel, 5, 1, 5};
  GroupSse2 sse(c);
  GroupPortable port(c);  // first 8 lanes
  // match(5): SSE sees all 16; portable sees first 8. Compare low-8 of SSE.
  std::uint32_t sse5 = 0;
  for (std::uint32_t i : sse.match(5)) sse5 |= (1u << i);
  std::uint32_t port5 = 0;
  for (std::uint32_t i : port.match(5)) port5 |= (1u << i);
  EXPECT_EQ(sse5 & 0xFF, port5);  // agree on shared low 8 lanes
}

// match_empty MUST flag EXACTLY the empty lanes across all 4 ctrl classes
// (empty/deleted/sentinel/full). This guards the SSSE3 fast path
// (_mm_sign_epi8, compiled when -mssse3) being equivalent to the SSE2 cmpeq
// path: build this test BOTH ways (default SSE2 + a -mssse3 build) and it must
// pass identically. kDeleted/kSentinel/full must NOT be flagged as empty.
TEST(GroupSse2, MatchEmptyExactlyEmptyAcrossAllClasses) {
  ctrl_t c[16] = {kEmpty, kDeleted, kSentinel, 0,    1,  kEmpty, 127, 64,
                  kDeleted, kEmpty, kSentinel, 5,    kEmpty, 3, 2, kEmpty};
  GroupSse2 g(c);
  std::uint32_t got = 0;
  for (std::uint32_t i : g.match_empty()) got |= (1u << i);
  // empty lanes: 0, 5, 9, 12, 15
  std::uint32_t want = (1u << 0) | (1u << 5) | (1u << 9) | (1u << 12) | (1u << 15);
  EXPECT_EQ(got, want);
  // explicitly: no deleted(1,8) / sentinel(2,10) / full(3,4,6,7,11,13,14) flagged
  for (std::uint32_t bad : {1u, 2u, 3u, 4u, 6u, 7u, 8u, 10u, 11u, 13u, 14u})
    EXPECT_EQ(got & (1u << bad), 0u) << "lane " << bad << " wrongly empty";
}
#endif
