/* SPDX-License-Identifier: LicenseRef-HYLE-Proprietary */

#include "hyle/simd/provider/route.h"

#include <gtest/gtest.h>

#include <cstddef>

namespace hyle::simd::provider {
namespace {

struct ProbeLane final {};

template <Provider>
struct ProbeCandidate;

template <>
struct ProbeCandidate<Provider::kHyle> final {
  static int run(int value) noexcept { return value + 1; }
};

template <>
struct ProbeCandidate<Provider::kHighway> final {
  static int run(int value) noexcept { return value + 2; }
};

template <>
struct ProbeCandidate<Provider::kXsimd> final {
  static int run(int value) noexcept { return value + 3; }
};

}  // namespace

// Use an impossible product key so this test cannot collide with a later
// generated real-lane route specialization.
template <>
struct RouteFor<RouteKey<Operation::kAdd>, ProbeLane, Target::kScalar,
                Compiler::kOther, 0> final {
  inline static constexpr Route value{Provider::kHighway,
                                      Decision::kRouteHighway};
};

namespace {

static_assert(kOperationCount == 387);
static_assert(static_cast<std::size_t>(Operation::kCount) == 387);
static_assert(kRouteFor<Operation::kAbs, LaneVariant<float>, Target::kScalar,
                        Compiler::kOther, 0>
                  .qualified());
static_assert(kRouteFor<Operation::kAbs, LaneVariant<float>, Target::kScalar,
                        Compiler::kOther, 0>
                  .provider == Provider::kHighway);

// Exact material exceptions are finite type/target/compiler routes. The same
// operation immediately falls back to Highway outside the signed selector.
static_assert(
    kRouteFor<Operation::kBitShuffle, LaneVariant<std::uint64_t>,
              Target::kAvx2, Compiler::kMsvc, 195000000>
        .provider == Provider::kHyle);
static_assert(
    kRouteFor<Operation::kBitShuffle, LaneVariant<std::uint64_t>,
              Target::kAvx2, Compiler::kClang, 210000>
        .provider == Provider::kHighway);

// Stream is the sole type-agnostic material exception: the current Apple
// NEON compiler retains HYLE's plain store, while tied AVX2 and every other
// compiler/target follow the Highway default.
static_assert(kRouteFor<Operation::kStream, AnyVariant, Target::kNeon,
                        Compiler::kAppleClang, 210000>
                  .provider == Provider::kHyle);
static_assert(kRouteFor<Operation::kStream, AnyVariant, Target::kNeon,
                        Compiler::kClang, 210000>
                  .provider == Provider::kHighway);
static_assert(kRouteFor<Operation::kStream, AnyVariant, Target::kAvx2,
                        Compiler::kMsvc, 195000000>
                  .provider == Provider::kHighway);

static_assert(
    kRouteFor<Family::kTanhF32Batch, LaneVariant<float>, Target::kAvx2,
              Compiler::kMsvc, 195000000>
        .provider == Provider::kXsimd);
static_assert(
    kRouteFor<Family::kTanhF32Batch, LaneVariant<float>, Target::kAvx2,
              Compiler::kGcc, 140200>
        .provider == Provider::kHighway);

static_assert(kRouteFor<Operation::kZipLower>.qualified());
static_assert(kRouteFor<Operation::kZipLower>.provider == Provider::kNone);
static_assert(kRouteFor<Operation::kZipLower>.decision ==
              Decision::kRemoveRedundant);

TEST(SimdProviderRoute, ExplicitCandidateSelectionIsCompileTime) {
  EXPECT_EQ((HYLE_SIMD_INVOKE_CANDIDATE(Provider::kHyle, ProbeCandidate, 10)),
            11);
  EXPECT_EQ((HYLE_SIMD_INVOKE_CANDIDATE(Provider::kHighway, ProbeCandidate, 10)),
            12);
  EXPECT_EQ((HYLE_SIMD_INVOKE_CANDIDATE(Provider::kXsimd, ProbeCandidate, 10)),
            13);
}

TEST(SimdProviderRoute, QualifiedRouteSelectsOneCandidate) {
  constexpr Provider selected =
      kQualifiedProviderFor<Operation::kAdd, ProbeLane, Target::kScalar,
                            Compiler::kOther, 0>;
  EXPECT_EQ((HYLE_SIMD_INVOKE_CANDIDATE(selected, ProbeCandidate, 10)), 12);
}

}  // namespace
}  // namespace hyle::simd::provider
