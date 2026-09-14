#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include "hyle/compute/route_profile.h"

namespace {

using namespace hyle::compute;

constexpr ProfileFingerprint fingerprint(std::uint64_t value) {
  return {{value, value + 1, value + 2, value + 3}};
}

TensorDesc matrix(DataType type = DataType::kFloat32) {
  return TensorDesc::dense(type, std::array<std::int64_t, 2>{2, 3}).value();
}

struct Fixture final {
  std::array<ProfileTargetIdentity, 2> targets{
      ProfileTargetIdentity{BackendId{1, 2}, fingerprint(10), fingerprint(20),
                            fingerprint(30), fingerprint(40)},
      ProfileTargetIdentity{BackendId{1, 3}, fingerprint(11), fingerprint(21),
                            fingerprint(31), fingerprint(41)}};
  std::array<ProfileProviderIdentity, 2> providers{
      ProfileProviderIdentity{ProviderId{2, 1}, fingerprint(50),
                              fingerprint(60), ProviderRouteKey{1}},
      ProfileProviderIdentity{ProviderId{2, 2}, fingerprint(51),
                              fingerprint(61), ProviderRouteKey{2}}};
  std::array<TensorDesc, 2> inputs{matrix(), matrix(DataType::kFloat16)};
  std::array<TensorDesc, 1> outputs{matrix()};

  RouteProfileIdentity identity() const {
    return {.hyle_build = fingerprint(100),
            .semantics = fingerprint(200),
            .shape_domain = fingerprint(300),
            .route = fingerprint(400),
            .workload_data = fingerprint(500),
            .targets = targets,
            .providers = providers,
            .inputs = inputs,
            .outputs = outputs,
            .completion = ProfileCompletionBoundary::kGraphTerminal};
  }
  RouteProfile profile() const {
    return {.identity = identity(),
            .replay = ProfileTimingInterval{100, 110},
            .compilation = ProfileTimingInterval{1000, 2000}};
  }
};

TEST(RouteProfile, MatchesIndependentEqualDataWithoutRetainingIt) {
  RouteProfileMatch match;
  {
    Fixture request, observation;
    match = match_route_profile(request.identity(), observation.profile());
    EXPECT_EQ(match.reason, ProfileMatchReason::kMatched);
    ASSERT_TRUE(match.cost);
  }
  EXPECT_EQ(match.cost, (ProfileTimingInterval{100, 110}));
  EXPECT_TRUE(WorkloadIntent{}.valid());
  // A copy-only plan need not invent a Provider; it still needs a qualified
  // backend, physical route and complete boundary identity.
  Fixture fixture;
  auto request = fixture.identity();
  auto profile = fixture.profile();
  request.providers = {};
  profile.identity.providers = {};
  EXPECT_EQ(match_route_profile(request, profile).reason,
            ProfileMatchReason::kMatched);
}

TEST(RouteProfile, SeparatesEveryOpaqueIdentityDimension) {
  Fixture fixture;
  const auto request = fixture.identity();
  struct Mutation {
    ProfileFingerprint RouteProfileIdentity::*field;
    ProfileMatchReason reason;
  };
  const std::array mutations{Mutation{&RouteProfileIdentity::hyle_build,
                                      ProfileMatchReason::kHyleBuildMismatch},
                             Mutation{&RouteProfileIdentity::semantics,
                                      ProfileMatchReason::kSemanticsMismatch},
                             Mutation{&RouteProfileIdentity::shape_domain,
                                      ProfileMatchReason::kShapeDomainMismatch},
                             Mutation{&RouteProfileIdentity::route,
                                      ProfileMatchReason::kRouteMismatch},
                             Mutation{&RouteProfileIdentity::workload_data,
                                      ProfileMatchReason::kWorkloadMismatch}};
  for (const auto mutation : mutations) {
    auto profile = fixture.profile();
    profile.identity.*mutation.field = fingerprint(900);
    const auto mismatch = match_route_profile(request, profile);
    EXPECT_EQ(mismatch.reason, mutation.reason);
    EXPECT_FALSE(mismatch.cost);
    // Equal unknowns do NOT become equal qualified identities.
    profile.identity.*mutation.field = {};
    auto unknown_request = request;
    unknown_request.*mutation.field = {};
    EXPECT_EQ(match_route_profile(unknown_request, profile).reason,
              ProfileMatchReason::kIncompleteIdentity);
  }
}

TEST(RouteProfile, ChecksBuildDeviceRuntimeConfigurationAndCanonicalOrder) {
  Fixture request;
  for (std::size_t member = 0; member < 5; ++member) {
    Fixture observation;
    auto &target = observation.targets[1];
    switch (member) {
    case 0:
      target.backend = BackendId{9, 9};
      break;
    case 1:
      target.build = fingerprint(999);
      break;
    case 2:
      target.hardware = fingerprint(999);
      break;
    case 3:
      target.runtime = fingerprint(999);
      break;
    case 4:
      target.configuration = fingerprint(999);
      break;
    }
    EXPECT_EQ(
        match_route_profile(request.identity(), observation.profile()).reason,
        ProfileMatchReason::kTargetMismatch);
  }
  for (std::size_t member = 0; member < 4; ++member) {
    Fixture observation;
    auto &provider = observation.providers[1];
    switch (member) {
    case 0:
      provider.provider = ProviderId{9, 9};
      break;
    case 1:
      provider.build = fingerprint(999);
      break;
    case 2:
      provider.configuration = fingerprint(999);
      break;
    case 3:
      provider.implementation = ProviderRouteKey{999};
      break;
    }
    EXPECT_EQ(
        match_route_profile(request.identity(), observation.profile()).reason,
        ProfileMatchReason::kProviderMismatch);
  }
  Fixture observation;
  std::swap(observation.targets[0], observation.targets[1]);
  EXPECT_EQ(
      match_route_profile(request.identity(), observation.profile()).reason,
      ProfileMatchReason::kTargetMismatch);
  std::swap(observation.targets[0], observation.targets[1]);
  std::swap(observation.providers[0], observation.providers[1]);
  EXPECT_EQ(
      match_route_profile(request.identity(), observation.profile()).reason,
      ProfileMatchReason::kProviderMismatch);
  auto profile = request.profile();
  profile.identity.targets = profile.identity.targets.first(1);
  EXPECT_EQ(match_route_profile(request.identity(), profile).reason,
            ProfileMatchReason::kTargetMismatch);
  profile = request.profile();
  profile.identity.providers = profile.identity.providers.first(1);
  EXPECT_EQ(match_route_profile(request.identity(), profile).reason,
            ProfileMatchReason::kProviderMismatch);
}

TEST(RouteProfile, MissingBackendAndImplementationFactsStayUnknown) {
  Fixture fixture;
  fixture.providers[0].implementation = {};
  EXPECT_EQ(match_route_profile(fixture.identity(), fixture.profile()).reason,
            ProfileMatchReason::kIncompleteIdentity);
  fixture.providers[0].implementation = ProviderRouteKey{1};
  fixture.targets[0].runtime = {};
  EXPECT_EQ(match_route_profile(fixture.identity(), fixture.profile()).reason,
            ProfileMatchReason::kIncompleteIdentity);
  fixture.targets[0].runtime = fingerprint(30);
  auto request = fixture.identity();
  auto profile = fixture.profile();
  request.targets = {};
  profile.identity.targets = {};
  EXPECT_EQ(match_route_profile(request, profile).reason,
            ProfileMatchReason::kIncompleteIdentity);
}

TEST(RouteProfile, TensorShapeDtypeAndLayoutAreNotOpaqueByteCounts) {
  Fixture request;
  for (int change : {0, 1, 2, 3, 4, 5}) {
    Fixture observation;
    if (change == 0)
      observation.inputs[0] = matrix(DataType::kFloat16);
    if (change == 1)
      observation.outputs[0] = observation.outputs[0].transposed().value();
    if (change == 2)
      std::swap(observation.inputs[0], observation.inputs[1]);
    if (change == 4) {
      // Equal dtype, shape AND allocation extent, different logical layout.
      observation.inputs[0] =
          TensorDesc::strided_bytes(DataType::kFloat32,
                                    std::array<std::int64_t, 2>{2, 3},
                                    std::array<std::int64_t, 2>{4, 8})
              .value();
      ASSERT_EQ(observation.inputs[0].required_bytes(),
                request.inputs[0].required_bytes());
    }
    if (change == 5) {
      observation.inputs[0] =
          TensorDesc::opaque(
              DataType::kFloat32, std::array<std::int64_t, 2>{2, 3},
              StorageFormat::opaque(StorageFormatId{99, 1}).value(), 24)
              .value();
      ASSERT_EQ(observation.inputs[0].required_bytes(),
                request.inputs[0].required_bytes());
    }
    auto profile = observation.profile();
    if (change == 3)
      profile.identity.inputs = profile.identity.inputs.first(1);
    EXPECT_EQ(match_route_profile(request.identity(), profile).reason,
              ProfileMatchReason::kTensorMismatch);
  }
  Fixture invalid;
  invalid.inputs[0] = {};
  EXPECT_EQ(match_route_profile(invalid.identity(), invalid.profile()).reason,
            ProfileMatchReason::kIncompleteIdentity);
}

TEST(RouteProfile, WorkloadAndTerminalBoundaryMustMatch) {
  Fixture fixture;
  for (int change : {0, 1, 2}) {
    auto profile = fixture.profile();
    if (change == 0)
      profile.identity.workload.objective =
          WorkloadObjective::kSteadyThroughput;
    if (change == 1)
      profile.identity.workload.concurrent_invocations = 2;
    if (change == 2)
      profile.identity.workload.require_determinism = true;
    EXPECT_EQ(match_route_profile(fixture.identity(), profile).reason,
              ProfileMatchReason::kWorkloadMismatch);
  }
  auto profile = fixture.profile();
  profile.identity.completion = ProfileCompletionBoundary::kOperationTerminal;
  EXPECT_EQ(match_route_profile(fixture.identity(), profile).reason,
            ProfileMatchReason::kCompletionMismatch);
  for (const auto boundary : {ProfileCompletionBoundary::kUnknown,
                              static_cast<ProfileCompletionBoundary>(255)}) {
    profile.identity.completion = boundary;
    auto request = profile.identity;
    EXPECT_EQ(match_route_profile(request, profile).reason,
              ProfileMatchReason::kIncompleteIdentity);
  }
  for (int invalid : {0, 1, 2}) {
    profile = fixture.profile();
    if (invalid == 0)
      profile.identity.workload.concurrent_invocations = 0;
    if (invalid == 1)
      profile.identity.workload.expected_replays = 0;
    if (invalid == 2)
      profile.identity.workload.objective = static_cast<WorkloadObjective>(255);
    EXPECT_EQ(match_route_profile(profile.identity, profile).reason,
              ProfileMatchReason::kIncompleteIdentity);
  }
}

TEST(RouteProfile, MissingTimingAndExactZeroAreDifferent) {
  Fixture fixture;
  auto profile = fixture.profile();
  profile.compilation.reset();
  EXPECT_EQ(match_route_profile(fixture.identity(), profile).reason,
            ProfileMatchReason::kMatched);
  profile.replay.reset();
  EXPECT_EQ(match_route_profile(fixture.identity(), profile).reason,
            ProfileMatchReason::kMissingTiming);
  profile.replay = ProfileTimingInterval{};
  const auto zero = match_route_profile(fixture.identity(), profile);
  ASSERT_TRUE(zero.cost);
  EXPECT_EQ(zero.cost->lower_ns, 0);
  EXPECT_EQ(zero.cost->upper_ns, 0);
}

TEST(RouteProfile, AmortizationIsExplicitAndNeedsTheWholeCompileCost) {
  Fixture fixture;
  auto profile = fixture.profile();
  // Default latency is not polluted by a 1000+ ns compile estimate.
  EXPECT_EQ(match_route_profile(profile.identity, profile).cost,
            (ProfileTimingInterval{100, 110}));
  auto request = profile.identity;
  request.workload.expected_replays = 10;
  EXPECT_EQ(match_route_profile(request, profile).cost,
            (ProfileTimingInterval{200, 310}));
  profile.compilation.reset();
  EXPECT_EQ(match_route_profile(request, profile).reason,
            ProfileMatchReason::kMissingTiming);
  profile.compilation = ProfileTimingInterval{};
  EXPECT_EQ(match_route_profile(request, profile).cost,
            (ProfileTimingInterval{100, 110}));
}

TEST(RouteProfile, RejectsNonfiniteNegativeInvertedAndOverflowedTiming) {
  Fixture fixture;
  const auto infinity = (std::numeric_limits<double>::infinity)();
  const auto nan = (std::numeric_limits<double>::quiet_NaN)();
  const std::array invalid{
      ProfileTimingInterval{-1, 2}, ProfileTimingInterval{2, 1},
      ProfileTimingInterval{1, infinity}, ProfileTimingInterval{nan, 1},
      ProfileTimingInterval{1, nan}};
  for (const auto interval : invalid) {
    EXPECT_FALSE(interval.valid());
    auto profile = fixture.profile();
    profile.replay = interval;
    EXPECT_EQ(match_route_profile(profile.identity, profile).reason,
              ProfileMatchReason::kInvalidTiming);
    profile = fixture.profile();
    profile.compilation = interval;
    EXPECT_EQ(match_route_profile(profile.identity, profile).reason,
              ProfileMatchReason::kInvalidTiming);
  }
  auto profile = fixture.profile();
  const auto maximum = (std::numeric_limits<double>::max)();
  profile.identity.workload.expected_replays = 1;
  profile.replay = ProfileTimingInterval{maximum, maximum};
  profile.compilation = ProfileTimingInterval{maximum, maximum};
  EXPECT_EQ(match_route_profile(profile.identity, profile).reason,
            ProfileMatchReason::kInvalidTiming);
}

TEST(RouteProfile, DuplicateEvidenceIsEnvelopedRegardlessOfOrder) {
  Fixture fixture;
  auto first = fixture.profile(), second = first, unrelated = first;
  second.replay = ProfileTimingInterval{90, 130};
  unrelated.identity.route = fingerprint(901);
  unrelated.replay = ProfileTimingInterval{1, 2};
  std::array profiles{first, unrelated, second};
  for (int repeat = 0; repeat < 2; ++repeat) {
    const auto lookup = find_route_profile(fixture.identity(), profiles);
    EXPECT_EQ(lookup.cost, (ProfileTimingInterval{90, 130}));
    EXPECT_EQ(lookup.matched_entries, 2u);
    EXPECT_EQ(lookup.rejected_entries, 1u);
    std::reverse(profiles.begin(), profiles.end());
  }
  EXPECT_FALSE(find_route_profile(fixture.identity(), {}).cost);
  EXPECT_FALSE(find_route_profile({}, profiles).cost);
  EXPECT_EQ(find_route_profile({}, profiles).rejected_entries, 3u);
}

} // namespace
