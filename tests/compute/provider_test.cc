#include <gtest/gtest.h>

#include <array>
#include <limits>

#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/provider.h"

namespace {

using hyle::compute::ProviderReplayClass;
using hyle::compute::ProviderRuntimeMode;
using hyle::compute::ProviderRuntimeProperties;
using hyle::compute::ProviderWorkspaceOwner;

TEST(ProviderImplementationId,
     IsAnOptionalImplementationIdentityNotAProviderSlot) {
  using hyle::compute::ProviderImplementationId;
  constexpr ProviderImplementationId unknown{}, known{42}, other{43};
  static_assert(!unknown.valid());
  static_assert(known.valid() && known.value() == 42);
  static_assert(known == ProviderImplementationId{42});
  static_assert(known != other);
  EXPECT_EQ(unknown, ProviderImplementationId{0});
  EXPECT_NE(unknown, known);
  EXPECT_TRUE(ProviderImplementationId{UINT64_MAX}.valid());
}

TEST(ProviderCostFacts, UnknownIsNotAnExactZero) {
  constexpr hyle::compute::ProviderRouteCostFacts unknown{};
  static_assert(unknown.valid());
  EXPECT_FALSE(unknown.workspace.has_value());
  EXPECT_FALSE(unknown.persistent_bytes.has_value());
  EXPECT_FALSE(unknown.copy_bytes.has_value());
  EXPECT_FALSE(unknown.repack_bytes.has_value());
  EXPECT_FALSE(unknown.synchronization_count.has_value());
  EXPECT_FALSE(unknown.dispatch_count.has_value());

  constexpr hyle::compute::ProviderRouteCostFacts known{
      .workspace = hyle::compute::WorkspaceRequirement{},
      .persistent_bytes = 0,
      .copy_bytes = 0,
      .repack_bytes = 0,
      .synchronization_count = 0,
      .dispatch_count = 1,
  };
  static_assert(known.valid());
  EXPECT_EQ(known.workspace->size, 0u);
  EXPECT_EQ(known.persistent_bytes, 0u);
  EXPECT_EQ(known.copy_bytes, 0u);
  EXPECT_EQ(known.repack_bytes, 0u);
  EXPECT_EQ(known.synchronization_count, 0u);
  EXPECT_EQ(known.dispatch_count, 1u);
}

TEST(ProviderCostFacts, WorkspaceAlignmentAndDispatchHaveOneSharedValidator) {
  hyle::compute::ProviderRouteCostFacts facts;
  for (std::size_t alignment : {std::size_t{0}, std::size_t{3}, std::size_t{6},
                                (std::numeric_limits<std::size_t>::max)()}) {
    facts.workspace = {.size = 0, .alignment = alignment};
    EXPECT_FALSE(facts.valid());
  }
  for (std::size_t alignment :
       {std::size_t{1}, std::size_t{64}, std::size_t{4096}}) {
    facts.workspace = {.size = 17, .alignment = alignment};
    EXPECT_TRUE(facts.valid());
  }
  facts.dispatch_count = 0;
  EXPECT_FALSE(facts.valid());
  facts.dispatch_count = (std::numeric_limits<std::uint32_t>::max)();
  EXPECT_TRUE(facts.valid());
  facts.dispatch_count.reset();
  EXPECT_TRUE(facts.valid());
}

TEST(ProviderCostFacts, CompilationStorageBoundsAreOptionalAndConsistent) {
  using hyle::compute::ProviderCompilationStorage;
  constexpr ProviderCompilationStorage unknown{}, zero{0, 0},
      max{SIZE_MAX, SIZE_MAX}, resident_only{64, std::nullopt},
      peak_only{std::nullopt, 64}, invalid{65, 64};
  static_assert(unknown.valid() && zero.valid() && max.valid());
  static_assert(resident_only.valid() && peak_only.valid() && !invalid.valid());
  EXPECT_FALSE(unknown.resident_bytes);
  EXPECT_FALSE(unknown.peak_bytes);
  EXPECT_EQ(zero.resident_bytes, 0U);
  hyle::compute::ProviderRouteCostFacts facts{.compilation_storage = invalid};
  EXPECT_FALSE(facts.valid());
  facts.compilation_storage = max;
  EXPECT_TRUE(facts.valid());
}

TEST(ProviderProperties, ReplayCostDoesNotFollowOwnershipOrCaptureFlags) {
  constexpr ProviderRuntimeProperties unspecified{};
  static_assert(unspecified.valid());
  static_assert(unspecified.replay_class == ProviderReplayClass::kUnknown);
  constexpr std::array classes{ProviderReplayClass::kUnknown,
                               ProviderReplayClass::kPrepared,
                               ProviderReplayClass::kManaged};
  constexpr std::array modes{ProviderRuntimeMode::kSharedQueue,
                             ProviderRuntimeMode::kDelegated};
  for (auto mode : modes) {
    for (auto replay_class : classes) {
      ProviderRuntimeProperties properties{.mode = mode,
                                           .replay_class = replay_class};
      EXPECT_TRUE(properties.valid());
      properties.workspace_owner = ProviderWorkspaceOwner::kProvider;
      EXPECT_TRUE(properties.valid());
      EXPECT_EQ(properties.replay_class, replay_class);
      properties.capture_compatible = true;
      EXPECT_EQ(properties.valid(), mode == ProviderRuntimeMode::kSharedQueue);
      EXPECT_EQ(properties.replay_class, replay_class);
    }
  }
}

TEST(ProviderProperties, EqualityIncludesAllFactsAndInvalidEnumsAreRejected) {
  const ProviderRuntimeProperties original{};
  EXPECT_EQ(original, ProviderRuntimeProperties{});
  auto changed = original;
  changed.replay_class = ProviderReplayClass::kPrepared;
  EXPECT_NE(changed, original);
  changed.replay_class = ProviderReplayClass::kManaged;
  EXPECT_NE(changed, original);
  changed.replay_class = static_cast<ProviderReplayClass>(255);
  EXPECT_FALSE(changed.valid());
  changed = original;
  changed.mode = static_cast<ProviderRuntimeMode>(255);
  EXPECT_FALSE(changed.valid());
  changed.mode = ProviderRuntimeMode::kDelegated;
  EXPECT_NE(changed, original);
  changed = original;
  changed.workspace_owner = static_cast<ProviderWorkspaceOwner>(255);
  EXPECT_FALSE(changed.valid());
  changed.workspace_owner = ProviderWorkspaceOwner::kProvider;
  EXPECT_NE(changed, original);
  changed = original;
  changed.blocks_host = true;
  EXPECT_NE(changed, original);
  changed = original;
  changed.uses_private_threads = true;
  EXPECT_NE(changed, original);
  changed = original;
  changed.supports_concurrent_invocations = false;
  EXPECT_NE(changed, original);
  changed = original;
  changed.capture_compatible = true;
  EXPECT_NE(changed, original);
}

} // namespace
