#include <gtest/gtest.h>

#include <array>

#include "hyle/compute/provider.h"

namespace {

using hyle::compute::ProviderReplayClass;
using hyle::compute::ProviderRuntimeMode;
using hyle::compute::ProviderRuntimeProperties;
using hyle::compute::ProviderWorkspaceOwner;

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
