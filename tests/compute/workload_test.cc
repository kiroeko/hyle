#include "hyle/compute/workload.h"

#include <array>
#include <string>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

#include "hyle/mem/quota_allocator.h"
#include "workload_fixture.h"

namespace {
using hyle::compute::ModelArtifactProfile;
using hyle::compute::ModelArtifactProfileDescription;
using hyle::compute::RegionSemanticId;
using hyle::compute::WorkloadContext;
using hyle::compute::WorkloadPhase;

constexpr auto kProfile = workload_test::kModel;

TEST(ModelArtifactProfile, OwnsAllFieldsAndSharesBackingUntilLastRelease) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator());
  ModelArtifactProfile retained;
  {
    std::array<std::string, 11> fields{
        std::string(kProfile.model_publisher),
        std::string(kProfile.model_repository),
        std::string(kProfile.model_family),
        std::string(kProfile.model_name),
        std::string(kProfile.model_revision),
        std::string(kProfile.configuration_digest),
        std::string(kProfile.artifact_repository),
        std::string(kProfile.artifact_revision),
        std::string(kProfile.variant),
        std::string(kProfile.manifest_digest),
        std::string(kProfile.format_family)};
    auto made = ModelArtifactProfile::create(
        {fields[0], fields[1], fields[2], fields[3], fields[4], fields[5],
         fields[6], fields[7], fields[8], fields[9], fields[10]},
        quota.get());
    ASSERT_TRUE(made.has_value());
    const auto allocated = quota.used();
    ASSERT_GT(allocated, 0u);
    retained = *made;
    for (auto &field : fields)
      field.clear();
    ModelArtifactProfile copied = *made;
    auto moved = std::move(copied);
    EXPECT_TRUE(copied.empty());
    EXPECT_EQ(moved.description(), kProfile);
    EXPECT_EQ(retained.description(), kProfile);
    *made = std::move(moved);
    EXPECT_TRUE(moved.empty());
    EXPECT_EQ(quota.used(), allocated);
    EXPECT_EQ(quota.peak(), allocated);
  }
  EXPECT_TRUE(retained.matches(kProfile));
  EXPECT_GT(quota.used(), 0u);
  retained = {};
  EXPECT_EQ(quota.used(), 0u);
}

TEST(ModelArtifactProfile, EmptyIsFreeAndPartialOrNonCanonicalIsRejected) {
  hyle::mem::QuotaAllocator denied(hyle::mem::system_allocator(), 0);
  auto empty = ModelArtifactProfile::create({}, denied.get());
  ASSERT_TRUE(empty.has_value());
  EXPECT_TRUE(empty->empty());
  EXPECT_EQ(denied.peak(), 0u);
  EXPECT_FALSE(empty->matches({}));
  EXPECT_FALSE(empty->matches(kProfile));

  auto partial = kProfile;
  partial.artifact_revision = {};
  EXPECT_FALSE(ModelArtifactProfile::create(partial).has_value());
  auto uppercase = kProfile;
  uppercase.manifest_digest =
      "A068044D8571635D8A86F1E96A43DAD104EFC7089169065F735601E350008155";
  EXPECT_FALSE(ModelArtifactProfile::create(uppercase).has_value());
  EXPECT_FALSE(ModelArtifactProfile::create(kProfile, {}).has_value());
}

TEST(ModelArtifactProfile, ExactIdentitySeparatesModelAndArtifactVariants) {
  auto profile = ModelArtifactProfile::create(kProfile);
  auto equal = ModelArtifactProfile::create(kProfile);
  ASSERT_TRUE(profile.has_value());
  ASSERT_TRUE(equal.has_value());
  EXPECT_EQ(*profile, *equal);
  EXPECT_NE(profile->description().model_repository.data(),
            equal->description().model_repository.data());

  auto model_revision = kProfile;
  model_revision.model_revision = "other-model-revision";
  auto artifact_revision = kProfile;
  artifact_revision.artifact_revision = "other-artifact-revision";
  auto variant = kProfile;
  variant.variant = "UD-Q4_K_XL";
  auto format = kProfile;
  format.format_family = "gguf";
  for (const auto &other :
       {model_revision, artifact_revision, variant, format}) {
    auto different = ModelArtifactProfile::create(other);
    ASSERT_TRUE(different.has_value());
    EXPECT_NE(*profile, *different);
    EXPECT_FALSE(profile->matches(other));
  }
}

TEST(ModelArtifactProfile, OomDoesNotCreatePartialOwner) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 0);
  auto denied = ModelArtifactProfile::create(kProfile, quota.get());
  ASSERT_FALSE(denied.has_value());
  EXPECT_TRUE(denied.error().IsResourceExhausted());
  EXPECT_EQ(quota.used(), 0u);
  EXPECT_EQ(quota.peak(), 0u);
}

TEST(ModelArtifactProfile, IndependentHandlesSupportConcurrentReads) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator());
  {
    auto profile = ModelArtifactProfile::create(kProfile, quota.get());
    ASSERT_TRUE(profile.has_value());
    const auto allocated = quota.used();
    std::array<std::thread, 4> workers;
    for (auto &worker : workers)
      worker = std::thread([held = *profile] {
        for (int iteration = 0; iteration < 128; ++iteration) {
          auto copy = held;
          EXPECT_TRUE(copy.matches(kProfile));
        }
      });
    for (auto &worker : workers)
      worker.join();
    EXPECT_EQ(quota.used(), allocated);
    EXPECT_EQ(quota.peak(), allocated);
  }
  EXPECT_EQ(quota.used(), 0u);
}

TEST(WorkloadContext, ProfilePhaseAndRegionAreIndependentColdFacts) {
  auto profile = ModelArtifactProfile::create(kProfile);
  ASSERT_TRUE(profile.has_value());
  WorkloadContext prefill{*profile, WorkloadPhase::kPrefill, {1, 7, 1}};
  auto decode = prefill;
  decode.phase = WorkloadPhase::kDecode;
  EXPECT_TRUE(prefill.valid());
  EXPECT_TRUE(decode.valid());
  EXPECT_NE(prefill, decode);
  auto other_region = prefill;
  other_region.region.version = 2;
  EXPECT_NE(prefill, other_region);
  other_region.region.version = 0;
  EXPECT_FALSE(other_region.valid());
  other_region.region = RegionSemanticId{0, 0, 1};
  EXPECT_FALSE(other_region.valid());
  decode.phase = static_cast<WorkloadPhase>(255);
  EXPECT_FALSE(decode.valid());
  EXPECT_TRUE(WorkloadContext{}.valid());
  EXPECT_EQ(WorkloadContext{}, WorkloadContext{});
}

} // namespace
