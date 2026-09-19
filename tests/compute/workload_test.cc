#include "hyle/compute/workload.h"

#include <array>
#include <limits>
#include <string>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

#include "hyle/mem/quota_allocator.h"

namespace {
using hyle::compute::ModelDescription;
using hyle::compute::ModelIdentity;
using hyle::compute::RegionSemanticId;
using hyle::compute::WorkloadContext;
using hyle::compute::WorkloadPhase;

constexpr ModelDescription kModel{"qwen3_5", "snapshot-a", "configuration-a"};

TEST(ModelIdentity, OwnsStringsAndSharesBackingThroughMovesAndLastRelease) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator());
  ModelIdentity retained;
  {
    std::string family(kModel.family), revision(kModel.revision),
        configuration(kModel.configuration);
    auto made = ModelIdentity::create({family, revision, configuration}, quota.get());
    ASSERT_TRUE(made.has_value());
    const auto allocated = quota.used();
    ASSERT_GT(allocated, 0u);
    retained = *made;
    family.assign(family.size(), 'x');
    revision.clear();
    configuration.clear();
    ModelIdentity copied = *made;
    auto moved = std::move(copied);
    EXPECT_TRUE(copied.empty());
    EXPECT_EQ(moved.description(), kModel);
    EXPECT_EQ(retained.description(), kModel);
    *made = std::move(moved);
    EXPECT_TRUE(moved.empty());
    EXPECT_EQ(quota.used(), allocated);
    EXPECT_EQ(quota.peak(), allocated);
  }
  EXPECT_TRUE(retained.matches(kModel));
  EXPECT_GT(quota.used(), 0u);
  retained = {};
  EXPECT_EQ(quota.used(), 0u);
}

TEST(ModelIdentity, EmptyPartialAndUnknownDoNotMatchCompleteRules) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 0);
  auto empty = ModelIdentity::create({}, quota.get());
  ASSERT_TRUE(empty.has_value());
  EXPECT_TRUE(empty->empty());
  EXPECT_EQ(quota.peak(), 0u);
  EXPECT_FALSE(empty->matches({}));
  EXPECT_FALSE(empty->matches(kModel));
  auto partial = ModelIdentity::create({kModel.family, {}, {}});
  ASSERT_TRUE(partial.has_value());
  EXPECT_FALSE(partial->description().complete());
  EXPECT_FALSE(partial->matches(partial->description()));
  EXPECT_FALSE(partial->matches(kModel));
  auto unknown = ModelIdentity::create({"unknown-family", kModel.revision,
                                       kModel.configuration});
  ASSERT_TRUE(unknown.has_value());
  EXPECT_FALSE(unknown->matches(kModel));
  EXPECT_FALSE(ModelIdentity::create({{}, "orphan-revision", {}}).has_value());
  EXPECT_FALSE(ModelIdentity::create(kModel, {}).has_value());
}

TEST(ModelIdentity, ExactFieldsDistinguishVersionsConfigurationsAndBoundaries) {
  auto model = ModelIdentity::create(kModel);
  auto equal = ModelIdentity::create(kModel);
  ASSERT_TRUE(model.has_value());
  ASSERT_TRUE(equal.has_value());
  EXPECT_EQ(*model, *equal);
  EXPECT_NE(model->description().family.data(), equal->description().family.data());
  for (const auto other : std::array{
           ModelDescription{"other", kModel.revision, kModel.configuration},
           ModelDescription{kModel.family, "snapshot-b", kModel.configuration},
           ModelDescription{kModel.family, kModel.revision, "configuration-b"}}) {
    auto different = ModelIdentity::create(other);
    ASSERT_TRUE(different.has_value());
    EXPECT_NE(*model, *different);
    EXPECT_FALSE(model->matches(other));
  }
  EXPECT_NE(*ModelIdentity::create({"ab", "c", "d"}),
            *ModelIdentity::create({"a", "bc", "d"}));
}

TEST(ModelIdentity, OomAndSizeOverflowDoNotCreatePartialOwners) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 0);
  auto denied = ModelIdentity::create(kModel, quota.get());
  ASSERT_FALSE(denied.has_value());
  EXPECT_TRUE(denied.error().IsResourceExhausted());
  EXPECT_EQ(quota.used(), 0u);
  const ModelDescription excessive{
      std::string_view("x", (std::numeric_limits<std::size_t>::max)()), {}, {}};
  auto overflow = ModelIdentity::create(excessive, quota.get());
  ASSERT_FALSE(overflow.has_value());
  EXPECT_TRUE(overflow.error().IsResourceExhausted());
  EXPECT_EQ(quota.peak(), 0u);
}

TEST(ModelIdentity, IndependentHandlesSupportConcurrentCopiesAndReads) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator());
  {
    auto model = ModelIdentity::create(kModel, quota.get());
    ASSERT_TRUE(model.has_value());
    const auto allocated = quota.used();
    std::array<std::thread, 4> workers;
    for (auto &worker : workers)
      worker = std::thread([held = *model] {
        for (int iteration = 0; iteration < 128; ++iteration) {
          auto copy = held;
          EXPECT_TRUE(copy.matches(kModel));
        }
      });
    for (auto &worker : workers)
      worker.join();
    EXPECT_EQ(quota.used(), allocated);
    EXPECT_EQ(quota.peak(), allocated);
  }
  EXPECT_EQ(quota.used(), 0u);
}

TEST(WorkloadContext, PhaseAndVersionedRegionAreIndependentQualificationFacts) {
  auto model = ModelIdentity::create(kModel);
  ASSERT_TRUE(model.has_value());
  WorkloadContext prefill{*model, WorkloadPhase::kPrefill, {1, 7, 1}};
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
