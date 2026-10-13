#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <span>

#include "allocation_fault_fixture.h"
#include "hyle/compute/batch_paged_native.h"

namespace {
namespace c = hyle::compute;
c::BatchSchema schema(c::DataType type = c::DataType::kFloat16) {
  return c::BatchSchema::create({.storage = c::BatchStorage::kPaged, .data_type = type,
      .max_requests = 3, .max_query_rows = 8, .max_kv_rows = 96, .max_pages = 4,
      .max_page_refs = 8, .query_heads = 8, .kv_heads = 2, .head_dim = 64, .page_size = 16}).value();
}
c::BatchLayoutVersion layout(const c::BatchSchema &schema) {
  const std::array<std::int32_t, 3> q{0, 1, 4}, k{0, 2, 4};
  const std::array<std::int32_t, 4> pages{3, 1, 1, 0};
  const std::array<std::int32_t, 2> tails{7, 15};
  return schema.version({q, k, pages, tails}).value();
}
TEST(NativePagedMetadata, CompactOwnedPageOrderAndTailGenerationAreIndependentOfQueryHeads) {
  for (auto type : {c::DataType::kFloat16, c::DataType::kBFloat16}) {
    auto selected = schema(type);
    auto recipe = c::NativePagedBatchProgram::prepare(selected,
        {.max_host_metadata_bytes = 32, .scale = .125F, .causal = false}).value();
    EXPECT_EQ(recipe.descriptions().size(), 4u);
    EXPECT_EQ(recipe.descriptions()[3].data_type(), c::DataType::kInt32);
    EXPECT_EQ(recipe.descriptions()[3].required_bytes().value(), 32u);
    EXPECT_EQ(recipe.costs().index_bytes, 32u);
    EXPECT_TRUE(recipe.costs().native_private_bytes_unknown);
    EXPECT_EQ(recipe.scale(), .125F); EXPECT_FALSE(recipe.causal());
    std::array<std::int32_t, 3> q{0, 1, 4}, k{0, 2, 4};
    std::array<std::int32_t, 4> pages{3, 1, 1, 0};
    std::array<std::int32_t, 2> tails{7, 15};
    auto owned = selected.version({q, k, pages, tails}).value();
    auto version = recipe.version(owned).value();
    pages.fill(2); tails.fill(16); q[1] = 2;
    EXPECT_EQ(version.layout(), owned);
    EXPECT_TRUE(version.program().schema().owns(version.layout()));
    const auto ids = version.indices().value();
    std::array<std::int32_t, 8> actual;
    ASSERT_TRUE(ids.view()->read(std::as_writable_bytes(std::span(actual))).ok());
    EXPECT_EQ(actual, (std::array<std::int32_t, 8>{3,1,1,0,0,0,0,0}));
    EXPECT_EQ(version.layout().host().last_page_lengths[0], 7);
    auto next = recipe.version(layout(selected)).value();
    EXPECT_NE(next.layout().generation(), version.layout().generation());
    std::array<std::int32_t, 8> repeated;
    ASSERT_TRUE(next.indices()->view()->read(std::as_writable_bytes(std::span(repeated))).ok());
    EXPECT_EQ(repeated, actual);
  }
  auto capacity = schema().capacity();
  capacity.max_query_rows = 8192; capacity.query_heads = 16; capacity.kv_heads = 4;
  capacity.head_dim = 128;
  auto wider = c::BatchSchema::create(capacity).value();
  auto compact = c::NativePagedBatchProgram::prepare(wider, {.max_host_metadata_bytes = 32}).value();
  EXPECT_EQ(compact.costs().index_bytes, 32u); // Query/head capacity does not expand metadata.
  const std::array<std::int32_t, 2> q{0, 2}, k{0, 6};
  const std::array<std::int32_t, 6> repeated_pages{3,1,3,1,3,1};
  const std::array<std::int32_t, 1> tail{15};
  auto repeated_layout = wider.version({q,k,repeated_pages,tail}).value();
  EXPECT_GT(repeated_layout.facts().kv_rows, capacity.max_pages * capacity.page_size);
  auto repeated_version = compact.version(repeated_layout).value();
  std::array<std::int32_t, 8> actual;
  ASSERT_TRUE(repeated_version.indices()->view()->read(std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, (std::array<std::int32_t, 8>{3,1,3,1,3,1,0,0}));
}
TEST(NativePagedMetadata, ExportedSnapshotRetainsOwnersAndEmptyVersionHasNoPageReads) {
  c::ReadLease exported;
  {
    auto selected = schema();
    auto recipe = c::NativePagedBatchProgram::prepare(selected, {.max_host_metadata_bytes = 32}).value();
    auto version = recipe.version(layout(selected)).value();
    exported = version.indices().value();
    auto moved = std::move(version);
    EXPECT_FALSE(version.valid()); EXPECT_TRUE(moved.valid());
    auto copied = moved;
    EXPECT_EQ(copied.layout(), moved.layout());
  }
  std::array<std::int32_t, 8> actual;
  ASSERT_TRUE(exported.view()->read(std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, (std::array<std::int32_t, 8>{3,1,1,0,0,0,0,0}));
  auto selected = schema();
  auto recipe = c::NativePagedBatchProgram::prepare(selected, {.max_host_metadata_bytes = 32}).value();
  const std::array<std::int32_t, 1> empty{0};
  auto empty_layout = selected.version({empty, empty, {}, {}}).value();
  auto empty_version = recipe.version(empty_layout).value();
  ASSERT_TRUE(empty_version.indices()->view()->read(std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, (std::array<std::int32_t, 8>{}));
  EXPECT_EQ(empty_version.layout().facts().query_rows, 0u);
  EXPECT_FALSE(c::NativePagedBatchVersion{}.indices());
}
TEST(NativePagedMetadata, ForeignSchemaBudgetAndNonfiniteScaleRejectBeforePublishing) {
  auto selected = schema(), foreign = schema();
  auto recipe = c::NativePagedBatchProgram::prepare(selected, {.max_host_metadata_bytes = 32}).value();
  auto rejected = recipe.version(layout(foreign));
  ASSERT_FALSE(rejected); EXPECT_TRUE(rejected.error().IsInvalidArgument());
  auto small = c::NativePagedBatchProgram::prepare(selected, {.max_host_metadata_bytes = 31});
  ASSERT_FALSE(small); EXPECT_TRUE(small.error().IsResourceExhausted());
  auto invalid = c::NativePagedBatchProgram::prepare(selected,
      {.max_host_metadata_bytes = 32, .scale = std::numeric_limits<float>::infinity()});
  ASSERT_FALSE(invalid); EXPECT_TRUE(invalid.error().IsInvalidArgument());
  EXPECT_FALSE(c::NativePagedBatchProgram{}.version(layout(selected)));
}
TEST(NativePagedMetadata, EveryColdAllocationFailureReclaimsMetadataAndExportCanOutliveRecipe) {
  auto selected = schema(); auto owned = layout(selected);
  hyle_f5_fault_fixture::Allocator measured;
  {
    auto recipe = c::NativePagedBatchProgram::prepare(selected,
        {.max_host_metadata_bytes = 32, .allocator = measured.ref()}).value();
    auto version = recipe.version(owned).value();
    auto exported = version.indices().value();
    version = {}; recipe = {};
    EXPECT_GT(measured.live, 0u);
    exported.reset();
  }
  EXPECT_EQ(measured.live, 0u);
  const auto count = measured.attempts;
  ASSERT_GT(count, 0u);
  for (std::size_t fail = 0; fail < count; ++fail) {
    hyle_f5_fault_fixture::Allocator fault; fault.fail = fail;
    {
      auto recipe = c::NativePagedBatchProgram::prepare(selected,
          {.max_host_metadata_bytes = 32, .allocator = fault.ref()});
      if (!recipe) EXPECT_TRUE(recipe.error().IsResourceExhausted()) << fail;
      else {
        auto version = recipe->version(owned);
        ASSERT_FALSE(version) << fail;
        EXPECT_TRUE(version.error().IsResourceExhausted()) << fail;
      }
    }
    EXPECT_EQ(fault.live, 0u) << fail;
  }
}
} // namespace
