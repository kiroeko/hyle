#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "../../engine/inference/paged_layout_scratch.h"

namespace {
namespace c = hyle::compute;
namespace i = hyle::inference;

// IDs describe metadata only; the scratch never receives page permissions.
struct Page final {
  std::uint64_t id;
  std::uint64_t page() const noexcept { return id; }
};
c::BatchSchema schema() {
  const auto plain = c::CacheStorage::plain(c::DataType::kBFloat16).value();
  return c::BatchSchema::create({.query_type = c::DataType::kBFloat16, .output_type = c::DataType::kBFloat16,
      .key_storage = plain, .value_storage = plain, .max_requests = 3, .max_query_rows = 8, .max_kv_rows = 32,
      .max_pages = 8, .max_page_refs = 8, .query_heads = 1, .kv_heads = 1,
      .qk_head_dim = 64, .value_head_dim = 64, .page_size = 4}).value();
}
using Integers = std::array<std::vector<std::int32_t>, 4>;
Integers integers(c::HostBatchLayout layout) {
  return {std::vector<std::int32_t>(layout.query_indptr.begin(), layout.query_indptr.end()),
      std::vector<std::int32_t>(layout.kv_indptr.begin(), layout.kv_indptr.end()),
      std::vector<std::int32_t>(layout.page_indices.begin(), layout.page_indices.end()),
      std::vector<std::int32_t>(layout.last_page_lengths.begin(), layout.last_page_lengths.end())};
}
auto backing(c::HostBatchLayout layout) {
  return std::array{layout.query_indptr.data(), layout.kv_indptr.data(), layout.page_indices.data(),
      layout.last_page_lengths.data()};
}

TEST(InferencePagedLayout, MixedRowsSharedPagesAndEmptyHistoryUseActualSchemaVersions) {
  auto actual = schema(); i::PagedLayoutScratch scratch; scratch.prepare(3, 8);
  const std::array first{Page{5}, Page{1}};
  const std::array second{Page{1}};
  ASSERT_TRUE(scratch.append(2, 6, 4, std::span<const Page>(first)).ok());
  ASSERT_TRUE(scratch.append(1, 4, 4, std::span<const Page>(second)).ok());
  ASSERT_TRUE(scratch.append(0, 0, 4, std::span<const Page>{}).ok());
  const Integers expected{{{0, 2, 3, 3}, {0, 2, 3, 3}, {5, 1, 1}, {2, 4, 0}}};
  EXPECT_EQ(integers(scratch.layout()), expected);
  auto version = actual.version(scratch.layout()); ASSERT_TRUE(version) << version.error().message();
  EXPECT_EQ(integers(version->host()), expected);
  EXPECT_EQ(version->facts().requests, 3u); EXPECT_EQ(version->facts().query_rows, 3u);
  EXPECT_EQ(version->facts().kv_rows, 10u); EXPECT_EQ(version->facts().page_refs, 3u);
  auto member = scratch.member_version(actual, 1); ASSERT_TRUE(member) << member.error().message();
  EXPECT_EQ(integers(member->host()), (Integers{{{0, 1}, {0, 1}, {1}, {4}}}));
  EXPECT_EQ(member->facts().kv_rows, 4u); EXPECT_TRUE(actual.owns(*member));
  auto empty = scratch.member_version(actual, 2); ASSERT_TRUE(empty) << empty.error().message();
  EXPECT_EQ(integers(empty->host()), (Integers{{{0, 0}, {0, 0}, {}, {0}}}));
  EXPECT_EQ(empty->facts().requests, 1u); EXPECT_EQ(empty->facts().kv_rows, 0u);
  const auto absent = scratch.member_version(actual, 3); ASSERT_FALSE(absent); EXPECT_TRUE(absent.error().IsOutOfRange());
}

TEST(InferencePagedLayout, CopiedSchemaReusesScratchWithoutChangingPriorImmutableVersion) {
  auto actual = schema(), alias = actual;
  i::PagedLayoutScratch scratch; scratch.prepare(2, 4);
  const std::array first{Page{6}, Page{2}};
  const std::array second{Page{7}};
  const std::array third{Page{0}, Page{2}};
  ASSERT_TRUE(scratch.append(1, 5, 4, std::span<const Page>(first)).ok());
  auto old = alias.version(scratch.layout()).value();
  const auto old_values = integers(old.host()); const auto old_generation = old.generation();
  scratch.reset();
  ASSERT_TRUE(scratch.append(3, 4, 4, std::span<const Page>(second)).ok());
  ASSERT_TRUE(scratch.append(2, 6, 4, std::span<const Page>(third)).ok());
  auto next = actual.version(scratch.layout()).value();
  EXPECT_TRUE(alias.owns(next)); EXPECT_TRUE(actual.owns(old)); EXPECT_NE(next, old);
  EXPECT_EQ(next.generation(), old_generation + 1); EXPECT_EQ(next.facts().query_rows, 5u);
  EXPECT_EQ(integers(next.host()), (Integers{{{0, 3, 5}, {0, 1, 3}, {7, 0, 2}, {4, 2}}}));
  scratch.reset(); actual = {};
  EXPECT_EQ(integers(old.host()), old_values); EXPECT_EQ(old.generation(), old_generation);
  EXPECT_TRUE(alias.owns(old)); EXPECT_TRUE(alias.owns(next));
}

TEST(InferencePagedLayout, RejectionsPreserveScratchAndDelegatePhysicalAndKvLegalityToSchema) {
  auto actual = schema(); i::PagedLayoutScratch scratch; scratch.prepare(2, 3);
  const std::array first{Page{0}}, short_pages{Page{1}};
  const std::array too_many{Page{1}, Page{2}, Page{3}};
  const auto maximum = static_cast<std::size_t>((std::numeric_limits<std::int32_t>::max)());
  const std::array wide_id{Page{std::uint64_t(maximum) + 1}}, foreign_id{Page{77}};
  ASSERT_TRUE(scratch.append(1, 2, 4, std::span<const Page>(first)).ok());
  auto old = actual.version(scratch.layout()).value(); const auto original = integers(old.host());
  const auto pointers = backing(scratch.layout());
  const auto unchanged = [&] {
    EXPECT_EQ(integers(scratch.layout()), original); EXPECT_EQ(backing(scratch.layout()), pointers);
    EXPECT_EQ(integers(old.host()), original);
  };
  EXPECT_THROW(scratch.prepare(maximum + 1, 3), std::out_of_range); unchanged();
  EXPECT_THROW(scratch.prepare(2, maximum + 1), std::out_of_range); unchanged();
  EXPECT_TRUE(scratch.append(1, 5, 4, std::span<const Page>(short_pages)).IsInvalidArgument()); unchanged();
  EXPECT_TRUE(scratch.append(1, 1, 0, std::span<const Page>(short_pages)).IsInvalidArgument()); unchanged();
  EXPECT_TRUE(scratch.append(maximum, 1, 4, std::span<const Page>(short_pages)).IsOutOfRange()); unchanged();
  EXPECT_TRUE(scratch.append(0, 1, 4, std::span<const Page>(wide_id)).IsOutOfRange()); unchanged();
  EXPECT_TRUE(scratch.append(1, 12, 4, std::span<const Page>(too_many)).IsOutOfRange()); unchanged();
  // A representable physical ID is not a permission or a schema range proof.
  ASSERT_TRUE(scratch.append(0, 1, 4, std::span<const Page>(foreign_id)).ok());
  auto foreign = actual.version(scratch.layout()); ASSERT_FALSE(foreign); EXPECT_TRUE(foreign.error().IsOutOfRange());
  const auto full = integers(scratch.layout());
  EXPECT_TRUE(scratch.append(0, 0, 4, std::span<const Page>{}).IsOutOfRange());
  EXPECT_EQ(integers(scratch.layout()), full); EXPECT_EQ(integers(old.host()), original);
  scratch.reset();
  ASSERT_TRUE(scratch.append(1, 0, 4, std::span<const Page>{}).ok());
  auto no_kv = scratch.member_version(actual, 0); ASSERT_FALSE(no_kv); EXPECT_TRUE(no_kv.error().IsInvalidArgument());
  EXPECT_EQ(integers(old.host()), original);
}

TEST(InferencePagedLayout, ColdPreparationRetainsLargestBoundsAndHotIntegerBackingDoesNotGrow) {
  i::PagedLayoutScratch scratch; scratch.reset();
  EXPECT_TRUE(scratch.layout().query_indptr.empty()); EXPECT_EQ(scratch.max_page_refs(), 0u);
  EXPECT_TRUE(scratch.append(0, 0, 4, std::span<const Page>{}).IsFailedPrecondition());
  scratch.prepare(2, 3);
  const std::array first{Page{5}, Page{1}};
  const std::array second{Page{1}};
  ASSERT_TRUE(scratch.append(2, 6, 4, std::span<const Page>(first)).ok());
  ASSERT_TRUE(scratch.append(1, 4, 4, std::span<const Page>(second)).ok());
  const auto pointers = backing(scratch.layout());
  // Only scratch operations are hot here. BatchSchema::version deliberately
  // allocates its independent immutable snapshot and is outside this loop.
  for (unsigned repeat = 0; repeat != 64; ++repeat) {
    scratch.reset();
    if (repeat % 2) {
      ASSERT_TRUE(scratch.append(0, 0, 4, std::span<const Page>{}).ok());
      ASSERT_TRUE(scratch.append(2, 6, 4, std::span<const Page>(first)).ok());
    } else {
      ASSERT_TRUE(scratch.append(2, 6, 4, std::span<const Page>(first)).ok());
      ASSERT_TRUE(scratch.append(1, 4, 4, std::span<const Page>(second)).ok());
    }
    EXPECT_EQ(backing(scratch.layout()), pointers);
  }
  scratch.prepare(1, 1); EXPECT_EQ(scratch.max_page_refs(), 3u);
  ASSERT_TRUE(scratch.append(2, 6, 4, std::span<const Page>(first)).ok());
  ASSERT_TRUE(scratch.append(1, 4, 4, std::span<const Page>(second)).ok());
  EXPECT_EQ(backing(scratch.layout()), pointers);
  scratch.prepare(3, 5); scratch.prepare(2, 4); EXPECT_EQ(scratch.max_page_refs(), 5u);
  const std::array three{Page{2}, Page{3}, Page{4}};
  ASSERT_TRUE(scratch.append(1, 9, 4, std::span<const Page>(three)).ok());
  ASSERT_TRUE(scratch.append(1, 4, 4, std::span<const Page>(second)).ok());
  ASSERT_TRUE(scratch.append(1, 4, 4, std::span<const Page>(second)).ok());
  auto actual = schema(); auto grown = actual.version(scratch.layout()); ASSERT_TRUE(grown) << grown.error().message();
  EXPECT_EQ(grown->facts().requests, 3u); EXPECT_EQ(grown->facts().page_refs, 5u); EXPECT_EQ(grown->facts().kv_rows, 17u);
}
} // namespace
