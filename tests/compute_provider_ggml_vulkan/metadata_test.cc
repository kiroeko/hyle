#include "metadata_arena.h"

#include <array>
#include <limits>
#include <utility>

#include <gtest/gtest.h>

#include "ggml.h"
#include "hyle/mem/quota_allocator.h"

namespace {
using hyle::compute::ggml_vulkan::detail::MetadataArena;

TEST(GgmlVulkanMetadata, RejectsInvalidAndExhaustedBacking) {
  EXPECT_FALSE(MetadataArena::create(1024, {}));
  EXPECT_FALSE(MetadataArena::create(0, hyle::mem::system_allocator()));
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 1024);
  auto exhausted = MetadataArena::create(2048, quota.get());
  ASSERT_FALSE(exhausted);
  EXPECT_EQ(exhausted.error().code(), hyle::StatusCode::ResourceExhausted);
  EXPECT_EQ(quota.used(), 0u);
  auto oversized = MetadataArena::create(
      (std::numeric_limits<std::size_t>::max)(), quota.get());
  ASSERT_FALSE(oversized);
  EXPECT_EQ(quota.used(), 0u);
}

TEST(GgmlVulkanMetadata,
     MoveAndIndependentContextsRetireTheirExactAllocations) {
  constexpr std::size_t capacity = 16 * 1024;
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 2 * capacity);
  {
    auto first = MetadataArena::create(capacity, quota.get());
    auto second = MetadataArena::create(capacity, quota.get());
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_EQ(quota.used(), 2 * capacity);
    auto *x = ggml_new_tensor_2d(first->get(), GGML_TYPE_F32, 2048, 17);
    auto *y = ggml_new_tensor_2d(second->get(), GGML_TYPE_F32, 2048, 128);
    ASSERT_NE(x, nullptr);
    ASSERT_NE(y, nullptr);
    EXPECT_EQ(x->data, nullptr);
    EXPECT_EQ(y->data, nullptr);
    auto moved = std::move(*first);
    EXPECT_EQ(first->get(), nullptr);
    EXPECT_EQ(first->capacity(), 0u);
    *second = std::move(moved);
    EXPECT_EQ(quota.used(), capacity);
    EXPECT_EQ(ggml_get_first_tensor(second->get()), x);
    EXPECT_EQ(moved.get(), nullptr);
  }
  EXPECT_EQ(quota.used(), 0u);
}

TEST(GgmlVulkanMetadata,
     PatchedContextRejectsOverflowAndMisalignmentWithoutAbort) {
  EXPECT_EQ(
      ggml_init({(std::numeric_limits<std::size_t>::max)(), nullptr, true}),
      nullptr);
  alignas(GGML_MEM_ALIGN) std::array<std::byte, 256> storage{};
  EXPECT_EQ(ggml_init({128, storage.data() + 1, true}), nullptr);
  auto *context = ggml_init({storage.size(), storage.data(), true});
  ASSERT_NE(context, nullptr);
  ggml_free(context);
  storage[0] =
      std::byte{0x7b}; // ggml_free must leave caller-owned storage live.
  EXPECT_EQ(storage[0], std::byte{0x7b});
}
} // namespace
