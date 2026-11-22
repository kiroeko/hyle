#include "hyle/compute/apple_memory.h"
#include "hyle/compute/detail/apple_plan.h"
#include "hyle/compute/detail/storage_owner_access.h"
#include "../../src/compute_route_apple/apple_bfloat_projection.h"

#include <array>
#include <cstdint>
#include <optional>
#include <utility>

#include <gtest/gtest.h>

namespace {
namespace a = hyle::compute::apple;
namespace c = hyle::compute;
namespace m = hyle::mem;
namespace mx = mlx::core;

TEST(AppleMemoryLifetime, BufferSurvivesWrapperDestructionAndMoveReplacement) {
  auto context = a::Context::create().value();
  m::Buffer first, replaced, empty;
  const std::array<float, 4> expected{1.25F, -2.F, 3.5F, 8.F};
  void *first_handle = nullptr;
  {
    auto original = a::MemoryResource::create(context).value();
    auto destination = a::MemoryResource::create(context).value();
    first = m::Buffer::create(original.ref(), sizeof(expected), alignof(float)).value();
    replaced = m::Buffer::create(destination.ref(), sizeof(expected), alignof(float)).value();
    empty = m::Buffer::create(original.ref(), 0, alignof(float)).value();
    ASSERT_TRUE(first.view().write(std::as_bytes(std::span(expected))).ok());
    ASSERT_TRUE(replaced.view().write(std::as_bytes(std::span(expected))).ok());
    first_handle = first.native_handle();
    destination = std::move(original);
    EXPECT_FALSE(original.valid());
    EXPECT_EQ(first.native_handle(), first_handle);
    auto moved = std::move(destination);
    EXPECT_TRUE(moved.valid());
    EXPECT_FALSE(destination.valid());
  }
  std::array<float, 4> actual{};
  ASSERT_TRUE(first.view().read(std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, expected);
  ASSERT_TRUE(replaced.view().read(std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(first.native_handle(), first_handle);
  EXPECT_TRUE(first.reset().ok());
  EXPECT_TRUE(replaced.reset().ok());
  EXPECT_TRUE(empty.valid());
  EXPECT_TRUE(empty.reset().ok());
}

TEST(AppleMemoryLifetime, TypedArrayAndImportedLeaseRetainBackingWithoutWrapper) {
  auto context = a::Context::create().value();
  const auto description = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}).value();
  const std::array<float, 4> expected{.25F, -4.F, 19.F, 2.F};
  std::optional<mx::array> native;
  c::ReadLease retained;
  c::ResourceBudget budget;
  m::MemoryResourceRef resource;
  const void *handle = nullptr;
  {
    auto memory = a::MemoryResource::create(context).value();
    resource = memory.ref();
    const std::array limits{c::ResourceBudgetLimit{resource, 4096},
        c::ResourceBudgetLimit{m::system_resource(), 1 << 20}};
    budget = c::ResourceBudget::create({.limits = limits, .allow_unattributed = true}).value();
    native.emplace(memory.allocate(description, budget).value());
    auto host = m::Buffer::create(m::system_resource(), sizeof(expected), alignof(float)).value();
    ASSERT_TRUE(host.view().write(std::as_bytes(std::span(expected))).ok());
    ASSERT_TRUE(memory.copy_from_host(*native, c::ConstTensorView::create(description, host.view()).value()).ok());
    handle = memory.storage_info(*native)->buffer;
    retained = memory.import_readonly(*native, description).value();
    EXPECT_EQ(budget.usage(resource)->live_bytes, sizeof(expected));
  }
  EXPECT_EQ(native->buffer().ptr(), handle);
  std::array<float, 4> actual{};
  ASSERT_TRUE(retained.view()->read(std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, expected);
  native.reset();
  EXPECT_EQ(budget.usage(resource)->live_bytes, sizeof(expected));
  ASSERT_TRUE(retained.view()->read(std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, expected);
  retained = {};
  // Reservation release uses the original budget row, not the dead wrapper.
  EXPECT_EQ(budget.usage(resource)->live_bytes, 0u);
  EXPECT_EQ(budget.unattributed_usage().unknown_private_owners, 0u);
}

TEST(AppleMemoryLifetime, PackedBFloatProjectionSharesOnlyItsOriginalBudgetAndBacking) {
  auto context = a::Context::create().value();
  const auto &runtime = a::detail::ContextAccess::runtime(context);
  auto owner = c::detail::StorageOwnerAccess::adopt_native(
      runtime, a::MemoryResource::create(context).value()).value();
  const auto resource = owner.ref();
  const auto description = c::TensorDesc::dense(c::DataType::kBFloat16,
      std::array<std::int64_t, 2>{4, 64}).value();
  std::array<std::uint16_t, 256> values{};
  values.fill(0x3f80);
  values[1] = 0x8000; // Preserve the sign of zero.
  values[2] = 0xbf80;
  values[64] = 0x0080; // This block cannot use the short exponent range.
  auto buffer = m::Buffer::create(resource, sizeof(values), 256).value();
  ASSERT_TRUE(buffer.view().write(std::as_bytes(std::span(values))).ok());
  auto source = owner.adopt_buffer(std::move(buffer)).value();
  const auto original = source.tensor(description).value();
  constexpr auto packed_bytes = 4 * 97;
  const auto make_budget = [&](std::size_t limit) {
    const std::array limits{c::ResourceBudgetLimit{resource, limit},
        c::ResourceBudgetLimit{m::system_resource(), 1 << 20}};
    return c::ResourceBudget::create({.limits = limits, .allow_unattributed = true}).value();
  };
  auto budget = make_budget(packed_bytes);
  auto first = a::detail::prepare_bfloat_projection(runtime, original, owner, source, budget).value();
  ASSERT_TRUE(first);
  auto second = a::detail::prepare_bfloat_projection(runtime, original, owner, source, budget).value();
  EXPECT_EQ(first, second);
  EXPECT_EQ(budget.usage(resource)->live_bytes, packed_bytes);
  const auto *exponents = first->packed[1].data<std::uint8_t>();
  EXPECT_EQ(exponents[0], 127u);
  EXPECT_EQ(exponents[1], 255u);
  EXPECT_EQ(first->packed[0].data<std::uint32_t>()[0] & 0xffffffu, 0x808000u);

  auto smaller = make_budget(packed_bytes - 1);
  EXPECT_FALSE(a::detail::prepare_bfloat_projection(runtime, original, owner, source, smaller).value());
  EXPECT_EQ(smaller.usage(resource)->live_bytes, 0u);
  first.reset();
  EXPECT_EQ(budget.usage(resource)->live_bytes, packed_bytes);
  second.reset();
  EXPECT_EQ(budget.usage(resource)->live_bytes, 0u);
  EXPECT_EQ(budget.unattributed_usage().unknown_private_owners, 0u);
}
} // namespace
