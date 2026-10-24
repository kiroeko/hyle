#include "hyle/compute/apple_memory.h"

#include <array>
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
} // namespace
