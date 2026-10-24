#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "hyle/compute/vulkan_memory.h"
#include "hyle/compute/detail/generated_result.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/vulkan.h"
#include "../../src/compute_route_vulkan/vulkan_internal.h"
#include "../../src/compute_route_vulkan/vulkan_memory_internal.h"

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace g = hyle::graph;
namespace d = v::detail;

// Intercept only the original Context's native PFNs. The accepted prefix and
// managed RMS are real device work; only the rejected native suffix and an
// optional unknown drain result are injected. Never fabricate native success.
struct NativeFailure final {
  d::Device &device;
  d::DeviceFunctions original;
  unsigned submissions = 0, drains = 0;
  bool unknown_drain = false, unknown_wait = false, reject_suffix = true;
  static inline NativeFailure *active = nullptr;
  explicit NativeFailure(v::Context &context)
      : device(*d::Access::runtime(context)->device), original(device.fn) {
    active = this;
    device.fn.QueueSubmit = submit;
    device.fn.QueueWaitIdle = drain;
    device.fn.WaitForFences = wait;
  }
  ~NativeFailure() { device.fn = original; active = nullptr; }
  static VKAPI_ATTR VkResult VKAPI_CALL submit(VkQueue queue,
      std::uint32_t count, const VkSubmitInfo *info, VkFence fence) {
    if (++active->submissions == 2 && active->reject_suffix) return VK_ERROR_OUT_OF_HOST_MEMORY;
    return active->original.QueueSubmit(queue, count, info, fence);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL drain(VkQueue queue) {
    ++active->drains;
    if (std::exchange(active->unknown_drain, false))
      return VK_ERROR_OUT_OF_HOST_MEMORY;
    return active->original.QueueWaitIdle(queue);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL wait(VkDevice device, std::uint32_t count,
      const VkFence *fences, VkBool32 all, std::uint64_t timeout) {
    if (std::exchange(active->unknown_wait, false)) return VK_ERROR_OUT_OF_HOST_MEMORY;
    return active->original.WaitForFences(device, count, fences, all, timeout);
  }
};

void abandoned_prefix(bool unknown_first_drain) {
  auto context = v::Context::create().value();
  ::testing::Test::RecordProperty("vulkan_device", context.environment().device_name);
  ASSERT_FALSE(context.environment().software_device);
  auto memory = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  const auto description = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{17, 2048}).value();
  std::vector<float> data(17 * 2048);
  for (std::size_t i = 0; i < data.size(); ++i)
    data[i] = float(int(i % 31) - 15) * .1F;
  auto native_input = context.upload(description, std::as_bytes(std::span(data))).value();
  std::weak_ptr<const void> owner = native_input.native_owner();
  auto input = memory.import_readonly(native_input).value();
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(description).value();
  const auto first = builder.add(c::Unary{c::UnaryKind::kRelu}, x).value();
  const auto normalized = builder.add(c::RmsNorm{}, first).value();
  const auto output = builder.add(c::Unary{c::UnaryKind::kRelu}, normalized).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();
  const std::array domains{g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{
      g::vulkan::execution_source(context).value(), g::StorageDomainId{0}}};
  auto compiled = g::compile(graph, {domains, targets}).value();
  ASSERT_EQ(compiled.executable.region_count(), 1U);
  auto invocation = compiled.executable.create_invocation().value();
  ASSERT_TRUE(invocation.prepare_resources().ok());
  ASSERT_TRUE(invocation.bind(x, description, input).ok());
  ASSERT_TRUE(invocation.prepare_replay({}).ok());
  NativeFailure failure(context);
  // Even an assertion failure must explicitly retire accepted native work
  // before this test's callback/PFN interceptor and allocation owners die.
  struct Retire final {
    v::Context &context;
    ~Retire() { context.drain().IgnoreError(); }
  } retire{context};
  auto completion = invocation.submit();
  ASSERT_TRUE(completion) << completion.error().message();
  ASSERT_EQ(failure.submissions, 2U);
  ASSERT_TRUE(invocation.in_flight());
  ASSERT_TRUE(completion->request_cancel().ok());
  auto polled = completion->poll();
  ASSERT_TRUE(polled);
  EXPECT_EQ(*polled, c::CompletionPoll::kPending);
  completion = c::Completion{};
  EXPECT_TRUE(invocation.close().IsUnavailable());
  input.reset(); native_input = {};
  EXPECT_FALSE(owner.expired());
  EXPECT_TRUE(context.collect().IsResourceExhausted());
  EXPECT_EQ(failure.drains, 0U);
  EXPECT_TRUE(invocation.in_flight());
  if (unknown_first_drain) {
    failure.unknown_drain = true;
    EXPECT_TRUE(context.drain().IsResourceExhausted());
    EXPECT_EQ(failure.drains, 1U);
    EXPECT_TRUE(invocation.in_flight());
    EXPECT_NE(d::Access::runtime(context)->pending, nullptr);
    EXPECT_FALSE(owner.expired());
  }
  // Status preserves the rejected suffix; the typed Graph guard, not that
  // error, proves the first real original-queue drain completed retirement.
  EXPECT_TRUE(context.drain().IsResourceExhausted());
  EXPECT_FALSE(invocation.in_flight());
  EXPECT_EQ(d::Access::runtime(context)->pending, nullptr);
  invocation = {};
  EXPECT_TRUE(owner.expired());

  // drain() must not revoke admission or poison this reusable Context merely
  // because one composed invocation failed. A fresh original Graph slot runs.
  auto fresh = compiled.executable.create_invocation().value();
  auto next = context.upload(description, std::as_bytes(std::span(data))).value();
  auto ready = memory.import_readonly(next).value();
  ASSERT_TRUE(fresh.prepare_resources().ok());
  ASSERT_TRUE(fresh.bind(x, description, ready).ok());
  ASSERT_TRUE(fresh.prepare_replay({}).ok());
  auto replay = fresh.submit();
  ASSERT_TRUE(replay) << replay.error().message();
  EXPECT_TRUE(replay->wait().ok());
  EXPECT_FALSE(fresh.in_flight());
  EXPECT_TRUE(context.drain().ok());
}

TEST(VulkanGraphRetirement, NonclosingDrainRetiresAbandonedPartialPrefix) {
  abandoned_prefix(false);
}
TEST(VulkanGraphRetirement, UnknownDrainRetainsGuardsUntilExplicitRetry) {
  abandoned_prefix(true);
}

TEST(VulkanGraphRetirement, OriginalAllocationControllersOutliveTheirWrappers) {
  auto context = v::Context::create().value();
  const auto description = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}).value();
  const std::array<float, 4> expected{1.25F, -2.5F, 3.75F, -4.0F};
  for (unsigned path = 0; path < 3; ++path) {
    SCOPED_TRACE(path);
    auto memory = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
    const auto resource = memory.ref();
    c::ReadLease ready;
    v::Tensor typed;
    if (path == 0) {
      auto backing = hyle::mem::Buffer::create(resource, sizeof(expected)).value();
      ASSERT_TRUE(backing.view().write(std::as_bytes(std::span(expected))).ok());
      ready = c::ReadLease::adopt(std::move(backing)).value();
    } else if (path == 1) {
      const std::array limits{c::ResourceBudgetLimit{resource, 64},
          c::ResourceBudgetLimit{hyle::mem::system_resource(), SIZE_MAX}};
      auto budget = c::ResourceBudget::create({.limits=limits,.allow_unattributed=true}).value();
      typed = memory.allocate(description, {.resource_budget=budget}).value();
      auto backing = d::MemoryAccess::export_storage(d::Access::runtime(context),
          resource, typed, hyle::mem::system_allocator()).value();
      ASSERT_TRUE(backing.view().write(std::as_bytes(std::span(expected))).ok());
      // Drop the export token before the wrapper: only the original typed
      // native Buffer/budget owner now keeps its controller, including views.
      ASSERT_TRUE(backing.reset().ok());
      typed = typed.view(c::View::reshape(c::TensorShape::create(
          std::array<std::int64_t, 2>{2, 2}).value()).value()).value();
    } else {
      auto original = context.upload(description, std::as_bytes(std::span(expected))).value();
      ready = memory.import_readonly(original).value();
    }
    memory = {};
    std::array<float, 4> actual{};
    if (typed) {
      auto backing = d::MemoryAccess::export_storage(d::Access::runtime(context),
          resource, typed, hyle::mem::system_allocator()).value();
      ASSERT_TRUE(backing.view().read(std::as_writable_bytes(std::span(actual))).ok());
    } else {
      ASSERT_TRUE(ready.view()->read(std::as_writable_bytes(std::span(actual))).ok());
    }
    EXPECT_EQ(actual, expected);
    ready.reset(); typed = {};
  }
  auto empty_resource = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  auto empty = hyle::mem::Buffer::create(empty_resource.ref(), 0).value();
  empty_resource = {};
  EXPECT_FALSE(empty.resource().resource().name().empty());
  EXPECT_TRUE(empty.reset().ok());
}

TEST(VulkanGraphRetirement, BlockingCallbackReentryIsRejectedWithoutInventingRetirement) {
  auto context = v::Context::create().value();
  const std::array<float, 4> values{1, 2, 3, 4};
  auto input = context.upload(c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}).value(),
      std::as_bytes(std::span(values))).value();
  auto program = v::prepare(context, c::Unary{c::UnaryKind::kRelu}, input.description()).value();
  auto invocation = program.create_invocation().value();
  ASSERT_TRUE(invocation.bind(input).ok());
  struct Reentry final {
    v::Context &context;
    hyle::Status drain, close;
    bool released = false;
    c::detail::CompletionRetention retention;
  } owner{context, {}, {}, false, {}};
  owner.retention.own = {&owner,
      +[](void *, const hyle::Status &) noexcept {},
      +[](void *opaque) noexcept {
        auto &state = *static_cast<Reentry *>(opaque);
        state.drain = state.context.drain();
        state.close = state.context.close();
        state.released = true;
      }};
  auto completion = invocation.submit().value();
  auto transfer = c::detail::ResultAccess::take_storage(std::move(completion));
  ASSERT_TRUE(c::detail::CompletionAccess::attach_retention(transfer.completion, owner.retention).ok());
  transfer.completion = {};
  ASSERT_TRUE(context.drain().ok());
  EXPECT_TRUE(owner.released);
  EXPECT_TRUE(owner.drain.IsFailedPrecondition());
  EXPECT_TRUE(owner.close.IsFailedPrecondition());
  EXPECT_EQ(d::Access::runtime(context)->pending, nullptr);
  EXPECT_TRUE(context.close().ok());
}

TEST(VulkanGraphRetirement, SynchronousIoPreservesObservationFailureAfterRealRetirement) {
  auto context = v::Context::create().value();
  auto memory = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  const auto description = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}).value();
  auto native = memory.allocate(description).value();
  std::weak_ptr<const void> owner = native.native_owner();
  auto backing = d::MemoryAccess::export_storage(d::Access::runtime(context),
      memory.ref(), native, hyle::mem::system_allocator()).value();
  NativeFailure failure(context);
  failure.reject_suffix = false;
  failure.unknown_wait = true;
  failure.unknown_drain = true;
  const std::array<float, 4> data{1, 2, 3, 4};
  const auto written = backing.view().write(std::as_bytes(std::span(data)));
  EXPECT_TRUE(written.IsResourceExhausted());
  // This synchronous API's original Completion::wait continues to an actual
  // fence proof after the exceptional wait/drain observation. It must keep
  // the first diagnosis, and never return while that original Run is pending.
  EXPECT_EQ(failure.submissions, 1U);
  EXPECT_EQ(failure.drains, 1U);
  EXPECT_EQ(d::Access::runtime(context)->pending, nullptr);
  memory = {}; native = {};
  ASSERT_TRUE(backing.reset().ok());
  EXPECT_TRUE(owner.expired()); // Cleared recording has no controller/backing cycle.
  EXPECT_TRUE(context.drain().ok());
  EXPECT_EQ(d::Access::runtime(context)->pending, nullptr);
  EXPECT_TRUE(owner.expired()); // No failed DeviceIO recording/controller cycle.
}
} // namespace
