#include <gtest/gtest.h>

#include <array>
#include <bit>

#include "hyle/graph/vulkan.h"
#include "hyle/compute/vulkan_memory.h"
#include "../../src/graph/execution_source_internal.h"
#include "../../src/graph_route_vulkan/vulkan_source.h"

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace g = hyle::graph;
namespace gd = g::detail;

TEST(VulkanExecutionSource, ActualOwnerResourceAdmissionAndStableBacking) {
  auto context = v::Context::create().value();
  auto foreign = v::Context::create().value();
  auto source = g::vulkan::execution_source(context).value();
  const auto again = g::vulkan::execution_source(context).value();
  const auto other = g::vulkan::execution_source(foreign).value();
  EXPECT_TRUE(source.same_owner(again));
  EXPECT_TRUE(source.same_device(again));
  EXPECT_FALSE(source.same_owner(other));
  EXPECT_FALSE(source.same_device(other));
  const g::vulkan::CustomLowering custom = [](const c::Operation &,
      std::span<const c::TensorDesc>, const c::TensorDesc &)
      -> hyle::Result<g::vulkan::CustomDispatch> {
    return std::unexpected(hyle::Status::Unimplemented("unused cold custom configuration"));
  };
  const auto configured = g::vulkan::execution_source(context, {.custom = custom}).value();
  const auto configured_copy = configured;
  const auto independently_configured =
      g::vulkan::execution_source(context, {.custom = custom}).value();
  EXPECT_TRUE(configured.same_owner(configured_copy));
  EXPECT_FALSE(configured.same_owner(source));
  EXPECT_FALSE(configured.same_owner(independently_configured));
  EXPECT_TRUE(configured.same_device(source));
  EXPECT_TRUE(configured.same_device(independently_configured));
  const auto facts = source.device_properties().value();
  EXPECT_EQ(facts.name, context.environment().device_name);
  EXPECT_EQ(facts.backend, source.backend_id());
  EXPECT_FALSE(gd::ExecutionSourceAccess::compute(source));
  auto memory = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  auto staging = v::MemoryResource::create(context, v::MemoryKind::kStaging).value();
  auto wrong_memory = v::MemoryResource::create(foreign, v::MemoryKind::kDevice).value();
  EXPECT_TRUE(source.validate_resource(memory.ref()).ok());
  EXPECT_TRUE(source.validate_resource(staging.ref()).ok());
  EXPECT_FALSE(source.validate_resource(wrong_memory.ref()).ok());
  EXPECT_FALSE(source.validate_resource(hyle::mem::system_resource()).ok());
  static const unsigned capability = 7;
  const auto region_source = gd::GraphRegionSource::native(source, &capability).value();
  EXPECT_TRUE(region_source.is_native());
  EXPECT_EQ(region_source.shared_extension(), nullptr);
  EXPECT_FALSE(region_source.provider_instance_id().valid());
  struct CapabilityCall final {
    const void *owner;
    unsigned calls = 0;
  } native_call{gd::ExecutionSourceAccess::native_context(source).value()};
  const auto invoke_capability = [](void *owner, const void *table, void *opaque) noexcept {
    auto &call = *static_cast<CapabilityCall *>(opaque);
    if (owner != call.owner || table != &capability)
      return hyle::Status::InvalidArgument("native source lost its actual owner/capability");
    ++call.calls;
    return hyle::Status::Ok();
  };
  ASSERT_TRUE(region_source.call(invoke_capability, &native_call).ok());

  const auto description = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}).value();
  const std::array<float, 4> values{0.25f, -1.5f, 3.75f, -0.0f};
  const auto input = context.upload(description, std::as_bytes(std::span(values))).value();
  const auto program = v::prepare(context, c::Unary{c::UnaryKind::kNeg}, description).value();
  auto slot = program.create_invocation().value();
  EXPECT_FALSE(g::vulkan::detail::output_backing(other, slot, 0));
  // This handle is uninitialized backing, not a ready ReadLease. Do not read
  // or import it until the original native result reports successful retirement.
  const auto backing = g::vulkan::detail::output_backing(source, slot, 0).value();
  const auto native_buffer = backing.native_buffer();
  const auto native_offset = backing.native_offset();
  ASSERT_TRUE(g::vulkan::detail::seal_outputs(source, slot).ok());
  ASSERT_TRUE(slot.bind(input).ok());
  v::Tensor old_output;
  {
    auto submitted = slot.submit();
    ASSERT_TRUE(submitted) << submitted.error().message();
    auto result = submitted->wait();
    ASSERT_TRUE(result) << result.error().message();
    old_output = result->get()->tensors()[0];
    EXPECT_EQ(old_output.native_buffer(), native_buffer);
    EXPECT_EQ(old_output.native_offset(), native_offset);
    std::array<float, 4> observed{};
    ASSERT_TRUE(context.download(old_output, std::as_writable_bytes(std::span(observed))).ok());
    for (std::size_t i = 0; i < values.size(); ++i)
      EXPECT_EQ(std::bit_cast<std::uint32_t>(observed[i]),
          std::bit_cast<std::uint32_t>(-values[i]));
  }
  EXPECT_FALSE(slot.submit()); // Native Tensor owner still blocks overwrite.
  old_output = {};
  {
    auto submitted = slot.submit();
    ASSERT_TRUE(submitted) << submitted.error().message();
    ASSERT_TRUE(submitted->wait());
  }
  unsigned calls = 0;
  const auto invoke = [](void *state) noexcept {
    ++*static_cast<unsigned *>(state);
    return hyle::Status::Ok();
  };
  EXPECT_TRUE(gd::ExecutionSourceAccess::call(source, invoke, &calls).ok());
  ASSERT_TRUE(context.close().ok());
  EXPECT_FALSE(source.is_open());
  EXPECT_FALSE(gd::ExecutionSourceAccess::call(source, invoke, &calls).ok());
  EXPECT_FALSE(region_source.call(invoke_capability, &native_call).ok());
  EXPECT_EQ(native_call.calls, 1U);
  EXPECT_EQ(calls, 1U);
  EXPECT_FALSE(g::vulkan::execution_source(context));
  // Closing admission does not invalidate a retained backing owner.
  std::array<float, 4> observed{};
  EXPECT_TRUE(context.download(backing, std::as_writable_bytes(std::span(observed))).ok());
  EXPECT_TRUE(source.same_owner(again));
}
} // namespace
