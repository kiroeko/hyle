#include <gtest/gtest.h>

#include <array>
#include <bit>

#include "hyle/graph/vulkan.h"
#include "hyle/graph/executable.h"
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

TEST(VulkanExecutionSource, RetiredCrossDomainCopyUnbindsAndUsesNewBacking) {
  auto context = v::Context::create().value();
  auto device = v::MemoryResource::create(context, v::MemoryKind::kDevice).value();
  auto staging = v::MemoryResource::create(context, v::MemoryKind::kStaging).value();
  const auto source = g::vulkan::execution_source(context).value();
  const auto description = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{4}).value();
  auto builder = g::GraphBuilder::create().value();
  const auto input = builder.input(description).value();
  const auto output = builder.add(c::Unary{c::UnaryKind::kNeg}, input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  const std::array domains{g::StorageDomain{staging.ref()}, g::StorageDomain{device.ref()}};
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{1}}};
  const std::array external{g::ExternalResourceContract{input, g::StorageDomainId{0}}};
  auto compiled = g::compile(graph, {.storage_domains = domains, .targets = targets},
      {.external_resources = external});
  ASSERT_TRUE(compiled) << compiled.error().message();
  auto invocation = compiled->executable.create_invocation().value();
  ASSERT_TRUE(invocation.prepare_resources().ok());
  const std::array<std::array<float, 4>, 2> values{{
      {0.25f, -1.5f, 3.75f, -0.0f}, {2.5f, -8.0f, 0.0f, 9.0f}}};
  std::array<hyle::mem::Buffer, 2> backings;
  for (auto &backing : backings)
    backing = hyle::mem::Buffer::create(staging.ref(), sizeof(values[0]), 64).value();
  ASSERT_NE(backings[0].view().native_handle(), backings[1].view().native_handle());
  for (std::size_t episode = 0; episode < values.size(); ++episode) {
    SCOPED_TRACE(episode);
    ASSERT_TRUE(backings[episode].view().write(std::as_bytes(std::span(values[episode]))).ok());
    const auto binding = c::ConstTensorView::create(description, backings[episode].view()).value();
    ASSERT_TRUE(invocation.bind(input, binding).ok());
    ASSERT_TRUE(invocation.prepare_replay({}).ok());
    {
      auto done = invocation.submit();
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      auto result = invocation.result(output).value();
      auto native = device.binding(result.storage_lease().value(), description).value();
      std::array<float, 4> actual{};
      ASSERT_TRUE(context.download(native, std::as_writable_bytes(std::span(actual))).ok());
      for (std::size_t index = 0; index < actual.size(); ++index)
        EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[index]),
            std::bit_cast<std::uint32_t>(-values[episode][index]));
    }
    // This must release the native copy's actual VkBuffer pins. The second
    // episode reuses this slot with another backing, without reset or drain.
    const auto released = invocation.unbind(input);
    ASSERT_TRUE(released.ok()) << released.message();
  }
}
} // namespace
