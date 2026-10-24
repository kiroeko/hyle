#include <gtest/gtest.h>

#include "hyle/graph/apple.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/result.h"

#include <mlx/ops.h>

#include <array>
#include <atomic>
#include <memory>

namespace {
namespace c = hyle::compute;
namespace a = c::apple;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace mx = mlx::core;

struct Affine final {
  static constexpr c::OperationId operation_id{0x48594c4541505053ULL, 0x435553544f4d0001ULL};
  static constexpr auto operation_name = "apple-configured-source-affine";
  c::OperationTraits operation_traits() const noexcept {
    return {.inputs = {1, 1}, .outputs = 1};
  }
  hyle::Status infer_output_descriptions(std::span<const c::TensorDesc> inputs,
      std::span<c::TensorDesc> outputs) const noexcept {
    if (inputs.size() != 1 || outputs.size() != 1)
      return hyle::Status::InvalidArgument("configured affine arity");
    outputs[0] = inputs[0];
    return hyle::Status::Ok();
  }
};
const auto description = c::TensorDesc::dense(c::DataType::kFloat32,
    std::array<std::int64_t, 1>{4}).value();

c::ReadLease upload(const a::MemoryResource &memory, const std::array<float, 4> &values) {
  auto array = memory.allocate(description).value();
  auto host = m::Buffer::create(m::system_resource(), sizeof(values), alignof(float)).value();
  EXPECT_TRUE(host.view().write(std::as_bytes(std::span(values))).ok());
  EXPECT_TRUE(memory.copy_from_host(array,
      c::ConstTensorView::create(description, host.view()).value()).ok());
  return memory.import_readonly(array, description).value();
}
void expect(const c::ReadLease &lease, const std::array<float, 4> &wanted) {
  std::array<float, 4> actual{};
  ASSERT_TRUE(lease.view()->read(std::as_writable_bytes(std::span(actual))).ok());
  EXPECT_EQ(actual, wanted);
}

TEST(AppleConfiguredSource, KeepsConfigurationOwnerDistinctOnTheActualRuntime) {
  auto context = a::Context::create().value();
  const auto plain = g::apple::execution_source(context).value();
  const auto again = g::apple::execution_source(context).value();
  EXPECT_TRUE(plain.same_owner(again));
  std::weak_ptr<int> observed;
  {
    auto payload = std::make_shared<int>(3);
    observed = payload;
    g::apple::CustomLowering custom = [payload](const auto &, auto, mx::Stream)
        -> hyle::Result<std::vector<mx::array>> {
      (void)payload;
      return std::unexpected(hyle::Status::Unimplemented("unused owned custom configuration"));
    };
    auto configured = g::apple::execution_source(context, {.custom = custom}).value();
    const auto copy = configured;
    const auto independent = g::apple::execution_source(context, {.custom = std::move(custom)}).value();
    EXPECT_TRUE(configured.same_owner(copy));
    EXPECT_FALSE(configured.same_owner(plain));
    EXPECT_FALSE(configured.same_owner(independent));
    EXPECT_TRUE(configured.same_device(plain));
    EXPECT_TRUE(configured.same_device(independent));
    payload.reset();
    EXPECT_FALSE(observed.expired());
  }
  EXPECT_TRUE(observed.expired());
}

TEST(AppleConfiguredSource, LowersColdAndReplaysTheOriginalInvocation) {
  auto context = a::Context::create().value();
  RecordProperty("device", context.environment()->device_name);
  auto memory = a::MemoryResource::create(context).value();
  auto builder = g::GraphBuilder::create().value();
  const auto input = builder.input(description).value();
  const auto affine = builder.add(c::Operation::custom(Affine{}), input).value();
  const auto output = builder.add(c::Unary{c::UnaryKind::kNeg}, affine).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph = std::move(builder).finish().value();
  const std::array domains{g::StorageDomain{memory.ref()}};
  const std::array plain{g::ExecutionTarget{g::apple::execution_source(context).value(), g::StorageDomainId{0}}};
  EXPECT_FALSE(g::compile(graph, {domains, plain}));

  std::atomic<unsigned> calls{0};
  auto source = g::apple::execution_source(context, {.custom = [&calls](
      const c::Operation &operation, std::span<const mx::array> inputs,
      mx::Stream stream) -> hyle::Result<std::vector<mx::array>> {
    if (!operation.custom_as<Affine>())
      return std::unexpected(hyle::Status::Unimplemented("unknown configured custom schema"));
    ++calls;
    return std::vector{mx::add(mx::multiply(inputs[0], mx::array(2.0F), stream), mx::array(1.0F), stream)};
  }}).value();
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
  auto compiled = g::compile(graph, {domains, targets});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->executable.region_count(), 1U);
  EXPECT_EQ(compiled->executable.required_queue_count(), 0U);
  EXPECT_EQ(calls.load(), 1U);
  auto invocation = compiled->executable.create_invocation({.resources = g::ResourceOptions{}}).value();
  c::ReadLease old;
  for (unsigned iteration = 1; iteration <= 2; ++iteration) {
    const std::array<float, 4> values{float(iteration), -2.0F, 0.5F, 4.0F};
    auto owned = upload(memory, values);
    ASSERT_TRUE(invocation.bind(input, description, owned).ok());
    ASSERT_TRUE(invocation.prepare_replay({}).ok());
    {
      auto completed = invocation.submit();
      ASSERT_TRUE(completed) << completed.error().message();
      ASSERT_TRUE(completed->wait().ok());
    }
    auto result = invocation.result(output).value();
    auto lease = result.storage_lease().value();
    expect(lease, std::array<float, 4>{-2.0F*values[0]-1.0F, 3.0F, -2.0F, -9.0F});
    EXPECT_FALSE(invocation.bind(input, description, owned).ok());
    if (iteration == 1) old = std::move(lease);
  }
  EXPECT_EQ(calls.load(), 1U); // No callback during replay/rebind/submit.
  ASSERT_TRUE(context.close().ok());
  expect(old, std::array<float, 4>{-3.0F, 3.0F, -2.0F, -9.0F});
}

TEST(AppleConfiguredSource, PreservesACustomUnimplementedErrorWithoutRouteFallback) {
  auto context = a::Context::create().value();
  auto memory = a::MemoryResource::create(context).value();
  auto builder = g::GraphBuilder::create().value();
  const auto input = builder.input(description).value();
  const auto output = builder.add(c::Operation::custom(Affine{}), input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  unsigned calls = 0;
  const auto source = g::apple::execution_source(context, {.custom = [&calls](
      const auto &, auto, mx::Stream) -> hyle::Result<std::vector<mx::array>> {
    ++calls;
    return std::unexpected(hyle::Status::Unimplemented("application custom deliberately failed"));
  }}).value();
  const std::array domains{g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
  const auto rejected = g::compile(graph, {domains, targets});
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsUnimplemented());
  EXPECT_EQ(rejected.error().message(), "application custom deliberately failed");
  EXPECT_EQ(calls, 1U);
}
} // namespace
