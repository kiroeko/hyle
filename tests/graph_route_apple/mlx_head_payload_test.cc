#include "../compute_models/mlx_head_payload_fixture.h"
#include "hyle/compute/apple_memory.h"
#include "hyle/graph/apple.h"

TEST(AppleMlxHeadPayload, PackedLayer0RolesKeepOriginalBytesAndHeadOrder) {
  namespace f = mlx_head_payload_test;
  namespace a = hyle::compute::apple;
  auto context = a::Context::create().value();
  ::testing::Test::RecordProperty("device", context.environment()->device_name);
  auto memory = a::MemoryResource::create(context).value();
  const std::array domains{f::g::StorageDomain{memory.ref()}};
  const std::array targets{f::g::ExecutionTarget{
      f::g::apple::execution_source(context).value(), f::g::StorageDomainId{0}}};
  f::run(memory.ref(), std::span<const f::g::QueueBinding>{},
      [&](const f::g::Graph &graph, f::g::CompileOptions options) {
        return f::g::compile(graph, {domains, targets}, options);
      },
      [](f::m::BufferView destination, std::span<const std::byte> source) {
        return destination.write(source);
      },
      [](f::m::ConstBufferView source, std::span<std::byte> destination) {
        return source.read(destination);
      },
      [](f::g::Invocation &invocation,
         std::span<const f::g::QueueBinding> bindings) {
        return invocation.prepare_replay(bindings);
      });
}

TEST(AppleMlxHeadPayload, StridedReadonlyAliasesKeepExactBorrowedAndOwnedPermission) {
  namespace f = mlx_head_payload_test;
  namespace a = hyle::compute::apple;
  auto context = a::Context::create().value();
  auto memory = a::MemoryResource::create(context).value();
  const auto input_description = f::c::TensorDesc::dense(
      f::Type::kFloat32, std::array<std::int64_t, 2>{2, 3}).value();
  auto builder = f::g::GraphBuilder::create().value();
  const auto x = builder.input(input_description).value();
  const auto negative = builder.add(f::c::Unary{f::c::UnaryKind::kNeg}, x).value();
  const auto gap = builder.add(f::g::ViewSpec::slice(1,
      f::g::Dimension::fixed(0), f::g::Dimension::fixed(2)).value(), x).value();
  const auto alias = builder.add(f::g::ViewSpec::permute(
      std::array<std::int32_t, 2>{1, 0}).value(), gap).value();
  ASSERT_TRUE(builder.publish(alias).ok());
  ASSERT_TRUE(builder.publish(negative).ok());
  auto graph = std::move(builder).finish().value();
  const auto output_description = graph.output_description(0).value();
  const std::array domains{f::g::StorageDomain{memory.ref()}};
  const std::array targets{f::g::ExecutionTarget{
      f::g::apple::execution_source(context).value(), f::g::StorageDomainId{0}}};
  auto compiled = f::g::compile(graph, {domains, targets}).value();
  std::array runs{
      compiled.executable.create_invocation().value(),
      compiled.executable.create_invocation({.resources = f::g::ResourceOptions{}}).value()};
  const std::array<float, 8> original{99, 98, 1, 2, 88, 3, 4, 77};
  auto storage = f::m::Buffer::create(memory.ref(), sizeof(original), 256).value();
  ASSERT_TRUE(storage.view().write(std::as_bytes(std::span(original))).ok());
  auto backing = f::c::ReadLease::adopt(std::move(storage)).value();
  auto exact = backing.subview(8, 24).value();
  const auto permission = exact.view().value();
  auto too_short = permission.subview(0, 20).value();
  EXPECT_FALSE(f::c::ConstTensorView::create(input_description, too_short));
  ASSERT_TRUE(runs[0].bind(x, exact.tensor(input_description).value()).ok());
  ASSERT_TRUE(runs[1].bind(x, input_description, exact).ok());
  for (auto &run : runs) {
    const auto status = run.prepare_replay({});
    ASSERT_TRUE(status.ok()) << status.message();
  }
  std::array<f::g::GraphOutput, 2> retained;
  const auto verify = [&](const f::g::GraphOutput &output) {
    auto tensor = output.tensor().value();
    EXPECT_EQ(tensor.description(), output_description);
    EXPECT_EQ(tensor.storage().native_handle(), permission.native_handle());
    EXPECT_EQ(tensor.storage().offset(), permission.offset());
    EXPECT_EQ(tensor.storage().size(), 20U);
    std::array<float, 5> physical{};
    ASSERT_TRUE(tensor.storage().read(std::as_writable_bytes(std::span(physical))).ok());
    EXPECT_EQ(physical, (std::array<float, 5>{1, 2, 88, 3, 4}));
    EXPECT_EQ(tensor.description().byte_strides()[0], 4);
    EXPECT_EQ(tensor.description().byte_strides()[1], 12);
    EXPECT_FALSE(tensor.storage().subview(0, 21));
  };
  for (unsigned slot = 0; slot < runs.size(); ++slot) {
    auto done = runs[slot].submit().value();
    ASSERT_TRUE(done.wait().ok());
    retained[slot] = runs[slot].result(alias).value();
    verify(retained[slot]);
    EXPECT_FALSE(runs[slot].submit());
  }
  EXPECT_FALSE(retained[0].storage_lease());
  auto escaped = retained[1].storage_lease().value();
  EXPECT_EQ(escaped.view()->offset(), permission.offset());
  EXPECT_EQ(escaped.view()->size(), 20U);
  EXPECT_FALSE(escaped.subview(0, 21));
  runs = {}; compiled = {}; graph = {};
  verify(retained[0]); // Raw borrowed owner is still alive here.
  retained[0] = {}; exact = {}; backing = {};
  verify(retained[1]); // The original owned subrange survives its caller.
  ASSERT_TRUE(context.close().ok());
  verify(retained[1]);
  RecordProperty("successful_submissions", 2);
}
