#include "../compute_models/nvfp4_reference_fixture.h"
#include "hyle/compute/apple_memory.h"
#include "hyle/graph/apple.h"

namespace {
namespace f = nvfp4_reference_test;
namespace a = hyle::compute::apple;
template<class Run> void run_apple(Run run) {
  auto context = f::take(a::Context::create(), "Apple context");
  ::testing::Test::RecordProperty("device", context.environment()->device_name);
  auto memory = f::take(a::MemoryResource::create(context), "Apple shared memory");
  const std::array domains{f::g::StorageDomain{memory.ref()}};
  const std::array targets{f::g::ExecutionTarget{
      f::take(f::g::apple::execution_source(context), "Apple execution source"), f::g::StorageDomainId{0}}};
  run(memory.ref(), std::span<const f::g::QueueBinding>{},
      [&](const f::g::Graph &graph, f::g::CompileOptions options) {
        return f::g::compile(graph, {domains,targets}, options);
      },
      [](f::m::BufferView destination, std::span<const std::byte> source) { return destination.write(source); },
      [](f::m::ConstBufferView source, std::span<std::byte> destination) { return source.read(destination); },
      [](f::g::Invocation &invocation, std::span<const f::g::QueueBinding> bindings) { return invocation.prepare_replay(bindings); });
}
}

TEST(AppleNvfp4ReferenceGraph, OriginalPackedBanksAndBfloatBoundariesSurviveSlotsAndUpdates) {
  run_apple([](auto resource, auto queues, auto compile, auto write, auto read, auto prepare) {
    f::run_lifetimes(resource, queues, compile, write, read, prepare);
  });
}
TEST(AppleNvfp4ReferenceGraph, RuntimeDomainFailuresRecoverWithoutChangingTheReferenceTree) {
  run_apple([](auto resource, auto queues, auto compile, auto write, auto read, auto prepare) {
    f::run_runtime_domain(resource, queues, compile, write, read, prepare);
  });
}
TEST(AppleNvfp4ReferenceGraph, ByteCarrierDoesNotTurnLogicalFp8IntoIntegerArithmetic) {
  auto context = f::take(a::Context::create(), "Apple context");
  auto memory = f::take(a::MemoryResource::create(context), "Apple memory");
  const auto description = f::dense(f::Type::kFloat8E4M3FN, {8});
  auto spec = a::input_spec(description);
  ASSERT_TRUE(spec) << spec.error().message();
  EXPECT_EQ(spec->dtype, mlx::core::uint8);
  EXPECT_EQ(spec->logical_description, description);
  auto builder = f::take(f::g::GraphBuilder::create(), "FP8 graph builder");
  const auto input = f::take(builder.input(description), "logical FP8 input");
  const auto output = f::take(builder.add(f::c::Cast{f::Type::kFloat32}, input), "logical FP8 Cast");
  f::check(builder.publish(output), "publish logical FP8 Cast");
  const auto tie_description = f::dense(f::Type::kFloat32,{4});
  const auto tie_input = f::take(builder.input(tie_description), "FP8 midpoint input");
  const auto tie_output = f::take(builder.add(f::c::Cast{f::Type::kFloat8E4M3FN},tie_input),
      "numerical FP8 midpoint Cast");
  f::check(builder.publish(tie_output), "publish FP8 midpoint codes");
  const auto graph = f::take(std::move(builder).finish(), "finish logical FP8 graph");
  const std::array domains{f::g::StorageDomain{memory.ref()}};
  const std::array targets{f::g::ExecutionTarget{
      f::take(f::g::apple::execution_source(context), "Apple source"), f::g::StorageDomainId{0}}};
  auto compiled = f::take(f::g::compile(graph, {domains,targets}), "logical FP8 Cast compile");
  auto invocation = f::take(compiled.executable.create_invocation(
      {.resources=f::g::ResourceOptions{}}), "logical FP8 invocation");
  auto storage = f::take(f::m::Buffer::create(memory.ref(),8,256), "FP8 byte storage");
  const std::array<std::uint8_t,8> codes{0x00,0x80,0x38,0xb8,0x01,0x7e,0x7f,0xff};
  f::check(storage.view().write(std::as_bytes(std::span(codes))), "write original FP8 codes");
  f::check(invocation.bind(input,f::take(f::c::ConstTensorView::create(
      description,storage.view()), "logical FP8 carrier")), "bind FP8 carrier");
  auto tie_storage=f::take(f::m::Buffer::create(memory.ref(),16,256), "FP8 midpoint storage");
  const std::array<float,4> ties{1.0625F,1.1875F,-1.0625F,-1.1875F};
  f::check(tie_storage.view().write(std::as_bytes(std::span(ties))), "write FP8 midpoints");
  f::check(invocation.bind(tie_input,f::take(f::c::ConstTensorView::create(
      tie_description,tie_storage.view()), "FP8 midpoint carrier")), "bind FP8 midpoints");
  f::check(invocation.prepare_replay({}), "prepare numerical FP8 Cast");
  auto done=f::take(invocation.submit(), "submit numerical FP8 Cast");
  f::check(done.wait(), "numerical FP8 Cast completion");
  auto result=f::take(invocation.result(output), "numerical FP8 result");
  auto tensor=f::take(result.tensor(), "numerical FP8 result storage");
  std::array<std::uint32_t,8> decoded{};
  f::check(tensor.storage().read(std::as_writable_bytes(std::span(decoded))), "read FP8 decoded bits");
  const std::array<std::uint32_t,6> expected{
      0x00000000u,0x80000000u,0x3f800000u,0xbf800000u,0x3b000000u,0x43e00000u};
  for (std::size_t i=0;i<expected.size();++i) EXPECT_EQ(decoded[i],expected[i]);
  for (std::size_t i=expected.size();i<decoded.size();++i)
    EXPECT_GT(decoded[i]&0x7fffffffu,0x7f800000u); // NaN class, not integer127/255.
  auto tie_result=f::take(invocation.result(tie_output), "FP8 midpoint result");
  auto tie_tensor=f::take(tie_result.tensor(), "FP8 midpoint result storage");
  std::array<std::uint8_t,4> tie_codes{};
  f::check(tie_tensor.storage().read(std::as_writable_bytes(std::span(tie_codes))),
      "read RNE FP8 midpoint codes");
  EXPECT_EQ(tie_codes,(std::array<std::uint8_t,4>{0x38,0x3a,0xb8,0xba}));

  auto unsupported=f::take(f::g::GraphBuilder::create(), "FP8 arithmetic graph");
  const auto bytes=f::take(unsupported.input(description), "FP8 arithmetic input");
  auto arithmetic=unsupported.add(f::c::Binary{f::c::BinaryKind::kAdd},bytes,bytes);
  if (arithmetic) {
    f::check(unsupported.publish(*arithmetic), "publish unsupported FP8 arithmetic");
    auto arithmetic_graph=f::take(std::move(unsupported).finish(), "finish FP8 arithmetic");
    EXPECT_FALSE(f::g::compile(arithmetic_graph,{domains,targets}))
        << "logical FP8 bytes must not become ordinary unsigned integer arithmetic";
  } else {
    EXPECT_TRUE(arithmetic.error().IsInvalidArgument() || arithmetic.error().IsUnimplemented());
  }
}
