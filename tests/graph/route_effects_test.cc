// Reuse the existing consumer Provider/owner fixture. The validation target
// runs only FoundationEffects and the declared common effect/alias cases.
#include "region_provider_test.cc"

#include "native_effect_fixture.h"

namespace {
#if defined(HYLE_COMPUTE_VULKAN)
namespace native = hyle::compute::vulkan;
#elif defined(HYLE_COMPUTE_APPLE)
namespace native = hyle::compute::apple;
#endif

#if defined(HYLE_COMPUTE_VULKAN) || defined(HYLE_COMPUTE_APPLE)
TEST(FoundationEffects, CpuResourceWriteGpuMathCpuReadAndCrossFragmentAlias) {
  RegionProbe probe;
  Context cpu = make_context(probe);
  auto queue = cpu.create_queue().value();
  auto gpu = native::Context::create().value();
#if defined(HYLE_COMPUTE_VULKAN)
  auto device = native::MemoryResource::create(gpu, native::MemoryKind::kDevice).value();
  auto staging = native::MemoryResource::create(gpu, native::MemoryKind::kStaging).value();
  const auto staging_resource = staging.ref();
#else
  auto device = native::MemoryResource::create(gpu).value();
  const auto staging_resource = device.ref();
#endif
  hyle::sched::Scheduler scheduler(1, hyle::mem::system_allocator(), 32);
  hyle::sched::BlockingPool completion_waiters(hyle::mem::system_allocator(),1);
  struct Drain { hyle::sched::Scheduler &scheduler; ~Drain(){scheduler.drain();} } drain{scheduler};
  ResourceStorage storage{7};
  GraphBuilder builder = GraphBuilder::create().value();
  const auto resource = builder.effect_resource().value();
  const auto write = add_resource_touch(builder,resource,storage,ResourceAccess::kWrite,41);
  const auto desc = TensorDesc::dense(DataType::kFloat32,std::array<std::int64_t,1>{1}).value();
  const auto input = builder.input(desc).value();
  const auto middle = builder.add_node(Operation(Unary{UnaryKind::kNeg}),std::array{input}).value();
  const auto read = add_resource_touch(builder,resource,storage,ResourceAccess::kRead);
  ASSERT_TRUE(builder.depends_on(middle,write).ok());
  ASSERT_TRUE(builder.depends_on(read,middle).ok());
  const auto output = builder.result(middle).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  const std::array placements{
      hyle::graph::NodePlacement{write,effect_test::kCpuTarget,hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{middle,effect_test::kNativeTarget,hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{read,effect_test::kCpuTarget,hyle::graph::PlacementStrength::kRequired}};
  const std::array bindings{EffectResourceBinding{resource,storage.footprint()}};
  hyle::graph::CompileOptions options{.effect_bindings=bindings,.node_placements=placements};
  auto program = effect_test::compile(graph,cpu,gpu,device.ref(),staging_resource,options);
  ASSERT_TRUE(program) << program.error().message();
  std::size_t cpu_nodes = 0, native_nodes = 0;
  for (std::size_t i = 0; i < program->report.step_count(); ++i) {
    const auto step = program->report.step(i).value();
    if (step.kind == hyle::graph::CompileStepKind::kCopy) continue;
    if (step.target == effect_test::kCpuTarget) cpu_nodes += step.node_count;
    else native_nodes += step.node_count;
  }
  EXPECT_EQ(cpu_nodes,2u);
  EXPECT_EQ(native_nodes,1u);
  // The effect footprint must reject tensor aliasing even when the tensor is
  // only consumed by the GPU fragment between two CPU effect fragments.
  auto slot = program->executable.create_invocation().value();
  const auto queues = effect_test::queues(program->report,queue).value();
  auto alias = ConstTensorView::create(desc,storage.buffer.view()).value();
  EXPECT_TRUE(slot.bind(input,alias).IsInvalidArgument());
  auto host = Buffer::create(hyle::mem::system_resource(),4,64).value();
  const std::array<float,1> value{3};
  ASSERT_TRUE(host.view().write(std::as_bytes(std::span(value))).ok());
  auto tensor = ConstTensorView::create(desc,host.view()).value();
  ASSERT_TRUE(slot.bind(input,tensor).ok());
  ASSERT_TRUE(slot.prepare_replay(queues).ok());
  ASSERT_TRUE(hyle::graph::prepare_progress(slot,scheduler,completion_waiters).ok());
  {
    { auto completion=slot.submit(queues); ASSERT_TRUE(completion)<<completion.error().message();
      ASSERT_TRUE(completion->wait().ok()); }
    auto result=slot.result(output);
    ASSERT_TRUE(result)<<result.error().message();
    EXPECT_EQ(*storage.value(),41);
    EXPECT_EQ(probe.last_resource_read,41);
    EXPECT_EQ(probe.effect_submit_count,2);
    EXPECT_FALSE(slot.reset().ok());
    const auto actual=effect_test::read(*result,gpu,device,desc);
    ASSERT_TRUE(actual)<<actual.error().message();
    EXPECT_EQ(*actual,-3);
  }
  ASSERT_TRUE(slot.reset().ok());
  auto missing=effect_test::compile(graph,cpu,gpu,device.ref(),staging_resource,{.node_placements=placements});
  EXPECT_FALSE(missing);
  const auto attempts=probe.effect_submit_count;
  const std::array forbidden{hyle::graph::NodePlacement{write,effect_test::kNativeTarget,
      hyle::graph::PlacementStrength::kRequired}};
  auto gpu_effect=effect_test::compile(graph,cpu,gpu,device.ref(),staging_resource,
      {.effect_bindings=bindings,.node_placements=forbidden});
  ASSERT_FALSE(gpu_effect);
  EXPECT_TRUE(gpu_effect.error().IsUnimplemented());
  EXPECT_EQ(probe.effect_submit_count,attempts);

  GraphBuilder only=GraphBuilder::create().value();
  const auto only_resource=only.effect_resource().value();
  const auto only_write=add_resource_touch(only,only_resource,storage,ResourceAccess::kWrite,17);
  const auto only_graph=std::move(only).finish().value();
  const std::array only_placement{hyle::graph::NodePlacement{only_write,effect_test::kCpuTarget,hyle::graph::PlacementStrength::kRequired}};
  const std::array only_binding{EffectResourceBinding{only_resource,storage.footprint()}};
  auto only_program=effect_test::compile(only_graph,cpu,gpu,device.ref(),staging_resource,
      {.effect_bindings=only_binding,.node_placements=only_placement});
  ASSERT_TRUE(only_program)<<only_program.error().message();
  auto only_slot=only_program->executable.create_invocation().value();
  const auto only_queues=effect_test::queues(only_program->report,queue).value();
  ASSERT_TRUE(only_slot.prepare_replay(only_queues).ok());
  ASSERT_TRUE(hyle::graph::prepare_progress(only_slot,scheduler,completion_waiters).ok());
  {auto completion=only_slot.submit(only_queues);ASSERT_TRUE(completion)<<completion.error().message();
   ASSERT_TRUE(completion->wait().ok());EXPECT_EQ(*storage.value(),17);}
  EXPECT_TRUE(only_slot.close().ok());
}
#endif
} // namespace
