// Reuse the existing consumer Provider/owner fixture. The validation target
// runs only FoundationEffects and the declared common effect/alias cases.
#include "region_provider_test.cc"

#if defined(HYLE_COMPUTE_VULKAN)
#include "hyle/graph/cpu_vulkan.h"
#include "hyle/sched/blocking.h"
#elif defined(HYLE_COMPUTE_APPLE)
#include "hyle/graph/cpu_apple.h"
#endif
#include "hyle/sched/scheduler.h"

namespace {
#if defined(HYLE_COMPUTE_VULKAN)
namespace bridge = hyle::graph::cpu_vulkan;
namespace native = hyle::compute::vulkan;
#elif defined(HYLE_COMPUTE_APPLE)
namespace bridge = hyle::graph::cpu_apple;
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
  bridge::Environment environment{.cpu=&cpu, .vulkan=&gpu, .device=&device, .staging=&staging};
#else
  auto device = native::MemoryResource::create(gpu).value();
  bridge::Environment environment{.cpu=&cpu, .apple=&gpu, .native=&device};
#endif
  hyle::sched::Scheduler scheduler(1, hyle::mem::system_allocator(), 32);
#if defined(HYLE_COMPUTE_VULKAN)
  hyle::sched::BlockingPool completion_waiters(hyle::mem::system_allocator(),1);
#endif
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
      hyle::graph::NodePlacement{write,bridge::kCpuTarget,hyle::graph::PlacementStrength::kRequired},
      hyle::graph::NodePlacement{read,bridge::kCpuTarget,hyle::graph::PlacementStrength::kRequired}};
  const std::array bindings{EffectResourceBinding{resource,storage.footprint()}};
  bridge::CompileOptions options{.node_placements=placements,.effect_bindings=bindings};
  bridge::CompileReport facts;
  auto program = bridge::compile(graph,environment,options,&facts);
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_EQ(facts.fragments.size(),3);
  // The effect footprint must reject tensor aliasing even when the tensor is
  // only consumed by the GPU fragment between two CPU effect fragments.
#if defined(HYLE_COMPUTE_VULKAN)
  auto slot = program->create_invocation(scheduler,queue,completion_waiters).value();
#else
  auto slot = program->create_invocation(scheduler,queue).value();
#endif
  auto alias = ConstTensorView::create(desc,storage.buffer.view()).value();
  EXPECT_TRUE(slot.bind(input,alias).IsInvalidArgument());
  auto host = Buffer::create(hyle::mem::system_resource(),4,64).value();
  const std::array<float,1> value{3};
  ASSERT_TRUE(host.view().write(std::as_bytes(std::span(value))).ok());
  auto tensor = ConstTensorView::create(desc,host.view()).value();
  ASSERT_TRUE(slot.bind(input,tensor).ok());
  ASSERT_TRUE(slot.prepare_replay().ok());
  {
    auto result=slot.run();
    ASSERT_TRUE(result) << result.error().message();
    EXPECT_EQ(*storage.value(),41);
    EXPECT_EQ(probe.last_resource_read,41);
    EXPECT_EQ(probe.effect_submit_count,2);
    EXPECT_FALSE(slot.reset().ok());
#if defined(HYLE_COMPUTE_VULKAN)
    std::array<float,1> actual;
    ASSERT_TRUE(gpu.download(*(*result)->tensor(output),std::as_writable_bytes(std::span(actual))).ok());
    EXPECT_EQ(actual[0],-3);
#else
    auto array=(*result)->array(output).value();
    EXPECT_EQ(array.data<float>()[0],-3);
#endif
  }
  ASSERT_TRUE(slot.reset().ok());
  auto missing=bridge::compile(graph,environment,{.node_placements=placements});
  EXPECT_FALSE(missing);
  const auto attempts=probe.effect_submit_count;
  auto gpu_effect=bridge::compile(graph,environment,{.effect_bindings=bindings});
  ASSERT_FALSE(gpu_effect);
  EXPECT_TRUE(gpu_effect.error().IsUnimplemented());
  EXPECT_EQ(probe.effect_submit_count,attempts);

  GraphBuilder only=GraphBuilder::create().value();
  const auto only_resource=only.effect_resource().value();
  const auto only_write=add_resource_touch(only,only_resource,storage,ResourceAccess::kWrite,17);
  const auto only_graph=std::move(only).finish().value();
  const std::array only_placement{hyle::graph::NodePlacement{only_write,bridge::kCpuTarget,hyle::graph::PlacementStrength::kRequired}};
  const std::array only_binding{EffectResourceBinding{only_resource,storage.footprint()}};
  auto only_program=bridge::compile(only_graph,environment,{.node_placements=only_placement,.effect_bindings=only_binding});
  ASSERT_TRUE(only_program)<<only_program.error().message();
#if defined(HYLE_COMPUTE_VULKAN)
  auto only_slot=only_program->create_invocation(scheduler,queue,completion_waiters).value();
#else
  auto only_slot=only_program->create_invocation(scheduler,queue).value();
#endif
  ASSERT_TRUE(only_slot.prepare_replay().ok());
  {auto result=only_slot.run();ASSERT_TRUE(result)<<result.error().message();EXPECT_EQ(*storage.value(),17);}
  EXPECT_TRUE(only_slot.close().ok());
}
#endif
} // namespace
