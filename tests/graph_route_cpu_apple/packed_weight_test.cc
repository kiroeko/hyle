#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"
#include "hyle/graph/cpu.h"
#include "hyle/graph/apple.h"
#include "hyle/graph/progress.h"
#include "hyle/graph/result.h"
#include "hyle/sched/blocking.h"
#include "hyle/sched/scheduler.h"

namespace {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace a=c::apple;
namespace mem=hyle::mem;

TEST(CpuApplePackedWeights, ResidentFp8BetweenCpuStages64StepsAndRetainedResult) {
  constexpr std::int64_t m=2,n=19,k=128;
  auto cpu=g::cpu::create_context().value();
  auto apple=a::Context::create().value();
  auto memory=a::MemoryResource::create(apple).value();
  hyle::sched::Scheduler scheduler(1,mem::system_allocator(),32);
  hyle::sched::BlockingPool workers(mem::system_allocator(),1);
  auto queue=cpu.create_queue().value();
  const auto row=c::TensorDesc::dense(c::DataType::kFloat32,std::array{m,k}).value();
  const auto wd=c::float8_weight_description(k,n,c::Float8ScaleMode::kFloat32Block128).value();
  auto input=mem::Buffer::create(mem::system_resource(),m*k*sizeof(float),64).value();
  auto weight=mem::Buffer::create(mem::system_resource(),wd.required_bytes().value(),64).value();
  std::vector<std::byte> packed(wd.required_bytes().value());
  for(int column=0;column<n;++column)for(int j=0;j<k;++j)
    packed[column*k+j]=std::byte(0x30+(column+j)%8);
  const float scale=.25F;std::memcpy(packed.data()+n*k,&scale,sizeof(scale));
  ASSERT_TRUE(weight.view().write(packed).ok());
  auto builder=g::GraphBuilder::create().value();
  const auto x=builder.input(row).value(),w=builder.constant(wd).value();
  const auto positive=builder.add(c::Unary{c::UnaryKind::kAbs},x).value();
  const auto product=builder.add(c::model_ops::float8_block_linear(),positive,w).value();
  const auto output=builder.add(c::Unary{c::UnaryKind::kNeg},product).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto graph=std::move(builder).finish().value();
  constexpr g::ExecutionTargetId native_target{0},cpu_target{1};
  constexpr g::StorageDomainId host_storage{0},native_storage{1};
  const std::array placement{
    g::NodePlacement{graph.node_id(0).value(),cpu_target,g::PlacementStrength::kRequired},
    g::NodePlacement{graph.node_id(1).value(),native_target,g::PlacementStrength::kRequired},
    g::NodePlacement{graph.node_id(2).value(),cpu_target,g::PlacementStrength::kRequired}};
  const std::array constants{g::ConstantBinding{w,c::ConstTensorView::create(wd,weight.view()).value(),g::ConstantPolicy::kMaterialize}};
  const std::array domains{g::StorageDomain{mem::system_resource()},g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{g::apple::execution_source(apple).value(),native_storage},
      g::ExecutionTarget{g::ExecutionSource::acquire(cpu).value(),host_storage}};
  const std::array external{g::ExternalResourceContract{x,host_storage},g::ExternalResourceContract{output,host_storage}};
  auto prepared=g::compile(graph,{.storage_domains=domains,.targets=targets},
      {.constants=constants,.node_placements=placement,.external_resources=external});
  ASSERT_TRUE(prepared)<<prepared.error().message();
  std::size_t cpu_nodes=0,native_nodes=0,copies=0,copy_bytes=0;
  for(std::size_t i=0;i<prepared->report.step_count();++i){
    const auto step=prepared->report.step(i).value();
    if(step.kind==g::CompileStepKind::kCopy){
      ++copies;copy_bytes+=step.copy_bytes;
      ASSERT_EQ(step.input_count,1u);ASSERT_EQ(step.output_count,1u);
      const auto from=prepared->report.physical_value(prepared->report.step_input(i,0).value()).value();
      const auto to=prepared->report.physical_value(prepared->report.step_output(i,0).value()).value();
      EXPECT_EQ(from.description,to.description);
      EXPECT_TRUE(from.description==row || from.description==graph.value_description(output).value());
      continue;
    }
    if(step.target==cpu_target)cpu_nodes+=step.node_count;else native_nodes+=step.node_count;
  }
  EXPECT_EQ(cpu_nodes,2u);EXPECT_EQ(native_nodes,1u);EXPECT_EQ(copies,2u);
  // Only the two dense CPU/native boundaries exist; no host repack graph or
  // extra materialization is hidden among the immutable plan steps.
  EXPECT_EQ(prepared->report.step_count(),5u);
  EXPECT_EQ(copy_bytes,(m*k+m*n)*sizeof(float));
  weight={}; // The resident constant must own the cold source independently.
  std::vector<g::QueueBinding> queues;
  for(std::size_t i=0;i<prepared->report.lane_count();++i){
    const auto lane=prepared->report.lane(i).value();
    if(lane.queue_required){
      ASSERT_EQ(lane.target,cpu_target);ASSERT_EQ(lane.runtime_mode,c::ProviderRuntimeMode::kSharedQueue);
      queues.push_back({lane.lane,&queue});
    }
  }
  auto first=prepared->executable.create_invocation().value();
  auto second=prepared->executable.create_invocation().value();
  ASSERT_TRUE(g::prepare_progress(first,scheduler,workers).ok());
  ASSERT_TRUE(g::prepare_progress(second,scheduler,workers).ok());
  const auto output_desc=graph.value_description(output).value();
  auto output0=mem::Buffer::create(mem::system_resource(),m*n*sizeof(float),64).value();
  auto output1=mem::Buffer::create(mem::system_resource(),m*n*sizeof(float),64).value();
  ASSERT_TRUE(first.bind(output,c::TensorView::create(output_desc,output0.view()).value()).ok());
  ASSERT_TRUE(second.bind(output,c::TensorView::create(output_desc,output1.view()).value()).ok());
  struct Drain { hyle::sched::Scheduler& scheduler; ~Drain(){scheduler.drain();} } drain{scheduler};
  const auto view=c::ConstTensorView::create(row,input.view()).value();
  ASSERT_TRUE(first.bind(x,view).ok());ASSERT_TRUE(second.bind(x,view).ok());
  ASSERT_TRUE(first.prepare_replay(queues).ok());ASSERT_TRUE(second.prepare_replay(queues).ok());
  std::vector<float> activation(m*k,0);
  activation[0]=activation[k]=-448.F;
  ASSERT_TRUE(input.view().write(std::as_bytes(std::span(activation))).ok());
  {auto done=first.submit(queues).value();ASSERT_TRUE(done.wait().ok());}
  auto retained=first.result(output).value();
  const auto old=retained.tensor().value();
  std::vector<float> before(m*n),after(m*n);
  ASSERT_TRUE(old.storage().read(std::as_writable_bytes(std::span(before))).ok());
  for(int step=0;step<64;++step) {
    SCOPED_TRACE(step);
    std::fill(activation.begin(),activation.end(),0.F);
    for(int row_index=0;row_index<m;++row_index)activation[row_index*k+(step+row_index)%k]=-448.F;
    ASSERT_TRUE(input.view().write(std::as_bytes(std::span(activation))).ok());
    auto done=second.submit(queues);ASSERT_TRUE(done)<<done.error().message();
    ASSERT_TRUE(done->wait().ok());
    auto result=second.result(output);ASSERT_TRUE(result)<<result.error().message();
    auto observed=result->tensor().value();
    std::vector<float> actual(m*n);
    ASSERT_TRUE(observed.storage().read(std::as_writable_bytes(std::span(actual))).ok());
    for(int row_index=0;row_index<m;++row_index)for(int column=0;column<n;++column)
      EXPECT_EQ(actual[row_index*n+column],-448.F*(.5F+float((column+step+row_index)%8)/16)*scale);
  }
  EXPECT_FALSE(first.bind(x,view).ok());
  const auto replay=second.replay_report().value();
  EXPECT_GT(replay.host_wait_dependency_count,0u);
  ASSERT_TRUE(first.close().ok());ASSERT_TRUE(second.close().ok());
  ASSERT_TRUE(old.storage().read(std::as_writable_bytes(std::span(after))).ok());
  EXPECT_EQ(before,after);
  scheduler.drain();
}
} // namespace
