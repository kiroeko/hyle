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
#include "hyle/graph/cpu_apple.h"
#include "hyle/sched/scheduler.h"

namespace {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace a=c::apple;
namespace hybrid=g::cpu_apple;
namespace mem=hyle::mem;

TEST(CpuApplePackedWeights, ResidentFp8BetweenCpuStages64StepsAndRetainedResult) {
  constexpr std::int64_t m=2,n=19,k=128;
  auto cpu=g::cpu::create_context().value();
  auto apple=a::Context::create().value();
  auto memory=a::MemoryResource::create(apple).value();
  hyle::sched::Scheduler scheduler(1,mem::system_allocator(),32);
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
  const std::array placement{
    g::NodePlacement{graph.node_id(0).value(),hybrid::kCpuTarget,g::PlacementStrength::kRequired},
    g::NodePlacement{graph.node_id(2).value(),hybrid::kCpuTarget,g::PlacementStrength::kRequired}};
  const std::array constants{g::ConstantBinding{w,c::ConstTensorView::create(wd,weight.view()).value(),g::ConstantPolicy::kMaterialize}};
  hybrid::CompileReport report;
  auto prepared=hybrid::compile(graph,{.cpu=&cpu,.apple=&apple,.native=&memory},
      {.node_placements=placement,.constants=constants},&report);
  ASSERT_TRUE(prepared)<<prepared.error().message();
  ASSERT_EQ(report.fragments.size(),3u);
  EXPECT_EQ(report.fragments[1].target,hybrid::kAppleTarget);
  weight={}; // The resident constant must own the cold source independently.
  auto first=prepared->create_invocation(scheduler,queue).value();
  auto second=prepared->create_invocation(scheduler,queue).value();
  struct Drain { hyle::sched::Scheduler& scheduler; ~Drain(){scheduler.drain();} } drain{scheduler};
  const auto view=c::ConstTensorView::create(row,input.view()).value();
  ASSERT_TRUE(first.bind(x,view).ok());ASSERT_TRUE(second.bind(x,view).ok());
  ASSERT_TRUE(first.prepare_replay().ok());ASSERT_TRUE(second.prepare_replay().ok());
  std::vector<float> activation(m*k,0);
  activation[0]=activation[k]=-448.F;
  ASSERT_TRUE(input.view().write(std::as_bytes(std::span(activation))).ok());
  auto retained=first.run().value();
  const auto old=retained->output(output).value();
  std::vector<float> before(m*n),after(m*n);
  ASSERT_TRUE(old.storage().read(std::as_writable_bytes(std::span(before))).ok());
  for(int step=0;step<64;++step) {
    SCOPED_TRACE(step);
    std::fill(activation.begin(),activation.end(),0.F);
    for(int row_index=0;row_index<m;++row_index)activation[row_index*k+(step+row_index)%k]=-448.F;
    ASSERT_TRUE(input.view().write(std::as_bytes(std::span(activation))).ok());
    auto result=second.run();ASSERT_TRUE(result)<<result.error().message();
    auto observed=result->get()->output(output).value();
    std::vector<float> actual(m*n);
    ASSERT_TRUE(observed.storage().read(std::as_writable_bytes(std::span(actual))).ok());
    for(int row_index=0;row_index<m;++row_index)for(int column=0;column<n;++column)
      EXPECT_EQ(actual[row_index*n+column],-448.F*(.5F+float((column+step+row_index)%8)/16)*scale);
  }
  EXPECT_FALSE(first.bind(x,view).ok());
  const auto replay=second.replay_report().value();
  EXPECT_EQ(replay.cpu_fragments,2u);EXPECT_EQ(replay.apple_fragments,1u);
  EXPECT_EQ(replay.host_copy_count,2u);EXPECT_EQ(replay.host_repack_count,0u);
  ASSERT_TRUE(first.close().ok());ASSERT_TRUE(second.close().ok());
  ASSERT_TRUE(old.storage().read(std::as_writable_bytes(std::span(after))).ok());
  EXPECT_EQ(before,after);
  scheduler.drain();
}
} // namespace
