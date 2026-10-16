#pragma once

#include "sampling_fixture.h"
#include <hyle/graph/cpu.h>
#include <hyle/mem/buffer.h>
#include <hyle/sched/scheduler.h>

#if defined(HYLE_G08_APPLE)
#include <hyle/graph/cpu_apple.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>
#else
#include <hyle/graph/cpu_vulkan.h>
#endif

namespace sampling_test {
namespace g=hyle::graph;
namespace m=hyle::mem;
#if defined(HYLE_G08_APPLE)
namespace native=c::apple;
namespace bridge=g::cpu_apple;
namespace mx=mlx::core;
#else
namespace native=c::vulkan;
namespace bridge=g::cpu_vulkan;
#endif
struct JointDrain final {
  hyle::sched::Scheduler &scheduler;
  ~JointDrain() { scheduler.drain(); }
};
inline void native_joint() {
  constexpr int rows=3,width=129;
  auto cpu=take(g::cpu::create_context());
  auto context=take(native::Context::create());
#if defined(HYLE_G08_APPLE)
  auto memory=take(native::MemoryResource::create(context));
#else
  auto memory=take(native::MemoryResource::create(context,native::MemoryKind::kDevice));
  auto staging=take(native::MemoryResource::create(context,native::MemoryKind::kStaging));
#endif
  hyle::sched::Scheduler scheduler(2,m::system_allocator(),64);
  auto table_buffer=take(m::Buffer::create(m::system_resource(),width*sizeof(float),256));
  std::array<float,width> table_data;
  for(int j=0;j<width;++j)table_data[std::size_t(j)]=float(j)*2+.25F;
  require(table_buffer.view().write(std::as_bytes(std::span(table_data))));
  const auto table_view=take(c::ConstTensorView::create(dense(c::DataType::kFloat32,{width}),table_buffer.view()));
  auto builder=take(g::GraphBuilder::create());
  const auto table=take(builder.constant(table_view.description()));
  const std::array constants{g::ConstantBinding{table,table_view,g::ConstantPolicy::kMaterialize}};
  const auto logits_desc=dense(c::DataType::kFloat32,{rows,width});
  const auto keys_desc=dense(c::DataType::kUInt32,{rows,4});
  const auto x=take(builder.input(logits_desc)),key=take(builder.input(keys_desc));
  const auto produced=take(builder.add(c::Copy{},x));
  const auto counter=take(builder.add(c::Copy{},key));
  const auto sampled=take(builder.add_node(c::sampling_ops::categorical(true),std::array{produced,counter}));
  const auto index=take(builder.result(sampled,0)),score=take(builder.result(sampled,1));
  const auto gathered=take(builder.add(c::Gather{},table,index));
  const auto sum=take(builder.add(c::Binary{c::BinaryKind::kAdd},gathered,score));
  require(builder.publish(sum));auto logical=take(std::move(builder).finish());
  const std::array placements{
      g::NodePlacement{take(logical.node_id(3)),bridge::kCpuTarget,g::PlacementStrength::kRequired},
      g::NodePlacement{take(logical.node_id(4)),bridge::kCpuTarget,g::PlacementStrength::kRequired}};
  bridge::CompileReport report;
#if defined(HYLE_G08_APPLE)
  auto program=take(bridge::compile(logical,{.cpu=&cpu,.apple=&context,.native=&memory},
      {.node_placements=placements,.constants=constants},&report));
#else
  auto program=take(bridge::compile(logical,{.cpu=&cpu,.vulkan=&context,.device=&memory,.staging=&staging},
      {.node_placements=placements,.constants=constants},&report));
#endif
  check(report.fragments.size()==2,"native joint lost GPU selection/CPU math partition");
  std::array<c::Queue,2> queues;
  std::array<bridge::Invocation,2> slots;
  for(std::size_t bank=0;bank<slots.size();++bank) {
    queues[bank]=take(cpu.create_queue());
    slots[bank]=take(program.create_invocation(scheduler,queues[bank]));
  }
  JointDrain drain{scheduler};
  for(int step=0;step<65;++step) {
    std::vector<float> logits(rows*width,-INFINITY);
    for(int r=0;r<rows;++r) {
      const int first=(step*7+r*13)%width,second=(first+65)%width;
      logits[std::size_t(r*width+first)]=16.F;
      logits[std::size_t(r*width+second)]=16.F;
    }
    const auto keys=identities(rows,0xffffffffULL+std::uint64_t(step));
#if defined(HYLE_G08_APPLE)
    auto input=mx::array(logits.data(),{rows,width},mx::float32);
    auto ids=mx::array(reinterpret_cast<const std::uint32_t *>(keys.data()),{rows,4},mx::uint32);
    mx::async_eval({input,ids});mx::synchronize(mx::default_stream(mx::Device::gpu));
    input.wait();ids.wait();
#else
    const auto input=take(context.upload(logits_desc,std::as_bytes(std::span(logits))));
    const auto ids=take(context.upload(keys_desc,std::as_bytes(std::span(keys))));
#endif
    auto &slot=slots[std::size_t(step%2)];
    require(slot.bind(x,input));require(slot.bind(key,ids));require(slot.prepare_replay());
    const auto replay=take(slot.replay_report());
    check(replay.cpu_fragments==1&&replay.host_copy_count==2&&replay.host_copy_bytes==rows*8,
          "native joint copied more than the selected I32/F32 result boundary");
    auto work=take(slot.submit());auto lease=take(work.wait());
    const auto output=take(lease->output(sum));
    std::array<float,rows> actual;
    require(output.storage().read(std::as_writable_bytes(std::span(actual))));
    for(int r=0;r<rows;++r) {
      const int first=(step*7+r*13)%width,second=(first+65)%width;
      const auto error=std::min(std::abs(double(actual[std::size_t(r)])-(2*first+.25-std::log(2.))),
                                std::abs(double(actual[std::size_t(r)])-(2*second+.25-std::log(2.))));
      check(error<2e-5,"GPU selection -> CPU lookup/add independent numerical oracle differs");
    }
  }
  for(auto &slot:slots)require(slot.close());
  require(context.close());require(cpu.close());
  std::puts("PASS native joint: resident GPU producer/checked categorical -> 24-byte CPU lookup+add, 65 steps/two slots");
}
} // namespace sampling_test
