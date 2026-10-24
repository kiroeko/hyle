#pragma once

#include "sampling_fixture.h"
#include <hyle/graph/cpu.h>
#include <hyle/graph/progress.h>
#include <hyle/graph/result.h>
#include <hyle/mem/buffer.h>
#include <hyle/sched/blocking.h>
#include <hyle/sched/scheduler.h>

#if defined(HYLE_G08_APPLE)
#include <hyle/graph/apple.h>
#include <hyle/compute/apple_memory.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>
#else
#include <hyle/graph/vulkan.h>
#endif

namespace sampling_test {
namespace g=hyle::graph;
namespace m=hyle::mem;
#if defined(HYLE_G08_APPLE)
namespace native=c::apple;
namespace source=g::apple;
namespace mx=mlx::core;
#else
namespace native=c::vulkan;
namespace source=g::vulkan;
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
  hyle::sched::BlockingPool completion_waiters(m::system_allocator(),2);
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
  constexpr g::ExecutionTargetId native_target{0},cpu_target{1};
  constexpr g::StorageDomainId host_storage{0},native_storage{1};
  const std::array placements{
      g::NodePlacement{take(logical.node_id(0)),native_target,g::PlacementStrength::kRequired},
      g::NodePlacement{take(logical.node_id(1)),native_target,g::PlacementStrength::kRequired},
      g::NodePlacement{sampled,native_target,g::PlacementStrength::kRequired},
      g::NodePlacement{take(logical.node_id(3)),cpu_target,g::PlacementStrength::kRequired},
      g::NodePlacement{take(logical.node_id(4)),cpu_target,g::PlacementStrength::kRequired}};
  auto native_source=take(source::execution_source(context));
  auto cpu_source=take(g::ExecutionSource::acquire(cpu));
#if defined(HYLE_G08_APPLE)
  const std::array domains{g::StorageDomain{m::system_resource()},g::StorageDomain{memory.ref()}};
#else
  const std::array domains{g::StorageDomain{m::system_resource()},g::StorageDomain{memory.ref()},g::StorageDomain{staging.ref()}};
#endif
  const std::array targets{g::ExecutionTarget{native_source,native_storage},g::ExecutionTarget{cpu_source,host_storage}};
  const std::array external{g::ExternalResourceContract{x,native_storage},g::ExternalResourceContract{key,native_storage},
      g::ExternalResourceContract{table,host_storage},g::ExternalResourceContract{sum,host_storage}};
  auto program=take(g::compile(logical,{.storage_domains=domains,.targets=targets},
      {.constants=constants,.node_placements=placements,.external_resources=external}));
  // Count only copies feeding the actual selected CPU input ports. Apple can
  // use shared Metal storage here; Vulkan may have an additional staging hop.
  // Storage ordinal alone identifies neither of these mathematical boundaries.
  std::vector<std::uint32_t> cpu_inputs;
  for(std::size_t i=0;i<program.report.step_count();++i){
    const auto step=take(program.report.step(i));
    if(step.target!=cpu_target || !step.node_count)continue;
    for(std::size_t port=0;port<step.input_count;++port){
      const auto input_id=take(program.report.step_input(i,port));
      const auto physical=take(program.report.physical_value(input_id));
      if((physical.value==index || physical.value==score) &&
          std::find(cpu_inputs.begin(),cpu_inputs.end(),input_id)==cpu_inputs.end())
        cpu_inputs.push_back(input_id);
    }
  }
  std::size_t boundary_copies=0,boundary_bytes=0;
  for(std::size_t i=0;i<program.report.step_count();++i){
    const auto step=take(program.report.step(i));
    if(step.kind!=g::CompileStepKind::kCopy)continue;
    const auto output_id=take(program.report.step_output(i,0));
    if(std::find(cpu_inputs.begin(),cpu_inputs.end(),output_id)==cpu_inputs.end())continue;
    const auto physical=take(program.report.physical_value(output_id));
    check(physical.value==index || physical.value==score,"native sampling copied an unselected GPU payload to the CPU");
    check(physical.storage==step.destination_storage && step.destination_storage.ordinal()<domains.size() &&
        domains[step.destination_storage.ordinal()].resource.properties().has(m::MemoryProperty::kHostAccessible),
        "native sampling CPU boundary is not its actual host-readable resource");
    ++boundary_copies;boundary_bytes+=step.copy_bytes;
  }
  if(cpu_inputs.size()!=2 || boundary_copies!=2 || boundary_bytes!=rows*8){
    std::fprintf(stderr,"sampling boundary diagnostic: cpu-inputs=%zu copies=%zu bytes=%zu expected=2/%d index=%u:%u score=%u:%u\n",
        cpu_inputs.size(),boundary_copies,boundary_bytes,rows*8,unsigned(index.kind()),unsigned(index.ordinal()),
        unsigned(score.kind()),unsigned(score.ordinal()));
    for(std::size_t i=0;i<program.report.step_count();++i){
      const auto step=take(program.report.step(i));
      if(step.kind!=g::CompileStepKind::kCopy)continue;
      const auto input_id=take(program.report.step_input(i,0)),output_id=take(program.report.step_output(i,0));
      const auto input=take(program.report.physical_value(input_id)),output=take(program.report.physical_value(output_id));
      std::fprintf(stderr,"sampling copy step=%zu source=%u destination=%u bytes=%zu input=%u value=%u:%u dtype=%u dims=[",
          i,unsigned(step.source_storage.ordinal()),unsigned(step.destination_storage.ordinal()),step.copy_bytes,
          unsigned(input_id),unsigned(input.value.kind()),unsigned(input.value.ordinal()),unsigned(input.description.data_type()));
      for(const auto dimension:input.description.dimensions())std::fprintf(stderr,"%lld,",static_cast<long long>(dimension));
      std::fprintf(stderr,"] alias=%u output=%u value=%u:%u dtype=%u dims=[",
          unsigned(input.alias_source.value_or(UINT32_MAX)),unsigned(output_id),unsigned(output.value.kind()),
          unsigned(output.value.ordinal()),unsigned(output.description.data_type()));
      for(const auto dimension:output.description.dimensions())std::fprintf(stderr,"%lld,",static_cast<long long>(dimension));
      std::fprintf(stderr,"] alias=%u\n",unsigned(output.alias_source.value_or(UINT32_MAX)));
    }
  }
  check(cpu_inputs.size()==2 && boundary_copies==2 && boundary_bytes==rows*8,
      "native sampling copied more than the selected I32/F32 CPU boundary");
  std::array<c::Queue,2> queues;
  std::array<m::Buffer,2> output_buffers;
  std::array<c::TensorView,2> output_views;
  std::array<g::Invocation,2> slots;
  std::array<std::vector<g::QueueBinding>,2> bindings;
  const auto output_description=take(logical.value_description(sum));
  const auto output_bytes=take(output_description.required_bytes());
  for(std::size_t bank=0;bank<slots.size();++bank) {
    queues[bank]=take(cpu.create_queue());
    for(std::size_t i=0;i<program.report.lane_count();++i){
      const auto lane=take(program.report.lane(i));
      if(lane.queue_required){
        check(lane.target==cpu_target && lane.runtime_mode==c::ProviderRuntimeMode::kSharedQueue,
            "native sampling requested a fabricated native Compute Queue");
        bindings[bank].push_back({lane.lane,&queues[bank]});
      }
    }
    slots[bank]=take(program.executable.create_invocation({.resources=g::ResourceOptions{}}));
    require(g::prepare_progress(slots[bank],scheduler,completion_waiters));
    output_buffers[bank]=take(m::Buffer::create(m::system_resource(),output_bytes,64));
    output_views[bank]=take(c::TensorView::create(output_description,output_buffers[bank].view()));
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
    const auto bank=std::size_t(step%2);auto &slot=slots[bank];
#if defined(HYLE_G08_APPLE)
    auto input_owner=take(memory.import_readonly(input,logits_desc));
    auto key_owner=take(memory.import_readonly(ids,keys_desc));
#else
    auto input_owner=take(memory.import_readonly(input));
    auto key_owner=take(memory.import_readonly(ids));
#endif
    require(slot.bind(x,logits_desc,std::move(input_owner)));require(slot.bind(key,keys_desc,std::move(key_owner)));
    require(slot.bind(sum,output_views[bank]));require(slot.prepare_replay(bindings[bank]));
    auto work=take(slot.submit(bindings[bank]));require(work.wait());
    auto result=take(slot.result(sum));const auto output=take(result.tensor());
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
