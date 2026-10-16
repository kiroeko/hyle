#include "../compute_models/sampling_fixture.h"
#include <hyle/graph/cpu.h>
#include <hyle/graph/cuda.h>
#include <hyle/graph/cpu_cuda.h>
#include <hyle/mem/buffer.h>

namespace {
namespace t=sampling_test;
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace b=g::cpu_cuda;
namespace m=hyle::mem;
struct Drain final {hyle::sched::Scheduler &scheduler;~Drain(){scheduler.drain();}};
void run() {
  constexpr int rows=3,width=129;
  auto cpu=t::take(g::cpu::create_context()),cuda=t::take(g::cuda::create_context());
  auto device=t::take(c::cuda::MemoryResource::create(cuda,c::cuda::MemoryKind::kDevice));
  auto staging=t::take(c::cuda::MemoryResource::create(cuda,c::cuda::MemoryKind::kPinnedHost));
  hyle::sched::Scheduler scheduler(2,m::system_allocator(),64);
  const auto logits_desc=t::dense(c::DataType::kFloat32,{rows,width});
  const auto key_desc=t::dense(c::DataType::kUInt32,{rows,4});
  const auto output_desc=t::dense(c::DataType::kFloat32,{rows});
  auto input=t::take(m::Buffer::create(device.ref(),t::take(logits_desc.required_bytes()),256));
  auto keys=t::take(m::Buffer::create(device.ref(),t::take(key_desc.required_bytes()),256));
  const auto input_view=t::take(c::ConstTensorView::create(logits_desc,input.view()));
  const auto key_view=t::take(c::ConstTensorView::create(key_desc,keys.view()));
  auto table_buffer=t::take(m::Buffer::create(m::system_resource(),width*sizeof(float),256));
  std::array<float,width> table_data;
  for(int j=0;j<width;++j)table_data[std::size_t(j)]=float(j)*2+.25F;
  t::require(table_buffer.view().write(std::as_bytes(std::span(table_data))));
  const auto table_view=t::take(c::ConstTensorView::create(t::dense(c::DataType::kFloat32,{width}),table_buffer.view()));
  auto builder=t::take(g::GraphBuilder::create());
  const auto table=t::take(builder.constant(table_view.description()));
  const std::array constants{g::ConstantBinding{table,table_view,g::ConstantPolicy::kMaterialize}};
  const auto x=t::take(builder.input(logits_desc)),key=t::take(builder.input(key_desc));
  const auto produced=t::take(builder.add(c::Copy{},x));
  const auto counter=t::take(builder.add(c::Copy{},key));
  const auto sampled=t::take(builder.add_node(c::sampling_ops::categorical(true),std::array{produced,counter}));
  const auto index=t::take(builder.result(sampled,0)),score=t::take(builder.result(sampled,1));
  const auto gathered=t::take(builder.add(c::Gather{},table,index));
  const auto sum=t::take(builder.add(c::Binary{c::BinaryKind::kAdd},gathered,score));
  t::require(builder.publish(sum));auto logical=t::take(std::move(builder).finish());
  const std::array placements{
      g::NodePlacement{t::take(logical.node_id(3)),b::kCpuTarget,g::PlacementStrength::kRequired},
      g::NodePlacement{t::take(logical.node_id(4)),b::kCpuTarget,g::PlacementStrength::kRequired}};
  const std::array external{g::ExternalResourceContract{x,b::kDeviceStorage},
      g::ExternalResourceContract{key,b::kDeviceStorage},g::ExternalResourceContract{sum,b::kHostStorage}};
  auto compiled=t::take(b::compile(logical,{.cpu=&cpu,.cuda=&cuda,.staging=&staging,.device=&device,
      .staging_workspace_budget_bytes=512},{.constants=constants,.node_placements=placements,.external_resources=external}));
  std::array<c::Queue,2> gpu_queues,cpu_queues;
  std::array<m::Buffer,2> outputs;
  std::array<b::Invocation,2> slots;
  for(std::size_t bank=0;bank<slots.size();++bank) {
    gpu_queues[bank]=t::take(cuda.create_queue());cpu_queues[bank]=t::take(cpu.create_queue());
    outputs[bank]=t::take(m::Buffer::create(m::system_resource(),rows*sizeof(float),256));
    auto ordinary=t::take(compiled.executable.create_invocation());
    const std::array queues{g::QueueBinding{g::ExecutionLaneId{0},&gpu_queues[bank]},
                            g::QueueBinding{g::ExecutionLaneId{1},&cpu_queues[bank]}};
    slots[bank]=t::take(b::Invocation::create(std::move(ordinary),scheduler,queues));
    t::require(slots[bank].bind(x,input_view));t::require(slots[bank].bind(key,key_view));
    t::require(slots[bank].bind(sum,t::take(c::TensorView::create(output_desc,outputs[bank].view()))));
    t::require(slots[bank].prepare_replay({.mode=g::ReplayMode::kCommands,.host_wait=g::HostWaitPolicy::kReject}));
    const auto replay=t::take(slots[bank].replay_report());
    t::check(replay.staging_copy_count==4&&replay.staging_copy_bytes==rows*16,
             "CUDA joint did not restrict migration to two small selection outputs (two-hop staging charged)");
  }
  Drain drain{scheduler};
  for(int step=0;step<65;++step) {
    std::vector<float> logits(rows*width,-INFINITY);
    for(int r=0;r<rows;++r) {
      const int first=(step*7+r*13)%width,second=(first+65)%width;
      logits[std::size_t(r*width+first)]=16.F;logits[std::size_t(r*width+second)]=16.F;
    }
    const auto identities=t::identities(rows,0xffffffffULL+std::uint64_t(step));
    t::require(input.view().write(std::as_bytes(std::span(logits))));
    t::require(keys.view().write(std::as_bytes(std::span(identities))));
    auto work=t::take(slots[std::size_t(step%2)].submit());auto lease=t::take(work.wait());
    const auto view=t::take(lease->output(sum));std::array<float,rows> actual;
    t::require(view.storage().read(std::as_writable_bytes(std::span(actual))));
    for(int r=0;r<rows;++r) {
      const int first=(step*7+r*13)%width,second=(first+65)%width;
      const auto error=std::min(std::abs(double(actual[std::size_t(r)])-(2*first+.25-std::log(2.))),
                                std::abs(double(actual[std::size_t(r)])-(2*second+.25-std::log(2.))));
      t::check(error<2e-5,"CUDA selection -> CPU lookup/add independent oracle differs");
    }
  }
  for(auto &slot:slots)t::require(slot.close());
  t::require(cuda.close());t::require(cpu.close());
  std::puts("PASS CPU+CUDA: GPU producer/categorical -> 24-byte CPU lookup+add; 65 steps/two slots; staging charged");
}
}
int main() {
  try {run();return 0;}
  catch(const std::exception &error) {
    std::fprintf(stderr,"FAIL CPU+CUDA sampling: %s\n",error.what());return 1;
  }
}
