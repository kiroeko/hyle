#include "../compute_models/grouped_fp8_power2_fixture.h"
#include "hyle/graph/cpu.h"

namespace {
void run_cpu(unsigned mode) {
  namespace f=grouped_fp8_power2_test;
  auto context=f::take(f::g::cpu::create_context(),"CPU grouped FP8 context");
  auto queue=f::take(context.create_queue(),"CPU grouped FP8 queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  auto compile=[&](const f::g::Graph &graph,f::g::CompileOptions options) {
    return f::g::cpu::compile(graph,context,options);
  };
  auto write=[](f::m::BufferView to,std::span<const std::byte> from) {return to.write(from);};
  auto read=[](f::m::ConstBufferView from,std::span<std::byte> to) {return from.read(to);};
  auto prepare=[](f::g::Invocation &run,std::span<const f::g::QueueBinding> bindings) {
    return run.prepare_replay(bindings);
  };
  if (mode==1) f::run_checked_ids(f::m::system_resource(),queues,compile,write,read,prepare);
  else f::run(f::m::system_resource(),queues,compile,write,read,prepare,{},mode==2);
}
}
TEST(CpuGroupedFp8Power2Graph, SharedAndPerSlotBfloatMathRetainsOldResults) {run_cpu(0);}
TEST(CpuGroupedFp8Power2Graph, CheckedIdsFailAndFreshInvocationRecovers) {run_cpu(1);}
TEST(CpuGroupedFp8Power2Graph, DefaultNoneKeepsItsOriginalQuantizer) {run_cpu(2);}
