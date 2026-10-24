#include "../compute_models/fp8_scale_policy_fixture.h"
#include "hyle/graph/cpu.h"

namespace {
template<class Run> void run_cpu(Run run) {
  namespace f=fp8_scale_policy_test;
  auto context=f::take(f::g::cpu::create_context(),"CPU scale-policy context");
  auto queue=f::take(context.create_queue(),"CPU scale-policy queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  run(f::m::system_resource(),queues,
      [&](const f::g::Graph &graph,f::g::CompileOptions options){return f::g::cpu::compile(graph,context,options);},
      [](f::m::BufferView to,std::span<const std::byte> from){return to.write(from);},
      [](f::m::ConstBufferView from,std::span<std::byte> to){return from.read(to);},
      [](f::g::Invocation &inv,std::span<const f::g::QueueBinding> q){return inv.prepare_replay(q);});
}
}
TEST(CpuFp8ScalePolicy, ThreePoliciesKeepDistinctFiniteQuantizers) {
  run_cpu([](auto r,auto q,auto compile,auto write,auto read,auto prepare){fp8_scale_policy_test::run_dense(r,q,compile,write,read,prepare);});
}
TEST(CpuFp8ScalePolicy, FixedHeaderRoleKeepsOriginalBlockAxes) {
  run_cpu([](auto r,auto q,auto compile,auto write,auto read,auto prepare){fp8_scale_policy_test::run_source_role(r,q,compile,write,read,prepare);});
}
TEST(CpuFp8ScalePolicy, GroupedSharedAndPerSlotInputsKeepReciprocalPolicy) {
  run_cpu([](auto r,auto q,auto compile,auto write,auto read,auto prepare){fp8_scale_policy_test::run_grouped(r,q,compile,write,read,prepare);});
}
TEST(CpuFp8ScalePolicy, InvalidIdsFailAndFreshInvocationRecovers) {
  run_cpu([](auto r,auto q,auto compile,auto write,auto read,auto prepare){fp8_scale_policy_test::run_checked_ids(r,q,compile,write,read,prepare);});
}
TEST(CpuFp8ScalePolicy, OriginalPowerOfTwoNeighbor) {
  run_cpu([](auto r,auto q,auto compile,auto write,auto read,auto prepare){grouped_fp8_power2_test::run(r,q,compile,write,read,prepare);});
}
TEST(CpuFp8ScalePolicy, OriginalDivideNeighbor) {
  run_cpu([](auto r,auto q,auto compile,auto write,auto read,auto prepare){grouped_fp8_power2_test::run(r,q,compile,write,read,prepare,{},true);});
}
