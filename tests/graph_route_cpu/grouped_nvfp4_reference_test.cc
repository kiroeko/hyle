#include "../compute_models/grouped_nvfp4_reference_fixture.h"
#include "hyle/graph/cpu.h"

namespace {
template<class Run> void run_cpu(Run run) {
  namespace f = grouped_nvfp4_reference_test;
  auto context=f::take(f::g::cpu::create_context(),"CPU grouped context");
  auto queue=f::take(context.create_queue(),"CPU grouped queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  run(f::m::system_resource(),queues,
      [&](const f::g::Graph &graph,f::g::CompileOptions options) {return f::g::cpu::compile(graph,context,options);},
      [](f::m::BufferView destination,std::span<const std::byte> source) {return destination.write(source);},
      [](f::m::ConstBufferView source,std::span<std::byte> destination) {return source.read(destination);},
      [](f::g::Invocation &invocation,std::span<const f::g::QueueBinding> bindings) {return invocation.prepare_replay(bindings);});
}
}
TEST(CpuGroupedNvfp4ReferenceGraph, SelectedExpertGlobalsAndSharedViewsSurviveSlotsAndUpdates) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read,auto prepare) {
    grouped_nvfp4_reference_test::run_lifetimes(resource,queues,compile,write,read,prepare);
  });
}
TEST(CpuGroupedNvfp4ReferenceGraph, CheckedSelectedRuntimeDomainUsesFreshInvocationAfterFailure) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read,auto prepare) {
    grouped_nvfp4_reference_test::run_runtime_domain(resource,queues,compile,write,read,prepare);
  });
}
