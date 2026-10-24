#include "../compute_models/ga_mhc_fixture.h"
#include "hyle/graph/cpu.h"

namespace {
template<class Run> void run_cpu(Run run) {
  namespace f=ga_mhc_test;
  auto context=f::take(f::g::cpu::create_context(),"CPU context");
  auto queue=f::take(context.create_queue(),"CPU queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  run(f::m::system_resource(),std::span<const f::g::QueueBinding>(queues),
      [&](const f::g::Graph &graph,f::g::CompileOptions options){return f::g::cpu::compile(graph,context,options);},
      [](f::m::BufferView dst,std::span<const std::byte> src){return dst.write(src);},
      [](f::m::ConstBufferView src,std::span<std::byte> dst){return src.read(dst);},
      [](f::g::Invocation &invocation,std::span<const f::g::QueueBinding> bindings){return invocation.prepare_replay(bindings);});
}
}
TEST(CpuGaMhc, NativeMhcAndFloatNormFactorKeepTypedResiduals) {
  run_cpu([](auto... args) { ga_mhc_test::run_mhc(args...); });
}
