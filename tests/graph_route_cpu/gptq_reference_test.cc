#include "../compute_models/gptq_reference_fixture.h"
#include "hyle/graph/cpu.h"

namespace {
template<class Run> void run_cpu(Run run,bool packed) {
  namespace f=gptq_reference_test;
  auto context=f::take(f::g::cpu::create_context(),"CPU GPTQ context");
  auto queue=f::take(context.create_queue(),"CPU GPTQ queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  run(f::m::system_resource(),queues,
      [&](const f::g::Graph &graph,f::g::CompileOptions options) {
        options.provider_replay=packed?f::g::ProviderReplayPolicy::kRequirePrepared:f::g::ProviderReplayPolicy::kAllowAny;
        return f::g::cpu::compile(graph,context,options);
      },[](f::m::BufferView d,std::span<const std::byte> s) {return d.write(s);},
      [](f::m::ConstBufferView s,std::span<std::byte> d) {return s.read(d);},
      [](f::g::Invocation &i,std::span<const f::g::QueueBinding> q) {return i.prepare_replay(q);});
}
}
TEST(CpuGptqReferenceGraph, PackedSourceHalfRoundAndDynamicIndicesSurviveSlots) {
  run_cpu([](auto r,auto q,auto compile,auto write,auto read,auto prepare) {
    gptq_reference_test::run_lifetimes(r,q,compile,write,read,prepare);
  },true);
}
TEST(CpuGptqReferenceGraph, ChecksAllIndicesOnEachSubmissionAndRetainsScaleDomain) {
  run_cpu([](auto r,auto q,auto compile,auto write,auto read,auto prepare) {
    gptq_reference_test::run_runtime_domain(r,q,compile,write,read,prepare);
  },true);
}
#if defined(HYLE_TEST_TORCH_CPU)
TEST(CpuGptqReferenceGraph, PublishedDecodeKeepsTheOriginalFourPortPath) {
  run_cpu([](auto r,auto q,auto compile,auto write,auto read,auto prepare) {
    gptq_reference_test::run_escape(r,q,compile,write,read,prepare);
  },false);
}
#endif
