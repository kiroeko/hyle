#include "../compute_models/local_nvfp4_reference_fixture.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/graph/cpu.h"

namespace {
template<class Run> void run_cpu(Run run) {
  namespace f=local_nvfp4_reference_test;
  auto context=f::take(f::g::cpu::create_context(),"CPU local NVFP4 context");
  auto queue=f::take(context.create_queue(),"CPU local NVFP4 queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  run(f::m::system_resource(),queues,
      [&](const f::g::Graph &graph,f::g::CompileOptions options) {return f::g::cpu::compile(graph,context,options);},
      [](f::m::BufferView destination,std::span<const std::byte> source) {return destination.write(source);},
      [](f::m::ConstBufferView source,std::span<std::byte> destination) {return source.read(destination);},
      [](f::g::Invocation &invocation,std::span<const f::g::QueueBinding> bindings) {return invocation.prepare_replay(bindings);});
}
}
TEST(CpuLocalNvfp4ReferenceGraph, OriginalDivisorBanksAndBothRoundingModesSurviveSlots) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read,auto prepare) {
    local_nvfp4_reference_test::run_lifetimes(resource,queues,compile,write,read,prepare);
  });
}
TEST(CpuLocalNvfp4ReferenceGraph, DynamicSelectedRuntimeDomainAndFailureRecovery) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read,auto prepare) {
    local_nvfp4_reference_test::run_runtime_domain(resource,queues,compile,write,read,prepare);
  });
}
TEST(CpuLocalNvfp4ReferenceDirect, CallerWorkspaceCannotOverwriteOriginalInputs) {
  namespace f=local_nvfp4_reference_test;
  auto context=f::take(f::g::cpu::create_context(),"CPU local direct context");
  auto queue=f::take(context.create_queue(),"CPU local direct queue");
  for(const auto rounding:{f::Rounding::kFloat32,f::Rounding::kBFloat16}) {
    f::Case sample(f::Layout::kDense,rounding);
    const auto data=sample.data();const auto payload=data.payloads();
    const std::array output_desc{sample.output_description};
    auto prepared=f::take(f::c::prepare(context,f::c::model_ops::local_nvfp4_linear(rounding),
        std::span(sample.descriptions).first(5),output_desc,f::m::system_resource()),"prepare local direct");
    EXPECT_EQ(prepared.provider().id,f::c::native::kProviderId);
    const auto requirement=prepared.workspace_requirement();
    EXPECT_EQ(requirement.size,std::size_t(2*sample.tokens*sample.depth));
    std::array<f::m::Buffer,5> buffers;
    std::array<f::c::ConstTensorView,5> inputs;
    for(std::size_t i=0;i<5;++i) {
      buffers[i]=f::take(f::m::Buffer::create(f::m::system_resource(),payload[i].size(),64),"direct original bank");
      f::check(buffers[i].view().write(payload[i]),"upload direct original bytes");
      inputs[i]=f::take(f::c::ConstTensorView::create(sample.descriptions[i],buffers[i].view()),"direct original view");
    }
    const auto size=sample.output_description.required_bytes().value();
    auto destination=f::take(f::m::Buffer::create(f::m::system_resource(),size,64),"direct output");
    const std::vector<std::byte> poison(size,std::byte{0x5a});
    f::check(destination.view().write(poison),"poison direct output");
    const std::array outputs{f::take(f::c::TensorView::create(sample.output_description,destination.view()),"direct output view")};
    auto aliased=f::take(prepared.create_invocation(buffers[0].view()),"aliased caller workspace");
    auto done=aliased.submit(queue,inputs,outputs);ASSERT_TRUE(done)<<done.error().message();
    const auto rejected=done->wait();EXPECT_TRUE(rejected.IsInvalidArgument())<<rejected.message();
    for(std::size_t i=0;i<5;++i) {
      std::vector<std::byte> actual(payload[i].size());f::check(buffers[i].view().read(actual),"read unchanged direct input");EXPECT_EQ(actual,payload[i]);
    }
    std::vector<std::byte> actual(size);f::check(destination.view().read(actual),"read unchanged direct output");EXPECT_EQ(actual,poison);
  }
}
