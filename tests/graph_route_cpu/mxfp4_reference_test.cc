#include "../compute_models/mxfp4_reference_fixture.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/graph/cpu.h"

namespace {
template<class Run> void run_cpu(Run run) {
  namespace f=mxfp4_reference_test;
  auto context=f::take(f::g::cpu::create_context(),"CPU MXFP4 context");
  auto queue=f::take(context.create_queue(),"CPU MXFP4 queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  run(f::m::system_resource(),queues,
      [&](const f::g::Graph &graph,f::g::CompileOptions options) {return f::g::cpu::compile(graph,context,options);},
      [](f::m::BufferView destination,std::span<const std::byte> source) {return destination.write(source);},
      [](f::m::ConstBufferView source,std::span<std::byte> destination) {return source.read(destination);},
      [](f::g::Invocation &invocation,std::span<const f::g::QueueBinding> bindings) {return invocation.prepare_replay(bindings);});
}
}
TEST(CpuMxfp4ReferenceGraph, OriginalPackedBanksSurviveSharedAndIndependentSlots) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read,auto prepare) {
    mxfp4_reference_test::run_lifetimes(resource,queues,compile,write,read,prepare);
  });
}
TEST(CpuMxfp4ReferenceGraph, TinyOperandsRuntimeFailuresAndIeeeRecovery) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read,auto prepare) {
    mxfp4_reference_test::run_runtime_domain(resource,queues,compile,write,read,prepare);
  });
}
TEST(CpuMxfp4ReferenceGraph, WeightOnlyKeepsOriginalActivationAndResultLifetimes) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read,auto prepare) {
    mxfp4_reference_test::run_lifetimes(resource,queues,compile,write,read,prepare,{},
        mxfp4_reference_test::ActivationMode::kBFloat16);
  });
}
TEST(CpuMxfp4ReferenceGraph, WeightOnlyTinyValuesCheckedFailuresAndPrescaledWeights) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read,auto prepare) {
    mxfp4_reference_test::run_runtime_domain(resource,queues,compile,write,read,prepare,
        mxfp4_reference_test::ActivationMode::kBFloat16);
  });
}
TEST(CpuMxfp4ReferenceDirect, QuantizesSharedRowsOnceAndRejectsWorkspaceAliasing) {
  namespace f=mxfp4_reference_test;
  auto context=f::take(f::g::cpu::create_context(),"CPU MX direct context");
  auto queue=f::take(context.create_queue(),"CPU MX direct queue");
  for(const auto layout:{f::Layout::kDense,f::Layout::kShared,f::Layout::kPerSlot,f::Layout::kSharedView}) {
    f::Case sample(layout);const auto data=sample.data();const auto payload=data.payloads();
    const auto operation=sample.grouped()?f::c::model_ops::grouped_mxfp4_linear():f::c::model_ops::mxfp4_linear();
    auto prepared=f::take(f::c::prepare(context,operation,std::span(sample.descriptions).first(sample.port_count()),
        std::array{sample.output_description},f::m::system_resource()),"prepare MX direct");
    EXPECT_EQ(prepared.provider().id,f::c::native::kProviderId);
    const auto requirement=prepared.workspace_requirement();
    EXPECT_EQ(requirement.size,std::size_t(2*sample.tokens*sample.depth*(layout==f::Layout::kPerSlot?sample.slots:1)));
    std::array<f::m::Buffer,4> buffers;std::array<f::c::ConstTensorView,4> inputs;
    for(std::size_t i=0;i<sample.port_count();++i) {
      buffers[i]=f::take(f::m::Buffer::create(f::m::system_resource(),std::max(payload[i].size(),requirement.size),64),"original MX direct input");
      f::check(buffers[i].view().subview(0,payload[i].size()).value().write(payload[i]),"upload original input");
      inputs[i]=f::take(f::c::ConstTensorView::create(sample.descriptions[i],buffers[i].view()),"original view");
    }
    const auto size=sample.output_description.required_bytes().value();
    auto destination=f::take(f::m::Buffer::create(f::m::system_resource(),size,64),"direct output");
    const std::vector<std::byte> poison(size,std::byte{0x5a});f::check(destination.view().write(poison),"poison output");
    const std::array outputs{f::take(f::c::TensorView::create(sample.output_description,destination.view()),"output view")};
    for(std::size_t alias=0;alias<sample.port_count();++alias) {
      auto aliased=f::take(prepared.create_invocation(buffers[alias].view()),"aliased workspace");
      auto done=aliased.submit(queue,std::span(inputs).first(sample.port_count()),outputs);ASSERT_TRUE(done)<<done.error().message();
      EXPECT_TRUE(done->wait().IsInvalidArgument());
      for(std::size_t i=0;i<sample.port_count();++i) {
        std::vector<std::byte> actual(payload[i].size());
        f::check(buffers[i].view().subview(0,payload[i].size()).value().read(actual),"unchanged original bytes");EXPECT_EQ(actual,payload[i]);
      }
      std::vector<std::byte> actual(size);f::check(destination.view().read(actual),"unchanged output");EXPECT_EQ(actual,poison);
    }
    auto scratch=f::take(f::m::Buffer::create(f::m::system_resource(),requirement.size,64),"distinct workspace");
    auto run=f::take(prepared.create_invocation(scratch.view()),"direct valid invocation");
    auto done=f::take(run.submit(queue,std::span(inputs).first(sample.port_count()),outputs),"direct valid submit");f::check(done.wait(),"direct completion");
    std::vector<std::uint16_t> q(requirement.size/2);f::check(scratch.view().read(std::as_writable_bytes(std::span(q))),"actual QDQ bytes");
    EXPECT_EQ(q[2],0x8000u)<<"source negative zero survives actual quantization";
  }
}
TEST(CpuMxfp4ReferenceDirect, WeightOnlyNeedsNoActivationWorkspace) {
  namespace f=mxfp4_reference_test;
  auto context=f::take(f::g::cpu::create_context(),"CPU W4A16 context");
  for(const auto layout:{f::Layout::kDense,f::Layout::kShared,f::Layout::kPerSlot,f::Layout::kSharedView}) {
    const f::Case sample(layout,3,64,f::ActivationMode::kBFloat16);
    const auto operation=sample.grouped()?f::c::model_ops::grouped_mxfp4_linear(sample.mode)
        :f::c::model_ops::mxfp4_linear(sample.mode);
    auto prepared=f::take(f::c::prepare(context,operation,std::span(sample.descriptions).first(sample.port_count()),
        std::array{sample.output_description},f::m::system_resource()),"prepare W4A16");
    EXPECT_EQ(prepared.provider().id,f::c::native::kProviderId);
    EXPECT_EQ(prepared.workspace_requirement().size,0u);
    EXPECT_TRUE(prepared.create_invocation());
  }
}
TEST(CpuMxfp4ReferenceSchema, RejectsWrongScaleAndIncompleteGroupWithoutGlobals) {
  namespace f=mxfp4_reference_test;f::Case sample;std::array<f::c::TensorDesc,1> output;
  auto input=sample.descriptions;
  EXPECT_TRUE(f::c::model_ops::Mxfp4Linear{}.infer_output_descriptions(std::span(input).first(3),output).ok());
  input[2]=f::dense(f::Type::kFloat8E4M3FN,{sample.columns,sample.depth/32});
  EXPECT_FALSE(f::c::model_ops::Mxfp4Linear{}.infer_output_descriptions(std::span(input).first(3),output).ok());
  input=sample.descriptions;input[0]=f::dense(f::Type::kBFloat16,{sample.tokens,16});
  EXPECT_FALSE(f::c::model_ops::Mxfp4Linear{}.infer_output_descriptions(std::span(input).first(3),output).ok());
  EXPECT_FALSE(f::c::model_ops::Mxfp4Linear{}.infer_output_descriptions(sample.descriptions,output).ok());
  f::Case grouped(f::Layout::kShared);input=grouped.descriptions;input[0]=f::dense(f::Type::kBFloat16,{grouped.tokens,2,grouped.depth});
  EXPECT_FALSE(f::c::model_ops::GroupedMxfp4Linear{}.infer_output_descriptions(input,output).ok());
  const auto bad_mode=static_cast<f::ActivationMode>(99);
  input=sample.descriptions;
  EXPECT_FALSE(f::c::model_ops::Mxfp4Linear{bad_mode}.infer_output_descriptions(std::span(input).first(3),output).ok());
  input=grouped.descriptions;
  EXPECT_FALSE(f::c::model_ops::GroupedMxfp4Linear{bad_mode}.infer_output_descriptions(input,output).ok());
}
