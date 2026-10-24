#include "../compute_models/blockwise_w4a8_fixture.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/graph/cpu.h"

namespace {
template<class Run> void run_cpu(Run run) {
  namespace f=blockwise_w4a8_test;
  auto context=f::take(f::g::cpu::create_context(),"CPU W4A8 context");
  auto queue=f::take(context.create_queue(),"CPU W4A8 queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  run(f::m::system_resource(),queues,
      [&](const f::g::Graph &graph,f::g::CompileOptions options) {return f::g::cpu::compile(graph,context,options);},
      [](f::m::BufferView destination,std::span<const std::byte> source) {return destination.write(source);},
      [](f::m::ConstBufferView source,std::span<std::byte> destination) {return source.read(destination);},
      [](f::g::Invocation &invocation,std::span<const f::g::QueueBinding> bindings) {return invocation.prepare_replay(bindings);});
}
}
TEST(CpuBlockwiseW4A8Graph, OriginalPackedBanksSurviveSharedAndIndependentSlots) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read,auto prepare) {
    blockwise_w4a8_test::run_lifetimes(resource,queues,compile,write,read,prepare);
  });
}
TEST(CpuBlockwiseW4A8Graph, TinyOperandsRuntimeFailuresAndIeeeRecovery) {
  run_cpu([](auto resource,auto queues,auto compile,auto write,auto read,auto prepare) {
    blockwise_w4a8_test::run_runtime_domain(resource,queues,compile,write,read,prepare);
  });
}
TEST(CpuBlockwiseW4A8Direct, SharedQuantizationHasBoundedWorkspaceAndRejectsAliasing) {
  namespace f=blockwise_w4a8_test;
  auto context=f::take(f::g::cpu::create_context(),"CPU W4A8 context");
  auto queue=f::take(context.create_queue(),"CPU W4A8 queue");
  for(const auto layout:{f::Layout::kDense,f::Layout::kShared,f::Layout::kPerSlot,f::Layout::kSharedView}) {
    f::Case sample(layout,17);const auto data=sample.data();const auto payload=data.payloads();
    const auto op=sample.grouped()?f::c::model_ops::grouped_blockwise_w4a8_linear():f::c::model_ops::blockwise_w4a8_linear();
    auto prepared=f::take(f::c::prepare(context,op,std::span(sample.descriptions).first(sample.port_count()),
        std::array{sample.output_description},f::m::system_resource()),"prepare direct W4A8");
    EXPECT_EQ(prepared.provider().id,f::c::native::kProviderId);
    const auto requirement=prepared.workspace_requirement();
    EXPECT_EQ(requirement.size,std::size_t(sample.tokens*(layout==f::Layout::kPerSlot?sample.slots:1)*(sample.depth+2*(sample.depth/512))));
    std::array<f::m::Buffer,4> buffers;std::array<f::c::ConstTensorView,4> inputs;
    for(std::size_t i=0;i<sample.port_count();++i) {
      buffers[i]=f::take(f::m::Buffer::create(f::m::system_resource(),std::max(payload[i].size(),requirement.size),64),"direct W4A8 input");
      f::check(buffers[i].view().subview(0,payload[i].size()).value().write(payload[i]),"upload direct input");
      inputs[i]=f::take(f::c::ConstTensorView::create(sample.descriptions[i],buffers[i].view()),"direct input view");
    }
    const auto size=sample.output_description.required_bytes().value();
    auto destination=f::take(f::m::Buffer::create(f::m::system_resource(),size,64),"direct output");
    const std::vector<std::byte> poison(size,std::byte{0x5a});f::check(destination.view().write(poison),"poison output");
    const std::array outputs{f::take(f::c::TensorView::create(sample.output_description,destination.view()),"direct output view")};
    for(std::size_t alias=0;alias<sample.port_count();++alias) {
      auto invocation=f::take(prepared.create_invocation(buffers[alias].view()),"aliased workspace");
      auto done=invocation.submit(queue,std::span(inputs).first(sample.port_count()),outputs);ASSERT_TRUE(done)<<done.error().message();
      EXPECT_TRUE(done->wait().IsInvalidArgument());
      for(std::size_t i=0;i<sample.port_count();++i) {
        std::vector<std::byte> actual(payload[i].size());
        f::check(buffers[i].view().subview(0,payload[i].size()).value().read(actual),"unchanged original input");EXPECT_EQ(actual,payload[i]);
      }
      std::vector<std::byte> actual(size);f::check(destination.view().read(actual),"unchanged output");EXPECT_EQ(actual,poison);
    }
    auto scratch=f::take(f::m::Buffer::create(f::m::system_resource(),requirement.size,64),"distinct workspace");
    auto invocation=f::take(prepared.create_invocation(scratch.view()),"valid direct invocation");
    auto done=f::take(invocation.submit(queue,std::span(inputs).first(sample.port_count()),outputs),"direct submit");f::check(done.wait(),"direct completion");
    std::vector<std::uint16_t> actual(size/2);f::check(destination.view().read(std::as_writable_bytes(std::span(actual))),"direct result");EXPECT_EQ(actual,sample.oracle(data));
  }
}
TEST(CpuBlockwiseW4A8Schema, RejectsWrongPackingScaleAndIncompleteBlocks) {
  namespace f=blockwise_w4a8_test;f::Case sample;std::array<f::c::TensorDesc,1> out;
  auto in=sample.descriptions;
  EXPECT_TRUE(f::c::model_ops::BlockwiseW4A8Linear{}.infer_output_descriptions(std::span(in).first(3),out).ok());
  in[2]=f::dense(f::Type::kBFloat16,{sample.columns,sample.depth/512});
  EXPECT_FALSE(f::c::model_ops::BlockwiseW4A8Linear{}.infer_output_descriptions(std::span(in).first(3),out).ok());
  in=sample.descriptions;in[1]=f::dense(f::Type::kUInt8,{sample.columns,sample.depth/2});
  EXPECT_FALSE(f::c::model_ops::BlockwiseW4A8Linear{}.infer_output_descriptions(std::span(in).first(3),out).ok());
  in=sample.descriptions;in[0]=f::dense(f::Type::kBFloat16,{sample.tokens,256});
  EXPECT_FALSE(f::c::model_ops::BlockwiseW4A8Linear{}.infer_output_descriptions(std::span(in).first(3),out).ok());
  EXPECT_FALSE(f::c::model_ops::BlockwiseW4A8Linear{}.infer_output_descriptions(sample.descriptions,out).ok());
  f::Case grouped(f::Layout::kShared);in=grouped.descriptions;
  in[0]=f::dense(f::Type::kBFloat16,{grouped.tokens,2,grouped.depth});
  EXPECT_FALSE(f::c::model_ops::GroupedBlockwiseW4A8Linear{}.infer_output_descriptions(in,out).ok());
}
