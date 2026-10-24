#include "../compute/native_context_fixture.h"
#include "../compute_models/grouped_nvfp4_reference_fixture.h"
#include "hyle/compute/prepared_operation.h"

namespace {
namespace f=grouped_nvfp4_reference_test;
namespace c=hyle::compute;
namespace m=hyle::mem;
struct Bindings final {
  std::array<m::Buffer,6> buffers;
  std::array<c::ConstTensorView,6> inputs;
  Bindings(const f::Case &geometry,const f::Data &data,std::size_t minimum_a=0) {
    const auto payload=data.payloads();
    for(std::size_t i=0;i<buffers.size();++i) {
      buffers[i]=f::take(m::Buffer::create(m::system_resource(),
          (std::max)(payload[i].size(),i==0 ? minimum_a : std::size_t{0})+16,64),"native original bank");
      f::check(buffers[i].view().fill(std::byte{0xa5}),"initial input guard");
      f::check(f::take(buffers[i].view().subview(0,payload[i].size()),"payload span").write(payload[i]),"write payload");
      inputs[i]=f::take(c::ConstTensorView::create(geometry.descriptions[i],buffers[i].view()),"native original view");
    }
  }
  void update(const f::Data &data) {
    const auto payload=data.payloads();
    for(const std::size_t i:{0u,5u})
      f::check(f::take(buffers[i].view().subview(0,payload[i].size()),"updated span").write(payload[i]),"update A/IDs");
  }
};
std::vector<std::byte> read_bytes(m::ConstBufferView view) {
  std::vector<std::byte> bytes(view.size());f::check(view.read(bytes),"read native bytes");return bytes;
}
void expect_bits(m::ConstBufferView view,const std::vector<std::uint16_t> &expected) {
  std::vector<std::uint16_t> actual(expected.size());
  f::check(view.subview(0,actual.size()*2).value().read(std::as_writable_bytes(std::span(actual))),"native result");
  for(std::size_t i=0;i<actual.size();++i) {
    if(std::isnan(f::scalar::wide(expected[i]))) {EXPECT_TRUE(std::isnan(f::scalar::wide(actual[i])))<<i;}
    else {EXPECT_EQ(actual[i],expected[i])<<i;}
  }
}
TEST(NativeGroupedNvfp4, ColdSchemaAndWorkspaceDependOnSelectedRowsNotExpertBank) {
  auto context=c::test::native_context().value();
  for(const auto experts:{3LL,17LL}) {
    f::Case geometry(f::Activation::kSharedView,experts);
    const std::array outputs{geometry.output_description};
    auto prepared=c::prepare(context,c::model_ops::grouped_nvfp4_linear(),geometry.descriptions,outputs,m::system_resource());
    ASSERT_TRUE(prepared)<<prepared.error().message();
    EXPECT_EQ(prepared->provider().id,c::native::kProviderId);
    EXPECT_EQ(prepared->workspace_requirement().size,2u*2u*3u*32u);
    auto bad=geometry.descriptions;
    bad[3]=f::dense(f::Type::kFloat32,{});
    std::array<c::TensorDesc,1> inferred;
    EXPECT_TRUE(c::model_ops::GroupedNvfp4Linear{}.infer_output_descriptions(bad,inferred).IsInvalidArgument());
    bad=geometry.descriptions;bad[5]=f::dense(f::Type::kInt32,{2,0});
    EXPECT_TRUE(c::model_ops::GroupedNvfp4Linear{}.infer_output_descriptions(bad,inferred).IsInvalidArgument());
  }
  // Real expert-bank scale is described without allocating or decoding it.
  const std::array descriptions{f::dense(f::Type::kBFloat16,{1,1,8192}),
      f::dense(f::Type::kUInt8,{512,2048,4096}),f::dense(f::Type::kFloat8E4M3FN,{512,2048,512}),
      f::dense(f::Type::kFloat32,{512}),f::dense(f::Type::kFloat32,{512}),f::dense(f::Type::kInt32,{1,10})};
  const std::array outputs{f::dense(f::Type::kBFloat16,{1,10,2048})};
  auto large=c::prepare(context,c::model_ops::grouped_nvfp4_linear(),descriptions,outputs,m::system_resource());
  ASSERT_TRUE(large)<<large.error().message();
  EXPECT_EQ(large->workspace_requirement().size,2u*10u*8192u);
}
TEST(NativeGroupedNvfp4, SelectedPackedRowsAndAllActivationLayoutsUseCurrentValues) {
  auto context=c::test::native_context().value();auto queue=context.create_queue().value();
  unsigned submits=0;
  for(const auto layout:{f::Activation::kShared,f::Activation::kPerSlot,f::Activation::kSharedView}) {
    for(const auto experts:{3LL,17LL}) {
      SCOPED_TRACE(::testing::Message()<<"layout="<<static_cast<int>(layout)<<" E="<<experts);
      f::Case geometry(layout,experts);const auto initial=geometry.data();
      ASSERT_NE(geometry.oracle(initial),geometry.oracle(initial,true));
      const std::array output_descriptions{geometry.output_description};
      auto prepared=c::prepare(context,c::model_ops::grouped_nvfp4_linear(),geometry.descriptions,output_descriptions,m::system_resource()).value();
      Bindings original(geometry,initial);
      const auto bytes=geometry.output_description.required_bytes().value();
      auto output=m::Buffer::create(m::system_resource(),bytes+16,64).value();
      auto scratch=m::Buffer::create(m::system_resource(),prepared.workspace_requirement().size+16,64).value();
      f::check(output.view().fill(std::byte{0xa5}),"output guard");
      f::check(scratch.view().fill(std::byte{0xa5}),"scratch guard");
      const std::array outputs{c::TensorView::create(geometry.output_description,output.view()).value()};
      auto invocation=prepared.create_invocation(scratch.view()).value();
      for(unsigned phase=0;phase<2;++phase) {
        const auto data=geometry.data(phase);original.update(data);
        auto done=invocation.submit(queue,original.inputs,outputs);ASSERT_TRUE(done)<<done.error().message();
        ASSERT_TRUE(done->wait().ok());++submits;
        expect_bits(output.view(),geometry.oracle(data));
        const auto payload=data.payloads();
        for(std::size_t i=0;i<payload.size();++i) {
          EXPECT_EQ(read_bytes(original.buffers[i].view().subview(0,payload[i].size()).value()),payload[i])<<i;
          EXPECT_EQ(read_bytes(original.buffers[i].view().subview(payload[i].size(),16).value()),std::vector<std::byte>(16,std::byte{0xa5}));
        }
        EXPECT_EQ(read_bytes(output.view().subview(bytes,16).value()),std::vector<std::byte>(16,std::byte{0xa5}));
        EXPECT_EQ(read_bytes(scratch.view().subview(prepared.workspace_requirement().size,16).value()),std::vector<std::byte>(16,std::byte{0xa5}));
      }
    }
  }
  ::testing::Test::RecordProperty("actual_submissions",submits);
}
TEST(NativeGroupedNvfp4, RejectsAllIdsAndInputWorkspaceOverlapBeforeWriting) {
  f::Case geometry;auto data=geometry.data();
  auto context=c::test::native_context().value();auto queue=context.create_queue().value();
  const std::array output_descriptions{geometry.output_description};
  auto prepared=c::prepare(context,c::model_ops::grouped_nvfp4_linear(),geometry.descriptions,output_descriptions,m::system_resource()).value();
  const auto requirement=prepared.workspace_requirement();
  Bindings original(geometry,data,requirement.size);
  auto output=m::Buffer::create(m::system_resource(),geometry.output_description.required_bytes().value(),64).value();
  const std::array outputs{c::TensorView::create(geometry.output_description,output.view()).value()};
  const auto before=read_bytes(original.buffers[0].view());
  {
    auto invocation=prepared.create_invocation(original.buffers[0].view()).value();
    auto done=invocation.submit(queue,original.inputs,outputs);ASSERT_TRUE(done)<<done.error().message();
    EXPECT_TRUE(done->wait().IsInvalidArgument());
    EXPECT_EQ(read_bytes(original.buffers[0].view()),before);
  }
  auto scratch=m::Buffer::create(m::system_resource(),requirement.size,64).value();
  f::check(scratch.view().fill(std::byte{0xa5}),"untouched checked-ID scratch");
  data.ids.back()=static_cast<std::int32_t>(geometry.experts);
  original.update(data);
  auto invocation=prepared.create_invocation(scratch.view()).value();
  auto done=invocation.submit(queue,original.inputs,outputs);ASSERT_TRUE(done)<<done.error().message();
  const auto status=done->wait();EXPECT_TRUE(status.IsOutOfRange());
  EXPECT_NE(status.message().find("expert id"),std::string_view::npos);
  EXPECT_EQ(read_bytes(scratch.view()),std::vector<std::byte>(requirement.size,std::byte{0xa5}));
}
} // namespace
