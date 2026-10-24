#include "../compute/native_context_fixture.h"
#include "../compute_models/gptq_reference_fixture.h"
#include "../../src/compute_packed_shared/artifact_packed_linear.h"
#include "hyle/compute/prepared_operation.h"

namespace {
namespace f=gptq_reference_test;
namespace c=hyle::compute;
namespace m=hyle::mem;
using Packed=c::packed::internal::ArtifactPackedLinear;
constexpr auto format=c::model_ops::WeightArtifactFormat::kGptqV1Int4;
struct Bindings final {
  std::array<m::Buffer,5> buffers;
  std::array<c::ConstTensorView,5> views;
  Bindings(const f::Case &geometry,const f::Data &data) {
    const auto payload=geometry.payloads(data);
    for(std::size_t i=0;i<5;++i) {
      buffers[i]=f::take(m::Buffer::create(m::system_resource(),payload[i].size()+16,64),"GPTQ original buffer");
      f::check(buffers[i].view().fill(std::byte{0xa5}),"GPTQ guard");
      f::check(buffers[i].view().subview(0,payload[i].size()).value().write(payload[i]),"GPTQ input bytes");
      views[i]=f::take(c::ConstTensorView::create(geometry.descriptions[i],buffers[i].view()),"GPTQ input view");
    }
  }
  void update(const f::Case &geometry,const f::Data &data) {
    const auto payload=geometry.payloads(data);
    for(std::size_t i=0;i<5;++i)f::check(buffers[i].view().subview(0,payload[i].size()).value().write(payload[i]),"updated GPTQ bytes");
  }
};
std::vector<std::byte> raw(m::ConstBufferView view) {
  std::vector<std::byte> b(view.size());f::check(view.read(b),"read GPTQ bytes");return b;
}
void expect(m::ConstBufferView view,f::Type type,std::span<const double> expected) {
  const auto bytes=raw(view.subview(0,expected.size()*c::data_type_size(type)).value());
  for(std::size_t i=0;i<expected.size();++i) {
    double value;
    if(type==f::Type::kFloat32) {float x;std::memcpy(&x,bytes.data()+i*4,4);value=x;}
    else {std::uint16_t x;std::memcpy(&x,bytes.data()+i*2,2);value=type==f::Type::kFloat16?half_test::decode_bits<f::F16>(x):half_test::decode_bits<f::BF16>(x);}
    if(std::isnan(expected[i])) {EXPECT_TRUE(std::isnan(value))<<i;}
    else {EXPECT_EQ(value,expected[i])<<i;}
  }
}
TEST(NativeGptq, OriginalSchemaDefaultAndPackedCapacityDoNotAllocateDenseBank) {
  f::Case geometry;const auto operation=c::model_ops::ArtifactWeightDecode{format,32};
  std::array<c::TensorDesc,1> output;
  ASSERT_TRUE(operation.infer_output_descriptions(std::span(geometry.descriptions).subspan(1),output).ok());
  EXPECT_EQ(output[0],f::dense(f::Type::kFloat16,{16,64}));
  auto invalid=geometry.descriptions;invalid[3]=f::dense(f::Type::kBFloat16,{2,16});
  EXPECT_TRUE(operation.infer_output_descriptions(std::span(invalid).subspan(1),output).IsInvalidArgument());
  invalid=geometry.descriptions;invalid[4]=f::dense(f::Type::kInt32,{63});
  EXPECT_TRUE(operation.infer_output_descriptions(std::span(invalid).subspan(1),output).IsInvalidArgument());
  auto context=c::test::native_context().value();
  const std::array descriptions{f::dense(f::Type::kFloat16,{1,5120}),f::dense(f::Type::kInt32,{640,17408}),
      f::dense(f::Type::kInt32,{80,2176}),f::dense(f::Type::kFloat16,{80,17408}),f::dense(f::Type::kInt32,{5120})};
  const std::array outputs{f::dense(f::Type::kFloat16,{1,17408})};
  auto prepared=c::prepare(context,c::Operation::custom(Packed{format,64,f::Type::kFloat16,f::Type::kFloat16}),descriptions,outputs,m::system_resource());
  ASSERT_TRUE(prepared)<<prepared.error().message();EXPECT_EQ(prepared->workspace_requirement().size,0u);EXPECT_EQ(prepared->invocation_private_bytes(),0u);
}
TEST(NativeGptq, BoundedPackedReadsOriginalWordsScalesAndCurrentGroupIndices) {
  auto context=c::test::native_context().value();auto queue=context.create_queue().value();unsigned submissions=0;
  for(const auto p:f::profiles) {
    f::Case geometry(p);auto data=geometry.data();Bindings b(geometry,data);const std::array outputs_desc{geometry.output};
    auto prepared=c::prepare(context,c::Operation::custom(Packed{format,static_cast<std::uint32_t>(p.group),p.decode,p.operand}),geometry.descriptions,outputs_desc,m::system_resource()).value();
    auto output=m::Buffer::create(m::system_resource(),geometry.output.required_bytes().value()+16,64).value();f::check(output.view().fill(std::byte{0xa5}),"output guard");
    const std::array outputs{c::TensorView::create(geometry.output,output.view()).value()};auto invocation=prepared.create_invocation().value();
    for(unsigned phase=0;phase<2;++phase) {
      data=geometry.data(phase);b.update(geometry,data);auto done=invocation.submit(queue,b.views,outputs);ASSERT_TRUE(done)<<done.error().message();ASSERT_TRUE(done->wait().ok());++submissions;expect(output.view(),p.operand,geometry.oracle(data));
      const auto payload=geometry.payloads(data);
      for(std::size_t i=0;i<5;++i) {EXPECT_EQ(raw(b.buffers[i].view().subview(0,payload[i].size()).value()),payload[i]);EXPECT_EQ(raw(b.buffers[i].view().subview(payload[i].size(),16).value()),std::vector<std::byte>(16,std::byte{0xa5}));}
      EXPECT_EQ(raw(output.view().subview(geometry.output.required_bytes().value(),16).value()),std::vector<std::byte>(16,std::byte{0xa5}));
    }
    data.ids.back()=static_cast<std::int32_t>(p.depth/p.group);b.update(geometry,data);f::check(output.view().fill(std::byte{0xa5}),"poison rejected output");
    auto rejected=invocation.submit(queue,b.views,outputs);ASSERT_TRUE(rejected)<<rejected.error().message();EXPECT_TRUE(rejected->wait().IsOutOfRange());EXPECT_EQ(raw(output.view()),std::vector<std::byte>(output.size(),std::byte{0xa5}));
  }
  ::testing::Test::RecordProperty("actual_submissions",submissions);
}
#if defined(HYLE_TEST_TORCH_CPU)
TEST(NativeGptq, OriginalFourPortDecoderKeepsHalfProductBeforeRequestedFloatOutput) {
  f::Case geometry;auto context=c::test::native_context().value();auto queue=context.create_queue().value();auto data=geometry.data();Bindings b(geometry,data);
  const auto op=c::model_ops::artifact_weight_decode(format,32,f::Type::kFloat32);const std::array descriptions{geometry.decoded};
  auto prepared=c::prepare(context,op,std::span(geometry.descriptions).subspan(1),descriptions,m::system_resource());ASSERT_TRUE(prepared)<<prepared.error().message();
  auto output=m::Buffer::create(m::system_resource(),geometry.decoded.required_bytes().value(),64).value();const std::array outputs{c::TensorView::create(geometry.decoded,output.view()).value()};auto invocation=prepared->create_invocation().value();
  for(unsigned phase=0;phase<2;++phase) {data=geometry.data(phase);b.update(geometry,data);auto done=invocation.submit(queue,std::span(b.views).subspan(1),outputs);ASSERT_TRUE(done)<<done.error().message();ASSERT_TRUE(done->wait().ok());expect(output.view(),f::Type::kFloat32,geometry.decoded_values(data));}
  data.ids.back()=-1;b.update(geometry,data);f::check(output.view().fill(std::byte{0xa5}),"poison original output");auto rejected=invocation.submit(queue,std::span(b.views).subspan(1),outputs);ASSERT_TRUE(rejected);EXPECT_TRUE(rejected->wait().IsOutOfRange());EXPECT_EQ(raw(output.view()),std::vector<std::byte>(output.size(),std::byte{0xa5}));
}
#endif
} // namespace
