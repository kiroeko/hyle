#include <gtest/gtest.h>
#include <array>
#include <vector>
#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
#include "hyle/graph/cpu.h"
#include "hyle/graph/executable.h"
#endif

namespace {
namespace c = hyle::compute;
namespace m = hyle::mem;
using Format = c::model_ops::WeightArtifactFormat;
c::TensorDesc dense(c::DataType type, std::initializer_list<std::int64_t> dims) {
  return c::TensorDesc::dense(type, {dims.begin(), dims.size()}).value();
}
template<class T> m::Buffer buffer(std::span<const T> values) {
  auto result = m::Buffer::create(m::system_resource(), values.size_bytes(), 64).value();
  EXPECT_TRUE(result.view().write(std::as_bytes(values)).ok());return result;
}
// Explicit cold result becomes immutable hot input. This is a real public
// CPU composition, not a hidden decode at every model projection.
template<class T> void constant_math(c::TensorDesc weight,m::Buffer &decoded,
                                     std::span<const float> expected) {
  const auto n=weight.dimensions()[0],k=weight.dimensions()[1];
  const auto type=weight.data_type();
  const auto a_desc=dense(type,{2,k}),out_desc=dense(type,{2,n});
  std::vector<T>a(static_cast<std::size_t>(2*k),half_test::encode<T>(0.25));
  auto activation=buffer<T>(a);
  auto output=m::Buffer::create(m::system_resource(),out_desc.required_bytes().value(),64).value();
  const auto av=c::ConstTensorView::create(a_desc,activation.view()).value();
  const auto wv=c::ConstTensorView::create(weight.transposed().value(),decoded.view()).value();
  const auto destination=c::TensorView::create(out_desc,output.view()).value();
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
  namespace g=hyle::graph;auto context=g::cpu::create_context().value();auto builder=g::GraphBuilder::create().value();
  const auto x=builder.input(a_desc).value(),w=builder.constant(wv.description()).value();
  const auto y=builder.add(c::MatMul{},x,w).value();ASSERT_TRUE(builder.publish(y).ok());
  const std::array constants{g::ConstantBinding{.value=w,.tensor=wv}};
  auto compiled=g::compile(std::move(builder).finish().value(),context,{.constants=constants});
  ASSERT_TRUE(compiled)<<compiled.error().message();auto invocation=compiled->executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(x,av).ok());ASSERT_TRUE(invocation.bind(y,destination).ok());
  auto queue=context.create_queue().value();const std::array queues{g::QueueBinding{.lane=g::ExecutionLaneId{0},.queue=&queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  const auto execute=[&]()->hyle::Status{auto done=invocation.submit(queue);if(!done)return std::move(done).error();return done->wait();};
#else
  auto context=c::cpu::create_context().value();
  auto prepared=c::prepare(context,c::MatMul{},a_desc,wv.description());
  ASSERT_TRUE(prepared)<<prepared.error().message();auto invocation=prepared->create_invocation().value();
  auto queue=context.create_queue().value();
  const auto execute=[&]()->hyle::Status{auto done=invocation.submit(queue,av,wv,destination);if(!done)return std::move(done).error();return done->wait();};
#endif
  for(int pass=0;pass<2;++pass){
    const auto factor=pass?0.5:0.25;
    std::fill(a.begin(),a.end(),half_test::encode<T>(factor));
    ASSERT_TRUE(activation.view().write(std::as_bytes(std::span(a))).ok());
    ASSERT_TRUE(execute().ok());const auto actual=half_test::read<T>(output);
    for(int token=0;token<2;++token)for(std::int64_t row=0;row<n;++row){
      double sum=0;for(std::int64_t col=0;col<k;++col)
        sum+=half_test::decode(half_test::encode<T>(expected[static_cast<std::size_t>(row*k+col)]))*factor;
      EXPECT_EQ(half_test::decode(actual[static_cast<std::size_t>(token*n+row)]),half_test::decode(half_test::encode<T>(sum)));
    }
  }
}
void check(Format format, std::uint32_t group,
    std::span<const c::TensorDesc> descriptions, std::span<m::Buffer> buffers,
    std::span<const float> expected, c::DataType output_type = c::DataType::kFloat32) {
  const auto op = c::model_ops::artifact_weight_decode(format, group, output_type);
  std::array<c::TensorDesc, 1> inferred;
  ASSERT_TRUE(c::infer_output_descriptions(op, descriptions, inferred).ok());
  auto result = m::Buffer::create(m::system_resource(), inferred[0].required_bytes().value(), 64).value();
  auto target = c::TensorView::create(inferred[0], result.view()).value();
  std::vector<c::ConstTensorView> sources;
  for (std::size_t i = 0; i < buffers.size(); ++i)
    sources.push_back(c::ConstTensorView::create(descriptions[i], buffers[i].view()).value());
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
  namespace g=hyle::graph;
  auto context=g::cpu::create_context().value();auto builder=g::GraphBuilder::create().value();
  std::vector<g::ValueId> arguments;
  for(const auto &d:descriptions)arguments.push_back(builder.input(d).value());
  const auto node=builder.add_node(op,arguments).value();const auto value=builder.result(node).value();
  ASSERT_TRUE(builder.publish(value).ok());auto compiled=g::compile(std::move(builder).finish().value(),context);
  ASSERT_TRUE(compiled)<<compiled.error().message();auto invocation=compiled->executable.create_invocation().value();
  for(std::size_t i=0;i<sources.size();++i)ASSERT_TRUE(invocation.bind(arguments[i],sources[i]).ok());
  ASSERT_TRUE(invocation.bind(value,target).ok());auto queue=context.create_queue().value();
  const std::array queues{g::QueueBinding{.lane=g::ExecutionLaneId{0},.queue=&queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  const auto execute=[&]()->hyle::Status{auto done=invocation.submit(queue);if(!done)return std::move(done).error();return done->wait();};
#else
  auto context = c::cpu::create_context().value();
  auto prepared = c::prepare(context, op, sources, std::span(&target, 1));
  ASSERT_TRUE(prepared) << prepared.error().message();
  auto invocation = prepared->create_invocation().value();auto queue = context.create_queue().value();
  const auto execute=[&]()->hyle::Status{auto done=invocation.submit(queue,sources,std::span(&target,1));if(!done)return std::move(done).error();return done->wait();};
#endif
  for (int repeat = 0; repeat < 2; ++repeat) {
    ASSERT_TRUE(execute().ok());
    if(inferred[0].data_type()==c::DataType::kFloat32){
      const auto actual = half_test::read<float>(result);ASSERT_EQ(actual.size(),expected.size());
      for(std::size_t i=0;i<actual.size();++i)EXPECT_EQ(actual[i],expected[i]);
    }else if(inferred[0].data_type()==c::DataType::kBFloat16){
      const auto actual=half_test::read<hyle::simd::bfloat16_t>(result);ASSERT_EQ(actual.size(),expected.size());
      for(std::size_t i=0;i<actual.size();++i)EXPECT_EQ(half_test::decode(actual[i]),half_test::decode(half_test::encode<hyle::simd::bfloat16_t>(expected[i])));
    }else{
      const auto actual=half_test::read<hyle::simd::float16_t>(result);ASSERT_EQ(actual.size(),expected.size());
      for(std::size_t i=0;i<actual.size();++i)EXPECT_EQ(half_test::decode(actual[i]),half_test::decode(half_test::encode<hyle::simd::float16_t>(expected[i])));
    }
  }
  if(inferred[0].data_type()==c::DataType::kFloat32)constant_math<float>(inferred[0],result,expected);
  else if(inferred[0].data_type()==c::DataType::kBFloat16)constant_math<hyle::simd::bfloat16_t>(inferred[0],result,expected);
  else constant_math<hyle::simd::float16_t>(inferred[0],result,expected);
}

TEST(CpuArtifactDecode, Nf4AndNestedScalesPreserveOriginalHighFirstOrder) {
  constexpr int n = 2, k = 256, group = 64, groups = n * k / group;
  std::vector<std::uint8_t> codes(n * k / 2);
  for (std::size_t i = 0; i < codes.size(); ++i) codes[i] = std::uint8_t((i % 16) * 16 + (15 - i % 16));
  std::array<float, 16> codebook{};
  for (std::size_t i = 0; i < codebook.size(); ++i) codebook[i] = (float(i) - 8) / 8;
  std::array<float, groups> absmax{};for (int i = 0; i < groups; ++i) absmax[i] = float(i + 1) / 8;
  std::vector<float> expected(n * k);
  for (std::size_t i = 0; i < codes.size(); ++i) {
    expected[i * 2] = codebook[codes[i] >> 4] * absmax[i * 2 / group];
    expected[i * 2 + 1] = codebook[codes[i] & 15] * absmax[(i * 2 + 1) / group];
  }
  const std::array descs{dense(c::DataType::kUInt8,{n,k/2}),dense(c::DataType::kFloat32,{groups}),dense(c::DataType::kFloat32,{16})};
  std::array inputs{buffer<std::uint8_t>(codes),buffer<float>(absmax),buffer<float>(codebook)};
  check(Format::kNf4, group, descs, inputs, expected);
  check(Format::kNf4, group, descs, inputs, expected,c::DataType::kUnknown);

  std::array<std::uint8_t, groups> indices{};std::array<float,256> nested{};
  for (int i = 0; i < 256; ++i) nested[i] = float(i) / 16;
  for (int i = 0; i < groups; ++i) indices[i] = std::uint8_t(i);
  const std::array<float,1> nested_absmax{0.5f},offset{0.125f};
  for (std::size_t i = 0; i < codes.size(); ++i) {
    expected[i*2] = codebook[codes[i]>>4] * (nested[indices[i*2/group]]*0.5f+0.125f);
    expected[i*2+1] = codebook[codes[i]&15] * (nested[indices[(i*2+1)/group]]*0.5f+0.125f);
  }
  const std::array nested_descs{descs[0],dense(c::DataType::kUInt8,{groups}),dense(c::DataType::kFloat32,{1}),dense(c::DataType::kFloat32,{256}),dense(c::DataType::kFloat32,{}),descs[2]};
  std::array nested_inputs{buffer<std::uint8_t>(codes),buffer<std::uint8_t>(indices),buffer<float>(nested_absmax),buffer<float>(nested),buffer<float>(offset),buffer<float>(codebook)};
  check(Format::kNf4DoubleQuant, group, nested_descs, nested_inputs, expected);
  check(Format::kNf4DoubleQuant, group, nested_descs, nested_inputs, expected,c::DataType::kUnknown);
}

TEST(CpuArtifactDecode, AwqAndCompressedTensorsUseDifferentOriginalWordLayouts) {
  constexpr int k = 256, n = 8, group = 128;
  std::vector<std::int32_t> awq_words(k, 0x76543210),zeros(k/group, 0x33333333);
  std::vector<float> scales(k/group*n, 0.125f),expected(n*k);
  constexpr std::array<int,8> reverse{0,4,1,5,2,6,3,7};
  for (int row = 0; row < n; ++row) for (int col = 0; col < k; ++col)
    expected[row*k+col] = float(reverse[row]-3)*0.125f;
  const std::array awq_descs{dense(c::DataType::kInt32,{k,n/8}),dense(c::DataType::kInt32,{k/group,n/8}),dense(c::DataType::kFloat32,{k/group,n})};
  std::array awq_inputs{buffer<std::int32_t>(awq_words),buffer<std::int32_t>(zeros),buffer<float>(scales)};
  check(Format::kAwqGemm4, group, awq_descs, awq_inputs, expected);
  std::vector<hyle::simd::float16_t> half_scale(scales.size(),half_test::encode<hyle::simd::float16_t>(0.125));
  auto half_descs=awq_descs;half_descs[2]=dense(c::DataType::kFloat16,{k/group,n});
  std::array half_inputs{buffer<std::int32_t>(awq_words),buffer<std::int32_t>(zeros),buffer<hyle::simd::float16_t>(half_scale)};
  check(Format::kAwqGemm4,group,half_descs,half_inputs,expected,c::DataType::kUnknown);

  std::vector<std::int32_t> ct_words(n*k/8,0x76543210);
  for (int row = 0; row < n; ++row) for (int col = 0; col < k; ++col)
    expected[row*k+col] = float(col%8-8)*0.125f;
  const std::array ct_descs{dense(c::DataType::kInt32,{n,k/8}),dense(c::DataType::kFloat32,{n,k/group})};
  std::array ct_inputs{buffer<std::int32_t>(ct_words),buffer<float>(scales)};
  check(Format::kCompressedTensorsSymmetric4, group, ct_descs, ct_inputs, expected);
  std::vector<hyle::simd::bfloat16_t> bf_scale(scales.size(),half_test::encode<hyle::simd::bfloat16_t>(0.125));
  auto bf_descs=ct_descs;bf_descs[1]=dense(c::DataType::kBFloat16,{n,k/group});
  std::array bf_inputs{buffer<std::int32_t>(ct_words),buffer<hyle::simd::bfloat16_t>(bf_scale)};
  check(Format::kCompressedTensorsSymmetric4,group,bf_descs,bf_inputs,expected,c::DataType::kUnknown);
}

TEST(CpuArtifactDecode, OriginalScaleDtypeRoundsBeforeExplicitFloat32Output) {
  constexpr int k=256,n=8,group=128;
  using Half=hyle::simd::float16_t;
  using Bfloat=hyle::simd::bfloat16_t;
  const auto hs=half_test::encode<Half>(0.1);
  const auto bs=half_test::encode<Bfloat>(0.1);
  std::vector<std::int32_t> words(k,0x76543210),zeros(2,0x33333333);
  constexpr std::array<int,8> reverse{0,4,1,5,2,6,3,7};
  std::vector<float> expected(n*k);
  for(int row=0;row<n;++row)for(int col=0;col<k;++col)
    expected[row*k+col]=float(half_test::decode(half_test::encode<Half>((reverse[row]-3)*half_test::decode(hs))));
  const std::array awq_descs{dense(c::DataType::kInt32,{k,n/8}),dense(c::DataType::kInt32,{2,n/8}),
                           dense(c::DataType::kFloat16,{2,n})};
  std::array awq_inputs{buffer<std::int32_t>(words),buffer<std::int32_t>(zeros),
                       buffer<Half>(std::vector<Half>(2*n,hs))};
  check(Format::kAwqGemm4,group,awq_descs,awq_inputs,expected);
  check(Format::kAwqGemm4,group,awq_descs,awq_inputs,expected,c::DataType::kUnknown);
  std::vector<std::int32_t> packed(n*k/8,0x76543210);
  for(int row=0;row<n;++row)for(int col=0;col<k;++col)
    expected[row*k+col]=float(half_test::decode(half_test::encode<Bfloat>((col%8-8)*half_test::decode(bs))));
  const std::array ct_descs{dense(c::DataType::kInt32,{n,k/8}),dense(c::DataType::kBFloat16,{n,2})};
  std::array ct_inputs{buffer<std::int32_t>(packed),buffer<Bfloat>(std::vector<Bfloat>(n*2,bs))};
  check(Format::kCompressedTensorsSymmetric4,group,ct_descs,ct_inputs,expected);
  check(Format::kCompressedTensorsSymmetric4,group,ct_descs,ct_inputs,expected,c::DataType::kUnknown);
}
} // namespace
