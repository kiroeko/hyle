#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <string_view>
#include <vector>
#include "../compute_models/mlx_affine_bitstream.h"
#include "../compute_models/mlx_grouped_shared_fixture.h"
#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
#include "hyle/graph/cpu.h"
#include "hyle/graph/executable.h"
#include "../compute_models/mlx_moe_fixture.h"
#endif

namespace {
namespace c = hyle::compute;
namespace m = hyle::mem;
using Format = c::model_ops::MlxWeightFormat;
using Kind = c::model_ops::MlxQuantizedKind;
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
TEST(CpuMlxQuantizedGraph, MixedBitMoeKeepsTypedMetadataAndIndependentSlots) {
  auto context = hyle::graph::cpu::create_context().value();
  auto queue = context.create_queue().value();
  const std::array queues{hyle::graph::QueueBinding{hyle::graph::ExecutionLaneId{0}, &queue}};
  mlx_moe_test::run(m::system_resource(), queues, [&](const hyle::graph::Graph &graph,
      hyle::graph::CompileOptions options) { return hyle::graph::cpu::compile(graph, context, options); },
      [](m::BufferView destination, std::span<const std::byte> source) { return destination.write(source); },
      [](m::ConstBufferView source, std::span<std::byte> destination) { return source.read(destination); });
}
#endif
c::TensorDesc desc(c::DataType type, std::initializer_list<std::int64_t> dims) {
  return c::TensorDesc::dense(type, {dims.begin(),dims.size()}).value();
}
template<class T> m::Buffer upload(std::span<const T> values) {
  auto result=m::Buffer::create(m::system_resource(),values.size_bytes(),64).value();
  EXPECT_TRUE(result.view().write(std::as_bytes(values)).ok());return result;
}
template<class T> void run(c::Operation op, std::span<const c::TensorDesc> descriptions,
    std::span<m::Buffer> buffers, std::span<const T> expected, bool indexed) {
  std::array<c::TensorDesc,1> inferred;
  ASSERT_TRUE(c::infer_output_descriptions(op,descriptions,inferred).ok());
  auto output=m::Buffer::create(m::system_resource(),inferred[0].required_bytes().value(),64).value();
  auto destination=c::TensorView::create(inferred[0],output.view()).value();
  std::vector<c::ConstTensorView> inputs;
  for(std::size_t i=0;i<buffers.size();++i)
    inputs.push_back(c::ConstTensorView::create(descriptions[i],buffers[i].view()).value());
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
  namespace g=hyle::graph;
  auto context=g::cpu::create_context().value();
  auto builder=g::GraphBuilder::create().value();std::vector<g::ValueId> arguments;
  for(const auto &d:descriptions) arguments.push_back(builder.input(d).value());
  const auto node=builder.add_node(op,arguments).value();
  const auto value=builder.result(node).value();ASSERT_TRUE(builder.publish(value).ok());
  auto compiled=g::compile(std::move(builder).finish().value(),context);
  ASSERT_TRUE(compiled)<<compiled.error().message();
  auto invocation=compiled->executable.create_invocation().value();
  for(std::size_t i=0;i<inputs.size();++i)ASSERT_TRUE(invocation.bind(arguments[i],inputs[i]).ok());
  ASSERT_TRUE(invocation.bind(value,destination).ok());auto queue=context.create_queue().value();
  const std::array queues{g::QueueBinding{.lane=g::ExecutionLaneId{0},.queue=&queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  const auto execute=[&]()->hyle::Status{auto done=invocation.submit(queue);if(!done)return std::move(done).error();return done->wait();};
#else
  auto context=c::cpu::create_context().value();
  auto prepared=c::prepare(context,op,inputs,std::span(&destination,1));
  ASSERT_TRUE(prepared)<<prepared.error().message();
  auto invocation=prepared->create_invocation().value();auto queue=context.create_queue().value();
  const auto execute=[&]()->hyle::Status{auto done=invocation.submit(queue,inputs,std::span(&destination,1));if(!done)return std::move(done).error();return done->wait();};
#endif
  for(int repeat=0;repeat<2;++repeat){
    if(repeat==1 && op.custom_as<c::model_ops::MlxQuantized>()->kind!=Kind::kEmbedding){
      auto activation=half_test::read<T>(buffers[0]);
      for(auto &sample:activation)sample=half_test::encode<T>(half_test::decode(sample)*2);
      ASSERT_TRUE(buffers[0].view().write(std::as_bytes(std::span(activation))).ok());
    }
    ASSERT_TRUE(execute().ok());const auto actual=half_test::read<T>(output);
    ASSERT_EQ(actual.size(),expected.size());for(std::size_t i=0;i<actual.size();++i)
      EXPECT_EQ(half_test::decode(actual[i]),half_test::decode(expected[i])*
          (repeat==1 && op.custom_as<c::model_ops::MlxQuantized>()->kind!=Kind::kEmbedding?2:1));
  }
  if(indexed){
    const auto before=half_test::read<T>(output);auto ids=half_test::read<std::int32_t>(buffers.back());
    ids[0]=-1;ASSERT_TRUE(buffers.back().view().write(std::as_bytes(std::span(ids))).ok());
    EXPECT_TRUE(execute().IsOutOfRange());const auto after=half_test::read<T>(output);
    for(std::size_t i=0;i<before.size();++i)EXPECT_EQ(half_test::decode(before[i]),half_test::decode(after[i]));
  }
}

template<class T> void matrices(){
  constexpr int k=64,n=3,e=2;
  const auto type=half_test::data_type<T>();
  for(const auto format:{Format::kAffine,Format::kMxfp4,Format::kMxfp8})
    for(const bool grouped:{false,true}){
      SCOPED_TRACE(static_cast<int>(format));
      SCOPED_TRACE(grouped);
      const bool affine=format==Format::kAffine;
      const int bits=format==Format::kMxfp8?8:4,pack=32/bits,group=affine?64:32;
      const int experts=grouped?e:1,rows=grouped?4:2;
      std::vector<std::uint32_t> words(experts*n*k/pack);
      for(std::size_t i=0;i<words.size();++i)words[i]=bits==8?0x4038b8c0u:(i%2?0xfedcba98u:0x76543210u);
      std::vector<T>a(rows*k),scale(experts*n*k/group),bias(scale.size());
      for(std::size_t i=0;i<a.size();++i)a[i]=half_test::encode<T>((int(i%7)-3)*0.25);
      std::vector<std::uint8_t> exponents(scale.size(),126);
      for(int expert=0;expert<experts;++expert)for(int row=0;row<n;++row)for(int b=0;b<k/group;++b){
        const auto index=(expert*n+row)*(k/group)+b;
        scale[index]=half_test::encode<T>((expert*n+row+1)*0.25);
        bias[index]=half_test::encode<T>((row-1)*0.125);
      }
      std::array<std::int32_t,4> ids{1,0,1,1};
      std::vector<T>expected(rows*n);
      constexpr std::array<float,16> fp4{0,.5f,1,1.5f,2,3,4,6,-0.f,-.5f,-1,-1.5f,-2,-3,-4,-6};
      constexpr std::array<float,4> fp8{-2,-1,1,2};
      for(int r=0;r<rows;++r)for(int row=0;row<n;++row){
        const int expert=grouped?ids[r]:0;double sum=0;
        for(int col=0;col<k;++col){
          const auto word=words[(expert*n+row)*(k/pack)+col/pack];
          const int code=int((word>>((col%pack)*bits))&((1u<<bits)-1));
          const auto index=(expert*n+row)*(k/group)+col/group;
          const double w=affine?code*half_test::decode(scale[index])+half_test::decode(bias[index]):
              (bits==4?fp4[code]:fp8[col%4])*0.5;
          sum+=half_test::decode(a[r*k+col])*w;
        }
        expected[r*n+row]=half_test::encode<T>(sum);
      }
      std::vector<c::TensorDesc> descriptions{
        grouped?desc(type,{2,2,k}):desc(type,{2,k}),
        grouped?desc(c::DataType::kUInt32,{e,n,k/pack}):desc(c::DataType::kUInt32,{n,k/pack}),
        grouped?desc(affine?type:c::DataType::kUInt8,{e,n,k/group}):desc(affine?type:c::DataType::kUInt8,{n,k/group})};
      std::vector<m::Buffer> buffers;buffers.push_back(upload<T>(a));buffers.push_back(upload<std::uint32_t>(words));
      buffers.push_back(affine?upload<T>(scale):upload<std::uint8_t>(exponents));
      if(affine){descriptions.push_back(descriptions[2]);buffers.push_back(upload<T>(bias));}
      if(grouped){descriptions.push_back(desc(c::DataType::kInt32,{2,2}));buffers.push_back(upload<std::int32_t>(ids));}
      run<T>(grouped?c::model_ops::mlx_grouped_matmul(format,bits,group):c::model_ops::mlx_quantized_matmul(format,bits,group),descriptions,buffers,expected,grouped);
    }
}

template<class T> void scale_boundaries(std::span<const std::uint8_t> exponents) {
  constexpr std::int64_t k=32;
  const auto type=half_test::data_type<T>();
  for (const auto exponent:exponents) {
    SCOPED_TRACE(exponent);
    const auto source_scale=half_test::encode<T>(std::ldexp(1.0, int(exponent)-127));
    const std::array expected{half_test::encode<T>(32*half_test::decode(source_scale))};
    for (const auto format:{Format::kMxfp4,Format::kMxfp8}) {
      SCOPED_TRACE(int(format));
      const auto pack=format==Format::kMxfp8?4:8;
      const auto word=format==Format::kMxfp8?0x38383838u:0x22222222u;
      const std::array descriptions{desc(type,{1,k}),desc(c::DataType::kUInt32,{1,k/pack}),
                                   desc(c::DataType::kUInt8,{1,1})};
      std::array buffers{upload<T>(std::vector<T>(k,half_test::encode<T>(1))),
          upload<std::uint32_t>(std::vector<std::uint32_t>(k/pack,word)),
          upload<std::uint8_t>(std::array{exponent})};
      run<T>(c::model_ops::mlx_quantized_matmul(format,format==Format::kMxfp8?8:4,32),descriptions,buffers,expected,false);
    }
  }
}

#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
TEST(CpuMlxQuantizedGraph, SharedActivationsUseOriginalPayloadAndRepeatedExpertIds) {
#else
TEST(CpuMlxQuantizedDirect, SharedActivationsUseOriginalPayloadAndRepeatedExpertIds) {
#endif
  for(const auto type:{c::DataType::kFloat32,c::DataType::kBFloat16})for(bool shared_view:{false,true}) {
    SCOPED_TRACE(static_cast<int>(type));
    SCOPED_TRACE(shared_view);
    const mlx_grouped_shared_test::Case data(type,shared_view);
    std::vector<m::Buffer> buffers;
    for(const auto &payload:data.payloads)buffers.push_back(upload<std::byte>(payload));
    if(type==c::DataType::kFloat32)run<float>(data.operation,data.descriptions,buffers,data.expected,true);
    else {
      std::vector<hyle::simd::bfloat16_t> expected;
      for(float value:data.expected)expected.push_back(half_test::encode<hyle::simd::bfloat16_t>(value));
      run<hyle::simd::bfloat16_t>(data.operation,data.descriptions,buffers,expected,true);
    }
  }
}

#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
TEST(CpuMlxQuantizedGraph, OriginalScaleCastPreservesUnderflowSubnormalAndOverflow){
#else
TEST(CpuMlxQuantizedDirect, OriginalScaleCastPreservesUnderflowSubnormalAndOverflow){
#endif
  scale_boundaries<hyle::simd::float16_t>(std::array<std::uint8_t,4>{102,103,127,143});
  scale_boundaries<hyle::simd::bfloat16_t>(std::array<std::uint8_t,3>{0,1,127});
  scale_boundaries<float>(std::array<std::uint8_t,3>{0,1,127});
}
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
TEST(CpuMlxQuantizedGraph, MixedFormatsAndSelectedExperts){
#else
TEST(CpuMlxQuantizedDirect, MixedFormatsAndSelectedExperts){
#endif
  matrices<float>();matrices<hyle::simd::bfloat16_t>();matrices<hyle::simd::float16_t>();
}

#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
TEST(CpuMlxQuantizedGraph, OriginalAffineEmbeddingUsesSelectedRows){
#else
TEST(CpuMlxQuantizedDirect, OriginalAffineEmbeddingUsesSelectedRows){
#endif
  using T=hyle::simd::bfloat16_t;
  constexpr int n=4,k=64;const auto type=half_test::data_type<T>();
  std::vector<std::uint32_t>words(n*k/8,0x76543210);
  std::vector<T>scales(n),bias(n),expected(3*k);
  for(int i=0;i<n;++i){scales[i]=half_test::encode<T>((i+1)*.25);bias[i]=half_test::encode<T>(i*.125);}
  const std::array<std::int32_t,3>ids{3,1,1};
  for(int r=0;r<3;++r)for(int col=0;col<k;++col)
    expected[r*k+col]=half_test::encode<T>((col%8)*half_test::decode(scales[ids[r]])+half_test::decode(bias[ids[r]]));
  const std::array descriptions{desc(c::DataType::kUInt32,{n,k/8}),desc(type,{n,1}),desc(type,{n,1}),desc(c::DataType::kInt32,{3})};
  std::array buffers{upload<std::uint32_t>(words),upload<T>(scales),upload<T>(bias),upload<std::int32_t>(ids)};
  run<T>(c::model_ops::mlx_quantized_embedding(4,64),descriptions,buffers,expected,true);
}

#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
TEST(CpuMlxQuantizedGraph, AffineMixedWidthsCrossWordsAndCompleteGroups) {
#else
TEST(CpuMlxQuantizedDirect, AffineMixedWidthsCrossWordsAndCompleteGroups) {
#endif
  constexpr int k = 256, n = 3, e = 3;
  for (const auto [bits, group] : {std::pair{5, 32}, std::pair{6, 64}, std::pair{8, 128}}) {
    SCOPED_TRACE(bits);
    SCOPED_TRACE(group);
    for (const bool grouped : {false, true}) {
      SCOPED_TRACE(grouped);
      const int experts = grouped ? e : 1, rows = grouped ? 4 : 2;
      const unsigned mask = (1u << bits) - 1;
      std::vector<std::uint32_t> codes(experts * n * k);
      for (std::size_t i = 0; i < codes.size(); ++i)
        codes[i] = static_cast<unsigned>(i * 13 + i / k * 7) & mask;
      for (int r = 0; r < experts * n; ++r)
        codes[r * k + (bits == 5 ? 6 : 5)] = mask;
      const auto words = mlx_affine_test::pack(codes, experts * n, k, bits);
      std::vector<float> activation(rows * k, 0), scales(experts * n * (k / group)), biases(scales.size());
      for (int r = 0; r < rows; ++r) for (int b = 0; b < k / group; ++b) {
        activation[r * k + b * group + 5] = float(r + 1) * 0.25f;
        activation[r * k + b * group + 6] = -0.5f;
      }
      for (std::size_t i = 0; i < scales.size(); ++i) {
        scales[i] = float(i % 3 + 1) * 0.25f;
        biases[i] = float(int(i % 5) - 2) * 0.125f;
      }
      const std::array<std::int32_t, 4> ids{2, 0, 2, 1};
      std::vector<float> expected(rows * n);
      for (int r = 0; r < rows; ++r) for (int col = 0; col < n; ++col) {
        const int ex = grouped ? ids[r] : 0;
        double sum = 0;
        for (int j = 0; j < k; ++j) {
          const auto meta = (ex * n + col) * (k / group) + j / group;
          sum += double(activation[r * k + j]) *
              (double(codes[(ex * n + col) * k + j]) * scales[meta] + biases[meta]);
        }
        expected[r * n + col] = static_cast<float>(sum);
      }
      std::vector<c::TensorDesc> descriptions{
          grouped ? desc(c::DataType::kFloat32, {2, 2, k}) : desc(c::DataType::kFloat32, {2, k}),
          grouped ? desc(c::DataType::kUInt32, {e, n, (k / 32) * bits}) : desc(c::DataType::kUInt32, {n, (k / 32) * bits}),
          grouped ? desc(c::DataType::kFloat32, {e, n, k / group}) : desc(c::DataType::kFloat32, {n, k / group})};
      descriptions.push_back(descriptions[2]);
      std::vector<m::Buffer> buffers;
      buffers.push_back(upload<float>(activation)); buffers.push_back(upload<std::uint32_t>(words));
      buffers.push_back(upload<float>(scales)); buffers.push_back(upload<float>(biases));
      if (grouped) { descriptions.push_back(desc(c::DataType::kInt32, {2, 2})); buffers.push_back(upload<std::int32_t>(ids)); }
      run<float>(grouped ? c::model_ops::mlx_grouped_matmul(Format::kAffine, bits, group)
                        : c::model_ops::mlx_quantized_matmul(Format::kAffine, bits, group),
                 descriptions, buffers, expected, grouped);
    }
  }
}

#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
TEST(CpuMlxQuantizedGraph, AffineMixedWidthEmbeddingKeepsTypedRounding) {
#else
TEST(CpuMlxQuantizedDirect, AffineMixedWidthEmbeddingKeepsTypedRounding) {
#endif
  using T = hyle::simd::bfloat16_t;
  constexpr int n = 3, k = 256;
  for (const auto [bits, group] : {std::pair{5, 32}, std::pair{6, 64}, std::pair{8, 128}}) {
    SCOPED_TRACE(bits);
    SCOPED_TRACE(group);
    std::vector<std::uint32_t> codes(n * k);
    const unsigned mask = (1u << bits) - 1;
    for (std::size_t i = 0; i < codes.size(); ++i) codes[i] = static_cast<unsigned>(i * 19) & mask;
    const auto words = mlx_affine_test::pack(codes, n, k, bits);
    std::vector<T> scales(n * (k / group)), biases(scales.size()), expected(3 * k);
    for (std::size_t i = 0; i < scales.size(); ++i) {
      scales[i] = half_test::encode<T>(0.75); biases[i] = half_test::encode<T>(256 + int(i % 3) * 2);
    }
    const std::array<std::int32_t, 3> ids{2, 0, 2};
    for (int row = 0; row < 3; ++row) for (int col = 0; col < k; ++col) {
      const auto meta = ids[row] * (k / group) + col / group;
      const auto product = half_test::encode<T>(double(codes[ids[row] * k + col]) * half_test::decode(scales[meta]));
      expected[row * k + col] = half_test::encode<T>(half_test::decode(product) + half_test::decode(biases[meta]));
    }
    const auto type = half_test::data_type<T>();
    const std::array descriptions{desc(c::DataType::kUInt32, {n, (k / 32) * bits}),
        desc(type, {n, k / group}), desc(type, {n, k / group}), desc(c::DataType::kInt32, {3})};
    std::array buffers{upload<std::uint32_t>(words), upload<T>(scales), upload<T>(biases), upload<std::int32_t>(ids)};
    run<T>(c::model_ops::mlx_quantized_embedding(bits, group), descriptions, buffers, expected, true);
  }
}

#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
TEST(CpuMlxQuantizedGraph, LowTypeActivationF32CapacityIsCheckedBeforePreparation) {
#else
TEST(CpuMlxQuantizedDirect, LowTypeActivationF32CapacityIsCheckedBeforePreparation) {
#endif
  constexpr auto limit = (std::numeric_limits<std::int64_t>::max)();
  constexpr std::int64_t k = (limit / 3 / 128) * 128;
  static_assert(k > limit / static_cast<std::int64_t>(sizeof(float)));
  const auto op = c::model_ops::mlx_quantized_matmul(Format::kAffine, 5, 32);
  for (const auto type : {c::DataType::kFloat16, c::DataType::kBFloat16}) {
    const std::array descriptions{desc(type, {1, k}),
        desc(c::DataType::kUInt32, {1, (k / 32) * 5}),
        desc(type, {1, k / 32}), desc(type, {1, k / 32})};
    std::array<c::TensorDesc, 1> outputs;
    ASSERT_TRUE(c::infer_output_descriptions(op, descriptions, outputs).ok());
    for (const auto &input : descriptions)
      ASSERT_LE(input.required_bytes().value(), static_cast<std::uint64_t>(limit));
    // No payload is allocated: all original carriers fit the native byte
    // domain, but expanding the whole low-type activation to F32 does not.
#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
    namespace g = hyle::graph;
    auto context = g::cpu::create_context().value();
    auto builder = g::GraphBuilder::create().value();
    std::vector<g::ValueId> inputs;
    for (const auto &input : descriptions)
      inputs.push_back(builder.input(input).value());
    const auto node = builder.add_node(op, inputs).value();
    ASSERT_TRUE(builder.publish(builder.result(node).value()).ok());
    auto prepared = g::compile(std::move(builder).finish().value(), context);
#else
    auto context = c::cpu::create_context().value();
    auto prepared = c::prepare(context, op, descriptions, outputs, m::system_resource());
#endif
    ASSERT_FALSE(prepared);
    EXPECT_TRUE(prepared.error().IsOutOfRange()) << prepared.error().message();
    EXPECT_NE(prepared.error().message().find("activation F32 temporary"),
              std::string_view::npos);
  }
}

#if defined(HYLE_TEST_GRAPH_ROUTE_CPU)
TEST(CpuMlxQuantizedGraph, EmptyEmbeddingHasNoSelectedRows) {
#else
TEST(CpuMlxQuantizedDirect, EmptyEmbeddingHasNoSelectedRows) {
#endif
  using T = hyle::simd::bfloat16_t;
  constexpr int n = 2, k = 64, group = 32;
  const auto type = half_test::data_type<T>();
  for (const unsigned bits : {5u, 6u}) {
    SCOPED_TRACE(bits);
    const std::array descriptions{desc(c::DataType::kUInt32, {n, (k / 32) * bits}),
        desc(type, {n, k / group}), desc(type, {n, k / group}),
        desc(c::DataType::kInt32, {0})};
    std::array buffers{
        upload<std::uint32_t>(std::vector<std::uint32_t>(n * (k / 32) * bits, ~0u)),
        upload<T>(std::vector<T>(n * k / group, half_test::encode<T>(1))),
        upload<T>(std::vector<T>(n * k / group, half_test::encode<T>(0))),
        upload<std::int32_t>(std::vector<std::int32_t>{})};
    run<T>(c::model_ops::mlx_quantized_embedding(bits, group), descriptions,
           buffers, std::span<const T>{}, false);
  }
}
} // namespace
