#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include <ATen/Context.h>

#include "../compute_provider_native/half_test_support.h"
#include "../compute_models/mlx_grouped_shared_fixture.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/cuda.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/torch_cuda.h"
#include "hyle/graph/cuda.h"
#include "../compute_models/mlx_moe_fixture.h"

namespace {
namespace c=hyle::compute;
namespace mem=hyle::mem;
using Format=c::model_ops::MlxWeightFormat;
using Half=hyle::simd::float16_t;
using Bfloat=hyle::simd::bfloat16_t;
TEST(CudaMlxMoeGraph, MixedBitMoeKeepsTypedMetadataAndIndependentSlots) {
  auto context = hyle::graph::cuda::create_context().value();
  auto resource = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  const auto owner = c::StorageOwner::adopt(context, std::move(resource)).value();
  auto queue = context.create_queue().value();
  const std::array queues{hyle::graph::QueueBinding{hyle::graph::ExecutionLaneId{0}, &queue}};
  mlx_moe_test::run(owner.ref(), queues, [&](const hyle::graph::Graph &graph,
      hyle::graph::CompileOptions options) { return hyle::graph::cuda::compile(graph, context, owner, options); },
      [](mem::BufferView destination, std::span<const std::byte> source) { return destination.write(source); },
      [](mem::ConstBufferView source, std::span<std::byte> destination) { return source.read(destination); }, owner);
}
template<class T> std::vector<std::byte> bytes(const std::vector<T> &v) {
  const auto view=std::as_bytes(std::span(v));return {view.begin(),view.end()};
}
float rounded(c::DataType type,float x) {
  if(type==c::DataType::kFloat16)return float(half_test::decode(half_test::encode<Half>(x)));
  if(type==c::DataType::kBFloat16)return float(half_test::decode(half_test::encode<Bfloat>(x)));
  return x;
}
std::vector<std::byte> floating(c::DataType type,const std::vector<float> &v) {
  if(type==c::DataType::kFloat16) {
    std::vector<Half> result;for(float x:v)result.push_back(half_test::encode<Half>(x));return bytes(result);
  }
  if(type==c::DataType::kBFloat16) {
    std::vector<Bfloat> result;for(float x:v)result.push_back(half_test::encode<Bfloat>(x));return bytes(result);
  }
  return bytes(v);
}
class OriginalTf32Policy final {
public:
  OriginalTf32Policy():previous_(at::globalContext().allowTF32CuBLAS()) {
    at::globalContext().setAllowTF32CuBLAS(true);
  }
  ~OriginalTf32Policy(){at::globalContext().setAllowTF32CuBLAS(previous_);}
private:
  bool previous_;
};
class CudaMlxPrecision:public ::testing::TestWithParam<std::tuple<Format,c::DataType,std::int64_t>>{};
TEST_P(CudaMlxPrecision, SourceF32EquationAndGroupScaleArePreservedUnderForeignTf32Policy) {
  const auto [format,type,rows]=GetParam();
  const bool affine=format==Format::kAffine;
  const auto group=affine?64:32,pack=format==Format::kMxfp8?4:8;
  constexpr std::int64_t n=8,k=64;
  auto context=c::cuda::create_context().value();
  auto resource=c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice).value();
  auto queue=context.create_queue().value();
  const auto dense=[](c::DataType t,std::initializer_list<std::int64_t> dimensions) {
    return c::TensorDesc::dense(t,{dimensions.begin(),dimensions.size()}).value();
  };
  std::vector<c::TensorDesc> descriptors{
    dense(type,{rows,k}),dense(c::DataType::kUInt32,{n,k/pack}),
    dense(affine?type:c::DataType::kUInt8,{n,k/group})};
  if(affine)descriptors.push_back(descriptors.back());
  std::vector<std::vector<std::byte>> payloads;
  constexpr std::array factors{1.F,-1.F,2.F,-2.F,.5F};
  std::vector<float> activation(rows*k);
  for(std::int64_t row=0;row<rows;++row) {
    activation[row*k]=factors[row];activation[row*k+1]=-1.375f*factors[row];
  }
  payloads.push_back(floating(type,activation));
  const auto word=format==Format::kMxfp8?0x00004038U:0x00000057U;
  payloads.push_back(bytes(std::vector<std::uint32_t>(n*k/pack,word)));
  payloads.push_back(affine?floating(type,std::vector<float>(n*k/group,.1f)):
      bytes(std::vector<std::uint8_t>(n*k/group,127)));
  if(affine)payloads.push_back(floating(type,std::vector<float>(n*k/group,.03f)));
  std::vector<mem::Buffer> buffers;std::vector<c::ConstTensorView> inputs;
  for(std::size_t i=0;i<descriptors.size();++i) {
    buffers.push_back(mem::Buffer::create(resource.ref(),payloads[i].size(),256).value());
    ASSERT_TRUE(buffers.back().view().write(payloads[i]).ok());
    inputs.push_back(c::ConstTensorView::create(descriptors[i],buffers.back().view()).value());
  }
  const auto operation=c::model_ops::mlx_quantized_matmul(format,format==Format::kMxfp8?8:4,group);
  std::array<c::TensorDesc,1> descriptions;
  ASSERT_TRUE(c::infer_output_descriptions(operation,descriptors,descriptions).ok());
  auto prepared_result=c::prepare(context,operation,descriptors,descriptions,resource.ref());
  ASSERT_TRUE(prepared_result)<<prepared_result.error().message();
  auto prepared=std::move(*prepared_result);
  EXPECT_EQ(prepared.provider().id,c::torch_cuda::kProviderId);
  auto invocation=prepared.create_invocation().value();
  auto output=mem::Buffer::create(resource.ref(),descriptions[0].required_bytes().value(),256).value();
  const std::array destinations{c::TensorView::create(descriptions[0],output.view()).value()};
  {
    OriginalTf32Policy foreign_policy;
    auto submitted=invocation.submit(queue,inputs,destinations);
    ASSERT_TRUE(submitted)<<submitted.error().message();
    const auto status=submitted->wait();ASSERT_TRUE(status.ok())<<status.message();
    EXPECT_TRUE(at::globalContext().allowTF32CuBLAS());
  }
  const auto scale=rounded(type,.1f),bias=rounded(type,.03f);
  const auto source=affine?(7*scale+bias)-1.375f*(5*scale+bias):
      format==Format::kMxfp8?1-1.375f*2:6-1.375f*3;
  std::vector<float> expected(rows*n);
  for(std::int64_t row=0;row<rows;++row)
    for(std::int64_t column=0;column<n;++column)
      expected[row*n+column]=rounded(type,source*factors[row]);
  const auto expected_bytes=floating(type,expected);
  std::vector<std::byte> actual(output.size());ASSERT_TRUE(output.view().read(actual).ok());
  if(type==c::DataType::kFloat32) {
    // The original contract permits F32 reduction association, not bitwise
    // SIMD lane order. Charge cancellation against the two source terms.
    std::vector<float> result(rows*n);ASSERT_TRUE(output.view().read(std::as_writable_bytes(std::span(result))).ok());
    const auto magnitude=affine?std::abs(7*scale+bias)+1.375f*std::abs(5*scale+bias):8.0f;
    for(std::int64_t row=0;row<rows;++row)for(std::int64_t column=0;column<n;++column)
      EXPECT_NEAR(result[row*n+column],expected[row*n+column],
          8*std::numeric_limits<float>::epsilon()*magnitude*std::abs(factors[row]));
  } else EXPECT_EQ(actual,expected_bytes);
  if(!affine) {
    for(const int exponent:{0,143,255}) {
      SCOPED_TRACE(int(exponent));
      std::vector<std::uint8_t> codes(n*k/group,127);
      for(std::int64_t row=0;row<n;++row)codes[row*(k/group)]=static_cast<std::uint8_t>(exponent);
      ASSERT_TRUE(buffers[2].view().write(bytes(codes)).ok());
      auto submitted=invocation.submit(queue,inputs,destinations);
      ASSERT_TRUE(submitted)<<submitted.error().message();
      const auto status=submitted->wait();ASSERT_TRUE(status.ok())<<status.message();
      const auto source_scale=rounded(type,std::ldexp(1.0f,int(exponent)-127));
      for(std::int64_t row=0;row<rows;++row) {
        auto boundary=rounded(type,(format==Format::kMxfp8?-1.75f:1.875f)*factors[row]*source_scale);
        // Adding a negative-zero group to the initial positive zero gives +0.
        if(boundary==0)boundary=0;
        for(std::int64_t column=0;column<n;++column)expected[row*n+column]=boundary;
      }
      std::vector<std::byte> result(output.size());ASSERT_TRUE(output.view().read(result).ok());
      EXPECT_EQ(result,floating(type,expected));
    }
  }
}
INSTANTIATE_TEST_SUITE_P(OriginalDtypes,CudaMlxPrecision,
  ::testing::Combine(::testing::Values(Format::kAffine,Format::kMxfp4,Format::kMxfp8),
                    ::testing::Values(c::DataType::kFloat16,c::DataType::kBFloat16,c::DataType::kFloat32),
                    ::testing::Values(std::int64_t{1},std::int64_t{5})));

TEST(CudaMlxSharedActivation, OriginalPayloadStridesAndCheckedIdsSurviveReuse) {
  auto context=c::cuda::create_context().value();
  auto resource=c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice).value();
  auto queue=context.create_queue().value();
  for(const auto type:{c::DataType::kFloat32,c::DataType::kBFloat16})for(bool shared_view:{false,true}) {
    SCOPED_TRACE(static_cast<int>(type));
    SCOPED_TRACE(shared_view);
    const mlx_grouped_shared_test::Case data(type,shared_view);
    std::vector<mem::Buffer> buffers;std::vector<c::ConstTensorView> inputs;
    for(std::size_t port=0;port<data.payloads.size();++port) {
      buffers.push_back(mem::Buffer::create(resource.ref(),data.payloads[port].size(),256).value());
      ASSERT_TRUE(buffers.back().view().write(data.payloads[port]).ok());
      inputs.push_back(c::ConstTensorView::create(data.descriptions[port],buffers.back().view()).value());
    }
    std::array<c::TensorDesc,1> out;
    const auto inferred=c::infer_output_descriptions(data.operation,data.descriptions,out);
    ASSERT_TRUE(inferred.ok())<<inferred.message();
    EXPECT_EQ(out[0].dimensions()[1],3);
    auto prepared=c::prepare(context,data.operation,data.descriptions,out,resource.ref());
    ASSERT_TRUE(prepared)<<prepared.error().message();
    auto slot=prepared->create_invocation().value();
    auto output=mem::Buffer::create(resource.ref(),out[0].required_bytes().value(),256).value();
    const std::array outputs{c::TensorView::create(out[0],output.view()).value()};
    for(unsigned repeat=0;repeat<2;++repeat) {
      if(repeat) {
        const auto count=buffers[0].size()/c::data_type_size(type);
        std::vector<float> a(count,-777.F);
        std::fill_n(a.begin(),64,2.F);std::fill_n(a.begin()+(shared_view?80:64),64,4.F);
        ASSERT_TRUE(buffers[0].view().write(floating(type,a)).ok());
      }
      auto done=slot.submit(queue,inputs,outputs);ASSERT_TRUE(done)<<done.error().message();
      const auto status=done->wait();ASSERT_TRUE(status.ok())<<status.message();
      auto expected=data.expected;for(auto &value:expected)value*=repeat?2.F:1.F;
      std::vector<std::byte> actual(output.size());ASSERT_TRUE(output.view().read(actual).ok());
      EXPECT_EQ(actual,floating(type,expected));
    }
    std::vector<std::byte> before(output.size());ASSERT_TRUE(output.view().read(before).ok());
    const std::array<std::int32_t,6> invalid{-1,0,2,1,1,0};
    ASSERT_TRUE(buffers.back().view().write(std::as_bytes(std::span(invalid))).ok());
    auto failed=slot.submit(queue,inputs,outputs);ASSERT_TRUE(failed)<<failed.error().message();
    EXPECT_TRUE(failed->wait().IsOutOfRange());
    std::vector<std::byte> after(output.size());ASSERT_TRUE(output.view().read(after).ok());EXPECT_EQ(after,before);
  }
}
} // namespace
