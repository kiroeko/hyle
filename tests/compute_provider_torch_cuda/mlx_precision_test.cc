#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include <ATen/Context.h>

#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/cuda.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/torch_cuda.h"

namespace {
namespace c=hyle::compute;
namespace mem=hyle::mem;
using Format=c::model_ops::MlxWeightFormat;
using Half=hyle::simd::float16_t;
using Bfloat=hyle::simd::bfloat16_t;
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
class CudaMlxPrecision:public ::testing::TestWithParam<std::tuple<Format,c::DataType>>{};
TEST_P(CudaMlxPrecision, SourceF32EquationAndGroupScaleArePreservedUnderForeignTf32Policy) {
  const auto [format,type]=GetParam();
  const bool affine=format==Format::kAffine4;
  const auto group=affine?64:32,pack=format==Format::kMxfp8?4:8;
  constexpr std::int64_t n=8,k=64;
  auto context=c::cuda::create_context().value();
  auto resource=c::cuda::MemoryResource::create(context,c::cuda::MemoryKind::kDevice).value();
  auto queue=context.create_queue().value();
  const auto dense=[](c::DataType t,std::initializer_list<std::int64_t> dimensions) {
    return c::TensorDesc::dense(t,{dimensions.begin(),dimensions.size()}).value();
  };
  std::vector<c::TensorDesc> descriptors{
    dense(type,{1,k}),dense(c::DataType::kUInt32,{n,k/pack}),
    dense(affine?type:c::DataType::kUInt8,{n,k/group})};
  if(affine)descriptors.push_back(descriptors.back());
  std::vector<std::vector<std::byte>> payloads;
  std::vector<float> activation(k);activation[0]=1;activation[1]=-1.375f;
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
  const auto operation=c::model_ops::mlx_quantized_matmul(format,group);
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
  const auto expected=affine?rounded(type,(7*scale+bias)-1.375f*(5*scale+bias)):
      rounded(type,format==Format::kMxfp8?1-1.375f*2:6-1.375f*3);
  const auto expected_bytes=floating(type,std::vector<float>(n,expected));
  std::vector<std::byte> actual(output.size());ASSERT_TRUE(output.view().read(actual).ok());
  EXPECT_EQ(actual,expected_bytes);
}
INSTANTIATE_TEST_SUITE_P(OriginalDtypes,CudaMlxPrecision,
  ::testing::Combine(::testing::Values(Format::kAffine4,Format::kMxfp4,Format::kMxfp8),
                    ::testing::Values(c::DataType::kFloat16,c::DataType::kBFloat16,c::DataType::kFloat32)));
} // namespace
