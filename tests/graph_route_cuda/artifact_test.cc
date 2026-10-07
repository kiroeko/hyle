#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <vector>

#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/cuda.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/torch_cuda.h"
#include "hyle/graph/cuda.h"
#include "hyle/compute/backends/cuda.h"

#if defined(HYLE_COMPUTE_TORCH_CUDA)
namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace mem = hyle::mem;
using Format = c::model_ops::WeightArtifactFormat;
using Half = hyle::simd::float16_t;
using Bfloat = hyle::simd::bfloat16_t;
struct Input { c::TensorDesc desc; std::vector<std::byte> bytes; };
template<class T> std::vector<std::byte> bytes(const std::vector<T> &values) {
  auto data = std::as_bytes(std::span(values)); return {data.begin(), data.end()};
}
float rounded(c::DataType type, float value) {
  if (type == c::DataType::kFloat16) return float(half_test::decode(half_test::encode<Half>(value)));
  if (type == c::DataType::kBFloat16) return float(half_test::decode(half_test::encode<Bfloat>(value)));
  return value;
}
std::vector<std::byte> floating(c::DataType type, const std::vector<float> &values) {
  if (type == c::DataType::kFloat16) {
    std::vector<Half> v; for (float x : values) v.push_back(half_test::encode<Half>(x)); return bytes(v);
  }
  if (type == c::DataType::kBFloat16) {
    std::vector<Bfloat> v; for (float x : values) v.push_back(half_test::encode<Bfloat>(x)); return bytes(v);
  }
  return bytes(values);
}
Input input(c::DataType type, std::initializer_list<std::int64_t> dims, std::vector<std::byte> payload) {
  return {c::TensorDesc::dense(type, {dims.begin(), dims.size()}).value(), std::move(payload)};
}
class CudaArtifactLifecycle : public ::testing::TestWithParam<std::tuple<Format,c::DataType>> {};
TEST_P(CudaArtifactLifecycle, ChangedPayloadTwoSlotsAndRetainedOwnersUseOriginalPrecision) {
  const auto [format, type] = GetParam();
  constexpr std::int64_t n = 16, k = 256, group = 64, groups = n*k/group;
  auto context = g::cuda::create_context().value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  auto queue = context.create_queue().value();
  std::vector<Input> source;
  const bool nf4 = format == Format::kNf4 || format == Format::kNf4DoubleQuant;
  if (nf4) {
    source.push_back(input(c::DataType::kUInt8, {n,k/2}, bytes(std::vector<std::uint8_t>(n*k/2,0xf0))));
    if (format == Format::kNf4DoubleQuant) {
      source.push_back(input(c::DataType::kUInt8, {groups}, bytes(std::vector<std::uint8_t>(groups,128))));
      source.push_back(input(c::DataType::kFloat32, {1}, bytes(std::vector<float>{.5f})));
      std::vector<float> nested(256); for(int i=0;i<256;++i)nested[i]=float(i)/256;
      source.push_back(input(c::DataType::kFloat32, {256}, bytes(nested)));
      source.push_back(input(c::DataType::kFloat32, {}, bytes(std::vector<float>{.125f})));
    } else source.push_back(input(c::DataType::kFloat32, {groups}, bytes(std::vector<float>(groups,.375f))));
    source.push_back(input(c::DataType::kFloat32, {16}, bytes(std::vector<float>{
        -1.f,-.6961928f,-.52507305f,-.3949175f,-.28444138f,-.18477343f,-.09105004f,0.f,
        .0795803f,.1609302f,.2461123f,.33791524f,.44070983f,.562617f,.72295684f,1.f})));
  } else if (format == Format::kAwqGemm4) {
    source.push_back(input(c::DataType::kInt32, {k,n/8}, bytes(std::vector<std::int32_t>(n*k/8,0x76543210))));
    source.push_back(input(c::DataType::kInt32, {k/group,n/8}, bytes(std::vector<std::int32_t>(n*k/group/8,0x33333333))));
    source.push_back(input(type, {k/group,n}, floating(type,std::vector<float>(n*k/group,.1f))));
  } else {
    source.push_back(input(c::DataType::kInt32, {n,k/8}, bytes(std::vector<std::int32_t>(n*k/8,0x76543210))));
    source.push_back(input(type, {n,k/group}, floating(type,std::vector<float>(n*k/group,.1f))));
  }
  std::vector<c::TensorDesc> descriptors; std::vector<mem::Buffer> storage;
  std::vector<c::ConstTensorView> views;
  for (const auto &value : source) {
    descriptors.push_back(value.desc);
    storage.push_back(mem::Buffer::create(memory.ref(),value.bytes.size(),256).value());
    ASSERT_TRUE(storage.back().view().write(value.bytes).ok());
    views.push_back(c::ConstTensorView::create(value.desc,storage.back().view()).value());
  }
  const auto operation = c::model_ops::artifact_weight_decode(format,group,type);
  std::array<c::TensorDesc,1> output_desc;
  ASSERT_TRUE(c::infer_output_descriptions(operation,descriptors,output_desc).ok());
  auto prepare_result = c::prepare(context,operation,descriptors,output_desc,memory.ref());
  ASSERT_TRUE(prepare_result) << prepare_result.error().message();
  auto prepared = std::move(*prepare_result);
  EXPECT_EQ(prepared.provider().id,c::torch_cuda::kProviderId);
  EXPECT_EQ(prepared.runtime_properties().replay_class,c::ProviderReplayClass::kManaged);
  EXPECT_FALSE(prepared.runtime_properties().capture_compatible);
  auto direct = prepared.create_invocation().value();
  auto builder = g::GraphBuilder::create().value();
  std::vector<g::ValueId> ids;
  for (const auto &desc : descriptors) ids.push_back(builder.input(desc).value());
  const auto node = builder.add_node(operation,ids).value();
  const auto result_id = builder.result(node).value();
  ASSERT_TRUE(builder.publish(result_id).ok());
  auto logical = std::move(builder).finish().value();
  auto compile_result = g::cuda::compile(logical,context,memory.ref());
  ASSERT_TRUE(compile_result) << compile_result.error().message();
  auto compiled = std::move(*compile_result);
  EXPECT_EQ(compiled.report.step(0)->provider.id,c::torch_cuda::kProviderId);
  auto slot = compiled.executable.create_invocation().value();
  auto second = compiled.executable.create_invocation().value();
  std::array<mem::Buffer,3> outputs;
  for (auto &buffer : outputs) buffer = mem::Buffer::create(memory.ref(),output_desc[0].required_bytes().value(),256).value();
  for (std::size_t i=0;i<ids.size();++i) {
    ASSERT_TRUE(slot.bind(ids[i],views[i]).ok()); ASSERT_TRUE(second.bind(ids[i],views[i]).ok());
  }
  ASSERT_TRUE(slot.bind(result_id,c::TensorView::create(output_desc[0],outputs[1].view()).value()).ok());
  ASSERT_TRUE(second.bind(result_id,c::TensorView::create(output_desc[0],outputs[2].view()).value()).ok());
  ASSERT_TRUE(g::cuda::prepare_replay(slot,queue).ok());
  EXPECT_EQ(slot.replay_route(),g::ReplayRoute::kCommands);
  // Context handle can disappear; program/queue/resource keep its backend alive.
  context = {};
  for (int pass : {0,1}) {
    if (pass) {
      if (nf4) {
        const auto payload=bytes(std::vector<std::uint8_t>(n*k/2,0x0f));
        ASSERT_TRUE(storage[0].view().write(payload).ok());
      } else {
        const auto payload=floating(type,std::vector<float>(n*k/group,.3f));
        ASSERT_TRUE(storage.back().view().write(payload).ok());
      }
    }
    const std::array destination{c::TensorView::create(output_desc[0],outputs[0].view()).value()};
    auto direct_result=direct.submit(queue,views,destination);
    ASSERT_TRUE(direct_result) << direct_result.error().message();
    auto done=std::move(*direct_result); ASSERT_TRUE(done.wait().ok());
    auto first_result=slot.submit(queue); ASSERT_TRUE(first_result) << first_result.error().message();
    auto second_result=second.submit(queue); ASSERT_TRUE(second_result) << second_result.error().message();
    auto first_done=std::move(*first_result); auto second_done=std::move(*second_result);
    ASSERT_TRUE(first_done.wait().ok()); ASSERT_TRUE(second_done.wait().ok());
    std::vector<float> expected(n*k);
    constexpr std::array<int,8> order{0,4,1,5,2,6,3,7};
    const float scale=rounded(type,pass?.3f:.1f);
    for(std::int64_t row=0;row<n;++row)for(std::int64_t col=0;col<k;++col) {
      float value;
      if(nf4)value=((col%2==0)^bool(pass))?.375f:-.375f;
      else value=float(format==Format::kAwqGemm4?order[row%8]-3:int(col%8)-8)*scale;
      expected[row*k+col]=rounded(type,value);
    }
    const auto expected_bytes=floating(type,expected);
    for(const auto &buffer:outputs) {
      std::vector<std::byte> actual(buffer.size()); ASSERT_TRUE(buffer.view().read(actual).ok());
      EXPECT_EQ(actual,expected_bytes);
    }
  }
}
INSTANTIATE_TEST_SUITE_P(OriginalPrecision,CudaArtifactLifecycle,
  ::testing::Combine(::testing::Values(Format::kNf4,Format::kNf4DoubleQuant,Format::kAwqGemm4,Format::kCompressedTensorsSymmetric4),
                    ::testing::Values(c::DataType::kFloat16,c::DataType::kBFloat16,c::DataType::kFloat32)));
} // namespace
#endif
