#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include <c10/cuda/CUDACachingAllocator.h>

#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/torch_cuda.h"

namespace {
namespace c = hyle::compute;
namespace mem = hyle::mem;
template<class T> auto bytes(const std::vector<T> &values) {
  const auto view = std::as_bytes(std::span(values));
  return std::vector<std::byte>(view.begin(), view.end());
}
c::TensorDesc dense(c::DataType type, std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}
struct InputSet {
  // The kernel unit tests need only its real public Provider. Natural default
  // Direct/Graph selection is exercised by the full-width N2 model fixtures.
  c::Context context = [] {
    const std::array providers{c::ProviderBinding{c::torch_cuda::provider(),0}};
    return c::Context::create(c::cuda::device(0).value(),{.providers=providers}).value();
  }();
  c::cuda::MemoryResource resource = c::cuda::MemoryResource::create(context,
      c::cuda::MemoryKind::kDevice).value();
  c::Queue queue = context.create_queue().value();
  std::vector<c::TensorDesc> descriptions;
  std::vector<mem::Buffer> buffers;
  std::vector<c::ConstTensorView> views;
  InputSet(std::vector<c::TensorDesc> desc, const std::vector<std::vector<std::byte>> &payloads)
      : descriptions(std::move(desc)) {
    for (std::size_t i = 0; i < descriptions.size(); ++i) {
      buffers.push_back(mem::Buffer::create(resource.ref(), payloads[i].size(), 256).value());
      EXPECT_TRUE(buffers.back().view().write(payloads[i]).ok());
      views.push_back(c::ConstTensorView::create(descriptions[i], buffers.back().view()).value());
    }
  }
};
class NoTorchAllocation final {
public:
  NoTorchAllocation() {
    // Native packed preparation deliberately never enters ATen. Initialize
    // only the original allocator metadata before applying its real limit.
    c10::cuda::CUDACachingAllocator::init(1);
    previous_=c10::cuda::CUDACachingAllocator::getMemoryFraction(0);
    c10::cuda::CUDACachingAllocator::emptyCache();
    c10::cuda::CUDACachingAllocator::setMemoryFraction(0, 0);
  }
  ~NoTorchAllocation() { c10::cuda::CUDACachingAllocator::setMemoryFraction(previous_, 0); }
private:
  double previous_=1;
};

TEST(CudaPackedWeights, Grouped64StepsObserveChangedWordsAndIdsWithoutTorchTemporaries) {
  constexpr std::int64_t rows=4,n=9,k=128,experts=2,group=64;
  std::vector<float> activation(rows*k, 0);
  for (int row=0;row<rows;++row) {
    activation[row*k]=1; activation[row*k+1]=-1.375F; activation[row*k+65]=.75F;
  }
  std::vector<std::uint32_t> words(experts*n*k/8, 0x55555555U);
  std::vector<float> scales(experts*n*k/group,.1F),biases(scales.size(),.03F);
  std::vector<std::int32_t> ids{1,0,1,0};
  InputSet data({dense(c::DataType::kFloat32,{2,2,k}),
      dense(c::DataType::kUInt32,{experts,n,k/8}),
      dense(c::DataType::kFloat32,{experts,n,k/group}),
      dense(c::DataType::kFloat32,{experts,n,k/group}),
      dense(c::DataType::kInt32,{2,2})},
      {bytes(activation),bytes(words),bytes(scales),bytes(biases),bytes(ids)});
  const auto op=c::model_ops::mlx_grouped_matmul(c::model_ops::MlxWeightFormat::kAffine4,group);
  std::array<c::TensorDesc,1> output_desc;
  ASSERT_TRUE(c::infer_output_descriptions(op,data.descriptions,output_desc).ok());
  // A real original allocator limit catches any return to per-group ATen
  // unpack/index_select/matmul even when the finite numerical answer matches.
  NoTorchAllocation no_torch;
  auto prepared=c::prepare(data.context,op,data.descriptions,output_desc,data.resource.ref()).value();
  ASSERT_EQ(prepared.provider().id,c::torch_cuda::kProviderId);
  EXPECT_FALSE(prepared.runtime_properties().blocks_host);
  auto first=prepared.create_invocation().value(),second=prepared.create_invocation().value();
  std::vector<mem::Buffer> outputs;
  for(int i=0;i<3;++i)outputs.push_back(mem::Buffer::create(data.resource.ref(),rows*n*sizeof(float),256).value());
  std::vector<float> old_result;
  for(int step=0;step<64;++step) {
    SCOPED_TRACE(step);
    for(std::size_t i=0;i<words.size();++i)words[i]=0x11111111U*unsigned((i+step)%16);
    for(int row=0;row<rows;++row)ids[row]=(row+step)%2;
    ASSERT_TRUE(data.buffers[1].view().write(bytes(words)).ok());
    ASSERT_TRUE(data.buffers[4].view().write(bytes(ids)).ok());
    const auto slot=step==0?0:1+step%2;
    const std::array target{c::TensorView::create(output_desc[0],outputs[slot].view()).value()};
    auto submitted=(step%2?second:first).submit(data.queue,data.views,target);
    ASSERT_TRUE(submitted)<<submitted.error().message();
    const auto status=submitted->wait(); ASSERT_TRUE(status.ok())<<status.message();
    std::vector<float> result(rows*n);
    ASSERT_TRUE(outputs[slot].view().read(std::as_writable_bytes(std::span(result))).ok());
    for(int row=0;row<rows;++row)for(int column=0;column<n;++column) {
      double expected=0,magnitude=0;
      for(int j=0;j<k;++j) {
        const auto wrow=ids[row]*n+column;
        const float code=float((words[wrow*(k/8)+j/8]>>(4*(j%8)))&15U);
        const float decoded=code*scales[wrow*(k/group)+j/group]+biases[wrow*(k/group)+j/group];
        const double term=double(activation[row*k+j])*decoded;
        expected+=term; magnitude+=std::abs(term);
      }
      EXPECT_NEAR(result[row*n+column],expected,8*std::numeric_limits<float>::epsilon()*magnitude);
    }
    if(step==0)old_result=result;
  }
  std::vector<float> retained(rows*n);
  ASSERT_TRUE(outputs[0].view().read(std::as_writable_bytes(std::span(retained))).ok());
  EXPECT_EQ(retained,old_result);
  for(const std::int32_t bad:{-1,2,(std::numeric_limits<std::int32_t>::max)()}) {
    SCOPED_TRACE(bad);
    ids[1]=bad;
    ASSERT_TRUE(data.buffers[4].view().write(bytes(ids)).ok());
    std::vector<float> before(rows*n),after(rows*n);
    ASSERT_TRUE(outputs[1].view().read(std::as_writable_bytes(std::span(before))).ok());
    auto queue=data.context.create_queue().value();
    const std::array target{c::TensorView::create(output_desc[0],outputs[1].view()).value()};
    auto checked=prepared.create_invocation().value();
    auto submitted=checked.submit(queue,data.views,target);
    ASSERT_TRUE(submitted)<<submitted.error().message();
    EXPECT_TRUE(submitted->wait().IsOutOfRange());
    EXPECT_FALSE(checked.submit(queue,data.views,target));
    ASSERT_TRUE(outputs[1].view().read(std::as_writable_bytes(std::span(after))).ok());
    EXPECT_EQ(before,after);
  }
}

TEST(CudaPackedWeights, Mxfp8CoversAllCodesWithIndependentScalarOracle) {
  constexpr std::int64_t rows=5,n=256,k=32;
  constexpr std::array factors{1.F,-1.F,2.F,-2.F,.5F};
  std::vector<float> a(rows*k,0);
  for(std::int64_t row=0;row<rows;++row)a[row*k]=factors[row];
  std::vector<std::uint32_t> words(n*k/4,0);
  for(unsigned code=0;code<256;++code)words[code*(k/4)]=code;
  InputSet data({dense(c::DataType::kFloat32,{rows,k}),dense(c::DataType::kUInt32,{n,k/4}),
      dense(c::DataType::kUInt8,{n,1})},{bytes(a),bytes(words),bytes(std::vector<std::uint8_t>(n,127))});
  const auto op=c::model_ops::mlx_quantized_matmul(c::model_ops::MlxWeightFormat::kMxfp8,32);
  std::array<c::TensorDesc,1> output_desc;
  ASSERT_TRUE(c::infer_output_descriptions(op,data.descriptions,output_desc).ok());
  auto prepared=c::prepare(data.context,op,data.descriptions,output_desc,data.resource.ref()).value();
  auto invocation=prepared.create_invocation().value();
  auto output=mem::Buffer::create(data.resource.ref(),rows*n*sizeof(float),256).value();
  const std::array target{c::TensorView::create(output_desc[0],output.view()).value()};
  auto completion=invocation.submit(data.queue,data.views,target).value();ASSERT_TRUE(completion.wait().ok());
  std::vector<float> result(rows*n);ASSERT_TRUE(output.view().read(std::as_writable_bytes(std::span(result))).ok());
  for(std::int64_t row=0;row<rows;++row)for(unsigned code=0;code<256;++code) {
    SCOPED_TRACE(row);SCOPED_TRACE(code);const auto exponent=(code>>3)&15,mantissa=code&7;
    if(exponent==15&&mantissa==7) { EXPECT_TRUE(std::isnan(result[row*n+code]));continue; }
    const auto magnitude=exponent?std::ldexp(1.F+float(mantissa)/8,int(exponent)-7):std::ldexp(float(mantissa),-9);
    EXPECT_EQ(result[row*n+code],(code&128?-magnitude:magnitude)*factors[row]);
  }
}

TEST(CudaPackedWeights, EmbeddingPreservesStorageRoundingBeforeBiasAndCheckedOutput) {
  using Half=hyle::simd::float16_t;
  using Bfloat=hyle::simd::bfloat16_t;
  for(const auto type:{c::DataType::kFloat16,c::DataType::kBFloat16}) {
    SCOPED_TRACE(int(type));
    constexpr std::int64_t n=3,k=128,group=32;
    auto rounded=[&](float x) { return float(type==c::DataType::kFloat16
        ? half_test::decode(half_test::encode<Half>(x))
        : half_test::decode(half_test::encode<Bfloat>(x))); };
    auto floating=[&](const std::vector<float>& values) {
      std::vector<std::uint16_t> result;
      for(float x:values)result.push_back(type==c::DataType::kFloat16
          ? half_test::encode<Half>(x).bits : half_test::encode<Bfloat>(x).bits);
      return bytes(result);
    };
    std::vector<std::uint32_t> words(n*k/8,0x76543210U);
    std::vector<float> scales(n*k/group,.1F),biases(scales.size(),-.03F);
    std::vector<std::int32_t> ids{2,0,2};
    InputSet data({dense(c::DataType::kUInt32,{n,k/8}),dense(type,{n,k/group}),
        dense(type,{n,k/group}),dense(c::DataType::kInt32,{3})},
        {bytes(words),floating(scales),floating(biases),bytes(ids)});
    const auto op=c::model_ops::mlx_quantized_embedding(group);
    std::array<c::TensorDesc,1> desc;
    ASSERT_TRUE(c::infer_output_descriptions(op,data.descriptions,desc).ok());
    NoTorchAllocation no_torch;
    auto prepared=c::prepare(data.context,op,data.descriptions,desc,data.resource.ref()).value();
    auto invocation=prepared.create_invocation().value();
    auto output=mem::Buffer::create(data.resource.ref(),desc[0].required_bytes().value(),256).value();
    const std::array target{c::TensorView::create(desc[0],output.view()).value()};
    auto completion=invocation.submit(data.queue,data.views,target).value();
    ASSERT_TRUE(completion.wait().ok());
    std::vector<float> expected(3*k);
    for(int row=0;row<3;++row)for(int j=0;j<k;++j)
      expected[row*k+j]=rounded(rounded(float(j%8)*rounded(.1F))+rounded(-.03F));
    std::vector<std::byte> actual(output.size());ASSERT_TRUE(output.view().read(actual).ok());
    EXPECT_EQ(actual,floating(expected));
    ids[2]=-1;ASSERT_TRUE(data.buffers[3].view().write(bytes(ids)).ok());
    auto failed=invocation.submit(data.queue,data.views,target).value();
    EXPECT_TRUE(failed.wait().IsOutOfRange());
    std::vector<std::byte> retained(output.size());ASSERT_TRUE(output.view().read(retained).ok());
    EXPECT_EQ(actual,retained);
  }
}

TEST(CudaPackedWeights, MixedAffinePromotesWithoutRoundingDecodedWeights) {
  using Half=hyle::simd::float16_t;
  using Bfloat=hyle::simd::bfloat16_t;
  constexpr std::int64_t n=7,k=256;
  for(const unsigned group:{32U,64U,128U}) {
    SCOPED_TRACE(group);
    std::vector<Half> activation(k);
    activation[0]=half_test::encode<Half>(1);
    activation[1]=half_test::encode<Half>(-1.375);
    activation[129]=half_test::encode<Half>(.75);
    std::vector<Bfloat> scales(n*k/group,half_test::encode<Bfloat>(.1));
    std::vector<Bfloat> biases(scales.size(),half_test::encode<Bfloat>(.03));
    const std::vector<std::uint32_t> words(n*k/8,0x76543210U);
    InputSet data({dense(c::DataType::kFloat16,{1,k}),dense(c::DataType::kUInt32,{n,k/8}),
        dense(c::DataType::kBFloat16,{n,k/group}),dense(c::DataType::kBFloat16,{n,k/group})},
        {bytes(activation),bytes(words),bytes(scales),bytes(biases)});
    const auto op=c::model_ops::mlx_quantized_matmul(c::model_ops::MlxWeightFormat::kAffine4,group);
    std::array<c::TensorDesc,1> desc;
    ASSERT_TRUE(c::infer_output_descriptions(op,data.descriptions,desc).ok());
    ASSERT_EQ(desc[0].data_type(),c::DataType::kFloat32);
    NoTorchAllocation no_torch;
    auto prepared=c::prepare(data.context,op,data.descriptions,desc,data.resource.ref()).value();
    auto slot=prepared.create_invocation().value();
    auto output=mem::Buffer::create(data.resource.ref(),n*sizeof(float),256).value();
    const std::array target{c::TensorView::create(desc[0],output.view()).value()};
    auto work=slot.submit(data.queue,data.views,target).value();ASSERT_TRUE(work.wait().ok());
    std::vector<float> actual(n);ASSERT_TRUE(output.view().read(std::as_writable_bytes(std::span(actual))).ok());
    double expected=0,magnitude=0;
    for(int j=0;j<k;++j) {
      const float decoded=float(j%8)*float(half_test::decode(scales[j/group]))+float(half_test::decode(biases[j/group]));
      const double term=half_test::decode(activation[j])*decoded;expected+=term;magnitude+=std::abs(term);
    }
    for(float value:actual)EXPECT_NEAR(value,expected,8*std::numeric_limits<float>::epsilon()*magnitude);
  }
}
} // namespace
