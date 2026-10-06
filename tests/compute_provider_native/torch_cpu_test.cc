#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#include <c10/core/CPUAllocator.h>

#include "half_test_support.h"
#include "../compute/native_context_fixture.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/native.h"
#include "hyle/mem/quota_allocator.h"

namespace {
namespace c = hyle::compute;
namespace mem = hyle::mem;

// Isolated test process only. Preserve native allocation and deleter ownership;
// fail one allocation through the real pinned CPU allocator's failure path.
class AllocationFailure final : public at::Allocator {
public:
  at::Allocator *original = c10::GetCPUAllocator();
  std::atomic<bool> fail{false};
  at::DataPtr allocate(std::size_t bytes) override {
    return original->allocate(fail.exchange(false) ? std::size_t{1} << 62 : bytes);
  }
  void copy_data(void *dest, const void *src, std::size_t bytes) const override {
    original->copy_data(dest, src, bytes);
  }
};

AllocationFailure &allocation_failure() {
  static AllocationFailure allocation;
  return allocation;
}

TEST(NativeTorchCpu, ManagedReplayReportsRealAllocationFailureAndReleasesProgram) {
  auto &allocation = allocation_failure();
  c10::SetCPUAllocator(&allocation, 255);
  mem::QuotaAllocator metadata(mem::system_allocator(), 4 * 1024 * 1024);
  const auto initial = metadata.remaining();
  {
    auto context = c::test::native_context({.metadata_allocator = metadata.get()}).value();
    const auto desc = c::TensorDesc::dense(c::DataType::kBFloat16,
        std::array<std::int64_t, 2>{1, 128}).value();
    const std::array descriptors{desc};
    const auto op = c::model_ops::block_quantization_roundtrip(
        c::model_ops::BlockQuantizationFormat::kFloat8E4M3Block128);
    auto prepared = c::prepare(context, op, descriptors, descriptors, mem::system_resource()).value();
    EXPECT_EQ(prepared.provider().id, c::native::kProviderId);
    EXPECT_EQ(prepared.runtime_properties().replay_class, c::ProviderReplayClass::kManaged);
    EXPECT_TRUE(prepared.runtime_properties().uses_private_threads);
    EXPECT_TRUE(prepared.runtime_properties().blocks_host);
    EXPECT_FALSE(prepared.invocation_private_bytes());
    EXPECT_FALSE(prepared.capture_compatible());
    auto input = half_test::buffer<std::uint16_t>(std::vector<std::uint16_t>(128, 0x3f80));
    auto output = half_test::buffer<std::uint16_t>(std::vector<std::uint16_t>(128));
    const auto x = c::ConstTensorView::create(desc, input.view()).value();
    const auto y = c::TensorView::create(desc, output.view()).value();
    auto queue = context.create_queue().value();
    auto lane = prepared.create_invocation().value();
    allocation.fail = true;
    {
      auto done = lane.submit(queue, x, y);
      ASSERT_TRUE(done) << done.error().message();
      EXPECT_TRUE(done->wait().IsResourceExhausted());
    }
    EXPECT_FALSE(allocation.fail.load());
    EXPECT_TRUE(lane.submit(queue, x, y).error().IsFailedPrecondition());
    lane = prepared.create_invocation().value();
    for (int replay = 0; replay < 3; ++replay) {
      auto done = lane.submit(queue, x, y);
      ASSERT_TRUE(done);
      EXPECT_TRUE(done->wait().ok());
      for (auto value : half_test::read<std::uint16_t>(output)) EXPECT_EQ(value, 0x3f80);
    }
    {
      auto reserved = metadata.try_reserve_bytes(metadata.remaining()).value();
      auto failed = c::prepare(context, op, descriptors, descriptors, mem::system_resource());
      ASSERT_FALSE(failed);
      EXPECT_TRUE(failed.error().IsResourceExhausted());
    }
    EXPECT_TRUE(c::prepare(context, op, descriptors, descriptors, mem::system_resource()));
  }
  EXPECT_EQ(metadata.remaining(), initial);
}

TEST(NativeTorchCpu, LowPrecisionConvolutionHasRealFailureRecoveryAndIndependentSlots) {
  auto &allocation=allocation_failure();c10::SetCPUAllocator(&allocation,255);
  for(auto type:{c::DataType::kFloat16,c::DataType::kBFloat16}) {
    mem::QuotaAllocator metadata(mem::system_allocator(),4*1024*1024);
    const auto initial=metadata.remaining();
    {
      auto context=c::test::native_context({.metadata_allocator=metadata.get()}).value();
      auto samples=c::TensorDesc::dense(type,std::array<std::int64_t,3>{1,4,8}).value();
      auto weights=c::TensorDesc::dense(type,std::array<std::int64_t,2>{8,4}).value();
      auto result=c::TensorDesc::dense(type,std::array<std::int64_t,3>{1,1,8}).value();
      const std::array descs{samples,weights};const std::array outputs{result};
      auto prepared=c::prepare(context,c::model_ops::causal_conv1d(),descs,outputs,mem::system_resource()).value();
      EXPECT_EQ(prepared.implementation().value(),0x544f524348000009ULL);
      EXPECT_EQ(prepared.runtime_properties().replay_class,c::ProviderReplayClass::kManaged);
      EXPECT_FALSE(prepared.invocation_private_bytes());EXPECT_FALSE(prepared.capture_compatible());
      const std::uint16_t one=type==c::DataType::kFloat16?0x3c00:0x3f80;
      const std::uint16_t four=type==c::DataType::kFloat16?0x4400:0x4080;
      auto x=half_test::buffer<std::uint16_t>(std::vector<std::uint16_t>(32,one));
      auto w=half_test::buffer<std::uint16_t>(std::vector<std::uint16_t>(32,one));
      auto a=half_test::buffer<std::uint16_t>(std::vector<std::uint16_t>(8));
      auto b=half_test::buffer<std::uint16_t>(std::vector<std::uint16_t>(8));
      const std::array inputs{c::ConstTensorView::create(samples,x.view()).value(),c::ConstTensorView::create(weights,w.view()).value()};
      auto ya=c::TensorView::create(result,a.view()).value(),yb=c::TensorView::create(result,b.view()).value();
      auto queue=context.create_queue().value();auto lane=prepared.create_invocation().value();
      allocation.fail=true;auto failed=lane.submit(queue,inputs,std::span(&ya,1));ASSERT_TRUE(failed);
      EXPECT_TRUE(failed->wait().IsResourceExhausted());EXPECT_FALSE(allocation.fail.load());
      EXPECT_TRUE(lane.submit(queue,inputs,std::span(&ya,1)).error().IsFailedPrecondition());
      for(auto value:half_test::read<std::uint16_t>(a))EXPECT_EQ(value,0);
      lane=prepared.create_invocation().value();
      auto done=lane.submit(queue,inputs,std::span(&ya,1));ASSERT_TRUE(done);EXPECT_TRUE(done->wait().ok());
      std::thread first([&]{auto q=context.create_queue().value();auto run=prepared.create_invocation().value();
        auto completion=run.submit(q,inputs,std::span(&ya,1));EXPECT_TRUE(completion&&completion->wait().ok());});
      std::thread second([&]{auto q=context.create_queue().value();auto run=prepared.create_invocation().value();
        auto completion=run.submit(q,inputs,std::span(&yb,1));EXPECT_TRUE(completion&&completion->wait().ok());});
      first.join();second.join();
      for(auto value:half_test::read<std::uint16_t>(a))EXPECT_EQ(value,four);
      for(auto value:half_test::read<std::uint16_t>(b))EXPECT_EQ(value,four);
    }
    EXPECT_EQ(metadata.remaining(),initial);
  }
}

TEST(NativeTorchCpu, DuplicateExpertIdsAndFailedInvocationRecovery) {
  auto context = c::test::native_context().value();
  auto queue = context.create_queue().value();
  const auto w = c::float8_grouped_weight_description(2, 128, 128).value();
  const auto a = c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 3>{1, 1, 128}).value();
  const auto ids = c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 2>{1, 2}).value();
  const std::array inputs{w, a, ids};
  std::array<c::TensorDesc, 1> outputs;
  const auto op = c::model_ops::grouped_float8_linear(448.0F);
  ASSERT_TRUE(c::infer_output_descriptions(op, inputs, outputs).ok());
  auto prepared = c::prepare(context, op, inputs, outputs, mem::system_resource()).value();
  std::vector<std::byte> weights(w.required_bytes().value(), std::byte{0x38});
  const float scale = 1;
  std::memcpy(weights.data() + 2 * 128 * 128, &scale, 4);
  std::memcpy(weights.data() + 2 * 128 * 128 + 4, &scale, 4);
  auto wb = half_test::buffer<std::byte>(weights);
  auto ab = half_test::buffer<float>(std::vector<float>(128, 1));
  auto ib = half_test::buffer<std::int32_t>(std::vector<std::int32_t>{1, 1});
  auto ob = half_test::buffer<float>(std::vector<float>(256));
  const std::array x{c::ConstTensorView::create(w, wb.view()).value(),
      c::ConstTensorView::create(a, ab.view()).value(),
      c::ConstTensorView::create(ids, ib.view()).value()};
  const auto y = c::TensorView::create(outputs[0], ob.view()).value();
  auto lane = prepared.create_invocation().value();
  for (auto selected : {std::array<std::int32_t, 2>{1, 1}, {0, 1}, {0, 0}}) {
    ASSERT_TRUE(ib.view().write(std::as_bytes(std::span(selected))).ok());
    auto done = lane.submit(queue, x, std::span(&y, 1));
    ASSERT_TRUE(done);
    ASSERT_TRUE(done->wait().ok());
    for (auto value : half_test::read<float>(ob)) EXPECT_EQ(value, 128);
  }
  auto second = prepared.create_invocation().value();
  auto second_queue = context.create_queue().value();
  auto second_output = half_test::buffer<float>(std::vector<float>(256));
  const auto second_y = c::TensorView::create(outputs[0], second_output.view()).value();
  bool first_ok = false, second_ok = false;
  {
    std::jthread first_thread([&] {
      auto done = lane.submit(queue, x, std::span(&y, 1));
      first_ok = done && done->wait().ok();
    });
    std::jthread second_thread([&] {
      auto done = second.submit(second_queue, x, std::span(&second_y, 1));
      second_ok = done && done->wait().ok();
    });
  }
  EXPECT_TRUE(first_ok);
  EXPECT_TRUE(second_ok);
  for (auto value : half_test::read<float>(second_output)) EXPECT_EQ(value, 128);
  for (auto selected : {std::array<std::int32_t, 2>{-1, 0}, {0, 2}}) {
    ASSERT_TRUE(ib.view().write(std::as_bytes(std::span(selected))).ok());
    auto done = lane.submit(queue, x, std::span(&y, 1));
    ASSERT_TRUE(done);
    EXPECT_TRUE(done->wait().IsOutOfRange());
    lane = prepared.create_invocation().value();
  }
  const std::array<std::int32_t, 2> selected{0, 1};
  ASSERT_TRUE(ib.view().write(std::as_bytes(std::span(selected))).ok());
  auto nonfinite = std::vector<float>(128, 1);
  nonfinite[127] = std::numeric_limits<float>::infinity();
  ASSERT_TRUE(ab.view().write(std::as_bytes(std::span(nonfinite))).ok());
  auto done = lane.submit(queue, x, std::span(&y, 1));
  ASSERT_TRUE(done);
  EXPECT_TRUE(done->wait().IsInvalidArgument());
}

TEST(NativeTorchCpu, ArtifactAndMlxProgramsHaveIndependentSlotsAndReleaseFailedResources) {
  auto &allocation = allocation_failure();
  c10::SetCPUAllocator(&allocation, 255);
  using D = c::DataType;
  using F = c::model_ops::WeightArtifactFormat;
  using M = c::model_ops::MlxWeightFormat;
  const auto dense = [](D type, std::initializer_list<std::int64_t> shape) {
    return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
  };
  struct Case {
    c::Operation op;
    std::vector<c::TensorDesc> inputs;
  };
  std::vector<Case> cases;
  cases.push_back({c::model_ops::artifact_weight_decode(F::kNf4, 64, D::kFloat32),
      {dense(D::kUInt8,{2,128}),dense(D::kFloat32,{8}),dense(D::kFloat32,{16})}});
  cases.push_back({c::model_ops::artifact_weight_decode(F::kNf4DoubleQuant, 64, D::kFloat32),
      {dense(D::kUInt8,{2,128}),dense(D::kUInt8,{8}),dense(D::kFloat32,{1}),
       dense(D::kFloat32,{256}),dense(D::kFloat32,{}),dense(D::kFloat32,{16})}});
  cases.push_back({c::model_ops::artifact_weight_decode(F::kAwqGemm4, 128, D::kFloat32),
      {dense(D::kInt32,{256,1}),dense(D::kInt32,{2,1}),dense(D::kFloat32,{2,8})}});
  cases.push_back({c::model_ops::artifact_weight_decode(F::kCompressedTensorsSymmetric4,128,D::kFloat32),
      {dense(D::kInt32,{2,32}),dense(D::kFloat32,{2,2})}});
  for (auto format : {M::kAffine4,M::kMxfp4,M::kMxfp8}) {
    const bool affine = format == M::kAffine4;
    const auto packed = format == M::kMxfp8 ? 16 : 8;
    cases.push_back({c::model_ops::mlx_quantized_matmul(format,affine ? 64 : 32),
        {dense(D::kFloat32,{2,64}),dense(D::kUInt32,{3,packed}),
         dense(affine ? D::kFloat32 : D::kUInt8,{3,affine ? 1 : 2})}});
    if (affine) cases.back().inputs.push_back(cases.back().inputs[2]);
  }
  for (const auto &test : cases) {
    SCOPED_TRACE(test.op.custom_as<c::model_ops::ArtifactWeightDecode>() ? "artifact" : "mlx");
    mem::QuotaAllocator metadata(mem::system_allocator(), 4 * 1024 * 1024);
    const auto initial = metadata.remaining();
    {
      auto context = c::test::native_context({.metadata_allocator=metadata.get()}).value();
      std::array<c::TensorDesc,1> outputs;
      ASSERT_TRUE(c::infer_output_descriptions(test.op,test.inputs,outputs).ok());
      auto prepared = c::prepare(context,test.op,test.inputs,outputs,mem::system_resource()).value();
      std::vector<mem::Buffer> storage;
      std::vector<c::ConstTensorView> inputs;
      for (const auto &desc : test.inputs) {
        const auto bytes=desc.required_bytes().value();
        storage.push_back(half_test::buffer<std::byte>(std::vector<std::byte>(bytes)));
        inputs.push_back(c::ConstTensorView::create(desc,storage.back().view()).value());
      }
      auto output=half_test::buffer<float>(std::vector<float>(outputs[0].element_count().value(),17));
      const auto destination=c::TensorView::create(outputs[0],output.view()).value();
      auto queue=context.create_queue().value();
      auto lane=prepared.create_invocation().value();
      allocation.fail=true;
      auto failed=lane.submit(queue,inputs,std::span(&destination,1));
      ASSERT_TRUE(failed);
      EXPECT_TRUE(failed->wait().IsResourceExhausted());
      EXPECT_FALSE(allocation.fail.load());
      for (auto value:half_test::read<float>(output)) EXPECT_EQ(value,17);
      EXPECT_TRUE(lane.submit(queue,inputs,std::span(&destination,1)).error().IsFailedPrecondition());
      lane=prepared.create_invocation().value();
      auto second=prepared.create_invocation().value();
      auto second_queue=context.create_queue().value();
      auto second_output=half_test::buffer<float>(std::vector<float>(outputs[0].element_count().value(),19));
      const auto second_destination=c::TensorView::create(outputs[0],second_output.view()).value();
      bool first_ok=false,second_ok=false;
      {
        std::jthread a([&]{auto done=lane.submit(queue,inputs,std::span(&destination,1));first_ok=done && done->wait().ok();});
        std::jthread b([&]{auto done=second.submit(second_queue,inputs,std::span(&second_destination,1));second_ok=done && done->wait().ok();});
      }
      EXPECT_TRUE(first_ok);EXPECT_TRUE(second_ok);
      EXPECT_EQ(half_test::read<float>(output),half_test::read<float>(second_output));
      {
        auto reserved=metadata.try_reserve_bytes(metadata.remaining()).value();
        auto no_metadata=c::prepare(context,test.op,test.inputs,outputs,mem::system_resource());
        ASSERT_FALSE(no_metadata);EXPECT_TRUE(no_metadata.error().IsResourceExhausted());
      }
    }
    EXPECT_EQ(metadata.remaining(),initial);
  }
}
} // namespace
