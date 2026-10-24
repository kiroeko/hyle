#include <gtest/gtest.h>

#include <algorithm>
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
  at::DeleterFnPtr raw_deleter() const override { return original->raw_deleter(); }
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
      auto candidate=c::prepare(context,c::model_ops::causal_conv1d(),descs,outputs,mem::system_resource());
      ASSERT_TRUE(candidate)<<candidate.error().message();
      auto prepared=std::move(candidate).value();
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
      { auto warm=lane.submit(queue,inputs,std::span(&ya,1));ASSERT_TRUE(warm);EXPECT_TRUE(warm->wait().ok()); }
      const auto untouched=half_test::read<std::uint16_t>(a);
      allocation.fail=true;
      { auto failed=lane.submit(queue,inputs,std::span(&ya,1));ASSERT_TRUE(failed);
      EXPECT_TRUE(failed->wait().IsResourceExhausted());EXPECT_FALSE(allocation.fail.load());
      }
      EXPECT_TRUE(lane.submit(queue,inputs,std::span(&ya,1)).error().IsFailedPrecondition());
      EXPECT_EQ(half_test::read<std::uint16_t>(a),untouched);
      lane=prepared.create_invocation().value();
      { auto done=lane.submit(queue,inputs,std::span(&ya,1));
      ASSERT_TRUE(done);EXPECT_TRUE(done->wait().ok()); }
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
  const auto op = c::model_ops::grouped_float8_linear(c::model_ops::Float8ScalePolicy::kDivide, 448.0F);
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

TEST(NativeTorchCpu, GroupedBlocksKeepMultiTokenIdsScalesAndTailRows) {
  constexpr std::int64_t experts=17, slots=4, columns=129, reduction=512;
  constexpr std::array<std::uint8_t,8> fp8_codes{0x38,0xb8,0x40,0xc0,0x30,0xb0,0x44,0xc4};
  constexpr std::array<double,8> fp8_values{1,-1,2,-2,0.5,-0.5,3,-3};
  constexpr std::array<double,8> fp4_values{0,0.5,1,1.5,2,3,4,6};
  for(const std::int64_t tokens:{1,3})for(const bool four:{false,true})for(const bool shared:{false,true}) {
    SCOPED_TRACE(::testing::Message()<<"tokens="<<tokens<<" fp4="<<four<<" shared="<<shared);
    const auto group=four?32:128;
    const auto blocks=reduction/group, activations=shared?1:slots;
    const auto w=four?c::float4_grouped_weight_description(experts,columns,reduction).value()
                     :c::float8_grouped_weight_description(experts,columns,reduction).value();
    const auto type=four?c::DataType::kBFloat16:c::DataType::kFloat32;
    const auto a=c::TensorDesc::dense(type,std::array{tokens,activations,reduction}).value();
    const auto id_desc=c::TensorDesc::dense(c::DataType::kInt32,std::array{tokens,slots}).value();
    const std::array descriptions{w,a,id_desc};std::array<c::TensorDesc,1> output;
    const auto op=four?c::model_ops::grouped_float4_linear():c::model_ops::grouped_float8_linear(c::model_ops::Float8ScalePolicy::kDivide,448.F);
    ASSERT_TRUE(c::infer_output_descriptions(op,descriptions,output).ok());
    auto context=c::test::native_context().value();auto queue=context.create_queue().value();
    auto prepared=c::prepare(context,op,descriptions,output,mem::system_resource()).value();
    auto lane=prepared.create_invocation().value();
    auto weights=half_test::buffer<std::byte>(std::vector<std::byte>(w.required_bytes().value()));
    auto input=half_test::buffer<std::byte>(std::vector<std::byte>(a.required_bytes().value()));
    auto ids=half_test::buffer<std::int32_t>(std::vector<std::int32_t>(tokens*slots));
    auto result=half_test::buffer<std::byte>(std::vector<std::byte>(output[0].required_bytes().value()));
    const std::array inputs{c::ConstTensorView::create(w,weights.view()).value(),
        c::ConstTensorView::create(a,input.view()).value(),c::ConstTensorView::create(id_desc,ids.view()).value()};
    const auto destination=c::TensorView::create(output[0],result.view()).value();
    for(int phase=0;phase<2;++phase) {
      SCOPED_TRACE(phase);
      const auto payload=experts*columns*reduction/(four?2:1);
      const auto scale_rows=four?columns:(columns+127)/128;
      std::vector<std::byte> bytes(w.required_bytes().value());
      for(std::int64_t e=0;e<experts;++e)for(std::int64_t n=0;n<columns;++n)
        for(std::int64_t block=0;block<blocks;++block)for(int term=0;term<2;++term) {
          const auto k=block*group+(n*7+e*3+phase*5+term*(group/2))%group;
          const auto code=std::uint8_t(four?1+(e+n+block+term+phase)%7+((e+term)%2)*8:
              fp8_codes[(e+n+block+term+phase)%fp8_codes.size()]);
          const auto index=(e*columns+n)*(reduction/(four?2:1))+k/(four?2:1);
          bytes[index]|=std::byte(four?code<<((k%2)*4):code);
        }
      for(std::int64_t e=0;e<experts;++e)for(std::int64_t n=0;n<scale_rows;++n)
        for(std::int64_t block=0;block<blocks;++block) {
          const float scale=std::ldexp(1.F,int((e+n+block+phase)%4)-9);
          std::memcpy(bytes.data()+payload+((e*scale_rows+n)*blocks+block)*sizeof(float),&scale,sizeof(scale));
        }
      std::vector<float> values(tokens*activations*reduction);
      for(std::size_t i=0;i<values.size();++i)
        values[i]=i%128==0?448.F:float(fp8_values[(i+phase*3)%fp8_values.size()]);
      std::vector<std::int32_t> selected(tokens*slots);
      for(std::size_t i=0;i<selected.size();++i)selected[i]=std::int32_t((i*5+phase*3)%experts);
      if(tokens!=1||phase!=0)selected[0]=selected[1]=16;
      selected.back()=15-phase;
      ASSERT_TRUE(weights.view().write(bytes).ok());
      ASSERT_TRUE(ids.view().write(std::as_bytes(std::span(selected))).ok());
      if(four) {
        std::vector<hyle::simd::bfloat16_t> stored(values.size());
        for(std::size_t i=0;i<stored.size();++i)stored[i]=half_test::encode<hyle::simd::bfloat16_t>(values[i]);
        ASSERT_TRUE(input.view().write(std::as_bytes(std::span(stored))).ok());
      } else ASSERT_TRUE(input.view().write(std::as_bytes(std::span(values))).ok());
      auto completed=lane.submit(queue,inputs,std::span(&destination,1));
      ASSERT_TRUE(completed);ASSERT_TRUE(completed->wait().ok());
      const auto actual=four?std::vector<float>{}:half_test::read<float>(result);
      const auto actual_bfloat=four?half_test::read<hyle::simd::bfloat16_t>(result):std::vector<hyle::simd::bfloat16_t>{};
      for(std::int64_t row=0;row<tokens*slots;++row)for(std::int64_t n=0;n<columns;++n) {
        const auto e=selected[row];
        const auto activation=(row/slots)*activations+(shared?0:row%slots);
        double total=0;
        for(std::int64_t block=0;block<blocks;++block) {
          double dot=0;
          for(std::int64_t offset=0;offset<group;++offset) {
            const auto k=block*group+offset;
            const auto raw=std::to_integer<std::uint8_t>(bytes[(e*columns+n)*(reduction/(four?2:1))+k/(four?2:1)]);
            const auto code=four?std::uint8_t((raw>>((k%2)*4))&15):raw;
            double decoded=0;
            if(four)decoded=(code&8?-1:1)*fp4_values[code&7];
            else if(code)decoded=fp8_values[std::find(fp8_codes.begin(),fp8_codes.end(),code)-fp8_codes.begin()];
            dot+=double(values[activation*reduction+k])*decoded;
          }
          float scale;
          std::memcpy(&scale,bytes.data()+payload+((e*scale_rows+(four?n:n/128))*blocks+block)*sizeof(float),sizeof(scale));
          total+=dot*scale;
        }
        // All selected values/scales are dyadic and quantize exactly with
        // scale1; FP64 is independent of the ATen block gather/matmul sequence.
        const auto index=std::size_t(row*columns+n);
        if(four)EXPECT_EQ(actual_bfloat[index].bits,half_test::encode<hyle::simd::bfloat16_t>(total).bits);
        else EXPECT_EQ(actual[index],float(total));
      }
    }
    for(const std::int32_t bad:{-1,17}) {
      std::vector<std::int32_t> selected(tokens*slots,16);selected.back()=bad;
      ASSERT_TRUE(ids.view().write(std::as_bytes(std::span(selected))).ok());
      const std::vector<std::byte> poison(result.size(),std::byte{0x5a});
      ASSERT_TRUE(result.view().write(poison).ok());
      auto failed=lane.submit(queue,inputs,std::span(&destination,1));ASSERT_TRUE(failed);
      EXPECT_TRUE(failed->wait().IsOutOfRange());EXPECT_EQ(half_test::read<std::byte>(result),poison);
      lane=prepared.create_invocation().value();
    }
  }
}

TEST(NativeTorchCpu, DenseFloat8BlocksShareWeightsAcrossTokensAndRebindBothSlots) {
  constexpr std::int64_t tokens = 3, columns = 129, reduction = 256;
  constexpr std::array<std::uint8_t, 8> codes{0x38, 0xb8, 0x40, 0xc0, 0x30, 0xb0, 0x44, 0xc4};
  constexpr std::array<double, 8> values{1, -1, 2, -2, 0.5, -0.5, 3, -3};
  const auto mode = c::Float8ScaleMode::kFloat32Block128;
  const auto weight_desc = c::float8_weight_description(reduction, columns, mode).value();
  auto context = c::test::native_context().value();
  for (int kind = 0; kind < 3; ++kind) {
    SCOPED_TRACE(kind); // Encoded input, normal scale, power-of-two scale.
    const bool encoded = kind == 0;
    const auto activation_desc = encoded
        ? c::float8_activation_description(tokens, reduction, mode).value()
        : c::TensorDesc::dense(c::DataType::kFloat32, std::array{tokens, reduction}).value();
    const std::array descriptions{activation_desc, weight_desc};
    const auto op = encoded ? c::Operation(c::MatMul{})
        : c::model_ops::float8_block_linear(kind == 2
              ? c::model_ops::Float8ScalePolicy::kPowerOfTwo
              : c::model_ops::Float8ScalePolicy::kDivide);
    std::array<c::TensorDesc, 1> outputs;
    ASSERT_TRUE(c::infer_output_descriptions(op, descriptions, outputs).ok());
    auto prepared = c::prepare(context, op, descriptions, outputs, mem::system_resource()).value();
    std::array lanes{prepared.create_invocation().value(), prepared.create_invocation().value()};
    std::array queues{context.create_queue().value(), context.create_queue().value()};
    auto input = half_test::buffer<std::byte>(std::vector<std::byte>(activation_desc.required_bytes().value()));
    auto weight = half_test::buffer<std::byte>(std::vector<std::byte>(weight_desc.required_bytes().value()));
    std::array result{half_test::buffer<float>(std::vector<float>(tokens * columns)),
                      half_test::buffer<float>(std::vector<float>(tokens * columns))};
    const std::array inputs{c::ConstTensorView::create(activation_desc, input.view()).value(),
                            c::ConstTensorView::create(weight_desc, weight.view()).value()};
    for (int phase = 0; phase < 2; ++phase) {
      SCOPED_TRACE(phase);
      std::vector<float> activation(tokens * reduction);
      std::vector<std::byte> activation_bytes(activation_desc.required_bytes().value());
      std::vector<std::byte> weight_bytes(weight_desc.required_bytes().value());
      std::vector<double> decoded(columns * reduction);
      for (std::int64_t row = 0; row < tokens; ++row) for (int block = 0; block < 2; ++block) {
        const float scale = encoded ? std::ldexp(1.F, int(row) + block + phase - 2) : 1.F;
        for (int j = 0; j < 128; ++j) {
          const auto index = row * reduction + block * 128 + j;
          const auto v = (index + phase * 3) % values.size();
          activation[index] = encoded ? float(values[v]) * scale : (j == 0 ? 448.F : float(values[v]));
          if (encoded) activation_bytes[index] = std::byte(codes[v]);
        }
        if (encoded) std::memcpy(activation_bytes.data() + tokens * reduction + (row * 2 + block) * 4, &scale, 4);
      }
      if (!encoded) std::memcpy(activation_bytes.data(), activation.data(), activation_bytes.size());
      for (std::int64_t n = 0; n < columns; ++n) for (int block = 0; block < 2; ++block) {
        const float scale = std::ldexp(1.F, int(n / 128) + block + phase - 5);
        for (int term = 0; term < 2; ++term) {
          const auto k = block * 128 + (n * 7 + term * 61 + phase * 5) % 128;
          const auto v = (n + block + term + phase) % values.size();
          weight_bytes[n * reduction + k] = std::byte(codes[v]);
          decoded[n * reduction + k] = values[v] * scale;
        }
        std::memcpy(weight_bytes.data() + columns * reduction + ((n / 128) * 2 + block) * 4, &scale, 4);
      }
      ASSERT_TRUE(input.view().write(activation_bytes).ok());
      ASSERT_TRUE(weight.view().write(weight_bytes).ok());
      for (std::size_t slot = 0; slot < lanes.size(); ++slot) {
        const auto destination = c::TensorView::create(outputs[0], result[slot].view()).value();
        auto done = lanes[slot].submit(queues[slot], inputs, std::span(&destination, 1));
        ASSERT_TRUE(done); ASSERT_TRUE(done->wait().ok());
        const auto actual = half_test::read<float>(result[slot]);
        for (std::int64_t row = 0; row < tokens; ++row) for (std::int64_t n = 0; n < columns; ++n) {
          double expected = 0;
          for (std::int64_t k = 0; k < reduction; ++k)
            expected += double(activation[row * reduction + k]) * decoded[n * reduction + k];
          // Dyadic codes/scales and sparse weights make the FP64 result exact.
          EXPECT_EQ(actual[row * columns + n], float(expected));
        }
      }
    }
  }
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
  for (auto format : {M::kAffine,M::kMxfp4,M::kMxfp8}) {
    const bool affine = format == M::kAffine;
    const auto packed = format == M::kMxfp8 ? 16 : 8;
    cases.push_back({c::model_ops::mlx_quantized_matmul(format,format == M::kMxfp8 ? 8 : 4,affine ? 64 : 32),
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
