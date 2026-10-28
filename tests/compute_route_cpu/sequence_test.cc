#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>

#include "../compute_models/packed_sequence_case.h"
#include "../compute_models/packed_gdn_prefix_case.h"
#include "hyle/compute/cpu.h"

namespace {
namespace c = hyle::compute;
namespace p = packed_sequence_test;

TEST(ComputeSequenceSchema, PackedDescriptionsRejectInvalidPortsAtomically) {
  const auto f32 = c::DataType::kFloat32;
  const auto gdn = c::sequence_ops::packed_gated_delta_net(2, -0.0F);
  EXPECT_EQ(c::operation_name(gdn), "packed-gated-delta-net");
  EXPECT_EQ(c::operation_traits(gdn).effect, c::EffectKind::kPure);
  EXPECT_EQ(c::operation_traits(gdn).alias, c::AliasPolicy::kNone);
  const auto sentinel = p::desc(f32,{7});
  for (const auto type : {f32,c::DataType::kFloat16,c::DataType::kBFloat16}) {
    const std::array valid{p::desc(type,{8,2,3}),p::desc(type,{8,2,3}),p::desc(type,{8,4,2}),
        p::desc(f32,{8,4,2}),p::desc(f32,{8,4,1}),p::desc(f32,{2,4,2,3}),p::desc(c::DataType::kInt32,{3})};
    std::array<c::TensorDesc,2> outputs;
    ASSERT_TRUE(c::infer_output_descriptions(gdn,valid,outputs).ok());
    EXPECT_EQ(outputs[0],valid[2]); EXPECT_EQ(outputs[1],p::desc(f32,{2,2,4,2,3}));
    const auto reject = [&](const auto &inputs, c::Operation operation = c::sequence_ops::packed_gated_delta_net(2)) {
      std::array result{sentinel,sentinel};
      EXPECT_TRUE(c::infer_output_descriptions(operation,inputs,result).IsInvalidArgument());
      EXPECT_EQ(result[0],sentinel); EXPECT_EQ(result[1],sentinel);
    };
    auto inputs = valid; inputs[6] = p::desc(c::DataType::kInt64,{3}); reject(inputs);
    inputs = valid; inputs[5] = p::desc(f32,{2,4,3,2}); reject(inputs);
    inputs = valid; inputs[3] = p::desc(f32,{8,4,3}); reject(inputs);
    reject(std::span(valid).first(6));
    reject(valid,c::sequence_ops::packed_gated_delta_net(0));
    reject(valid,c::sequence_ops::packed_gated_delta_net(9));
    reject(valid,c::sequence_ops::packed_gated_delta_net(1,std::numeric_limits<float>::infinity()));
  }
  const std::array conv{p::desc(f32,{0,3}),p::desc(f32,{3,1}),p::desc(f32,{2,0,3}),p::desc(c::DataType::kInt32,{3})};
  std::array<c::TensorDesc,2> outputs;
  ASSERT_TRUE(c::infer_output_descriptions(c::sequence_ops::packed_causal_conv1d(),conv,outputs).ok());
  EXPECT_EQ(outputs[0],conv[0]); EXPECT_EQ(outputs[1],conv[2]);
  auto bad = conv; bad[2] = p::desc(f32,{2,1,3});
  outputs = {sentinel,sentinel};
  EXPECT_TRUE(c::infer_output_descriptions(c::sequence_ops::packed_causal_conv1d(),bad,outputs).IsInvalidArgument());
  EXPECT_EQ(outputs[0],sentinel); EXPECT_EQ(outputs[1],sentinel);
}

TEST(CpuPackedSequence, GdnPackedRowsAndPerChannelDecayMatchReference) {
  auto context = c::cpu::create_context().value();
  p::gdn_math<float>(context,hyle::mem::system_resource());
  p::gdn_math<float>(context,hyle::mem::system_resource(),3,std::nullopt);
  p::gdn_math<hyle::simd::float16_t>(context,hyle::mem::system_resource());
  p::gdn_math<hyle::simd::bfloat16_t>(context,hyle::mem::system_resource());
}

TEST(CpuPackedSequence, GdnSnapshotsFollowEachRealSequenceAndRejectShortTail) {
  auto context = c::cpu::create_context().value();
  p::gdn_snapshots(context,hyle::mem::system_resource());
}

TEST(CpuPackedSequence, EmptySequencesAndZeroPhysicalRowsKeepInitialStateBits) {
  auto context = c::cpu::create_context().value();
  p::gdn_empty(context,hyle::mem::system_resource());
}

TEST(CpuPackedSequence, ConvolutionUsesOnlyRealRowsAndKeepsExactHistory) {
  auto context = c::cpu::create_context().value();
  p::conv_math<float>(context,hyle::mem::system_resource());
  p::conv_math<hyle::simd::float16_t>(context,hyle::mem::system_resource());
  p::conv_math<hyle::simd::bfloat16_t>(context,hyle::mem::system_resource());
}

TEST(CpuPackedSequence, WholeOffsetValidationPrecedesEitherOutput) {
  auto context = c::cpu::create_context().value();
  p::invalid_offsets(context,hyle::mem::system_resource());
}

TEST(CpuPackedSequence, PrefixesMatchFp64AndContinueFromTheActualDeviceState) {
  auto context = c::cpu::create_context().value();
  packed_gdn_prefix_test::math<float>(context,hyle::mem::system_resource(),true,std::nullopt);
  packed_gdn_prefix_test::math<hyle::simd::float16_t>(context,hyle::mem::system_resource(),false);
  packed_gdn_prefix_test::math<hyle::simd::bfloat16_t>(context,hyle::mem::system_resource(),false);
}

TEST(CpuPackedSequence, PrefixCapacityEmptyMembersAndFailureRemainAtomic) {
  packed_gdn_prefix_test::schema();
  auto context = c::cpu::create_context().value();
  packed_gdn_prefix_test::empty_and_failure(context,hyle::mem::system_resource());
}

} // namespace
