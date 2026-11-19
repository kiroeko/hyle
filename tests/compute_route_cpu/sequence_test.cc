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

TEST(CpuPackedSequence, GdnChannelTilesMatchIdenticalPerChannelGates) {
  auto context = c::cpu::create_context().value();
  const auto resource = hyle::mem::system_resource();
  const auto f32 = c::DataType::kFloat32;
  constexpr std::int64_t rows = 8, batches = 2, hk = 2, hv = 4, dv = 16;
  for (const auto dk : {std::int64_t{17}, std::int64_t{128}, std::int64_t{256}, std::int64_t{257}}) {
    SCOPED_TRACE(dk);
    const auto q = p::values<float>(rows * hk * dk, 1);
    const auto k = p::values<float>(rows * hk * dk, 2);
    const auto v = p::values<float>(rows * hv * dv, 3);
    const auto gates = p::values<float>(rows * hv, 4);
    const auto beta = p::values<float>(rows * hv, 5);
    auto initial = p::values<float>(batches * hv * dv * dk, 6);
    initial[0] = -0.0F;
    std::vector<float> channel_gates;
    for (const auto gate : gates)
      channel_gates.insert(channel_gates.end(), dv, gate);

    std::array descriptions{p::desc(f32,{rows,hk,dk}), p::desc(f32,{rows,hk,dk}),
        p::desc(f32,{rows,hv,dv}), p::desc(f32,{rows,hv,1}), p::desc(f32,{rows,hv,1}),
        p::desc(f32,{batches,hv,dv,dk}), p::desc(c::DataType::kInt32,{batches+1})};
    std::vector data{p::bytes(q), p::bytes(k), p::bytes(v), p::bytes(gates), p::bytes(beta),
        p::bytes(initial), p::bytes(std::vector<std::int32_t>{0,1,4})};
    const auto operation = c::sequence_ops::packed_gated_delta_net();
    p::Case shared_gate(context, resource, operation, descriptions, data);
    descriptions[3] = p::desc(f32,{rows,hv,dv});
    data[3] = p::bytes(channel_gates);
    p::Case channel_gate(context, resource, operation, descriptions, data);
    for (const auto offsets : {std::array<std::int32_t,3>{0,1,4}, std::array<std::int32_t,3>{0,0,3}}) {
      shared_gate.offsets(offsets);
      channel_gate.offsets(offsets);
      ASSERT_TRUE(shared_gate.submit().ok());
      ASSERT_TRUE(channel_gate.submit().ok());
      EXPECT_EQ(shared_gate.read<std::uint32_t>(0), channel_gate.read<std::uint32_t>(0));
      EXPECT_EQ(shared_gate.read<std::uint32_t>(1), channel_gate.read<std::uint32_t>(1));
    }
    shared_gate.poison();
    shared_gate.offsets(std::array<std::int32_t,3>{0,4,9});
    EXPECT_TRUE(shared_gate.submit().IsInvalidArgument());
    shared_gate.expect_poison();
  }
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
