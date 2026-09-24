#include "hyle/compute/apple.h"

#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include <mlx/ops.h>

#include "../compute/workload_fixture.h"
#include "ffn_fixture.h"
#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"

namespace {
namespace c = hyle::compute;
namespace a = c::apple;
namespace mx = mlx::core;

TEST(AppleTyped, MatMulAndSiluGatesUseOrdinaryPreparation) {
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  const std::vector<mx::array> inputs{mx::array({1.f, -2.f, 3.f, -4.f}, {2, 2}),
                                      mx::array({2.f, 0.f, 0.f, 3.f}, {2, 2})};
  auto workload = workload_test::known(c::WorkloadPhase::kDecode);
  auto matmul = context->prepare(c::MatMul{}, inputs, workload);
  ASSERT_TRUE(matmul) << matmul.error().message();
  EXPECT_EQ(matmul->workload(), workload);
  workload.region.version = 0;
  EXPECT_FALSE(context->prepare(c::MatMul{}, inputs, workload));
  workload = {};
  EXPECT_TRUE(matmul->workload().profile.matches(workload_test::kModel));
  auto slot = matmul->create_invocation();
  ASSERT_TRUE(slot);
  ASSERT_TRUE(slot->bind(inputs).ok());
  auto run = slot->submit();
  ASSERT_TRUE(run);
  auto result = run->wait();
  ASSERT_TRUE(result);
  EXPECT_FLOAT_EQ(result->get()->arrays()[0].data<float>()[3], -12.f);

  const std::vector<mx::array> separate{mx::array({-3.f, -0.f, 1.f, 4.f}),
                                        mx::array({2.f, 3.f, 4.f, 5.f})};
  const std::vector<mx::array> packed{
      mx::array({-3.f, -0.f, 1.f, 4.f, 2.f, 3.f, 4.f, 5.f})};
  for (auto mode :
       {c::GatedInputKind::kSeparate, c::GatedInputKind::kPackedHalves}) {
    const auto &args = mode == c::GatedInputKind::kSeparate ? separate : packed;
    auto program = context->prepare(c::GatedActivation{.inputs = mode}, args);
    ASSERT_TRUE(program) << program.error().message();
    auto invocation = program->create_invocation();
    ASSERT_TRUE(invocation);
    ASSERT_TRUE(invocation->bind(args).ok());
    auto submitted = invocation->submit();
    ASSERT_TRUE(submitted);
    auto lease = submitted->wait();
    ASSERT_TRUE(lease);
    for (int i = 0; i < 4; ++i) {
      const auto x = separate[0].data<float>()[i];
      const auto y = separate[1].data<float>()[i];
      EXPECT_NEAR(lease->get()->arrays()[0].data<float>()[i],
                  x / (1.f + std::exp(-x)) * y, 1e-5f);
    }
  }
}

TEST(AppleTyped, MlxBatchedLinearPositionAndGqaAttentionReplay) {
  auto context = a::Context::create();
  ASSERT_TRUE(context);

  const std::vector<mx::array> linear_inputs{
      mx::array({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 2.f, 1.f, 0.f, 3.f, 2.f, 1.f},
                {2, 2, 3}, mx::float16),
      mx::array({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}, {1, 3, 2}, mx::float16)};
  auto linear = context->prepare(c::MatMul{}, linear_inputs);
  ASSERT_TRUE(linear) << linear.error().message();
  auto linear_slot = linear->create_invocation();
  ASSERT_TRUE(linear_slot);
  ASSERT_TRUE(linear_slot->bind(linear_inputs).ok());
  auto linear_work = linear_slot->submit();
  ASSERT_TRUE(linear_work);
  auto linear_result = linear_work->wait();
  ASSERT_TRUE(linear_result);
  auto linear_f32 = mx::astype(linear_result->get()->arrays()[0], mx::float32);
  linear_f32.eval();
  EXPECT_EQ(linear_f32.shape(), (mx::Shape{2, 2, 2}));
  EXPECT_NEAR(linear_f32.data<float>()[0], 22.f, 1.e-2f);
  EXPECT_NEAR(linear_f32.data<float>()[3], 64.f, 1.e-2f);
  EXPECT_NEAR(linear_f32.data<float>()[4], 5.f, 1.e-2f);

  const std::vector<mx::array> rope_inputs{
      mx::array({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f}, {1, 2, 1, 4}),
      mx::array({0, 2}, {2}, mx::int32), mx::array({0.5f, 0.25f})};
  auto rope = context->prepare(c::RoPE{.rotary_dimensions = 4,
                                       .sequence_axis = 1,
                                       .layout = c::RopeLayout::kInterleaved},
                               rope_inputs);
  ASSERT_TRUE(rope) << rope.error().message();
  auto rope_slot = rope->create_invocation();
  ASSERT_TRUE(rope_slot);
  ASSERT_TRUE(rope_slot->bind(rope_inputs).ok());
  auto rope_work = rope_slot->submit();
  ASSERT_TRUE(rope_work);
  auto rope_result = rope_work->wait();
  ASSERT_TRUE(rope_result);
  const auto *rotated = rope_result->get()->arrays()[0].data<float>();
  EXPECT_FLOAT_EQ(rotated[0], 1.f);
  EXPECT_NEAR(rotated[4], 5.f * std::cos(1.f) - 6.f * std::sin(1.f), 2.e-5f);
  EXPECT_NEAR(rotated[7], 7.f * std::sin(.5f) + 8.f * std::cos(.5f), 2.e-5f);

  const std::vector<mx::array> hadamard_inputs{
      mx::array({1.f, 2.f, 3.f, 4.f}, {1, 4})};
  auto hadamard = context->prepare(c::Hadamard{}, hadamard_inputs);
  ASSERT_TRUE(hadamard) << hadamard.error().message();
  auto hadamard_slot = hadamard->create_invocation();
  ASSERT_TRUE(hadamard_slot);
  ASSERT_TRUE(hadamard_slot->bind(hadamard_inputs).ok());
  auto hadamard_work = hadamard_slot->submit();
  ASSERT_TRUE(hadamard_work);
  auto hadamard_result = hadamard_work->wait();
  ASSERT_TRUE(hadamard_result);
  const std::array expected_hadamard{5.f, -1.f, -2.f, 0.f};
  for (std::size_t i = 0; i < expected_hadamard.size(); ++i)
    EXPECT_NEAR(hadamard_result->get()->arrays()[0].data<float>()[i],
                expected_hadamard[i], 2.e-6f);

  const std::vector<mx::array> attention_inputs{
      mx::array({0.f, 0.f}, {1, 1, 2, 1}),
      mx::array({0.f, 0.f, 0.f}, {1, 3, 1, 1}),
      mx::array({1.f, 2.f, 6.f}, {1, 3, 1, 1})};
  const c::ScaledDotProductAttention attention{
      .layout = c::AttentionLayout::kBatchSequenceHeadDepth,
      .scale = 1.f,
      .causal = true};
  auto attention_program = context->prepare(attention, attention_inputs);
  ASSERT_TRUE(attention_program) << attention_program.error().message();
  auto attention_slot = attention_program->create_invocation();
  ASSERT_TRUE(attention_slot);
  for (int replay = 0; replay < 2; ++replay) {
    ASSERT_TRUE(attention_slot->bind(attention_inputs).ok());
    auto attention_work = attention_slot->submit();
    ASSERT_TRUE(attention_work);
    auto attention_result = attention_work->wait();
    ASSERT_TRUE(attention_result);
    ASSERT_EQ(attention_result->get()->arrays()[0].size(), 2U);
    EXPECT_NEAR(attention_result->get()->arrays()[0].data<float>()[0], 3.f,
                2.e-5f);
    EXPECT_NEAR(attention_result->get()->arrays()[0].data<float>()[1], 3.f,
                2.e-5f);
  }
}

TEST(AppleTyped, AffineQ4MatchesIndependentPackedOracleAndRejectsOtherModes) {
  apple_test::Ffn fixture;
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  auto host = fixture.input(3, 2);
  const std::vector<mx::array> args{mx::array(host.data(), {3, fixture.k}),
                                    fixture.constants[0], fixture.constants[1],
                                    fixture.constants[2]};
  const c::QuantizedMatMul qmm{.mode =
                                   c::QuantizedWeightMode::kUnsignedAffine4};
  auto program = context->prepare(qmm, args);
  ASSERT_TRUE(program) << program.error().message();
  auto invocation = program->create_invocation();
  ASSERT_TRUE(invocation);
  ASSERT_TRUE(invocation->bind(args).ok());
  auto submitted = invocation->submit();
  ASSERT_TRUE(submitted);
  auto result = submitted->wait();
  ASSERT_TRUE(result);
  for (int row = 0; row < 3; ++row) {
    for (int n = 0; n < fixture.hidden; ++n) {
      double oracle = 0;
      for (int k = 0; k < fixture.k; ++k)
        oracle +=
            double(host[row * fixture.k + k]) * fixture.gate[n * fixture.k + k];
      EXPECT_NEAR(
          result->get()->arrays()[0].data<float>()[row * fixture.hidden + n],
          oracle, 1e-5);
    }
  }
  auto unsupported = qmm;
  unsupported.group_size = 32;
  EXPECT_FALSE(context->prepare(unsupported, args));
  unsupported = qmm;
  unsupported.mode = c::QuantizedWeightMode::kSignedSymmetric4;
  EXPECT_FALSE(context->prepare(unsupported, args));
  auto bad = args;
  bad[1] = mx::array({1.f});
  EXPECT_FALSE(context->prepare(qmm, bad));
  EXPECT_FALSE(a::affine_q4_description(1, 63));
}

TEST(AppleTyped, GgufPackedMatMulAndGroupedExpertsUseOrdinaryPreparation) {
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  constexpr int k = 256;
  constexpr int n = 8;
  constexpr int m = 2;
  const std::array activation{
      a::InputSpec{{m, k}, mx::float32}};
  for (const auto type : {c::GgufQuantizedType::kQ4K,
                          c::GgufQuantizedType::kIq4Xs,
                          c::GgufQuantizedType::kMxfp4}) {
    SCOPED_TRACE(static_cast<int>(type));
    const auto weight =
        c::gguf_weight_description(type, c::DataType::kFloat32, k, n)
            .value();
    const auto packed_spec = a::input_spec(weight);
    ASSERT_TRUE(packed_spec) << packed_spec.error().message();
    const std::array specs{activation[0], *packed_spec};
    auto program = context->prepare(c::MatMul{}, specs);
    ASSERT_TRUE(program) << program.error().message();
    std::vector<float> host(m * k, 0.25F);
    std::vector<std::uint8_t> bytes(weight.required_bytes().value());
    const std::array inputs{
        mx::array(host.data(), mx::Shape{m, k}),
        mx::array(bytes.data(), mx::Shape{static_cast<int>(bytes.size())},
                  mx::uint8)};
    EXPECT_FALSE(context->prepare(c::MatMul{}, inputs));
    auto invocation = program->create_invocation();
    ASSERT_TRUE(invocation);
    for (int replay = 0; replay < 2; ++replay) {
      ASSERT_TRUE(invocation->bind(inputs).ok());
      auto work = invocation->submit();
      ASSERT_TRUE(work) << work.error().message();
      auto result = work->wait();
      ASSERT_TRUE(result) << result.error().message();
      ASSERT_EQ(result->get()->arrays()[0].shape(), (mx::Shape{m, n}));
      for (std::size_t i = 0; i < result->get()->arrays()[0].size(); ++i)
        EXPECT_NEAR(result->get()->arrays()[0].data<float>()[i], 0.0F,
                    1.0e-12F);
    }
    auto wrong = inputs;
    wrong[1] = mx::array(bytes.data(),
                         mx::Shape{static_cast<int>(bytes.size() - 1)},
                         mx::uint8);
    EXPECT_TRUE(invocation->bind(wrong).IsInvalidArgument());
  }

  constexpr int experts = 4;
  constexpr int tokens = 2;
  constexpr int slots = 2;
  const auto grouped_weight = c::gguf_grouped_weight_description(
                                  c::GgufQuantizedType::kIq4Xs,
                                  c::DataType::kFloat32, experts, n, k)
                                  .value();
  const auto grouped_spec = a::input_spec(grouped_weight).value();
  const std::array grouped_specs{
      grouped_spec, a::InputSpec{{tokens, slots, k}, mx::float32},
      a::InputSpec{{tokens, slots}, mx::int32}};
  auto grouped =
      context->prepare(c::model_ops::grouped_matmul(), grouped_specs);
  ASSERT_TRUE(grouped) << grouped.error().message();
  std::vector<std::uint8_t> packed(grouped_weight.required_bytes().value());
  std::vector<float> samples(tokens * slots * k, 0.5F);
  const std::array<std::int32_t, tokens * slots> ids{0, 3, 1, 2};
  const std::array grouped_inputs{
      mx::array(packed.data(), mx::Shape{static_cast<int>(packed.size())},
                mx::uint8),
      mx::array(samples.data(), mx::Shape{tokens, slots, k}),
      mx::array(ids.data(), mx::Shape{tokens, slots})};
  auto invocation = grouped->create_invocation();
  ASSERT_TRUE(invocation);
  ASSERT_TRUE(invocation->bind(grouped_inputs).ok());
  auto work = invocation->submit();
  ASSERT_TRUE(work) << work.error().message();
  auto result = work->wait();
  ASSERT_TRUE(result) << result.error().message();
  ASSERT_EQ(result->get()->arrays()[0].shape(),
            (mx::Shape{tokens, slots, n}));
  for (std::size_t i = 0; i < result->get()->arrays()[0].size(); ++i)
    EXPECT_NEAR(result->get()->arrays()[0].data<float>()[i], 0.0F, 1.0e-12F);
}

TEST(AppleTyped, MetadataPreparationAndCheckedRegionError) {
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  auto environment = context->environment();
  ASSERT_TRUE(environment);
  EXPECT_FALSE(environment->device_name.empty());
  EXPECT_FALSE(environment->architecture.empty());
  EXPECT_FALSE(environment->os_runtime.empty());
  EXPECT_FALSE(environment->metal_compiler.empty());
  EXPECT_EQ(environment->mlx_version, "0.32.2");
  EXPECT_EQ(environment->mlx_revision,
            "1f8e74e3f12f31365464a6867c6579f0e9b29d85");
  EXPECT_GE(environment->build_sdk, environment->deployment_target);
  const std::array specs{a::InputSpec{{2, 2}, mx::float32}};
  auto failed = context->compile(
      [](const auto &, mx::Stream) -> hyle::Result<std::vector<mx::array>> {
        return std::unexpected(
            hyle::Status::Unimplemented("custom schema declined"));
      },
      specs);
  ASSERT_FALSE(failed);
  EXPECT_EQ(failed.error().code(), hyle::StatusCode::Unimplemented);
  auto identity = context->compile(
      [](const auto &inputs, mx::Stream) { return inputs; }, specs);
  ASSERT_TRUE(identity) << identity.error().message();
}
} // namespace
