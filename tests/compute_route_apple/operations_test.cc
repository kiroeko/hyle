#include "hyle/compute/apple.h"

#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "ffn_fixture.h"

namespace {
namespace c = hyle::compute;
namespace a = c::apple;
namespace mx = mlx::core;

TEST(AppleTyped, MatMulAndSiluGatesUseOrdinaryPreparation) {
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  const std::vector<mx::array> inputs{mx::array({1.f, -2.f, 3.f, -4.f}, {2, 2}),
                                      mx::array({2.f, 0.f, 0.f, 3.f}, {2, 2})};
  auto matmul = context->prepare(c::MatMul{}, inputs);
  ASSERT_TRUE(matmul) << matmul.error().message();
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
