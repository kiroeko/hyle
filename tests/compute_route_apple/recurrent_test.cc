#include "recurrent_fixture.h"

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <mlx/ops.h>

#include "hyle/compute/model_ops.h"
#include "hyle/mem/quota_allocator.h"
#include "../compute_models/d5_qwen38_gdn_fixture.h"
#include "../compute_models/d5_qwen38_profile_cases.h"
#include "../model_fragment_test_support.h"
#include "../sparse_fragment_test_support.h"

namespace {
namespace c = hyle::compute;
namespace a = c::apple;
namespace mx = mlx::core;

void check(std::span<const mx::array> actual,
           const std::array<std::vector<double>, 2> &expected) {
  ASSERT_EQ(actual.size(), 2u);
  for (std::size_t port = 0; port < 2; ++port) {
    ASSERT_EQ(actual[port].size(), expected[port].size());
    double max_error = 0;
    for (std::size_t i = 0; i < expected[port].size(); ++i)
      max_error =
          std::max(max_error, std::abs(double(actual[port].data<float>()[i]) -
                                       expected[port][i]));
    EXPECT_LT(max_error, 3e-6);
  }
}

std::vector<mx::array>
qwen38_inputs(const d5_qwen38_gdn_fixture::Fixture &fixture) {
  std::vector<mx::array> inputs;
  inputs.reserve(fixture.inputs.size());
  for (std::size_t port = 0; port < fixture.inputs.size(); ++port) {
    mx::Shape shape;
    for (const auto extent : fixture.inputs[port].dimensions())
      shape.push_back(static_cast<int>(extent));
    inputs.emplace_back(fixture.data[port].data(), std::move(shape));
  }
  return inputs;
}

TEST(AppleRecurrent, OrdinaryPreparationInterleavedRequestsAndOldOutputOwner) {
  for (int tokens : {1, 17}) {
    apple_test::Recurrent first(tokens, 16, 128);
    apple_test::Recurrent second(tokens, 16, 128, 7);
    auto context = a::Context::create();
    ASSERT_TRUE(context);
    EXPECT_FALSE(context->environment()->ggml_revision.empty());
    auto program = context->prepare(c::GatedDeltaNet{}, first.specs());
    ASSERT_TRUE(program) << program.error().message();
    auto left = program->create_invocation();
    auto right = program->create_invocation();
    ASSERT_TRUE(left && right);
    ASSERT_TRUE(left->bind(first.inputs).ok());
    ASSERT_TRUE(right->bind(second.inputs).ok());
    auto first_work = left->submit();
    auto b = right->submit();
    ASSERT_TRUE(first_work && b);
    auto old = first_work->wait();
    ASSERT_TRUE(old) << old.error().message();
    check(old->get()->arrays(), first.oracle());
    {
      auto result = b->wait();
      ASSERT_TRUE(result) << result.error().message();
      check(result->get()->arrays(), second.oracle());
    }
    b = c::OwnedResult<a::Results>{};
    for (int i = 0; i < 3; ++i) {
      auto repeat = right->submit();
      ASSERT_TRUE(repeat);
      auto result = repeat->wait();
      ASSERT_TRUE(result);
      check(result->get()->arrays(), second.oracle());
    }
    const auto arrays = old->get()->arrays();
    EXPECT_EQ(arrays[0].buffer().ptr(), arrays[1].buffer().ptr());
    EXPECT_EQ(arrays[1].offset(),
              static_cast<std::int64_t>(arrays[0].nbytes()));
    EXPECT_EQ(std::memcmp(first.inputs[5].data<float>(), first.data[5].data(),
                          first.data[5].size() * sizeof(float)),
              0);
    ASSERT_TRUE(context->close().ok());
    check(arrays, first.oracle());
    EXPECT_FALSE(right->submit());
  }
}

class AppleQwen38GdnProfiles
    : public ::testing::TestWithParam<
          const d5_qwen38_profile_cases::ProfileCase *> {};

TEST_P(AppleQwen38GdnProfiles,
       D5Qwen38GroupedHeadGdnUsesOrdinaryRouteAndReplays) {
  const d5_qwen38_gdn_fixture::Fixture fixture(
      0, GetParam()->gdn_key_heads, GetParam()->gdn_value_heads);
  const auto inputs = qwen38_inputs(fixture);
  const auto workload = d5_qwen38_gdn_fixture::workload(
      *GetParam()->profile, c::WorkloadPhase::kDecode);
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  auto program = context->prepare(c::GatedDeltaNet{}, inputs, workload);
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_EQ(program->workload(), workload);
  auto invocation = program->create_invocation();
  ASSERT_TRUE(invocation) << invocation.error().message();
  const auto expected = fixture.oracle();
  for (int repeat = 0; repeat < 2; ++repeat) {
    ASSERT_TRUE(invocation->bind(inputs).ok());
    auto work = invocation->submit();
    ASSERT_TRUE(work) << work.error().message();
    auto result = work->wait();
    ASSERT_TRUE(result) << result.error().message();
    check(result->get()->arrays(), expected);
  }
}

INSTANTIATE_TEST_SUITE_P(
    D5OfficialAndPrimaryGguf, AppleQwen38GdnProfiles,
    ::testing::Values(&d5_qwen38_profile_cases::kOfficial27B,
                      &d5_qwen38_profile_cases::kPrimaryGguf27B));

INSTANTIATE_TEST_SUITE_P(
    D5FidelityGguf, AppleQwen38GdnProfiles,
    ::testing::Values(&d5_qwen38_profile_cases::kFidelityGguf27B));

INSTANTIATE_TEST_SUITE_P(
    D5Official24T, AppleQwen38GdnProfiles,
    ::testing::Values(&d5_qwen38_profile_cases::kOfficial24T));

TEST(AppleRecurrent, StridedMlxInputAndPreparationDiagnostics) {
  apple_test::Recurrent fixture(5, 2, 32);
  std::vector<float> transposed(fixture.data[0].size());
  for (int b = 0; b < 2; ++b)
    for (int t = 0; t < 5; ++t)
      for (int h = 0; h < 2; ++h)
        for (int d = 0; d < 32; ++d)
          transposed[((b * 5 + t) * 32 + d) * 2 + h] =
              fixture.data[0][((b * 5 + t) * 2 + h) * 32 + d];
  fixture.inputs[0] =
      mx::transpose(mx::array(transposed.data(), {2, 5, 32, 2}), {0, 1, 3, 2});
  fixture.inputs[0].eval();
  fixture.inputs[0].wait();
  ASSERT_FALSE(fixture.inputs[0].flags().row_contiguous);
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  auto program = context->prepare(c::GatedDeltaNet{}, fixture.inputs);
  ASSERT_TRUE(program) << program.error().message();
  auto invocation = program->create_invocation();
  ASSERT_TRUE(invocation);
  ASSERT_TRUE(invocation->bind(fixture.inputs).ok());
  auto work = invocation->submit();
  ASSERT_TRUE(work);
  auto result = work->wait();
  ASSERT_TRUE(result) << result.error().message();
  check(result->get()->arrays(), fixture.oracle());
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 0);
  auto rejected = program->create_invocation(quota.get());
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsResourceExhausted());
  EXPECT_EQ(quota.used(), 0u);
  apple_test::Recurrent unsupported(1, 2, 96);
  auto missing = context->prepare(c::GatedDeltaNet{}, unsupported.specs());
  ASSERT_FALSE(missing);
  EXPECT_TRUE(missing.error().IsUnimplemented());
}

TEST(AppleRecurrent, AbandonedCompletionAndCancelCloseDrain) {
  apple_test::Recurrent fixture(17, 16, 128);
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  auto program = context->prepare(c::GatedDeltaNet{}, fixture.inputs);
  ASSERT_TRUE(program);
  auto abandoned = program->create_invocation();
  auto cancelled = program->create_invocation();
  ASSERT_TRUE(abandoned && cancelled);
  ASSERT_TRUE(abandoned->bind(fixture.inputs).ok());
  ASSERT_TRUE(cancelled->bind(fixture.inputs).ok());
  {
    auto pending = abandoned->submit();
    ASSERT_TRUE(pending);
    abandoned = a::Invocation{};
  }
  auto pending = cancelled->submit();
  ASSERT_TRUE(pending);
  ASSERT_TRUE(pending->request_cancel().ok());
  ASSERT_TRUE(context->close().ok());
  auto done = pending->wait();
  if (done)
    check(done->get()->arrays(), fixture.oracle());
  else
    EXPECT_TRUE(done.error().IsCancelled());
}

TEST(AppleRecurrent, PerChannelKdaGateUsesOriginalMetalRegion) {
  apple_test::Recurrent fixture(3, 2, 32, 0, true);
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  auto program = context->prepare(c::GatedDeltaNet{}, fixture.inputs);
  ASSERT_TRUE(program) << program.error().message();
  auto invocation = program->create_invocation();
  ASSERT_TRUE(invocation);
  ASSERT_TRUE(invocation->bind(fixture.inputs).ok());
  auto work = invocation->submit();
  ASSERT_TRUE(work) << work.error().message();
  auto result = work->wait();
  ASSERT_TRUE(result) << result.error().message();
  check(result->get()->arrays(), fixture.oracle());
}

TEST(AppleRecurrent, MatureCausalConvUsesMlxRegion) {
  const std::array<float, 10> samples{1, 10, 2, 20, 3, 30, 4, 40, 5, 50};
  const std::array<float, 6> weights{1, 2, 3, -1, 0, 1};
  const std::array inputs{mx::array(samples.data(), {1, 5, 2}),
                          mx::array(weights.data(), {2, 3})};
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  auto program = context->prepare(c::model_ops::causal_conv1d(), inputs);
  ASSERT_TRUE(program) << program.error().message();
  auto invocation = program->create_invocation();
  ASSERT_TRUE(invocation);
  ASSERT_TRUE(invocation->bind(inputs).ok());
  auto work = invocation->submit();
  ASSERT_TRUE(work) << work.error().message();
  auto result = work->wait();
  ASSERT_TRUE(result) << result.error().message();
  const auto arrays = result->get()->arrays();
  ASSERT_EQ(arrays.size(), 1U);
  const std::array expected{14.F, 20.F, 20.F, 20.F, 26.F, 20.F};
  for (std::size_t i = 0; i < expected.size(); ++i)
    EXPECT_FLOAT_EQ(arrays[0].data<float>()[i], expected[i]);
}

TEST(AppleRecurrent, MatureMoeAndHyperConnectionBatchUsesOriginalMetal) {
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  for (const auto &test : model_fragment_test::cases()) {
    SCOPED_TRACE(test.name);
    std::vector<mx::array> inputs;
    inputs.reserve(test.inputs.size());
    for (std::size_t i = 0; i < test.inputs.size(); ++i) {
      mx::Shape shape;
      for (auto extent : test.inputs[i].dimensions())
        shape.push_back(static_cast<int>(extent));
      if (test.inputs[i].data_type() == c::DataType::kInt32)
        inputs.emplace_back(
            reinterpret_cast<const std::int32_t *>(test.payloads[i].data()),
            shape);
      else
        inputs.emplace_back(
            reinterpret_cast<const float *>(test.payloads[i].data()), shape);
    }
    auto program = context->prepare(test.operation, inputs);
    ASSERT_TRUE(program) << program.error().message();
    auto invocation = program->create_invocation();
    ASSERT_TRUE(invocation);
    ASSERT_TRUE(invocation->bind(inputs).ok());
    auto work = invocation->submit();
    ASSERT_TRUE(work) << work.error().message();
    auto result = work->wait();
    ASSERT_TRUE(result) << result.error().message();
    const auto arrays = result->get()->arrays();
    ASSERT_EQ(arrays.size(), 1U);
    ASSERT_EQ(arrays[0].size(), test.expected.size());
    for (std::size_t i = 0; i < test.expected.size(); ++i)
      EXPECT_NEAR(arrays[0].data<float>()[i], test.expected[i], 2.0e-5F)
          << i;
  }
}

TEST(AppleRecurrent, MatureLightningIndexerUsesOriginalMetalAndReplays) {
  const sparse_fragment_test::Fixture test;
  std::vector<mx::array> inputs;
  for (std::size_t i = 0; i < test.inputs.size(); ++i) {
    mx::Shape shape;
    for (const auto extent : test.inputs[i].dimensions())
      shape.push_back(static_cast<int>(extent));
    if (test.inputs[i].data_type() == c::DataType::kFloat16) {
      inputs.emplace_back(
          std::initializer_list<float>{-1, 0, 0.5F, -2, 0.25F, -0.5F, 1, 0},
          shape, mx::float16);
    } else {
      inputs.emplace_back(
          reinterpret_cast<const float *>(test.payloads[i].data()), shape);
    }
  }
  auto context = a::Context::create();
  ASSERT_TRUE(context);
  auto program = context->prepare(test.operation, inputs);
  ASSERT_TRUE(program) << program.error().message();
  auto invocation = program->create_invocation().value();
  for (int repeat = 0; repeat < 2; ++repeat) {
    ASSERT_TRUE(invocation.bind(inputs).ok());
    auto work = invocation.submit();
    ASSERT_TRUE(work) << work.error().message();
    auto result = work->wait();
    ASSERT_TRUE(result) << result.error().message();
    const auto arrays = result->get()->arrays();
    ASSERT_EQ(arrays.size(), 1U);
    for (std::size_t i = 0; i < test.expected.size(); ++i)
      EXPECT_NEAR(arrays[0].data<float>()[i], test.expected[i], 2.0e-5F)
          << i;
  }
}
} // namespace
