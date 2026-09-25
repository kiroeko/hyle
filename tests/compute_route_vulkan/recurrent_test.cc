#include "recurrent_test_support.h"
#include "tensor_test_support.h"
#include "vulkan_program_internal.h"

#include "../model_fragment_test_support.h"
#include "../compute_models/d5_qwen38_gdn_fixture.h"
#include "../compute_models/d5_qwen38_profile_cases.h"
#include "../sparse_fragment_test_support.h"

#include "hyle/compute/model_ops.h"

namespace {
using namespace vulkan_recurrent_test;

class VulkanQwen38GdnProfiles
    : public ::testing::TestWithParam<
          const d5_qwen38_profile_cases::ProfileCase *> {};

TEST_P(VulkanQwen38GdnProfiles,
       D5Qwen38GroupedHeadGdnUsesOrdinaryRouteAndReplays) {
  const d5_qwen38_gdn_fixture::Fixture fixture(
      0, GetParam()->gdn_key_heads, GetParam()->gdn_value_heads);
  auto context = vulkan_test::context().value();
  std::vector<v::Argument> arguments;
  arguments.reserve(fixture.inputs.size());
  for (const auto &description : fixture.inputs)
    arguments.emplace_back(description);
  const auto workload = d5_qwen38_gdn_fixture::workload(
      *GetParam()->profile, c::WorkloadPhase::kDecode);
  auto program = v::prepare(context, c::Operation(c::GatedDeltaNet{}),
                            arguments, workload);
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_EQ(program->workload(), workload);
  std::array<v::Tensor, 6> resident;
  for (std::size_t port = 0; port < resident.size(); ++port)
    resident[port] =
        vulkan_test::upload(context, fixture.inputs[port], fixture.data[port])
            .value();
  auto invocation = program->create_invocation().value();
  ASSERT_TRUE(invocation.bind(resident).ok());
  const auto expected = fixture.oracle();
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto work = invocation.submit();
    ASSERT_TRUE(work) << work.error().message();
    auto result = work->wait();
    ASSERT_TRUE(result) << result.error().message();
    const auto outputs = result->get()->tensors();
    ASSERT_EQ(outputs.size(), expected.size());
    for (std::size_t port = 0; port < outputs.size(); ++port)
      expect(context, outputs[port], expected[port]);
  }
}

INSTANTIATE_TEST_SUITE_P(
    D5OfficialAndPrimaryGguf, VulkanQwen38GdnProfiles,
    ::testing::Values(&d5_qwen38_profile_cases::kOfficial27B,
                      &d5_qwen38_profile_cases::kPrimaryGguf27B));

INSTANTIATE_TEST_SUITE_P(
    D5FidelityGguf, VulkanQwen38GdnProfiles,
    ::testing::Values(&d5_qwen38_profile_cases::kFidelityGguf27B));

INSTANTIATE_TEST_SUITE_P(
    D5Official24T, VulkanQwen38GdnProfiles,
    ::testing::Values(&d5_qwen38_profile_cases::kOfficial24T));

TEST(VulkanRecurrent, OriginalDirectStateAndIndependentRetainedOutputs) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(),
                                  64 * 1024 * 1024);
  auto context = vulkan_test::context(&quota).value();
  for (const auto depth : {16, 32, 64, 128}) {
    for (const auto tokens : {1, 5}) {
      SCOPED_TRACE(depth);
      SCOPED_TRACE(tokens);
      Fixture original(tokens, 2, depth);
      Fixture changed(tokens, 2, depth, 13);
      const auto reference = original.oracle();
      const auto changed_reference = changed.oracle();
      const auto &d = original.descriptions;
      auto program = v::prepare(context, c::GatedDeltaNet{}, d[0], d[1], d[2],
                                d[3], d[4], d[5]);
      ASSERT_TRUE(program) << program.error().message();
      const auto &plan = v::detail::Access::program(*program);
      ASSERT_TRUE(plan->dispatches.back().region);
      auto first = program->create_invocation().value();
      auto second = program->create_invocation().value();
      auto inputs = original.upload(context);
      auto next_inputs = changed.upload(context);
      ASSERT_TRUE(first.bind(inputs).ok());
      ASSERT_TRUE(second.bind(next_inputs).ok());
      std::array<v::Tensor, 2> retained;
      std::size_t warmed_bytes = 0;
      for (int repeat = 0; repeat < 3; ++repeat) {
        auto &slot = repeat == 0 ? first : second;
        const auto &expected = repeat == 0 ? reference : changed_reference;
        auto work = slot.submit();
        ASSERT_TRUE(work) << work.error().message();
        auto result = work->wait();
        ASSERT_TRUE(result) << result.error().message();
        const auto outputs = result->get()->tensors();
        ASSERT_EQ(outputs.size(), 2U);
        for (std::size_t port = 0; port < outputs.size(); ++port) {
          expect(context, outputs[port], expected[port]);
          if (repeat == 0)
            retained[port] = outputs[port];
          else
            expect(context, retained[port], reference[port]);
        }
        if (repeat == 1)
          warmed_bytes = quota.used();
        else if (repeat == 2)
          EXPECT_EQ(quota.used(), warmed_bytes);
      }
      EXPECT_EQ(vulkan_test::read(context, inputs[5]).value(),
                original.data[5]);
      EXPECT_FALSE(first.submit());
      retained[0] = {};
      EXPECT_FALSE(
          first.submit()); // State alone retains the same execution slot.
      retained[1] = {};
      auto replay = first.submit();
      ASSERT_TRUE(replay);
      ASSERT_TRUE(replay->wait());
    }
  }
  EXPECT_TRUE(context.close().ok());
  context = {};
  EXPECT_EQ(quota.used(), 0U);
}

TEST(VulkanRecurrent, RejectsUnsupportedDomainAndPreservesSuccessfulBinding) {
  auto context = vulkan_test::context().value();
  Fixture supported(3, 2, 32);
  Fixture unsupported(3, 2, 24);
  auto rejected = v::prepare(context, c::Operation(c::GatedDeltaNet{}),
                             unsupported.arguments());
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code(), hyle::StatusCode::Unimplemented);
  auto arguments = supported.arguments();
  arguments.pop_back();
  EXPECT_FALSE(
      v::prepare(context, c::Operation(c::GatedDeltaNet{}), arguments));
  auto program = v::prepare(context, c::Operation(c::GatedDeltaNet{}),
                            supported.arguments())
                     .value();
  auto slot = program.create_invocation().value();
  auto inputs = supported.upload(context);
  ASSERT_TRUE(slot.bind(inputs).ok());
  auto invalid = inputs;
  invalid[5] = inputs[0];
  EXPECT_FALSE(slot.bind(invalid).ok());
  auto work = slot.submit().value();
  auto result = work.wait();
  ASSERT_TRUE(result);
  const auto expected = supported.oracle();
  expect(context, result->get()->tensors()[0], expected[0]);
  expect(context, result->get()->tensors()[1], expected[1]);
}

TEST(VulkanRecurrent, PerChannelKdaGateUsesMatureManagedRegion) {
  auto context = vulkan_test::context().value();
  Fixture fixture(3, 2, 32, 0, true);
  auto program = v::prepare(context, c::Operation(c::GatedDeltaNet{}),
                            fixture.arguments());
  ASSERT_TRUE(program) << program.error().message();
  auto slot = program->create_invocation().value();
  auto inputs = fixture.upload(context);
  ASSERT_TRUE(slot.bind(inputs).ok());
  auto work = slot.submit();
  ASSERT_TRUE(work) << work.error().message();
  auto result = work->wait();
  ASSERT_TRUE(result) << result.error().message();
  const auto expected = fixture.oracle();
  expect(context, result->get()->tensors()[0], expected[0]);
  expect(context, result->get()->tensors()[1], expected[1]);
}

TEST(VulkanRecurrent, MatureCausalConvUsesManagedRegion) {
  auto context = vulkan_test::context().value();
  const auto samples_desc =
      c::TensorDesc::dense(c::DataType::kFloat32,
                           std::array<std::int64_t, 3>{1, 5, 2})
          .value();
  const auto weights_desc =
      c::TensorDesc::dense(c::DataType::kFloat32,
                           std::array<std::int64_t, 2>{2, 3})
          .value();
  const std::array arguments{v::Argument(samples_desc),
                             v::Argument(weights_desc)};
  auto program = v::prepare(context, c::model_ops::causal_conv1d(), arguments);
  ASSERT_TRUE(program) << program.error().message();
  const std::array<float, 10> samples{1, 10, 2, 20, 3, 30, 4, 40, 5, 50};
  const std::array<float, 6> weights{1, 2, 3, -1, 0, 1};
  const std::array inputs{
      vulkan_test::upload(context, samples_desc, samples).value(),
      vulkan_test::upload(context, weights_desc, weights).value()};
  auto slot = program->create_invocation().value();
  ASSERT_TRUE(slot.bind(inputs).ok());
  auto work = slot.submit();
  ASSERT_TRUE(work) << work.error().message();
  auto result = work->wait();
  ASSERT_TRUE(result) << result.error().message();
  EXPECT_EQ(vulkan_test::read(context, result->get()->tensors()[0]).value(),
            (std::vector<float>{14, 20, 20, 20, 26, 20}));
}

TEST(VulkanRecurrent, MatureMoeAndHyperConnectionBatchUsesManagedRegions) {
  auto context = vulkan_test::context().value();
  for (const auto &test : model_fragment_test::cases()) {
    SCOPED_TRACE(test.name);
    std::vector<v::Tensor> inputs;
    inputs.reserve(test.inputs.size());
    for (std::size_t i = 0; i < test.inputs.size(); ++i)
      inputs.push_back(
          context.upload(test.inputs[i], test.payloads[i]).value());
    auto output =
        vulkan_tensor_test::execute(context, test.operation, inputs);
    ASSERT_TRUE(output) << output.error().message();
    const auto actual = vulkan_test::read(context, *output).value();
    ASSERT_EQ(actual.size(), test.expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], test.expected[i], 2.0e-5F) << i;
  }
}

TEST(VulkanRecurrent, MatureLightningIndexerUsesManagedRegion) {
  const sparse_fragment_test::Fixture test;
  auto context = vulkan_test::context().value();
  std::vector<v::Tensor> inputs;
  for (std::size_t i = 0; i < test.inputs.size(); ++i)
    inputs.push_back(context.upload(test.inputs[i], test.payloads[i]).value());
  auto output = vulkan_tensor_test::execute(context, test.operation, inputs);
  ASSERT_TRUE(output) << output.error().message();
  const auto actual = vulkan_test::read(context, *output).value();
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_NEAR(actual[i], test.expected[i], 2.0e-5F) << i;
}
} // namespace
