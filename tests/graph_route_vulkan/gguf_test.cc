#include "../compute_route_vulkan/gguf_test_support.h"
#include "hyle/graph/vulkan.h"

namespace {
using namespace vulkan_gguf_test;
namespace g = hyle::graph;
namespace gv = g::vulkan;

TEST(VulkanGraphGguf, PackedMatMulUsesCapturedNativeWeightAndReplays) {
  QuantizerCleanup cleanup;
  auto context = vulkan_test::context().value();
  constexpr int k = 256;
  constexpr int n = 8;
  constexpr int m = 2;
  for (const auto type : selected_matrix_types()) {
    SCOPED_TRACE(static_cast<int>(type));
    const auto input_desc =
        vulkan_tensor_test::dense(c::DataType::kFloat32, {m, k});
    const auto weight_desc =
        c::gguf_weight_description(type, c::DataType::kFloat32, k, n).value();
    const auto physical = weights(n, k, static_cast<int>(type));
    auto packed = pack(type, physical, n, k, *weight_desc.required_bytes());
    auto weight = context.upload(weight_desc, packed.bytes).value();
    auto builder = g::GraphBuilder::create().value();
    const auto input = builder.input(input_desc).value();
    const auto constant = builder.constant(weight_desc).value();
    const auto output = builder.add(c::MatMul{}, input, constant).value();
    ASSERT_TRUE(builder.publish(output).ok());
    auto graph = std::move(builder).finish().value();
    gv::CompileReport report;
    auto program =
        gv::compile(context, graph, std::array{weight}, {.report = &report});
    ASSERT_TRUE(program) << program.error().message();
    EXPECT_EQ(report.managed_region_count, 1U);
    EXPECT_NE(report.numerical_policy.find("GGUF"), std::string::npos);
    auto invocation = program->create_invocation().value();
    for (int replay = 1; replay <= 2; ++replay) {
      const auto values = activations(m, k, replay);
      auto activation = vulkan_tensor_test::make(context, input_desc, values);
      ASSERT_TRUE(invocation.bind(activation).ok());
      auto work = invocation.submit().value();
      auto result = work.wait();
      ASSERT_TRUE(result) << result.error().message();
      expect_native_projection(context, result->get()->tensors()[0], type,
                               values, packed, m, n, k);
    }
  }
}

TEST(VulkanGraphGguf, TargetEmbeddingFormatsUseCheckedNativeRowsAndKeepLeases) {
  QuantizerCleanup cleanup;
  auto context = vulkan_test::context().value();
  const auto ids_desc = vulkan_tensor_test::dense(c::DataType::kInt32, {2});
  for (const auto type : kEmbeddingTypes) {
    SCOPED_TRACE(static_cast<int>(type));
    const auto stored = c::gguf_weight_description(type, c::DataType::kFloat32, 256, 3).value();
    const auto table_desc = stored.transposed().value();
    auto packed = pack(type, weights(3, 256, 11), 3, 256, *stored.required_bytes());
    auto table = context.upload(table_desc, packed.bytes).value();
    auto builder = g::GraphBuilder::create().value();
    const auto weight = builder.constant(table_desc).value();
    const auto index = builder.input(ids_desc).value();
    const auto output = builder.add(c::Gather{}, weight, index).value();
    ASSERT_TRUE(builder.publish(output).ok());
    auto graph = std::move(builder).finish().value();
    auto program = gv::compile(context, graph, std::array{table});
    ASSERT_TRUE(program) << program.error().message();
    std::optional<v::Tensor> retained;
    std::vector<std::byte> old_bytes;
    for (const std::array<std::int32_t, 2> ids : {std::array<std::int32_t, 2>{2, 0}, {1, 1}}) {
      auto invocation = program->create_invocation().value();
      auto selected = context.upload(ids_desc, std::as_bytes(std::span(ids))).value();
      ASSERT_TRUE(invocation.bind(selected).ok());
      auto work = invocation.submit().value();
      auto result = work.wait(); ASSERT_TRUE(result) << result.error().message();
      std::vector<double> expected;
      for (auto row : ids)
        expected.insert(expected.end(), packed.decoded.begin() + row * 256,
                        packed.decoded.begin() + (row + 1) * 256);
      expect_close(context, result->get()->tensors()[0], expected);
      if (retained) EXPECT_EQ(vulkan_tensor_test::bytes(context, *retained), old_bytes);
      else {
        retained = result->get()->tensors()[0];
        old_bytes = vulkan_tensor_test::bytes(context, *retained);
      }
    }
    const std::array<std::int32_t, 2> bad{0, 3};
    auto invalid = context.upload(ids_desc, std::as_bytes(std::span(bad))).value();
    auto failed = program->create_invocation().value();
    ASSERT_TRUE(failed.bind(invalid).ok());
    auto work = failed.submit().value();
    auto result = work.wait(); ASSERT_FALSE(result);
    EXPECT_TRUE(result.error().IsOutOfRange());
    EXPECT_EQ(vulkan_tensor_test::bytes(context, *retained), old_bytes);
  }
}

TEST(VulkanGraphGguf, PackedFormatsGroupedExpertsUseCapturedMulMatId) {
  QuantizerCleanup cleanup;
  auto context = vulkan_test::context().value();
  for (const auto type : selected_matrix_types(true)) {
    SCOPED_TRACE(static_cast<int>(type));
    constexpr int experts = 4;
    constexpr int n = 8;
    constexpr int k = 256;
    constexpr int tokens = 2;
    constexpr int slots = 2;
    const auto weight_desc =
        c::gguf_grouped_weight_description(type,
                                           c::DataType::kFloat32, experts, n, k)
            .value();
    const auto activation_desc =
        vulkan_tensor_test::dense(c::DataType::kFloat32, {tokens, slots, k});
    const auto ids_desc =
        vulkan_tensor_test::dense(c::DataType::kInt32, {tokens, slots});
    const auto physical = weights(experts * n, k, 7);
    auto packed = pack(type, physical, experts * n, k,
                       *weight_desc.required_bytes());
    auto weight = context.upload(weight_desc, packed.bytes).value();
    auto builder = g::GraphBuilder::create().value();
    const auto constant = builder.constant(weight_desc).value();
    const auto activation = builder.input(activation_desc).value();
    const auto ids = builder.input(ids_desc).value();
    const std::array ports{constant, activation, ids};
    const auto node =
        builder.add_node(c::model_ops::grouped_matmul(), ports).value();
    ASSERT_TRUE(builder.publish(builder.result(node).value()).ok());
    auto graph = std::move(builder).finish().value();
    gv::CompileReport report;
    auto program =
        gv::compile(context, graph, std::array{weight}, {.report = &report});
    ASSERT_TRUE(program) << program.error().message();
    EXPECT_EQ(report.managed_region_count, 1U);
    const std::array<std::int32_t, tokens * slots> selected_values{0, 3, 1, 2};
    auto selected =
        context.upload(ids_desc, std::as_bytes(std::span(selected_values)))
            .value();
    auto invocation = program->create_invocation().value();
    for (int replay = 1; replay <= 2; ++replay) {
      const auto values = activations(tokens * slots, k, replay);
      auto samples = vulkan_tensor_test::make(context, activation_desc, values);
      ASSERT_TRUE(invocation.bind(samples, selected).ok());
      auto work = invocation.submit().value();
      auto result = work.wait();
      ASSERT_TRUE(result) << result.error().message();
      expect_native_projection(context, result->get()->tensors()[0], type,
                               values, packed, tokens * slots, n, k, selected_values);
    }
  }
}

TEST(VulkanGraphGguf,
     DenseBfloatGroupedExpertsUseCapturedMulMatIdAndReplay) {
  auto context = vulkan_test::context().value();
  constexpr int experts = 3;
  constexpr int n = 2;
  constexpr int k = 32;
  constexpr int tokens = 1;
  constexpr int slots = 2;
  const auto weight_desc = vulkan_tensor_test::dense(
      c::DataType::kBFloat16, {experts, n, k});
  const auto activation_desc = vulkan_tensor_test::dense(
      c::DataType::kFloat32, {tokens, slots, k});
  const auto ids_desc =
      vulkan_tensor_test::dense(c::DataType::kInt32, {tokens, slots});
  constexpr std::array<double, experts * n> row_values{1.0, 2.0, 3.0,
                                                        4.0, -1.0, 0.5};
  std::vector<double> weight_values(experts * n * k);
  for (std::size_t row = 0; row < row_values.size(); ++row)
    std::fill_n(weight_values.begin() + row * k, k, row_values[row]);
  auto weight =
      vulkan_tensor_test::make(context, weight_desc, weight_values);
  auto builder = g::GraphBuilder::create().value();
  const auto constant = builder.constant(weight_desc).value();
  const auto activation = builder.input(activation_desc).value();
  const auto ids = builder.input(ids_desc).value();
  const std::array ports{constant, activation, ids};
  const auto node =
      builder.add_node(c::model_ops::grouped_matmul(), ports).value();
  ASSERT_TRUE(builder.publish(builder.result(node).value()).ok());
  auto graph = std::move(builder).finish().value();
  gv::CompileReport report;
  auto program =
      gv::compile(context, graph, std::array{weight}, {.report = &report});
  ASSERT_TRUE(program) << program.error().message();
  EXPECT_EQ(report.managed_region_count, 1U);
  constexpr std::array<std::int32_t, tokens * slots> selected_values{0, 2};
  auto selected =
      context.upload(ids_desc, std::as_bytes(std::span(selected_values)))
          .value();
  auto invocation = program->create_invocation().value();
  for (int replay = 1; replay <= 2; ++replay) {
    const std::vector<double> activation_values(tokens * slots * k,
                                                0.25 * replay);
    auto samples =
        vulkan_tensor_test::make(context, activation_desc, activation_values);
    ASSERT_TRUE(invocation.bind(samples, selected).ok());
    auto work = invocation.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    const std::array expected{8.0 * replay, 16.0 * replay,
                              -8.0 * replay, 4.0 * replay};
    vulkan_tensor_test::expect(context, result->get()->tensors()[0], expected);
  }
}
} // namespace
