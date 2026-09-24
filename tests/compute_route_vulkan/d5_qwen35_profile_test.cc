#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "../compute_models/d5_qwen35_fixture.h"
#include "fixture.h"
#include "hyle/compute/model_ops.h"
#include "vulkan_program_internal.h"

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace d = v::detail;

c::TensorDesc dense(c::DataType type,
                    std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}

bool has_mature_dispatch(const v::Program &program) {
  const auto &dispatches = d::Access::program(program)->dispatches;
  return std::ranges::any_of(dispatches, [](const auto &dispatch) {
    return dispatch.region != nullptr || dispatch.iree.has_value();
  });
}

TEST(VulkanManaged,
     D5Qwen35OfficialAllRegionShapesReachMatureProvidersInOneBatch) {
  auto context = vulkan_test::context().value();
  const auto workload =
      d5_qwen35_fixture::workload(c::WorkloadPhase::kDecode, 0xd5);
  const auto check = [&](std::string_view name, c::Operation operation,
                         std::span<const c::TensorDesc> inputs) {
    SCOPED_TRACE(name);
    std::vector<v::Argument> arguments;
    arguments.reserve(inputs.size());
    for (const auto &input : inputs)
      arguments.emplace_back(input);
    auto program = v::prepare(context, operation, arguments, workload);
    ASSERT_TRUE(program) << program.error().message();
    EXPECT_TRUE(has_mature_dispatch(*program));
    EXPECT_EQ(program->workload(), workload);
  };

  constexpr auto hidden = d5_qwen35_fixture::kHidden;
  constexpr auto ffn = d5_qwen35_fixture::kFfnHidden;
  const auto token = dense(c::DataType::kBFloat16, {2, hidden});
  const auto scale = dense(c::DataType::kBFloat16, {hidden});
  const auto table = dense(c::DataType::kBFloat16,
                           {d5_qwen35_fixture::kEmbeddingRows, hidden});
  const auto ids = dense(c::DataType::kInt32, {2});
  check("R01-gather", c::Operation(c::Gather{}), std::array{table, ids});
  check("R01-norm", c::Operation(c::RmsNorm{.epsilon = 1.0e-6F}),
        std::array{token, scale});
  check("R01-residual", c::Operation(c::Binary{c::BinaryKind::kAdd}),
        std::array{token, token});
  check("R01-head", c::Operation(c::MatMul{}),
        std::array{token, table.transposed().value()});

  const auto activation = dense(c::DataType::kBFloat16, {1, hidden});
  const auto expanded = dense(c::DataType::kBFloat16, {1, ffn});
  const auto gate_weight =
      dense(c::DataType::kBFloat16, {ffn, hidden}).transposed().value();
  const auto down_weight =
      dense(c::DataType::kBFloat16, {hidden, ffn}).transposed().value();
  check("R02-norm", c::Operation(c::RmsNorm{}), std::array{activation, scale});
  check("R02-up", c::Operation(c::MatMul{}),
        std::array{activation, gate_weight});
  check(
      "R02-gate",
      c::Operation(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}),
      std::array{expanded, expanded});
  check("R02-down", c::Operation(c::MatMul{}),
        std::array{expanded, down_weight});

  constexpr std::int64_t query_heads = 8;
  constexpr std::int64_t kv_heads = 2;
  constexpr std::int64_t depth = 256;
  const auto query = dense(c::DataType::kBFloat16, {1, 1, query_heads, depth});
  const auto kv = dense(c::DataType::kBFloat16, {1, 1, kv_heads, depth});
  const auto head_scale = dense(c::DataType::kBFloat16, {depth});
  const auto position = dense(c::DataType::kInt32, {1});
  const auto frequency = dense(c::DataType::kFloat32, {depth / 2});
  check("R04-qnorm", c::Operation(c::RmsNorm{}), std::array{query, head_scale});
  const c::RoPE rope{.rotary_dimensions = depth,
                     .sequence_axis = 1,
                     .layout = c::RopeLayout::kHalfSplit};
  check("R04-rope", c::Operation(rope), std::array{query, position, frequency});
  check("R04-attention",
        c::Operation(c::ScaledDotProductAttention{
            .layout = c::AttentionLayout::kBatchSequenceHeadDepth,
            .scale = 0.0625F,
            .causal = true,
            .accumulation_type = c::DataType::kFloat32}),
        std::array{query, kv, kv});

  constexpr auto heads = d5_qwen35_fixture::kGdnHeads;
  constexpr auto state_depth = d5_qwen35_fixture::kGdnDepth;
  constexpr auto channels = d5_qwen35_fixture::kConvChannels;
  constexpr auto kernel = d5_qwen35_fixture::kConvKernel;
  const auto conv_samples = dense(c::DataType::kFloat32, {1, kernel, channels});
  const auto conv_weight = dense(c::DataType::kFloat32, {channels, kernel});
  check("R05-conv", c::model_ops::causal_conv1d(),
        std::array{conv_samples, conv_weight});
  const auto qkv = dense(c::DataType::kFloat32, {1, 1, heads, state_depth});
  const auto recurrent_gate = dense(c::DataType::kFloat32, {1, 1, heads, 1});
  const auto state =
      dense(c::DataType::kFloat32, {1, heads, state_depth, state_depth});
  check("R05-state", c::Operation(c::GatedDeltaNet{}),
        std::array{qkv, qkv, qkv, recurrent_gate, recurrent_gate, state});

  const auto joined =
      dense(c::DataType::kBFloat16, {1, d5_qwen35_fixture::kMtpInput});
  const auto mtp_weight =
      dense(c::DataType::kBFloat16, {hidden, d5_qwen35_fixture::kMtpInput})
          .transposed()
          .value();
  check("R13-concat", c::Operation(c::Concat{.axis = 1}),
        std::array{activation, activation});
  check("R14-projection", c::Operation(c::MatMul{}),
        std::array{joined, mtp_weight});
  check("R15-boundary", c::Operation(c::Cast{c::DataType::kFloat32}),
        std::array{activation});
}
} // namespace
