#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>

#include "../compute_models/d5_qwen35_fixture.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/compute/cuda.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/providers/cublas.h"
#include "hyle/compute/providers/cuda.h"
#include "hyle/compute/providers/flashinfer.h"
#include "hyle/compute/providers/ggml_cuda.h"

namespace {
namespace c = hyle::compute;
namespace cuda = hyle::compute::cuda;
namespace mem = hyle::mem;

c::TensorDesc dense(c::DataType type,
                    std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}

bool mature(c::ProviderId provider) {
  return provider == c::ggml_cuda::kProviderId ||
         provider == c::cublas::kProviderId ||
         provider == c::cuda_native::kSglangProviderId ||
         provider == c::cuda_native::kCcclProviderId ||
         provider == c::flashinfer::kProviderId;
}

TEST(CudaDefaultCompute,
     D5Qwen35OfficialAllRegionShapesReachMatureProvidersInOneBatch) {
  auto context = cuda::create_context().value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto workload =
      d5_qwen35_fixture::workload(c::WorkloadPhase::kDecode, 0xd5);
  const auto check = [&](std::string_view name, c::Operation operation,
                         std::span<const c::TensorDesc> inputs,
                         std::span<const c::TensorDesc> outputs) {
    SCOPED_TRACE(name);
    auto prepared = c::prepare(context, std::move(operation), inputs, outputs,
                               memory.ref(), workload);
    ASSERT_TRUE(prepared) << prepared.error().message();
    EXPECT_TRUE(mature(prepared->provider().id))
        << c::operation_name(prepared->kind()) << ": "
        << prepared->provider().name;
    EXPECT_NE(prepared->provider().id, c::cuda_native::kProviderId);
    EXPECT_EQ(prepared->workload(), workload);
  };

  constexpr auto hidden = d5_qwen35_fixture::kHidden;
  constexpr auto ffn = d5_qwen35_fixture::kFfnHidden;
  const auto token = dense(c::DataType::kBFloat16, {2, hidden});
  const auto scale = dense(c::DataType::kBFloat16, {hidden});
  const auto table = dense(c::DataType::kBFloat16,
                           {d5_qwen35_fixture::kEmbeddingRows, hidden});
  const auto ids = dense(c::DataType::kInt32, {2});
  const auto logits =
      dense(c::DataType::kBFloat16, {2, d5_qwen35_fixture::kEmbeddingRows});
  check("R01-gather", c::Operation(c::Gather{}), std::array{table, ids},
        std::array{token});
  check("R01-norm", c::Operation(c::RmsNorm{.epsilon = 1.0e-6F}),
        std::array{token, scale}, std::array{token});
  check("R01-residual", c::Operation(c::Binary{c::BinaryKind::kAdd}),
        std::array{token, token}, std::array{token});
  check("R01-head", c::Operation(c::MatMul{}),
        std::array{token, table.transposed().value()}, std::array{logits});

  const auto activation = dense(c::DataType::kBFloat16, {1, hidden});
  const auto expanded = dense(c::DataType::kBFloat16, {1, ffn});
  const auto gate_weight =
      dense(c::DataType::kBFloat16, {ffn, hidden}).transposed().value();
  const auto down_weight =
      dense(c::DataType::kBFloat16, {hidden, ffn}).transposed().value();
  check("R02-norm", c::Operation(c::RmsNorm{}), std::array{activation, scale},
        std::array{activation});
  check("R02-up", c::Operation(c::MatMul{}),
        std::array{activation, gate_weight}, std::array{expanded});
  check(
      "R02-gate",
      c::Operation(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}),
      std::array{expanded, expanded}, std::array{expanded});
  check("R02-down", c::Operation(c::MatMul{}),
        std::array{expanded, down_weight}, std::array{activation});

  constexpr std::int64_t query_heads = 8;
  constexpr std::int64_t kv_heads = 2;
  constexpr std::int64_t depth = 256;
  const auto query = dense(c::DataType::kBFloat16, {1, 1, query_heads, depth});
  const auto kv = dense(c::DataType::kBFloat16, {1, 1, kv_heads, depth});
  const auto head_scale = dense(c::DataType::kBFloat16, {depth});
  const auto position = dense(c::DataType::kInt32, {1});
  const auto frequency = dense(c::DataType::kFloat32, {depth / 2});
  check("R04-qnorm", c::Operation(c::RmsNorm{}), std::array{query, head_scale},
        std::array{query});
  const c::RoPE rope{.rotary_dimensions = depth,
                     .sequence_axis = 1,
                     .layout = c::RopeLayout::kHalfSplit};
  check("R04-rope", c::Operation(rope), std::array{query, position, frequency},
        std::array{query});
  check("R04-attention",
        c::Operation(c::ScaledDotProductAttention{
            .layout = c::AttentionLayout::kBatchSequenceHeadDepth,
            .scale = 0.0625F,
            .causal = true,
            .accumulation_type = c::DataType::kFloat32}),
        std::array{query, kv, kv}, std::array{query});

  constexpr auto heads = d5_qwen35_fixture::kGdnHeads;
  constexpr auto state_depth = d5_qwen35_fixture::kGdnDepth;
  constexpr auto channels = d5_qwen35_fixture::kConvChannels;
  constexpr auto kernel = d5_qwen35_fixture::kConvKernel;
  const auto conv_samples = dense(c::DataType::kFloat32, {1, kernel, channels});
  const auto conv_weight = dense(c::DataType::kFloat32, {channels, kernel});
  const auto conv_output = dense(c::DataType::kFloat32, {1, 1, channels});
  check("R05-conv", c::model_ops::causal_conv1d(),
        std::array{conv_samples, conv_weight}, std::array{conv_output});
  const auto qkv = dense(c::DataType::kFloat32, {1, 1, heads, state_depth});
  const auto recurrent_gate = dense(c::DataType::kFloat32, {1, 1, heads, 1});
  const auto state =
      dense(c::DataType::kFloat32, {1, heads, state_depth, state_depth});
  check("R05-state", c::Operation(c::GatedDeltaNet{}),
        std::array{qkv, qkv, qkv, recurrent_gate, recurrent_gate, state},
        std::array{qkv, state});

  const auto joined =
      dense(c::DataType::kBFloat16, {1, d5_qwen35_fixture::kMtpInput});
  const auto mtp_weight =
      dense(c::DataType::kBFloat16, {hidden, d5_qwen35_fixture::kMtpInput})
          .transposed()
          .value();
  const auto fp32 = dense(c::DataType::kFloat32, {1, hidden});
  check("R13-concat", c::Operation(c::Concat{.axis = 1}),
        std::array{activation, activation}, std::array{joined});
  check("R14-projection", c::Operation(c::MatMul{}),
        std::array{joined, mtp_weight}, std::array{activation});
  check("R15-boundary", c::Operation(c::Cast{c::DataType::kFloat32}),
        std::array{activation}, std::array{fp32});
}

TEST(CudaDefaultCompute, D5Qwen35OfficialFfnAdmissionDoesNotUseFrozenFallback) {
  constexpr auto ffn = d5_qwen35_fixture::kFfnHidden;
  const auto expanded = dense(c::DataType::kBFloat16, {1, ffn});
  const std::array inputs{expanded, expanded};
  const std::array outputs{expanded};
  const auto workload =
      d5_qwen35_fixture::workload(c::WorkloadPhase::kDecode, 2);
  auto context = cuda::create_context().value();
  auto resource =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  auto prepared = c::prepare(
      context,
      c::Operation(c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}),
      inputs, outputs, resource.ref(), workload);
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, c::ggml_cuda::kProviderId)
      << prepared->provider().name;
  EXPECT_NE(prepared->provider().id, c::cuda_native::kProviderId);
  EXPECT_EQ(prepared->workload(), workload);
}
} // namespace
