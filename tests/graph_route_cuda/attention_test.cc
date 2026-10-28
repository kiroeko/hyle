#include "hyle/compute/providers/ggml_cuda.h"
#include "hyle/compute/providers/torch_cuda.h"
#include "hyle/graph/cuda.h"

#include "../compute_provider_flashinfer/attention_test_support.h"

namespace {
using namespace flashinfer_test;
namespace graph = hyle::graph;

template <class T>
void run_attention(compute::DataType type, int query_length, int kv_length,
                   double tolerance) {
  auto context = graph::cuda::create_context().value();
  const auto environment = cuda::environment(context).value();
  const int sm = environment.sm_major * 10 + environment.sm_minor;
  auto expected_provider = compute::ggml_cuda::kProviderId;
  bool supported = type != compute::DataType::kBFloat16;
#if defined(HYLE_COMPUTE_TORCH_CUDA)
  if (type == compute::DataType::kBFloat16 && sm >= 80 && sm <= 121) {
    expected_provider = compute::torch_cuda::kProviderId;
    supported = true;
  }
#endif
#if defined(HYLE_COMPUTE_FLASHINFER)
  if (sm == 89) {
    expected_provider = compute::flashinfer::kProviderId;
    supported = true;
  }
#endif
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  Tensor query(type, {2, query_length, 8, 256}, memory.ref());
  Tensor key(type, {2, kv_length, 2, 256}, memory.ref());
  Tensor value(type, {2, kv_length, 2, 256}, memory.ref());
  Tensor direct_output(type, {2, query_length, 8, 256}, memory.ref());
  Tensor graph_output(type, {2, query_length, 8, 256}, memory.ref());
  const compute::ScaledDotProductAttention operation{.scale = 0.0625F,
                                                     .causal = true};
  const std::array inputs{query.view.as_const(), key.view.as_const(),
                          value.view.as_const()};
  const std::array outputs{direct_output.view};
  auto prepared_result = compute::prepare(context, compute::Operation(operation),
                                          inputs, outputs, memory.ref());
  if (!supported) {
    ASSERT_FALSE(prepared_result);
    EXPECT_TRUE(prepared_result.error().IsUnimplemented()) << prepared_result.error().message();
    return;
  }
  ASSERT_TRUE(prepared_result) << prepared_result.error().message();
  auto prepared = std::move(*prepared_result);
  ASSERT_EQ(prepared.provider().id, expected_provider);
  const auto requirement = prepared.workspace_requirement();
  mem::Buffer workspace;
  if (requirement.size != 0) {
    auto allocated = mem::Buffer::create(memory.ref(), requirement.size,
                                         requirement.alignment);
    ASSERT_TRUE(allocated) << allocated.error().message();
    workspace = std::move(*allocated);
  }
  auto direct_result = prepared.create_invocation(workspace.view());
  ASSERT_TRUE(direct_result) << direct_result.error().message();
  auto direct = std::move(*direct_result);

  auto builder = graph::GraphBuilder::create().value();
  auto q_id = builder.input(query.description).value();
  auto k_id = builder.input(key.description).value();
  auto v_id = builder.input(value.description).value();
  auto output_id = builder.add(operation, q_id, k_id, v_id).value();
  ASSERT_TRUE(builder.publish(output_id).ok());
  auto logical = std::move(builder).finish().value();
  auto compiled = graph::cuda::compile(logical, context, memory.ref());
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 1u);
  EXPECT_EQ(compiled->report.step(0)->provider.id, expected_provider);
  auto invocation_result = compiled->executable.create_invocation();
  ASSERT_TRUE(invocation_result) << invocation_result.error().message();
  auto invocation = std::move(*invocation_result);
  ASSERT_TRUE(invocation.bind(q_id, query.view.as_const()).ok());
  ASSERT_TRUE(invocation.bind(k_id, key.view.as_const()).ok());
  ASSERT_TRUE(invocation.bind(v_id, value.view.as_const()).ok());
  ASSERT_TRUE(invocation.bind(output_id, graph_output.view).ok());
  auto direct_queue = context.create_queue().value();
  auto graph_queue = context.create_queue().value();
  ASSERT_TRUE(graph::cuda::prepare_replay(invocation, graph_queue).ok());
  EXPECT_EQ(invocation.replay_route(), graph::ReplayRoute::kCommands);
  for (int seed : {0, 3, 9}) {
    const auto q = payload<T>(query, seed);
    const auto k = payload<T>(key, seed + 5);
    const auto v = payload<T>(value, seed + 13);
    auto direct_done = direct.submit(direct_queue, inputs, outputs).value();
    auto graph_done = invocation.submit(graph_queue).value();
    ASSERT_TRUE(graph_done.wait().ok());
    ASSERT_TRUE(direct_done.wait().ok());
    check_output(direct_output, q, k, v, query_length, kv_length, tolerance);
    check_output(graph_output, q, k, v, query_length, kv_length, tolerance);
  }
}

TEST(CudaDefaultGraph, MaturePrefillDecodeThroughNormalEntries) {
  run_attention<__half>(compute::DataType::kFloat16, 17, 31, 0.002);
  run_attention<__half>(compute::DataType::kFloat16, 1, 513, 0.002);
  run_attention<__nv_bfloat16>(compute::DataType::kBFloat16, 17, 31, 0.01);
  run_attention<__nv_bfloat16>(compute::DataType::kBFloat16, 1, 513, 0.01);
}
#if !defined(HYLE_COMPUTE_TORCH_CUDA)
TEST(CudaDefaultGraph, BroadBf16AttentionKeepsOriginalF32FallbackWithoutAten) {
  auto context = graph::cuda::create_context().value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  // Outside FlashInfer's domain, the existing ggml program keeps Q/K/V and
  // scores/softmax/PV in F32, then converts only the final BF16 output. It is
  // managed and may materialize scores; removing ATen does not remove it.
  Tensor query(compute::DataType::kBFloat16, {1, 3, 4, 64}, memory.ref());
  Tensor key(compute::DataType::kBFloat16, {1, 9, 2, 64}, memory.ref());
  Tensor value(compute::DataType::kBFloat16, {1, 9, 2, 64}, memory.ref());
  Tensor direct_output(compute::DataType::kBFloat16, {1, 3, 4, 64}, memory.ref());
  Tensor graph_output(compute::DataType::kBFloat16, {1, 3, 4, 64}, memory.ref());
  const compute::ScaledDotProductAttention operation{.scale = 0.125F, .causal = true};
  const std::array inputs{query.view.as_const(), key.view.as_const(), value.view.as_const()};
  const std::array outputs{direct_output.view};
  auto prepared = compute::prepare(context, compute::Operation(operation),
                                  inputs, outputs, memory.ref());
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, compute::ggml_cuda::kProviderId);
  EXPECT_EQ(prepared->runtime_properties().replay_class, compute::ProviderReplayClass::kManaged);
  const auto requirement = prepared->workspace_requirement();
  mem::Buffer workspace;
  if (requirement.size != 0) {
    auto allocated = mem::Buffer::create(memory.ref(), requirement.size, requirement.alignment);
    ASSERT_TRUE(allocated) << allocated.error().message();
    workspace = std::move(*allocated);
  }
  auto direct_result = prepared->create_invocation(workspace.view());
  ASSERT_TRUE(direct_result) << direct_result.error().message();
  auto direct = std::move(*direct_result);
  auto builder = graph::GraphBuilder::create().value();
  const auto q = builder.input(query.description).value(), k = builder.input(key.description).value(),
             v = builder.input(value.description).value();
  const auto output = builder.add(operation, q, k, v).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto logical = std::move(builder).finish().value();
  auto compiled = graph::cuda::compile(logical, context, memory.ref());
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 1u);
  EXPECT_EQ(compiled->report.step(0)->provider.id, compute::ggml_cuda::kProviderId);
  auto invocation_result = compiled->executable.create_invocation();
  ASSERT_TRUE(invocation_result) << invocation_result.error().message();
  auto invocation = std::move(*invocation_result);
  ASSERT_TRUE(invocation.bind(q, query.view.as_const()).ok());
  ASSERT_TRUE(invocation.bind(k, key.view.as_const()).ok());
  ASSERT_TRUE(invocation.bind(v, value.view.as_const()).ok());
  ASSERT_TRUE(invocation.bind(output, graph_output.view).ok());
  auto direct_queue = context.create_queue().value();
  auto graph_queue = context.create_queue().value();
  for (int seed : {0, 3, 9}) {
    auto query_values = payload<__nv_bfloat16>(query, seed);
    if (seed == 9) {
      // A hidden F16 conversion would overflow these finite BF16 queries.
      for (auto &entry : query_values)
        entry = encode<__nv_bfloat16>(decode(entry) * 1048576.0F);
      ASSERT_TRUE(query.storage.view().write(std::as_bytes(std::span(query_values))).ok());
    }
    const auto key_values = payload<__nv_bfloat16>(key, seed + 5);
    const auto value_values = payload<__nv_bfloat16>(value, seed + 13);
    auto direct_done = direct.submit(direct_queue, inputs, outputs).value();
    auto graph_done = invocation.submit(graph_queue).value();
    ASSERT_TRUE(graph_done.wait().ok());
    ASSERT_TRUE(direct_done.wait().ok());
    check_output(direct_output, query_values, key_values, value_values, 3, 9, 0.01, 0.125F);
    check_output(graph_output, query_values, key_values, value_values, 3, 9, 0.01, 0.125F);
  }
}
#endif
} // namespace
