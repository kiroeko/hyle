#include "hyle/compute/providers/ggml_cuda.h"
#include "hyle/graph/cuda.h"

#include "../compute_provider_flashinfer/attention_test_support.h"

namespace {
using namespace flashinfer_test;
namespace graph = hyle::graph;
#if defined(__linux__)
constexpr auto kAttentionProvider = compute::flashinfer::kProviderId;
#else
constexpr auto kAttentionProvider = compute::ggml_cuda::kProviderId;
#endif

template <class T>
void run_attention(compute::DataType type, int query_length, int kv_length,
                   double tolerance) {
  auto context = graph::cuda::create_context().value();
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
  auto prepared = compute::prepare(context, compute::Operation(operation),
                                   inputs, outputs, memory.ref())
                      .value();
  ASSERT_EQ(prepared.provider().id, kAttentionProvider);
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
  EXPECT_EQ(compiled->report.step(0)->provider.id, kAttentionProvider);
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
} // namespace
