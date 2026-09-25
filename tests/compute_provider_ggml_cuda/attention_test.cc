#include "hyle/compute/providers/ggml_cuda.h"

#include "../compute_provider_flashinfer/attention_test_support.h"

namespace {
using namespace flashinfer_test;

template <class T>
void exercise_attention(compute::DataType type, int query_length, int kv_length,
                        double tolerance, float scale = 0.0625F,
                        bool wide_exponent = false) {
  auto context = compute::Context::create(cuda::device(0).value(),
                                          compute::ggml_cuda::provider())
                     .value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  Tensor query(type, {2, query_length, 8, 256}, memory.ref());
  Tensor key(type, {2, kv_length, 2, 256}, memory.ref());
  Tensor value(type, {2, kv_length, 2, 256}, memory.ref());
  Tensor other_query(type, {2, query_length, 8, 256}, memory.ref());
  Tensor other_key(type, {2, kv_length, 2, 256}, memory.ref());
  Tensor other_value(type, {2, kv_length, 2, 256}, memory.ref());
  Tensor output(type, {2, query_length, 8, 256}, memory.ref());
  Tensor other_output(type, {2, query_length, 8, 256}, memory.ref());
  const compute::ScaledDotProductAttention operation{.scale = scale,
                                                     .causal = true};
  const std::array inputs{query.view.as_const(), key.view.as_const(),
                          value.view.as_const()};
  const std::array other_inputs{other_query.view.as_const(),
                                other_key.view.as_const(),
                                other_value.view.as_const()};
  const std::array outputs{output.view};
  const std::array other_outputs{other_output.view};
  auto prepared = compute::prepare(context, compute::Operation(operation),
                                   inputs, outputs, memory.ref());
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, compute::ggml_cuda::kProviderId);
  EXPECT_EQ(prepared->runtime_properties().workspace_owner,
            compute::ProviderWorkspaceOwner::kProvider);
  auto first = prepared->create_invocation();
  ASSERT_TRUE(first) << first.error().message();
  auto second = prepared->create_invocation();
  ASSERT_TRUE(second) << second.error().message();
  auto first_queue = context.create_queue().value();
  auto second_queue = context.create_queue().value();
  for (int seed : {0, 3, 9}) {
    auto q = payload<T>(query, seed);
    auto k = payload<T>(key, seed + 5);
    auto v = payload<T>(value, seed + 13);
    if (wide_exponent) {
      std::fill(q.begin(), q.end(), encode<T>(0));
      std::fill(k.begin(), k.end(), encode<T>(131072));
      for (std::size_t i = 0; i < v.size(); ++i)
        v[i] = encode<T>((i % 2 == 0 ? 1.0F : -1.0F) * 131072);
      ASSERT_TRUE(query.storage.view().write(std::as_bytes(std::span(q))).ok());
      ASSERT_TRUE(key.storage.view().write(std::as_bytes(std::span(k))).ok());
      ASSERT_TRUE(value.storage.view().write(std::as_bytes(std::span(v))).ok());
    }
    const auto oq = payload<T>(other_query, seed + 19);
    const auto ok = payload<T>(other_key, seed + 23);
    const auto ov = payload<T>(other_value, seed + 31);
    auto first_done = first->submit(first_queue, inputs, outputs);
    ASSERT_TRUE(first_done) << first_done.error().message();
    auto second_done =
        second->submit(second_queue, other_inputs, other_outputs);
    ASSERT_TRUE(second_done) << second_done.error().message();
    ASSERT_TRUE(second_done->wait().ok());
    ASSERT_TRUE(first_done->wait().ok());
    check_output(output, q, k, v, query_length, kv_length, tolerance, scale);
    check_output(other_output, oq, ok, ov, query_length, kv_length,
                 wide_exponent ? 0.01 : tolerance, scale);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    // A compiled graph's derived views must follow all changed port addresses.
    auto rebound = first->submit(first_queue, other_inputs, other_outputs);
    ASSERT_TRUE(rebound) << rebound.error().message();
    ASSERT_TRUE(rebound->wait().ok());
    check_output(other_output, oq, ok, ov, query_length, kv_length,
                 wide_exponent ? 0.01 : tolerance, scale);
    check_output(output, q, k, v, query_length, kv_length, tolerance, scale);
  }
  {
    auto abandoned = first->submit(first_queue, inputs, outputs);
    ASSERT_TRUE(abandoned);
  }
}

TEST(GgmlCudaAttention, OriginalPrefillDecodeAndIndependentReplays) {
  exercise_attention<__half>(compute::DataType::kFloat16, 17, 31, 0.002);
  exercise_attention<__half>(compute::DataType::kFloat16, 1, 513, 0.002);
  exercise_attention<__nv_bfloat16>(compute::DataType::kBFloat16, 17, 31, 0.01);
  exercise_attention<__nv_bfloat16>(compute::DataType::kBFloat16, 1, 513, 0.01);
}

TEST(GgmlCudaAttention, ZeroAndNegativeScalePreserveCausalMask) {
  for (float scale : {0.0F, -0.0625F}) {
    exercise_attention<__half>(compute::DataType::kFloat16, 17, 31, 0.002,
                               scale);
    exercise_attention<__nv_bfloat16>(compute::DataType::kBFloat16, 17, 31,
                                      0.01, scale);
  }
}

TEST(GgmlCudaAttention, Bfloat16ValuesOutsideFloat16RangeRemainFinite) {
  exercise_attention<__nv_bfloat16>(compute::DataType::kBFloat16, 17, 31, 0.01,
                                    0.0625F, true);
  exercise_attention<__nv_bfloat16>(compute::DataType::kBFloat16, 1, 513, 0.01,
                                    0.0625F, true);
}

TEST(GgmlCudaAttention, Qwen38GroupedQueryHeadRatioRuns) {
  auto context = compute::Context::create(cuda::device(0).value(),
                                          compute::ggml_cuda::provider())
                     .value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  Tensor query(compute::DataType::kBFloat16, {1, 1, 24, 256}, memory.ref());
  Tensor key(compute::DataType::kBFloat16, {1, 1, 4, 256}, memory.ref());
  Tensor value(compute::DataType::kBFloat16, {1, 1, 4, 256}, memory.ref());
  Tensor output(compute::DataType::kBFloat16, {1, 1, 24, 256}, memory.ref());
  (void)payload<__nv_bfloat16>(query, 3);
  (void)payload<__nv_bfloat16>(key, 7);
  const auto values = payload<__nv_bfloat16>(value, 11);
  const std::array inputs{query.view.as_const(), key.view.as_const(),
                          value.view.as_const()};
  const std::array outputs{output.view};
  auto prepared = compute::prepare(
      context,
      compute::Operation(compute::ScaledDotProductAttention{.scale = 0.0625F,
                                                            .causal = true}),
      inputs, outputs, memory.ref());
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, compute::ggml_cuda::kProviderId);
  auto invocation = prepared->create_invocation();
  ASSERT_TRUE(invocation) << invocation.error().message();
  auto queue = context.create_queue().value();
  auto done = invocation->submit(queue, inputs, outputs);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  std::vector<__nv_bfloat16> actual(24 * 256);
  ASSERT_TRUE(output.storage.view()
                  .read(std::as_writable_bytes(std::span(actual)))
                  .ok());
  for (std::size_t head = 0; head < 24; ++head)
    for (std::size_t depth = 0; depth < 256; ++depth)
      EXPECT_NEAR(decode(actual[head * 256 + depth]),
                  decode(values[(head / 6) * 256 + depth]), 0.01)
          << "head=" << head << " depth=" << depth;
}
} // namespace
