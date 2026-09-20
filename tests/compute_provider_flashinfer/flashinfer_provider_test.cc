#include "attention_test_support.h"

namespace {
using namespace flashinfer_test;

template <class T>
void exercise(compute::DataType type, int query_length, int kv_length,
              double tolerance) {
  auto context = compute::Context::create(cuda::device(0).value(),
                                          compute::flashinfer::provider())
                     .value();
  auto environment = cuda::environment(context).value();
  ASSERT_EQ(environment.sm_major, 8);
  ASSERT_EQ(environment.sm_minor, 9);
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  auto first_queue = context.create_queue().value();
  auto second_queue = context.create_queue().value();
  Tensor query(type, {2, query_length, 8, 256}, memory.ref());
  Tensor key(type, {2, kv_length, 2, 256}, memory.ref());
  Tensor value(type, {2, kv_length, 2, 256}, memory.ref());
  Tensor first_output(type, {2, query_length, 8, 256}, memory.ref());
  Tensor second_output(type, {2, query_length, 8, 256}, memory.ref());
  auto q = payload<T>(query, 0);
  auto k = payload<T>(key, 5);
  auto v = payload<T>(value, 13);
  const std::array inputs{query.view.as_const(), key.view.as_const(),
                          value.view.as_const()};
  const std::array first_outputs{first_output.view};
  const std::array second_outputs{second_output.view};
  auto prepared =
      compute::prepare(context,
                       compute::Operation(compute::ScaledDotProductAttention{
                           .scale = 0.0625F, .causal = true}),
                       inputs, first_outputs, memory.ref())
          .value();
  EXPECT_EQ(prepared.provider().id, compute::flashinfer::kProviderId);
  EXPECT_EQ(prepared.runtime_properties().replay_class,
            compute::ProviderReplayClass::kManaged);
  const auto requirement = prepared.workspace_requirement();
  ASSERT_GT(requirement.size, 0u);
  auto first_workspace =
      mem::Buffer::create(memory.ref(), requirement.size, requirement.alignment)
          .value();
  auto second_workspace =
      mem::Buffer::create(memory.ref(), requirement.size, requirement.alignment)
          .value();
  auto first = prepared.create_invocation(first_workspace.view()).value();
  auto second = prepared.create_invocation(second_workspace.view()).value();
  auto too_small = mem::Buffer::create(memory.ref(), requirement.size - 1,
                                       requirement.alignment)
                       .value();
  EXPECT_FALSE(prepared.create_invocation(too_small.view()));
  for (int replay = 0; replay < 3; ++replay) {
    auto first_done = first.submit(first_queue, inputs, first_outputs).value();
    auto second_done =
        second.submit(second_queue, inputs, second_outputs).value();
    ASSERT_TRUE(second_done.wait().ok());
    ASSERT_TRUE(first_done.wait().ok());
  }
  // A later run on the second invocation must not overwrite the old first
  // result.
  const auto old_v = v;
  v = payload<T>(value, 21);
  ASSERT_TRUE(second.submit(second_queue, inputs, second_outputs)->wait().ok());
  check_output(first_output, q, k, old_v, query_length, kv_length, tolerance);
  check_output(second_output, q, k, v, query_length, kv_length, tolerance);
  // Completion may be abandoned; invocation/Queue teardown retains accepted
  // work.
  {
    auto abandoned =
        prepared.create_invocation(second_workspace.view()).value();
    ASSERT_TRUE(abandoned.submit(second_queue, inputs, second_outputs));
  }
  ASSERT_TRUE(second_queue.drain().ok());
}

TEST(Flashinfer, Fp16PrefillAndDecodeThroughCompute) {
  exercise<__half>(compute::DataType::kFloat16, 17, 31, 0.002);
  exercise<__half>(compute::DataType::kFloat16, 1, 513, 0.002);
}

TEST(Flashinfer, Bf16PrefillAndDecodeThroughCompute) {
  exercise<__nv_bfloat16>(compute::DataType::kBFloat16, 17, 31, 0.01);
  exercise<__nv_bfloat16>(compute::DataType::kBFloat16, 1, 513, 0.01);
}

} // namespace
