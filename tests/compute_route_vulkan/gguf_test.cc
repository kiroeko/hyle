#include "gguf_test_support.h"
#include "vulkan_program_internal.h"

namespace {
using namespace vulkan_gguf_test;

TEST(VulkanGguf, EveryCatalogFormatUsesNativeDirectRegionAndReplays) {
  QuantizerCleanup cleanup;
  auto context = vulkan_test::context().value();
  constexpr int k = 256;
  constexpr int n = 8;
  constexpr int m = 2;
  const auto input_desc =
      vulkan_tensor_test::dense(c::DataType::kFloat32, {m, k});
  const auto physical = weights(n, k);
  for (const auto type : kTypes) {
    SCOPED_TRACE(static_cast<int>(type));
    const auto weight_desc =
        c::gguf_weight_description(type, c::DataType::kFloat32, k, n).value();
    auto packed = pack(type, physical, n, k, *weight_desc.required_bytes());
    auto weight = context.upload(weight_desc, packed.bytes).value();
    auto program = v::prepare(context, c::MatMul{}, input_desc, weight_desc);
    ASSERT_TRUE(program) << program.error().message();
    ASSERT_TRUE(v::detail::Access::program(*program)->dispatches.back().region);
    auto invocation = program->create_invocation().value();
    for (int replay = 1; replay <= 2; ++replay) {
      const auto values = activations(m, k, replay);
      auto input = vulkan_tensor_test::make(context, input_desc, values);
      ASSERT_TRUE(invocation.bind(input, weight).ok());
      auto work = invocation.submit().value();
      auto result = work.wait();
      ASSERT_TRUE(result) << result.error().message();
      expect_close(context, result->get()->tensors()[0],
                   matmul(values, packed.decoded, m, n, k));
    }
  }
}

TEST(VulkanGguf, PackedIq4XsGroupedExpertsUseNativeMulMatId) {
  QuantizerCleanup cleanup;
  auto context = vulkan_test::context().value();
  constexpr int experts = 4;
  constexpr int n = 8;
  constexpr int k = 256;
  constexpr int tokens = 2;
  constexpr int slots = 2;
  const auto weight_desc =
      c::gguf_grouped_weight_description(c::GgufQuantizedType::kIq4Xs,
                                         c::DataType::kFloat32, experts, n, k)
          .value();
  const auto activation_desc =
      vulkan_tensor_test::dense(c::DataType::kFloat32, {tokens, slots, k});
  const auto ids_desc =
      vulkan_tensor_test::dense(c::DataType::kInt32, {tokens, slots});
  const auto physical = weights(experts * n, k, 3);
  auto packed = pack(c::GgufQuantizedType::kIq4Xs, physical, experts * n, k,
                     *weight_desc.required_bytes());
  auto weight = context.upload(weight_desc, packed.bytes).value();
  const std::array<std::int32_t, tokens * slots> ids{0, 3, 1, 2};
  auto selected =
      context.upload(ids_desc, std::as_bytes(std::span(ids))).value();
  auto program = v::prepare(context, c::model_ops::grouped_matmul(),
                            weight_desc, activation_desc, ids_desc);
  ASSERT_TRUE(program) << program.error().message();
  ASSERT_TRUE(v::detail::Access::program(*program)->dispatches.back().region);
  auto invocation = program->create_invocation().value();
  for (int replay = 1; replay <= 2; ++replay) {
    const auto values = activations(tokens * slots, k, replay);
    auto activation =
        vulkan_tensor_test::make(context, activation_desc, values);
    ASSERT_TRUE(invocation.bind(weight, activation, selected).ok());
    auto work = invocation.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    expect_close(context, result->get()->tensors()[0],
                 grouped_matmul(values, packed.decoded, ids, n, k));
  }
}

TEST(VulkanGguf, RejectsMalformedFormatAndPackedStorageOnWrongPorts) {
  auto context = vulkan_test::context().value();
  const auto input = vulkan_tensor_test::dense(c::DataType::kFloat32, {2, 256});
  const auto malformed_format =
      c::StorageFormat::blocked(c::kGgufQ4K, 256, 143).value();
  const auto malformed =
      c::TensorDesc::blocked_strided(
          c::DataType::kFloat32, std::array<std::int64_t, 2>{256, 8},
          malformed_format, 0, std::array<std::int64_t, 2>{143, 143})
          .value();
  EXPECT_FALSE(v::prepare(context, c::MatMul{}, input, malformed));
  const auto packed = c::gguf_weight_description(c::GgufQuantizedType::kQ4K,
                                                 c::DataType::kFloat32, 256, 8)
                          .value();
  EXPECT_FALSE(v::prepare(context, c::Copy{}, packed));
  EXPECT_FALSE(v::prepare(context, c::MatMul{}, packed, input));
  EXPECT_FALSE(v::prepare(context,
                          c::MatMul{.accumulation_type = c::DataType::kFloat16},
                          input, packed));
}
} // namespace
