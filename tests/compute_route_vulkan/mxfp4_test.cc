#include "mxfp4_test_support.h"
#include "vulkan_program_internal.h"

namespace {
using namespace vulkan_mxfp4_test;

TEST(VulkanMxfp4, OriginalDirectRepeatedSlotsKeepCompressedInputsAndResults) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(),
                                  64 * 1024 * 1024);
  auto context = vulkan_test::context(&quota).value();
  for (auto shape :
       {std::array{1, 32, 5}, std::array{2, 128, 16}, std::array{7, 256, 32}}) {
    const auto [m, k, n] = shape;
    SCOPED_TRACE(m);
    SCOPED_TRACE(k);
    SCOPED_TRACE(n);
    Weights weights(k, n, 1);
    const auto input_desc =
        vulkan_tensor_test::dense(c::DataType::kFloat16, {m, k});
    auto program =
        v::prepare(context, c::MatMul{}, input_desc, weights.description);
    ASSERT_TRUE(program) << program.error().message();
    ASSERT_TRUE(v::detail::Access::program(*program)->dispatches.back().region);
    auto packed = weights.upload(context);
    const auto input_values = activation(m, k);
    const auto changed_values = activation(m, k, 1);
    auto a = vulkan_tensor_test::make(context, input_desc, input_values);
    auto b = vulkan_tensor_test::make(context, input_desc, changed_values);
    auto first = program->create_invocation().value();
    auto second = program->create_invocation().value();
    ASSERT_TRUE(first.bind(std::array{a, packed}).ok());
    ASSERT_TRUE(second.bind(std::array{b, packed}).ok());
    const auto expected = weights.multiply(input_values, m);
    const auto changed = weights.multiply(changed_values, m);
    v::Tensor retained;
    for (int repeat = 0; repeat < 3; ++repeat) {
      auto &slot = repeat == 0 ? first : second;
      auto work = slot.submit().value();
      auto result = work.wait();
      ASSERT_TRUE(result) << result.error().message();
      auto tensor = result->get()->tensors()[0];
      vulkan_tensor_test::expect(context, tensor,
                                 repeat == 0 ? expected : changed, true);
      if (repeat == 0)
        retained = tensor;
      else
        vulkan_tensor_test::expect(context, retained, expected, true);
    }
    EXPECT_FALSE(first.submit());
    EXPECT_EQ(vulkan_tensor_test::values(context, a), input_values);
    const auto original = std::as_bytes(std::span(weights.packed));
    EXPECT_EQ(vulkan_tensor_test::bytes(context, packed),
              std::vector<std::byte>(original.begin(), original.end()));
    retained = {};
    ASSERT_TRUE(first.submit().value().wait());
  }
  ASSERT_TRUE(context.close().ok());
  context = {};
  EXPECT_EQ(quota.used(), 0U);
}

TEST(VulkanMxfp4, RejectsUnrelatedEncodingAndPlainMathOnCompressedStorage) {
  auto context = vulkan_test::context().value();
  const auto input = vulkan_tensor_test::dense(c::DataType::kFloat16, {2, 128});
  const auto format = c::StorageFormat::blocked({123, 456}, 32, 17).value();
  const auto unknown =
      c::TensorDesc::blocked_strided(
          c::DataType::kFloat16, std::array<std::int64_t, 2>{128, 16}, format,
          0, std::array<std::int64_t, 2>{17, 68})
          .value();
  EXPECT_FALSE(v::prepare(context, c::MatMul{}, input, unknown));
  const auto known = c::mxfp4_weight_description(128, 16).value();
  EXPECT_FALSE(
      v::prepare(context, c::Unary{.kind = c::UnaryKind::kAbs}, known));
  EXPECT_FALSE(v::prepare(context,
                          c::MatMul{.accumulation_type = c::DataType::kFloat16},
                          input, known));
}
} // namespace
