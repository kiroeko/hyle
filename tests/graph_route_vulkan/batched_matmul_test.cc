#include "../compute_route_vulkan/tensor_test_support.h"

TEST(VulkanBatchedMatMul, TwoWayBatchBroadcastAndMatrixTransposeKeepStorageTypes) {
  namespace t = vulkan_tensor_test;
  namespace c = t::c;
  auto context = vulkan_test::context().value();
  constexpr std::int64_t tokens = 2, slots = 3, rows = 2, depth = 8, columns = 3;
  for (const auto type : {c::DataType::kFloat32, c::DataType::kFloat16,
                          c::DataType::kBFloat16}) {
    SCOPED_TRACE(static_cast<int>(type));
    const auto a_description = t::dense(type, {tokens, 1, depth, rows});
    const auto b_description = t::dense(type, {1, slots, columns, depth});
    std::vector<double> a_values(tokens * depth * rows), b_values(slots * columns * depth);
    for (std::int64_t token = 0; token < tokens; ++token)
      for (std::int64_t k = 0; k < depth; ++k)
        for (std::int64_t row = 0; row < rows; ++row)
          a_values[(token * depth + k) * rows + row] =
              double((token * 3 + row * 2 + k) % 9 - 4) * .5;
    for (std::int64_t slot = 0; slot < slots; ++slot)
      for (std::int64_t column = 0; column < columns; ++column)
        for (std::int64_t k = 0; k < depth; ++k)
          b_values[(slot * columns + column) * depth + k] =
              double((slot * 2 + column * 3 + k) % 7 - 3) * .25;
    const auto a_storage = t::make(context, a_description, a_values);
    const auto b_storage = t::make(context, b_description, b_values);
    const auto transpose = c::View::permute(
        std::array<std::int32_t, 4>{0, 1, 3, 2}).value();
    const auto a = t::execute(context, c::Operation(transpose), std::array{a_storage});
    const auto b = t::execute(context, c::Operation(transpose), std::array{b_storage});
    ASSERT_TRUE(a) << a.error().message();
    ASSERT_TRUE(b) << b.error().message();
    ASSERT_FALSE(a->description().is_dense());
    ASSERT_FALSE(b->description().is_dense());
    const auto product = t::execute(context,
        c::Operation(c::MatMul{.accumulation_type = c::DataType::kFloat32}), std::array{*a, *b});
    ASSERT_TRUE(product) << product.error().message();
    EXPECT_EQ(product->description(), t::dense(type, {tokens, slots, rows, columns}));
    std::vector<double> expected;
    for (std::int64_t token = 0; token < tokens; ++token)
      for (std::int64_t slot = 0; slot < slots; ++slot)
        for (std::int64_t row = 0; row < rows; ++row)
          for (std::int64_t column = 0; column < columns; ++column) {
            double sum = 0;
            for (std::int64_t k = 0; k < depth; ++k)
              sum += a_values[(token * depth + k) * rows + row] *
                  b_values[(slot * columns + column) * depth + k];
            expected.push_back(sum);
          }
    t::expect(context, *product, expected, true);
    EXPECT_EQ(t::values(context, a_storage), a_values);
    EXPECT_EQ(t::values(context, b_storage), b_values);
  }
}
