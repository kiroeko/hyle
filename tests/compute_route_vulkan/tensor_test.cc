#include <gtest/gtest.h>

#include "tensor_test_support.h"

namespace {
using namespace vulkan_tensor_test;
constexpr std::array types{c::DataType::kFloat32,  c::DataType::kFloat16,
                           c::DataType::kBFloat16, c::DataType::kInt32,
                           c::DataType::kInt64,    c::DataType::kBool};

TEST(VulkanTensor, ExactStorageHandlesOddStridesWordsAndEmptyTransfers) {
  auto context = vulkan_test::context().value();
  for (auto type : types) {
    const auto size = static_cast<std::int64_t>(c::data_type_size(type));
    for (int count : {0, 1, 3, 129, 257}) {
      SCOPED_TRACE(::testing::Message()
                   << static_cast<int>(type) << '/' << count);
      const auto desc =
          c::TensorDesc::strided_bytes(type, std::array<std::int64_t, 1>{count},
                                       std::array<std::int64_t, 1>{size + 1})
              .value();
      std::vector<std::uint64_t> raw(count);
      for (std::size_t i = 0; i < raw.size(); ++i) {
        raw[i] = type == c::DataType::kBool    ? i % 2
                 : type == c::DataType::kInt64 ? 0xffffeeee12340000ULL + i
                 : size == 2                   ? (i % 3 == 0   ? 0xffc3u
                                                  : i % 3 == 1 ? 0x8000u
                                                               : 0x7fc7u)
                                               : (i % 3 == 0   ? 0xffc00003u
                                                  : i % 3 == 1 ? 0x80000000u
                                                               : 0x7fc00007u);
      }
      const auto host = payload(desc, raw);
      const auto input = context.upload(desc, host).value();
      EXPECT_EQ(bytes(context, input), host);
      auto copy = execute(context, c::Operation(c::Copy{}), std::array{input});
      ASSERT_TRUE(copy) << copy.error().message();
      EXPECT_EQ(bits(context, *copy), raw);
      const auto mask_desc = dense(c::DataType::kBool, {count});
      std::vector<std::uint64_t> flags(count);
      for (int i = 0; i < count; ++i)
        flags[i] = i % 2;
      const auto mask =
          context.upload(mask_desc, payload(mask_desc, flags)).value();
      const auto scalar = c::ScalarValue::from_bits(
          type, type == c::DataType::kBool    ? 1
                : type == c::DataType::kInt64 ? 0x12345678abcdef01ULL
                : size == 2                   ? 0xffc3
                                              : 0xffc00003);
      c::Fill fill{.output_shape = c::TensorShape::create(
                                       std::array<std::int64_t, 1>{count})
                                       .value(),
                   .output_type = type,
                   .value = scalar};
      auto filled = execute(context, c::Operation(fill), {});
      ASSERT_TRUE(filled) << filled.error().message();
      EXPECT_EQ(bits(context, *filled),
                std::vector<std::uint64_t>(count, scalar.bits()));
      auto selected = execute(context, c::Operation(c::Select{}),
                              std::array{mask, input, *filled});
      ASSERT_TRUE(selected) << selected.error().message();
      auto expected = raw;
      for (int i = 0; i < count; ++i)
        if (!flags[i])
          expected[i] = scalar.bits();
      EXPECT_EQ(bits(context, *selected), expected);
      EXPECT_EQ(bytes(context, input), host); // Includes stride holes.
      const auto required = *copy->description().required_bytes();
      const auto capacity =
          std::max(std::size_t{4}, (required + 3) & ~std::size_t{3});
      EXPECT_FALSE(context.borrow_ready(
          copy->description(), copy->native_buffer(), copy->native_offset(),
          capacity - 1, copy->native_owner()));
      auto borrowed = context.borrow_ready(
          copy->description(), copy->native_buffer(), copy->native_offset(),
          capacity, copy->native_owner());
      ASSERT_TRUE(borrowed) << borrowed.error().message();
      EXPECT_EQ(bits(context, *borrowed), raw);
      EXPECT_FALSE(context.borrow_ready(
          copy->description(), copy->native_buffer(), copy->native_offset() + 1,
          capacity, copy->native_owner()));
    }
  }
  const auto empty = context.upload(dense(c::DataType::kBool, {0}), {}).value();
  EXPECT_TRUE(context.close().ok());
  EXPECT_FALSE(context.upload(empty, {}).ok());
  EXPECT_TRUE(context.download(empty, {}).ok());
}

TEST(VulkanTensor, UnaryBinaryAndCastUseIndependentFloatingOracles) {
  auto context = vulkan_test::context().value();
  const double inf = std::numeric_limits<double>::infinity(),
               nan = std::numeric_limits<double>::quiet_NaN();
  for (auto type :
       {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16}) {
    const auto size = static_cast<std::int64_t>(c::data_type_size(type));
    const auto desc =
        c::TensorDesc::strided_bytes(
            type, std::array<std::int64_t, 2>{17, 2},
            std::array<std::int64_t, 2>{size + 1, 17 * (size + 1) + 3})
            .value();
    std::vector<double> data(34);
    for (std::size_t i = 0; i < data.size(); ++i)
      data[i] = (static_cast<int>(i) - 17) * 0.37;
    data[0] = -inf;
    data[1] = inf;
    data[2] = nan;
    data[3] = -0.0;
    const auto input = make(context, desc, data);
    const auto logical = values(context, input);
    const auto original = bytes(context, input);
    for (int mode = 0; mode <= static_cast<int>(c::UnaryKind::kSoftplus);
         ++mode) {
      SCOPED_TRACE(::testing::Message()
                   << static_cast<int>(type) << '/' << mode);
      const auto kind = static_cast<c::UnaryKind>(mode);
      auto result =
          execute(context, c::Operation(c::Unary{kind}), std::array{input});
      ASSERT_TRUE(result) << result.error().message();
      std::vector<double> expected;
      for (double value : logical)
        expected.push_back(unary(kind, value));
      expect(context, *result, expected);
    }
    const auto rhs = make(context, dense(type, {1, 2}), std::array{0.5, nan});
    const auto right = values(context, rhs);
    for (int mode = 0; mode <= static_cast<int>(c::BinaryKind::kMaximum);
         ++mode) {
      auto result = execute(
          context, c::Operation(c::Binary{static_cast<c::BinaryKind>(mode)}),
          std::array{input, rhs});
      ASSERT_TRUE(result) << result.error().message();
      std::vector<double> expected(logical.size());
      for (std::size_t i = 0; i < expected.size(); ++i) {
        const auto x = logical[i], y = right[i % 2];
        expected[i] = mode == 0   ? x + y
                      : mode == 1 ? x - y
                      : mode == 2 ? x * y
                      : mode == 3 ? x / y
                      : mode == 4 ? std::fmin(x, y)
                                  : std::fmax(x, y);
      }
      expect(context, *result, expected);
    }
    for (auto output_type : {c::DataType::kFloat32, c::DataType::kFloat16,
                             c::DataType::kBFloat16}) {
      auto result = execute(context, c::Operation(c::Cast{output_type}),
                            std::array{input});
      ASSERT_TRUE(result) << result.error().message();
      expect(context, *result, logical, true);
    }
    EXPECT_EQ(bytes(context, input), original);
  }
  const std::array edges{0.0,
                         -0.0,
                         1.00048828125,
                         1.00146484375,
                         1.00390625,
                         1.01171875,
                         65504.0,
                         65520.0,
                         std::ldexp(1.0, -24),
                         std::ldexp(1.0, -25),
                         -std::ldexp(1.0, -24),
                         std::ldexp(1.0, -14),
                         std::ldexp(1.0, -126),
                         std::ldexp(1.0, -133),
                         nan,
                         inf,
                         -inf};
  const auto input = make(
      context,
      dense(c::DataType::kFloat32, {static_cast<std::int64_t>(edges.size())}),
      edges);
  for (auto type : {c::DataType::kFloat16, c::DataType::kBFloat16}) {
    auto result =
        execute(context, c::Operation(c::Cast{type}), std::array{input});
    ASSERT_TRUE(result) << result.error().message();
    expect(context, *result, edges, true);
  }
}

TEST(VulkanTensor, ScalarEmptyTailRankEightBroadcastAndColdRefusal) {
  auto context = vulkan_test::context().value();
  for (auto type :
       {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16}) {
    for (int count : {0, 1, 127, 128, 129, 257, 2048}) {
      const auto desc = dense(type, {count});
      std::vector<double> data(count);
      for (int i = 0; i < count; ++i)
        data[i] = (i % 31 - 15) * 1.3;
      const auto input = make(context, desc, data);
      auto result =
          execute(context, c::Operation(c::Unary{c::UnaryKind::kSoftplus}),
                  std::array{input});
      ASSERT_TRUE(result) << result.error().message();
      auto expected = values(context, input);
      for (auto &x : expected)
        x = unary(c::UnaryKind::kSoftplus, x);
      expect(context, *result, expected);
    }
    const auto scalar = make(context, dense(type, {}), std::array{-0.75});
    auto result =
        execute(context, c::Operation(c::Unary{c::UnaryKind::kSigmoid}),
                std::array{scalar});
    ASSERT_TRUE(result) << result.error().message();
    EXPECT_EQ(result->description().rank(), 0U);
    expect(context, *result, std::array{unary(c::UnaryKind::kSigmoid, -0.75)});
    const auto rank8 =
        c::TensorDesc::strided_bytes(
            type, std::array<std::int64_t, 8>{1, 2, 1, 3, 1, 1, 1, 1},
            std::array<std::int64_t, 8>{
                0, 0, 0, static_cast<std::int64_t>(c::data_type_size(type)) + 1,
                0, 0, 0, 0})
            .value();
    const std::array repeated{-2.0, 0.5, 1.75, -2.0, 0.5, 1.75};
    const auto input = make(context, rank8, repeated);
    auto sum =
        execute(context, c::Operation(c::Binary{}), std::array{input, scalar});
    ASSERT_TRUE(sum) << sum.error().message();
    auto expected = std::vector<double>(repeated.begin(), repeated.end());
    for (auto &x : expected)
      x -= 0.75;
    expect(context, *sum, expected, true);
  }
  const auto ints = dense(c::DataType::kInt32, {17});
  auto rejected = v::prepare(context, c::Unary{c::UnaryKind::kAbs}, ints);
  ASSERT_FALSE(rejected);
  EXPECT_TRUE(rejected.error().IsUnimplemented());
  const auto half = dense(c::DataType::kFloat16, {2, 4});
  EXPECT_FALSE(v::prepare(context, c::GatedActivation{}, half));
  EXPECT_FALSE(v::prepare(context, c::MatMul{}, half, half));
  EXPECT_FALSE(v::prepare(context, c::Binary{}, half,
                          dense(c::DataType::kFloat16, {3})));
  hyle::mem::QuotaAllocator empty(hyle::mem::system_allocator(), 0);
  auto limited = vulkan_test::context(&empty).value();
  auto program = v::prepare(limited, c::Unary{c::UnaryKind::kSoftplus}, half);
  ASSERT_TRUE(program)
      << program.error().message(); // Metadata-only preparation.
  auto slot = program->create_invocation();
  ASSERT_FALSE(slot);
  EXPECT_TRUE(slot.error().IsResourceExhausted());
  EXPECT_EQ(empty.used(), 0U);
}
} // namespace
