#include "packed_test_support.h"
#include "../compute_models/mlx_affine_bitstream.h"

namespace {
auto prepare(const hyle::compute::apple::Context &context,
             hyle::compute::Operation operation,
             const std::vector<mlx::core::array> &inputs) {
  return context.prepare(operation, inputs);
}
TEST(AppleMlxQuantized, GroupedSharesOriginalActivationAcrossSelectedExperts) {
  apple_packed_test::grouped_shared_activation(prepare);
}
TEST(AppleMlxQuantized, NativeFormatsUseOrdinaryEntryAndCheckedExperts) {
  apple_packed_test::formats(prepare);
}
TEST(AppleMlxQuantized, EmbeddingSelectsPackedRowsAndChecksChangedIds) {
  apple_packed_test::embedding(prepare);
}
} // namespace


namespace {
namespace apple_affine_contract_test {
using namespace apple_tensor_test;

// Every quantity in this witness is exactly representable in F32. There are
// only two nonzero dot terms, so neither FMA nor a different F32 reduction tree
// can account for the -1 versus -2 distinction.
void matmul_witness(mx::Dtype dtype, int rows) {
  constexpr int k = 128, n = 16, group = 64;
  constexpr std::uint32_t first_word = 0x21U;
  constexpr auto code0 = first_word & 15U;
  constexpr auto code1 = (first_word >> 4) & 15U;
  const double bias = dtype == mx::bfloat16 ? 256.0 : 2048.0;
  ASSERT_EQ(rounded(bias, dtype), bias);
  const double expected = rounded((double(code0) + bias) -
                                  (double(code1) + bias), dtype);
  const double narrowed = rounded(rounded(double(code0) + bias, dtype) -
                                  rounded(double(code1) + bias, dtype), dtype);
  ASSERT_EQ(expected, -1.0);
  ASSERT_EQ(narrowed, -2.0);

  std::vector<std::uint32_t> words(n * k / 8, 0U);
  for (int column = 0; column < n; ++column)
    words[column * (k / 8)] = first_word;
  std::vector<double> input(rows * k, 0.0);
  for (int row = 0; row < rows; ++row) {
    input[row * k] = 1.0;
    input[row * k + 1] = -1.0;
  }
  const std::vector<double> scales(n * k / group, 1.0);
  const std::vector<double> biases(scales.size(), bias);
  std::vector<mx::array> ports{
      ready(make(input, {rows, k}, dtype)),
      ready(mx::array(words.data(), {n, k / 8}, mx::uint32)),
      ready(make(scales, {n, k / group}, dtype)),
      ready(make(biases, {n, k / group}, dtype))};
  auto context = a::Context::create().value();
  auto result = execute(context,
      c::model_ops::mlx_quantized_matmul(
          c::model_ops::MlxWeightFormat::kAffine, 4, group), ports);
  ASSERT_TRUE(result) << result.error().message();
  EXPECT_EQ(result->dtype(), dtype);
  EXPECT_EQ(result->shape(), (mx::Shape{rows, n}));
  const auto actual = values(*result);
  ASSERT_EQ(actual.size(), static_cast<std::size_t>(rows * n));
  EXPECT_EQ(actual, std::vector<double>(actual.size(), expected))
      << "F32 affine decoded operands require -1; pre-narrowed decoded "
         "weights produce -2. M1 is the source-qualified QMV control and "
         "M64 is the source-qualified split-K QMM case.";
}

// An embedding returns decoded values in the declared low dtype; it is not
// the F32 operand stream consumed by matmul. Multiplication by one is exact
// here, deliberately avoiding a claim about general mul/add contraction.
void embedding_control(mx::Dtype dtype) {
  constexpr int k = 128, group = 64;
  const double bias = dtype == mx::bfloat16 ? 256.0 : 2048.0;
  std::vector<std::uint32_t> words(k / 8, 0U);
  words[0] = 0x21U;
  const std::vector<double> scales(k / group, 1.0);
  const std::vector<double> biases(k / group, bias);
  std::vector<mx::array> ports{
      ready(mx::array(words.data(), {1, k / 8}, mx::uint32)),
      ready(make(scales, {1, k / group}, dtype)),
      ready(make(biases, {1, k / group}, dtype)),
      ready(mx::array({std::int32_t{0}}, {1}))};
  std::vector<double> expected(k);
  for (int index = 0; index < k; ++index) {
    const auto code = (words[index / 8] >> (4 * (index % 8))) & 15U;
    expected[index] = rounded(rounded(double(code), dtype) + bias, dtype);
  }
  ASSERT_EQ(expected[0], bias);
  ASSERT_EQ(expected[1], bias + 2.0);
  ASSERT_NE(expected[0], 1.0 + bias);
  auto context = a::Context::create().value();
  auto result = execute(context, c::model_ops::mlx_quantized_embedding(4, group), ports);
  ASSERT_TRUE(result) << result.error().message();
  EXPECT_EQ(result->dtype(), dtype);
  EXPECT_EQ(result->shape(), (mx::Shape{1, k}));
  EXPECT_EQ(values(*result), expected);
}

void affine_width(int bits, int group) {
  constexpr int k = 256, n = 16, experts = 2, selected = 3;
  const int words_per_row = (k / 32) * bits;
  const auto mask = (std::uint32_t{1} << bits) - 1;
  const int crossing = bits == 5 ? 6 : bits == 6 ? 5 : 3;
  const auto code = [&](int expert, int channel, int column) {
    if (column == crossing || column == 31 || column == k - 1)
      return mask;
    return static_cast<std::uint32_t>(column * 13 + expert * 7 + channel * 3) & mask;
  };
  std::vector<std::uint32_t> codes(experts * n * k);
  std::vector<double> scales(experts * n * k / group), biases(scales.size());
  for (int expert = 0; expert < experts; ++expert)
    for (int channel = 0; channel < n; ++channel) {
      for (int column = 0; column < k; ++column)
        codes[(expert * n + channel) * k + column] = code(expert, channel, column);
      for (int g = 0; g < k / group; ++g) {
        const auto index = (expert * n + channel) * (k / group) + g;
        scales[index] = std::ldexp(1.0, g % 3 - 2);
        biases[index] = expert * .5 - .25;
      }
    }
  const auto words = mlx_affine_test::pack(codes, experts * n, k, bits);
  const auto weight = [&](int expert, int channel, int column) {
    const auto index = (expert * n + channel) * (k / group) + column / group;
    return double(code(expert, channel, column)) * scales[index] + biases[index];
  };
  std::vector<double> activation(selected * k, 0.0);
  for (int row = 0; row < selected; ++row)
    for (const int column : {0, crossing, 31, 32, group - 1, group, k - 1})
      activation[row * k + column] = (column % 2 ? -.5 : .25) * (row + 1);
  const std::array<std::int32_t, selected> ids{1, 0, 1};
  const auto dot = [&](int row, int expert, int channel) {
    double sum = 0;
    for (int column = 0; column < k; ++column)
      sum += activation[row * k + column] * weight(expert, channel, column);
    return sum;
  };
  // Small integer codes and dyadic scales/A make every decode and partial sum
  // exactly representable in F32. This checks packing, not general contraction.
  auto context = a::Context::create().value();
  std::vector<mx::array> ports{
      ready(make(std::span<const double>(activation.data(), k), {1, k}, mx::float32)),
      ready(mx::array(words.data(), {n, words_per_row}, mx::uint32)),
      ready(make(std::span<const double>(scales.data(), n * k / group),
                 {n, k / group}, mx::float32)),
      ready(make(std::span<const double>(biases.data(), n * k / group),
                 {n, k / group}, mx::float32))};
  auto result = execute(context, c::model_ops::mlx_quantized_matmul(
      c::model_ops::MlxWeightFormat::kAffine, bits, group), ports);
  ASSERT_TRUE(result) << result.error().message();
  std::vector<double> expected(n);
  for (int channel = 0; channel < n; ++channel)
    expected[channel] = dot(0, 0, channel);
  EXPECT_EQ(values(*result), expected);

  ports = {ready(make(activation, {1, selected, k}, mx::float32)),
      ready(mx::array(words.data(), {experts, n, words_per_row}, mx::uint32)),
      ready(make(scales, {experts, n, k / group}, mx::float32)),
      ready(make(biases, {experts, n, k / group}, mx::float32)),
      ready(mx::array(ids.data(), {1, selected}, mx::int32))};
  const auto grouped = c::model_ops::mlx_grouped_matmul(
      c::model_ops::MlxWeightFormat::kAffine, bits, group);
  result = execute(context, grouped, ports);
  ASSERT_TRUE(result) << result.error().message();
  expected.resize(selected * n);
  for (int row = 0; row < selected; ++row)
    for (int channel = 0; channel < n; ++channel)
      expected[row * n + channel] = dot(row, ids[row], channel);
  EXPECT_EQ(values(*result), expected);
  const std::array<std::int32_t, selected> invalid{1, experts, 1};
  ports.back() = ready(mx::array(invalid.data(), {1, selected}, mx::int32));
  auto failed = execute(context, grouped, ports);
  ASSERT_FALSE(failed);
  EXPECT_TRUE(failed.error().IsOutOfRange());

  const std::array<std::int32_t, selected> rows{n + 3, 2, n + 3};
  ports = {ready(mx::array(words.data(), {experts * n, words_per_row}, mx::uint32)),
      ready(make(scales, {experts * n, k / group}, mx::float32)),
      ready(make(biases, {experts * n, k / group}, mx::float32)),
      ready(mx::array(rows.data(), {selected}, mx::int32))};
  result = execute(context, c::model_ops::mlx_quantized_embedding(bits, group), ports);
  ASSERT_TRUE(result) << result.error().message();
  expected.resize(selected * k);
  for (int row = 0; row < selected; ++row)
    for (int column = 0; column < k; ++column)
      expected[row * k + column] = weight(rows[row] / n, rows[row] % n, column);
  EXPECT_EQ(values(*result), expected);
}
} // namespace apple_affine_contract_test

TEST(AppleMlxQuantized, AffineMatmulKeepsF32DecodedOperandsAcrossQmvAndQmm) {
  RecordProperty("contract", "F32 affine decode/dot, declared low output; no artifact-quality claim");
  for (const auto dtype : {mlx::core::bfloat16, mlx::core::float16})
    for (const int rows : {1, 64}) {
      SCOPED_TRACE(::testing::Message() << "dtype=" << dtype << " M=" << rows);
      apple_affine_contract_test::matmul_witness(dtype, rows);
    }
}

TEST(AppleMlxQuantized, AffineEmbeddingKeepsItsOwnDeclaredTypedOutput) {
  RecordProperty("contract", "Typed decoded embedding output; exact multiply control only");
  for (const auto dtype : {mlx::core::bfloat16, mlx::core::float16}) {
    SCOPED_TRACE(::testing::Message() << "dtype=" << dtype);
    apple_affine_contract_test::embedding_control(dtype);
  }
}

TEST(AppleMlxQuantized, AffineWidthsUseCrossWordCodesAndIndependentGroups) {
  for (const auto &[bits, group] :
       {std::pair{5, 32}, std::pair{6, 64}, std::pair{8, 128}}) {
    SCOPED_TRACE(::testing::Message() << "bits=" << bits << " group=" << group);
    apple_affine_contract_test::affine_width(bits, group);
  }
}
} // namespace
