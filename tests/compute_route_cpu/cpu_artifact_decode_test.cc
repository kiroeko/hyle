#include <gtest/gtest.h>
#include <array>
#include <vector>
#include "../compute_provider_native/half_test_support.h"
#include "hyle/compute/cpu.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"

namespace {
namespace c = hyle::compute;
namespace m = hyle::mem;
using Format = c::model_ops::WeightArtifactFormat;
c::TensorDesc dense(c::DataType type, std::initializer_list<std::int64_t> dims) {
  return c::TensorDesc::dense(type, {dims.begin(), dims.size()}).value();
}
template<class T> m::Buffer buffer(std::span<const T> values) {
  auto result = m::Buffer::create(m::system_resource(), values.size_bytes(), 64).value();
  EXPECT_TRUE(result.view().write(std::as_bytes(values)).ok());return result;
}
void check(Format format, std::uint32_t group,
    std::span<const c::TensorDesc> descriptions, std::span<m::Buffer> buffers,
    std::span<const float> expected) {
  auto context = c::cpu::create_context().value();
  const auto op = c::model_ops::artifact_weight_decode(format, group, c::DataType::kFloat32);
  std::array<c::TensorDesc, 1> inferred;
  ASSERT_TRUE(c::infer_output_descriptions(op, descriptions, inferred).ok());
  auto result = m::Buffer::create(m::system_resource(), inferred[0].required_bytes().value(), 64).value();
  auto target = c::TensorView::create(inferred[0], result.view()).value();
  std::vector<c::ConstTensorView> sources;
  for (std::size_t i = 0; i < buffers.size(); ++i)
    sources.push_back(c::ConstTensorView::create(descriptions[i], buffers[i].view()).value());
  auto prepared = c::prepare(context, op, sources, std::span(&target, 1));
  ASSERT_TRUE(prepared) << prepared.error().message();
  auto invocation = prepared->create_invocation().value();
  auto queue = context.create_queue().value();
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto done = invocation.submit(queue, sources, std::span(&target, 1));
    ASSERT_TRUE(done) << done.error().message();ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<float>(result);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) EXPECT_EQ(actual[i], expected[i]);
  }
}

TEST(CpuArtifactDecode, Nf4AndNestedScalesPreserveOriginalHighFirstOrder) {
  constexpr int n = 2, k = 256, group = 64, groups = n * k / group;
  std::vector<std::uint8_t> codes(n * k / 2);
  for (std::size_t i = 0; i < codes.size(); ++i) codes[i] = std::uint8_t((i % 16) * 16 + (15 - i % 16));
  std::array<float, 16> codebook{};
  for (std::size_t i = 0; i < codebook.size(); ++i) codebook[i] = (float(i) - 8) / 8;
  std::array<float, groups> absmax{};for (int i = 0; i < groups; ++i) absmax[i] = float(i + 1) / 8;
  std::vector<float> expected(n * k);
  for (std::size_t i = 0; i < codes.size(); ++i) {
    expected[i * 2] = codebook[codes[i] >> 4] * absmax[i * 2 / group];
    expected[i * 2 + 1] = codebook[codes[i] & 15] * absmax[(i * 2 + 1) / group];
  }
  const std::array descs{dense(c::DataType::kUInt8,{n,k/2}),dense(c::DataType::kFloat32,{groups}),dense(c::DataType::kFloat32,{16})};
  std::array inputs{buffer<std::uint8_t>(codes),buffer<float>(absmax),buffer<float>(codebook)};
  check(Format::kNf4, group, descs, inputs, expected);

  std::array<std::uint8_t, groups> indices{};std::array<float,256> nested{};
  for (int i = 0; i < 256; ++i) nested[i] = float(i) / 16;
  for (int i = 0; i < groups; ++i) indices[i] = std::uint8_t(i);
  const std::array<float,1> nested_absmax{0.5f},offset{0.125f};
  for (std::size_t i = 0; i < codes.size(); ++i) {
    expected[i*2] = codebook[codes[i]>>4] * (nested[indices[i*2/group]]*0.5f+0.125f);
    expected[i*2+1] = codebook[codes[i]&15] * (nested[indices[(i*2+1)/group]]*0.5f+0.125f);
  }
  const std::array nested_descs{descs[0],dense(c::DataType::kUInt8,{groups}),dense(c::DataType::kFloat32,{1}),dense(c::DataType::kFloat32,{256}),dense(c::DataType::kFloat32,{}),descs[2]};
  std::array nested_inputs{buffer<std::uint8_t>(codes),buffer<std::uint8_t>(indices),buffer<float>(nested_absmax),buffer<float>(nested),buffer<float>(offset),buffer<float>(codebook)};
  check(Format::kNf4DoubleQuant, group, nested_descs, nested_inputs, expected);
}

TEST(CpuArtifactDecode, AwqAndCompressedTensorsUseDifferentOriginalWordLayouts) {
  constexpr int k = 256, n = 8, group = 128;
  std::vector<std::int32_t> awq_words(k, 0x76543210),zeros(k/group, 0x33333333);
  std::vector<float> scales(k/group*n, 0.125f),expected(n*k);
  constexpr std::array<int,8> reverse{0,4,1,5,2,6,3,7};
  for (int row = 0; row < n; ++row) for (int col = 0; col < k; ++col)
    expected[row*k+col] = float(reverse[row]-3)*0.125f;
  const std::array awq_descs{dense(c::DataType::kInt32,{k,n/8}),dense(c::DataType::kInt32,{k/group,n/8}),dense(c::DataType::kFloat32,{k/group,n})};
  std::array awq_inputs{buffer<std::int32_t>(awq_words),buffer<std::int32_t>(zeros),buffer<float>(scales)};
  check(Format::kAwqGemm4, group, awq_descs, awq_inputs, expected);

  std::vector<std::int32_t> ct_words(n*k/8,0x76543210);
  for (int row = 0; row < n; ++row) for (int col = 0; col < k; ++col)
    expected[row*k+col] = float(col%8-8)*0.125f;
  const std::array ct_descs{dense(c::DataType::kInt32,{n,k/8}),dense(c::DataType::kFloat32,{n,k/group})};
  std::array ct_inputs{buffer<std::int32_t>(ct_words),buffer<float>(scales)};
  check(Format::kCompressedTensorsSymmetric4, group, ct_descs, ct_inputs, expected);
}
} // namespace
