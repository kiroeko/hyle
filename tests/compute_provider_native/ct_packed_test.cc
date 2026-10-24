#include "../compute/native_context_fixture.h"
#include "../compute_models/ct_packed_linear_fixture.h"
#include "../../src/compute_packed_shared/artifact_packed_linear.h"

#include "hyle/compute/prepared_operation.h"
#include "hyle/mem/buffer.h"
#include "hyle/mem/resource.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace m = hyle::mem;
namespace t = ct_packed_test;
using Type = c::DataType;
using Ct = c::packed::internal::ArtifactPackedLinear;

c::TensorDesc matrix(Type type, std::int64_t rows, std::int64_t columns) {
  return c::TensorDesc::dense(type, std::array{rows, columns}).value();
}
template <class Value> m::Buffer store(std::span<const Value> values) {
  auto result = m::Buffer::create(m::system_resource(), values.size_bytes(), 64).value();
  EXPECT_TRUE(result.view().write(std::as_bytes(values)).ok());
  return result;
}
m::Buffer floating(Type type, const std::vector<double> &values) {
  if (type == Type::kFloat32) {
    std::vector<float> converted;
    converted.reserve(values.size());
    for (const auto value : values) converted.push_back(static_cast<float>(value));
    return store<float>(converted);
  }
  std::vector<std::uint16_t> converted;
  converted.reserve(values.size());
  for (const auto value : values)
    converted.push_back(type == Type::kFloat16 ? half_test::encode<t::F16>(value).bits
                                              : half_test::encode<t::BF16>(value).bits);
  return store<std::uint16_t>(converted);
}
std::vector<double> read(Type type, const m::Buffer &buffer) {
  if (type == Type::kFloat32) {
    std::vector<float> values(buffer.size() / sizeof(float));
    EXPECT_TRUE(buffer.view().read(std::as_writable_bytes(std::span(values))).ok());
    return {values.begin(), values.end()};
  }
  std::vector<std::uint16_t> bits(buffer.size() / 2);
  EXPECT_TRUE(buffer.view().read(std::as_writable_bytes(std::span(bits))).ok());
  std::vector<double> values;
  for (const auto value : bits)
    values.push_back(type == Type::kFloat16 ? half_test::decode_bits<t::F16>(value)
                                          : half_test::decode_bits<t::BF16>(value));
  return values;
}

struct Profile final {
  bool asymmetric;
  Type scale, decode, operand;
};
struct Bindings final {
  t::Fixture fixture;
  std::vector<double> activation;
  std::array<m::Buffer, 4> buffers;
  std::vector<c::TensorDesc> descriptions;
  std::vector<c::ConstTensorView> views;
  c::TensorDesc output;
  Bindings(Profile profile, unsigned seed, std::int64_t rows = 2,
           bool special = false, bool witness = true)
      : fixture(profile.asymmetric, seed),
        activation(fixture.activation(rows, profile.operand, seed, witness)),
        output(matrix(profile.operand, rows, fixture.rows)) {
    if (special) {
      fixture.scales[0] = std::numeric_limits<double>::quiet_NaN();
      fixture.scales[2] = std::numeric_limits<double>::infinity();
    }
    descriptions.push_back(matrix(profile.operand, rows, fixture.depth));
    descriptions.push_back(matrix(Type::kInt32, fixture.rows, fixture.depth / 8));
    if (profile.asymmetric)
      descriptions.push_back(matrix(Type::kInt32, (fixture.rows + 7) / 8,
                                    fixture.depth / fixture.group));
    descriptions.push_back(matrix(profile.scale, fixture.rows,
                                  fixture.depth / fixture.group));
    buffers[0] = floating(profile.operand, activation);
    buffers[1] = store<std::int32_t>(fixture.codes);
    if (profile.asymmetric) buffers[2] = store<std::int32_t>(fixture.zeros);
    buffers[descriptions.size() - 1] = floating(profile.scale, fixture.scales);
    for (std::size_t i = 0; i < descriptions.size(); ++i)
      views.push_back(c::ConstTensorView::create(descriptions[i], buffers[i].view()).value());
  }
};
Ct descriptor(Profile profile) {
  return {profile.asymmetric ? c::model_ops::WeightArtifactFormat::kCompressedTensorsAsymmetric4
                            : c::model_ops::WeightArtifactFormat::kCompressedTensorsSymmetric4,
          32, profile.decode, profile.operand};
}
void expect_result(Profile profile, const Bindings &binding, const m::Buffer &output) {
  const auto weights = binding.fixture.decoded(profile.scale, profile.decode, profile.operand);
  const auto actual = read(profile.operand, output);
  const auto columns = static_cast<std::size_t>(binding.fixture.rows);
  const auto depth = static_cast<std::size_t>(binding.fixture.depth);
  for (std::size_t i = 0; i < actual.size(); ++i) {
    // The pointwise witnesses use one nonzero A per row. The replay inputs
    // use small dyadic values whose products/sums fit F32 exactly. NaN and
    // infinity still participate at zero lanes; never skip 0*NaN.
    float expected = 0.F;
    for (std::size_t k = 0; k < depth; ++k)
      expected += static_cast<float>(binding.activation[(i / columns) * depth + k]) *
                  static_cast<float>(weights[(i % columns) * depth + k]);
    const auto rounded = t::round(profile.operand, expected);
    if (std::isnan(rounded)) EXPECT_TRUE(std::isnan(actual[i])) << i;
    else EXPECT_EQ(actual[i], rounded) << i;
  }
}

TEST(NativeCtPackedLinear, SourceDecodeAndOperandRoundsWithPackedOutputAxisZeroTail) {
  auto context = c::test::native_context().value();
  auto queue = context.create_queue().value();
  constexpr std::array profiles{
      Profile{false, Type::kFloat32, Type::kFloat32, Type::kFloat32},
      Profile{true, Type::kBFloat16, Type::kBFloat16, Type::kBFloat16},
      Profile{true, Type::kFloat16, Type::kFloat32, Type::kFloat32},
      Profile{false, Type::kBFloat16, Type::kFloat16, Type::kFloat32},
      Profile{true, Type::kFloat32, Type::kBFloat16, Type::kFloat16},
      Profile{false, Type::kFloat32, Type::kFloat16, Type::kBFloat16}};
  for (const auto profile : profiles) {
    SCOPED_TRACE(::testing::Message() << profile.asymmetric << '/' << int(profile.scale)
                 << '/' << int(profile.decode) << '/' << int(profile.operand));
    Bindings binding(profile, 0);
    const std::array outputs{binding.output};
    auto prepared = c::prepare(context, c::Operation::custom(descriptor(profile)),
        binding.descriptions, outputs, m::system_resource()).value();
    EXPECT_EQ(prepared.provider().id, c::native::kProviderId);
    EXPECT_EQ(prepared.workspace_requirement().size, 0U);
    auto destination = m::Buffer::create(m::system_resource(),
        binding.output.required_bytes().value(), 64).value();
    const std::array output_views{c::TensorView::create(binding.output, destination.view()).value()};
    auto slot = prepared.create_invocation().value();
    auto done = slot.submit(queue, binding.views, output_views);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    expect_result(profile, binding, destination);
  }
  const Profile profile{true, Type::kFloat32, Type::kFloat32, Type::kFloat32};
  Bindings special(profile, 0, 2, true);
  const std::array outputs{special.output};
  auto prepared = c::prepare(context, c::Operation::custom(descriptor(profile)),
      special.descriptions, outputs, m::system_resource()).value();
  auto destination = m::Buffer::create(m::system_resource(),
      special.output.required_bytes().value(), 64).value();
  const std::array output_views{c::TensorView::create(special.output, destination.view()).value()};
  auto slot = prepared.create_invocation().value();
  auto done = slot.submit(queue, special.views, output_views);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  expect_result(profile, special, destination);
}

TEST(NativeCtPackedLinear, DynamicInputsUseIndependentSlotsAndRejectReadWriteAlias) {
  const Profile profile{true, Type::kBFloat16, Type::kFloat16, Type::kBFloat16};
  auto context = c::test::native_context().value();
  auto queue = context.create_queue().value();
  Bindings first(profile, 0), second(profile, 5, 2, false, false);
  const std::array outputs{first.output};
  auto prepared = c::prepare(context, c::Operation::custom(descriptor(profile)),
      first.descriptions, outputs, m::system_resource()).value();
  auto slot0 = prepared.create_invocation().value(), slot1 = prepared.create_invocation().value();
  auto result0 = m::Buffer::create(m::system_resource(), first.output.required_bytes().value(), 64).value();
  auto result1 = m::Buffer::create(m::system_resource(), first.output.required_bytes().value(), 64).value();
  const std::array views0{c::TensorView::create(first.output, result0.view()).value()};
  const std::array views1{c::TensorView::create(first.output, result1.view()).value()};
  for (std::size_t update = 0; update < 2; ++update) {
    SCOPED_TRACE(update);
    Bindings changed(profile, static_cast<unsigned>(update * 3), 2, false, false);
    EXPECT_NE(changed.activation, second.activation);
    auto done0 = slot0.submit(queue, changed.views, views0);
    ASSERT_TRUE(done0) << done0.error().message();
    ASSERT_TRUE(done0->wait().ok());
    expect_result(profile, changed, result0);
    const auto retained = read(profile.operand, result0);
    auto done1 = slot1.submit(queue, second.views, views1);
    ASSERT_TRUE(done1) << done1.error().message();
    ASSERT_TRUE(done1->wait().ok());
    expect_result(profile, second, result1);
    EXPECT_EQ(read(profile.operand, result0), retained);
  }
  const auto original = read(profile.operand, first.buffers[0]);
  const std::array aliased{c::TensorView::create(first.output, first.buffers[0].view()).value()};
  auto bad = prepared.create_invocation().value();
  auto done = bad.submit(queue, first.views, aliased);
  if (done) EXPECT_TRUE(done->wait().IsInvalidArgument());
  else EXPECT_TRUE(done.error().IsInvalidArgument());
  EXPECT_EQ(read(profile.operand, first.buffers[0]), original);
}

TEST(NativeCtPackedLinear, RealQ27GeometryPreparesWithoutADenseWeightWorkspace) {
  auto context = c::test::native_context().value();
  const std::array inputs{matrix(Type::kBFloat16, 1, 5120),
      matrix(Type::kInt32, 17408, 640), matrix(Type::kInt32, 2176, 160),
      matrix(Type::kBFloat16, 17408, 160)};
  const std::array outputs{matrix(Type::kBFloat16, 1, 17408)};
  const Profile profile{true, Type::kBFloat16, Type::kBFloat16, Type::kBFloat16};
  auto prepared = c::prepare(context, c::Operation::custom(descriptor(profile)),
      inputs, outputs, m::system_resource());
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, c::native::kProviderId);
  EXPECT_EQ(prepared->workspace_requirement().size, 0U);
  EXPECT_EQ(prepared->invocation_private_bytes(), 0U);
  // No payload allocation or execution is implied by this large shape check.
}
} // namespace
