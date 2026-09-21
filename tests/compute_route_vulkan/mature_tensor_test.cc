#include <gtest/gtest.h>

#include <algorithm>

#include "gate_norm_test_support.h"
#include "reduction_test_support.h"
#include "vulkan_program_internal.h"

namespace {
using namespace vulkan_gate_norm_test;
namespace d = v::detail;

void expect_mature(const v::Program &program) {
  const auto &dispatches = d::Access::program(program)->dispatches;
  ASSERT_FALSE(dispatches.empty());
  ASSERT_TRUE(dispatches.back().region);
  EXPECT_EQ(std::ranges::count_if(dispatches,
                                  [](const auto &dispatch) {
                                    return dispatch.region != nullptr;
                                  }),
            1);
}

std::size_t broadcast_index(const c::TensorDesc &input,
                            const c::TensorDesc &output, std::size_t index) {
  std::array<std::size_t, c::kMaxTensorRank> coordinates{};
  for (std::size_t axis = output.rank(); axis-- > 0;) {
    coordinates[axis] = index % output.dimensions()[axis];
    index /= output.dimensions()[axis];
  }
  std::size_t result = 0;
  for (std::size_t axis = 0; axis < input.rank(); ++axis) {
    result *= input.dimensions()[axis];
    if (input.dimensions()[axis] != 1)
      result += coordinates[output.rank() - input.rank() + axis];
  }
  return result;
}

TEST(VulkanMatureTensor, CrossBroadcastUsesOriginalRegionAtRankTwoSixAndEight) {
  auto context = vulkan_test::context().value();
  const std::array pairs{
      std::array{dense(c::DataType::kFloat32, {3, 1}),
                 dense(c::DataType::kFloat32, {1, 7})},
      std::array{dense(c::DataType::kFloat32, {2, 1, 3, 1, 2, 1}),
                 dense(c::DataType::kFloat32, {1, 2, 1, 5, 1, 7})},
      std::array{dense(c::DataType::kFloat32, {2, 1, 3, 1, 2, 1, 3, 1}),
                 dense(c::DataType::kFloat32, {1, 2, 1, 5, 1, 2, 1, 7})}};
  for (const auto &descriptions : pairs) {
    for (const auto kind : {c::BinaryKind::kAdd, c::BinaryKind::kSubtract,
                            c::BinaryKind::kMultiply, c::BinaryKind::kDivide}) {
      SCOPED_TRACE(::testing::Message()
                   << descriptions[0].rank() << '/' << int(kind));
      auto program = v::prepare(context, c::Binary{kind}, descriptions[0],
                                descriptions[1]);
      ASSERT_TRUE(program) << program.error().message();
      expect_mature(*program);
      auto first = program->create_invocation().value();
      auto second = program->create_invocation().value();
      v::Tensor retained;
      std::vector<double> saved;
      for (int phase = 0; phase != 3; ++phase) {
        std::array<std::vector<double>, 2> data;
        std::array<v::Tensor, 2> inputs;
        for (std::size_t port = 0; port != 2; ++port) {
          data[port].resize(*descriptions[port].element_count());
          for (std::size_t i = 0; i != data[port].size(); ++i)
            data[port][i] = 0.25 + ((i * 7 + phase * 3 + port) % 29) * 0.125;
          inputs[port] = make(context, descriptions[port], data[port]);
        }
        auto &slot = phase == 0 ? first : second;
        ASSERT_TRUE(slot.bind(inputs).ok());
        auto work = slot.submit().value();
        auto result = work.wait();
        ASSERT_TRUE(result) << result.error().message();
        const auto &output = result->get()->tensors()[0];
        std::vector<double> expected(*output.description().element_count());
        for (std::size_t i = 0; i != expected.size(); ++i) {
          const auto a = data[0][broadcast_index(descriptions[0],
                                                 output.description(), i)];
          const auto b = data[1][broadcast_index(descriptions[1],
                                                 output.description(), i)];
          expected[i] = kind == c::BinaryKind::kAdd        ? a + b
                        : kind == c::BinaryKind::kSubtract ? a - b
                        : kind == c::BinaryKind::kMultiply ? a * b
                                                           : a / b;
        }
        expect(context, output, expected);
        if (phase == 0) {
          retained = output;
          saved = expected;
        } else {
          expect(context, retained, saved);
        }
      }
      EXPECT_FALSE(first.submit());
    }
  }
}

TEST(VulkanMatureTensor, ArbitraryAxesKeepDimensionsAndExactRowTiles) {
  auto context = vulkan_test::context().value();
  const std::array descriptions{
      dense(c::DataType::kFloat32, {3, 7, 171}),
      dense(c::DataType::kFloat32, {2, 3, 2, 5, 2, 3}),
      dense(c::DataType::kFloat32, {2, 1, 3, 2, 1, 2, 3, 5})};
  for (const auto &description : descriptions) {
    std::vector<c::Axes> axes{c::Axes{}, c::Axes::from(std::array{1}).value()};
    if (description.rank() > 3)
      axes.push_back(c::Axes::from(std::array{1, 3, -1}).value());
    for (auto selected : axes)
      for (auto kind : {c::ReductionKind::kSum, c::ReductionKind::kMean})
        for (bool keep : {false, true}) {
          SCOPED_TRACE(::testing::Message() << description.rank() << '/'
                                            << int(kind) << '/' << keep);
          const c::Reduce operation{.kind = kind,
                                    .axes = selected,
                                    .keep_dimensions = keep,
                                    .accumulation_type = c::DataType::kFloat32};
          auto program = v::prepare(context, operation, description);
          ASSERT_TRUE(program) << program.error().message();
          expect_mature(*program);
          auto slot = program->create_invocation().value();
          for (int phase = 0; phase != 2; ++phase) {
            const auto data = vulkan_reduction_test::pattern(
                *description.element_count(), phase);
            auto input = make(context, description, data);
            ASSERT_TRUE(slot.bind(input).ok());
            auto work = slot.submit().value();
            auto result = work.wait();
            ASSERT_TRUE(result) << result.error().message();
            const auto expected = vulkan_reduction_test::reduce_reference(
                data, description, operation);
            expect(context, result->get()->tensors()[0], expected);
          }
        }
  }
}

TEST(VulkanMatureTensor, CopyPreservesBitsAndConcatRestoresHighRankAxes) {
  auto context = vulkan_test::context().value();
  const std::array<std::uint64_t, 8> edges{0x80000000, 0x00000001, 0x7f800000,
                                           0xff800000, 0x7fc12345, 0x7f812345,
                                           0xffc54321, 0x3f800000};
  const auto description = dense(c::DataType::kFloat32, {2, 1, 2, 1, 2, 1});
  const auto input =
      context.upload(description, payload(description, edges)).value();
  auto copy = v::prepare(context, c::Copy{}, description);
  ASSERT_TRUE(copy) << copy.error().message();
  expect_mature(*copy);
  auto copy_slot = copy->create_invocation().value();
  ASSERT_TRUE(copy_slot.bind(input).ok());
  auto copied_work = copy_slot.submit().value();
  auto copied = copied_work.wait().value();
  EXPECT_EQ(bits(context, copied->tensors()[0]),
            std::vector(edges.begin(), edges.end()));

  for (int axis = 0; axis != 6; ++axis) {
    SCOPED_TRACE(axis);
    std::array<std::int64_t, 6> shape{2, 1, 2, 1, 3, 5};
    std::array<c::TensorDesc, 3> descriptions;
    std::array<v::Tensor, 3> inputs;
    std::array<std::vector<double>, 3> data;
    for (std::size_t port = 0; port != 3; ++port) {
      shape[axis] = static_cast<std::int64_t>(port + 1);
      descriptions[port] =
          c::TensorDesc::dense(c::DataType::kFloat32, shape).value();
      data[port] = vulkan_reduction_test::pattern(
          *descriptions[port].element_count(), int(port));
      inputs[port] = make(context, descriptions[port], data[port]);
    }
    auto program = v::prepare(context, c::Concat{axis}, descriptions[0],
                              descriptions[1], descriptions[2]);
    ASSERT_TRUE(program) << program.error().message();
    expect_mature(*program);
    auto slot = program->create_invocation().value();
    ASSERT_TRUE(slot.bind(inputs).ok());
    auto work = slot.submit().value();
    auto result = work.wait().value();
    std::size_t inner = 1, outer = 1;
    for (int i = 0; i != axis; ++i)
      outer *= shape[i];
    for (int i = axis + 1; i != 6; ++i)
      inner *= shape[i];
    std::vector<double> expected;
    for (std::size_t row = 0; row != outer; ++row)
      for (std::size_t port = 0; port != 3; ++port) {
        const auto width = inner * (port + 1);
        const auto begin = data[port].begin() + row * width;
        expected.insert(expected.end(), begin, begin + width);
      }
    expect(context, result->tensors()[0], expected, true);
  }
}

TEST(VulkanMatureTensor, HighRankNormAndUnaryStayInMatureRegion) {
  auto context = vulkan_test::context().value();
  const auto description =
      dense(c::DataType::kFloat32, {2, 1, 3, 1, 2, 1, 1, 129});
  const auto data =
      vulkan_reduction_test::pattern(*description.element_count());
  const auto input = make(context, description, data);
  for (const auto operation :
       {c::Operation(c::RmsNorm{}), c::Operation(c::LayerNorm{}),
        c::Operation(c::Unary{c::UnaryKind::kTanh})}) {
    auto program = v::prepare(context, operation, description);
    ASSERT_TRUE(program) << program.error().message();
    expect_mature(*program);
    auto slot = program->create_invocation().value();
    ASSERT_TRUE(slot.bind(input).ok());
    auto work = slot.submit().value();
    auto result = work.wait().value();
    auto expected = data;
    if (operation.get_if<c::Unary>())
      for (auto &value : expected)
        value = std::tanh(value);
    else
      expected =
          norm(data, 129, operation.get_if<c::LayerNorm>() != nullptr, 1e-5F);
    expect(context, result->tensors()[0], expected);
  }
}
} // namespace
