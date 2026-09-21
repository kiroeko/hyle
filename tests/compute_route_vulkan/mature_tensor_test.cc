#include <gtest/gtest.h>

#include <algorithm>

#include "concat_test_support.h"
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
  for (const auto &operation :
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

hyle::Result<v::Tensor> run_prepared(const v::Program &program,
                                     std::span<const v::Tensor> inputs) {
  HYLE_ASSIGN_OR_RETURN(auto slot, program.create_invocation());
  HYLE_RETURN_IF_ERROR(slot.bind(inputs));
  HYLE_ASSIGN_OR_RETURN(auto work, slot.submit());
  HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
  return result->tensors()[0];
}

TEST(VulkanMatureTensor, IntegerCopyPreservesBitsInIndependentStorage) {
  auto context = vulkan_test::context().value();
  const auto description = dense(c::DataType::kInt32, {3, 1, 1, 1, 1, 171});
  const std::array<std::uint64_t, 8> edges{0,          1,          0x80000000,
                                           0x7fffffff, 0xffffffff, 0x7f800001,
                                           0xff800001, 0x12345678};
  std::vector<std::uint64_t> raw(*description.element_count());
  for (std::size_t i = 0; i != raw.size(); ++i)
    raw[i] = edges[i % edges.size()];
  const auto input =
      context.upload(description, payload(description, raw)).value();
  auto program = v::prepare(context, c::Copy{}, description);
  ASSERT_TRUE(program) << program.error().message();
  expect_mature(*program);
  auto output = run_prepared(*program, std::array{input});
  ASSERT_TRUE(output) << output.error().message();
  EXPECT_EQ(bits(context, *output), raw);
  EXPECT_NE(output->native_buffer(), input.native_buffer());
}

TEST(VulkanMatureTensor, StorageConcatPreservesBitsAcrossEmptyPortsAndAllAxes) {
  auto context = vulkan_test::context().value();
  const std::array<std::uint64_t, 12> edges16{0,      0x8000, 1,      0x7c00,
                                              0xfc00, 0x7e01, 0x7d01, 0x7f80,
                                              0xff80, 0x7f81, 0xffff, 0x7fff};
  const std::array<std::uint64_t, 12> edges32{
      0,          0x80000000, 1,          0x7f800000, 0xff800000, 0x7fc12345,
      0x7f800001, 0xff800001, 0x7fffffff, 0xffffffff, 0x12345678, 0xfedcba98};
  const std::array width_sets{std::array<std::int64_t, 6>{0, 1, 2, 0, 3, 0},
                              std::array<std::int64_t, 6>{0, 0, 0, 3, 0, 0},
                              std::array<std::int64_t, 6>{0, 0, 0, 0, 0, 0}};
  for (auto type : {c::DataType::kFloat32, c::DataType::kFloat16,
                    c::DataType::kBFloat16, c::DataType::kInt32}) {
    const auto &edges = c::data_type_size(type) == 2 ? edges16 : edges32;
    for (int axis = 0; axis != 6; ++axis)
      for (const auto &widths : width_sets) {
        SCOPED_TRACE(::testing::Message() << int(type) << '/' << axis << '/'
                                          << widths[3] << '/' << widths[4]);
        std::array<std::int64_t, 6> shape{2, 1, 2, 1, 3, 5};
        std::array<v::Tensor, 6> inputs;
        std::array<std::vector<std::uint64_t>, 6> parts;
        std::vector<v::Argument> arguments;
        for (std::size_t port = 0; port != inputs.size(); ++port) {
          shape[axis] = widths[port];
          const auto description = c::TensorDesc::dense(type, shape).value();
          auto &raw = parts[port];
          raw.resize(*description.element_count());
          for (std::size_t i = 0; i != raw.size(); ++i)
            raw[i] = i % 16 < edges.size()
                         ? edges[(i + port) % edges.size()]
                         : vulkan_concat_test::raw_value(i, port, type);
          inputs[port] =
              context.upload(description, payload(description, raw)).value();
          arguments.emplace_back(description);
        }
        auto program = v::prepare(context, c::Operation(c::Concat{axis}),
                                  std::span<const v::Argument>(arguments));
        ASSERT_TRUE(program) << program.error().message();
        if (widths[3] || widths[4])
          expect_mature(*program);
        auto output = run_prepared(*program, inputs);
        ASSERT_TRUE(output) << output.error().message();
        std::size_t inner = 1, outer = 1;
        for (int i = 0; i != axis; ++i)
          outer *= shape[i];
        for (int i = axis + 1; i != 6; ++i)
          inner *= shape[i];
        EXPECT_EQ(bits(context, *output),
                  vulkan_concat_test::concatenate(parts, widths, outer, inner));
        if (widths[3])
          EXPECT_NE(output->native_buffer(), inputs[3].native_buffer());
      }
  }
}

TEST(VulkanMatureTensor, EverySixteenBitPatternCopiesAndConvertsWithOddTail) {
  auto context = vulkan_test::context().value();
  std::vector<std::uint64_t> raw(65537);
  for (std::size_t i = 0; i != raw.size(); ++i)
    raw[i] = i & 0xffff;
  raw.back() = 0x8000;
  for (auto type : {c::DataType::kFloat16, c::DataType::kBFloat16}) {
    const auto description = dense(type, {65537});
    const auto input =
        context.upload(description, payload(description, raw)).value();
    auto copy = v::prepare(context, c::Copy{}, description);
    ASSERT_TRUE(copy) << copy.error().message();
    expect_mature(*copy);
    auto copied = run_prepared(*copy, std::array{input});
    ASSERT_TRUE(copied) << copied.error().message();
    EXPECT_EQ(bits(context, *copied), raw);
    std::vector<double> expected;
    for (auto value : raw)
      expected.push_back(decode(value, type));
    for (auto target : {c::DataType::kFloat32, type == c::DataType::kFloat16
                                                   ? c::DataType::kBFloat16
                                                   : c::DataType::kFloat16}) {
      SCOPED_TRACE(::testing::Message() << int(type) << '/' << int(target));
      auto program = v::prepare(context, c::Cast{target}, description);
      ASSERT_TRUE(program) << program.error().message();
      if (target == c::DataType::kBFloat16)
        ASSERT_TRUE(d::Access::program(*program)->dispatches.back().iree);
      else
        expect_mature(*program);
      auto output = run_prepared(*program, std::array{input});
      ASSERT_TRUE(output) << output.error().message();
      expect(context, *output, expected, true);
    }
  }
}

TEST(VulkanMatureTensor, NarrowingPreservesNanClassAndNearestEvenBoundaries) {
  auto context = vulkan_test::context().value();
  for (auto type : {c::DataType::kFloat16, c::DataType::kBFloat16}) {
    SCOPED_TRACE(::testing::Message() << "narrow target=" << int(type));
    // Independent decoded adjacent representable values and their midpoints;
    // keep raw signaling NaN inputs instead of converting them through double.
    std::vector<std::uint64_t> raw{
        0,          0x80000000, 1,          0x80000001, 0x7f800000,
        0xff800000, 0x7f800001, 0xff800001, 0x7fffffff, 0xffffffff,
        0x7f7fffff, 0xff7fffff, 0x477ff000, 0xc77ff000};
    const std::uint32_t last = type == c::DataType::kFloat16 ? 0x7bff : 0x7f7f;
    for (std::uint32_t i = 0; i < last; ++i) {
      const auto midpoint =
          static_cast<float>((decode(i, type) + decode(i + 1, type)) / 2);
      for (auto value : {std::nextafter(midpoint, -INFINITY), midpoint,
                         std::nextafter(midpoint, INFINITY)}) {
        raw.push_back(std::bit_cast<std::uint32_t>(value));
        raw.push_back(std::bit_cast<std::uint32_t>(-value));
      }
    }
    raw.push_back(0x80000000); // Non-multiple-of-workgroup odd tail.
    const auto description =
        dense(c::DataType::kFloat32, {static_cast<std::int64_t>(raw.size())});
    const auto input =
        context.upload(description, payload(description, raw)).value();
    auto program = v::prepare(context, c::Cast{type}, description);
    ASSERT_TRUE(program) << program.error().message();
    if (type == c::DataType::kBFloat16)
      ASSERT_TRUE(d::Access::program(*program)->dispatches.back().iree);
    else
      expect_mature(*program);
    auto output = run_prepared(*program, std::array{input});
    ASSERT_TRUE(output) << output.error().message();
    std::vector<double> expected;
    for (auto value : raw)
      expected.push_back(decode(value, c::DataType::kFloat32));
    expect(context, *output, expected, true);
  }
}

TEST(VulkanMatureTensor,
     LowPrecisionFamiliesUseOriginalProvidersAndFloatGamma) {
  auto context = vulkan_test::context().value();
  const auto gamma_description = dense(c::DataType::kFloat32, {129});
  std::vector<double> gamma(129);
  for (std::size_t i = 0; i != gamma.size(); ++i)
    gamma[i] = static_cast<float>(0.731 + 0.013 * (i % 19));
  const auto gamma_tensor = make(context, gamma_description, gamma);
  for (auto type : {c::DataType::kFloat16, c::DataType::kBFloat16}) {
    for (const auto &operation :
         {c::Operation(c::RmsNorm{}), c::Operation(c::LayerNorm{}),
          c::Operation(c::Unary{c::UnaryKind::kTanh}),
          c::Operation(
              c::GatedActivation{.inputs = c::GatedInputKind::kSeparate}),
          c::Operation(c::Binary{}),
          c::Operation(
              c::Reduce{.axes = c::Axes::from(std::array{0, -1}).value(),
                        .accumulation_type = c::DataType::kFloat32}),
          c::Operation(c::Reduce{.kind = c::ReductionKind::kMean,
                                 .axes = c::Axes::from(std::array{-1}).value(),
                                 .keep_dimensions = true,
                                 .accumulation_type = c::DataType::kFloat32}),
          c::Operation(
              c::Softmax{.accumulation_type = c::DataType::kFloat32})}) {
      const auto description = operation.get_if<c::Softmax>()
                                   ? dense(type, {3, 129})
                                   : dense(type, {3, 1, 3, 1, 1, 129});
      const bool affine =
          operation.get_if<c::RmsNorm>() || operation.get_if<c::LayerNorm>();
      const bool binary = operation.get_if<c::Binary>() != nullptr;
      const bool pair = binary || operation.get_if<c::GatedActivation>();
      const auto second_description =
          binary ? dense(type, {1, 2, 1, 1, 3, 1}) : description;
      std::vector<v::Argument> arguments{description};
      if (affine || pair)
        arguments.emplace_back(affine ? gamma_description : second_description);
      auto program = v::prepare(context, operation,
                                std::span<const v::Argument>(arguments));
      ASSERT_TRUE(program) << program.error().message();
      const auto &dispatches = d::Access::program(*program)->dispatches;
      ASSERT_EQ(std::ranges::count_if(dispatches,
                                      [](const auto &dispatch) {
                                        return dispatch.region != nullptr;
                                      }),
                1);
      EXPECT_EQ(dispatches.back().iree.has_value(),
                type == c::DataType::kBFloat16);
      auto slot = program->create_invocation().value();
      for (int phase = 0; phase != 2; ++phase) {
        auto data =
            vulkan_reduction_test::pattern(*description.element_count(), phase);
        std::vector<v::Tensor> inputs{make(context, description, data)};
        auto second = vulkan_reduction_test::pattern(
            *second_description.element_count(), phase + 3);
        if (affine)
          inputs.push_back(gamma_tensor);
        else if (pair)
          inputs.push_back(make(context, second_description, second));
        ASSERT_TRUE(slot.bind(inputs).ok());
        auto work = slot.submit().value();
        auto result = work.wait();
        ASSERT_TRUE(result) << result.error().message();
        const auto &output = result->get()->tensors()[0];
        std::vector<double> expected = data;
        if (affine)
          expected =
              norm(data, 129, operation.get_if<c::LayerNorm>() != nullptr,
                   1e-5F, gamma);
        else if (const auto *reduce = operation.get_if<c::Reduce>())
          expected = vulkan_reduction_test::reduce_reference(data, description,
                                                             *reduce);
        else if (const auto *softmax = operation.get_if<c::Softmax>())
          expected = vulkan_reduction_test::probability_reference(
              data, description, *softmax);
        else if (binary) {
          expected.resize(*output.description().element_count());
          for (std::size_t i = 0; i != expected.size(); ++i)
            expected[i] =
                data[broadcast_index(description, output.description(), i)] +
                second[broadcast_index(second_description, output.description(),
                                       i)];
        } else
          for (std::size_t i = 0; i != expected.size(); ++i)
            expected[i] = pair ? unary(c::UnaryKind::kSilu, data[i]) * second[i]
                               : std::tanh(data[i]);
        expect(context, output, expected);
      }
    }
  }
}

TEST(VulkanMatureTensor, BFloatConversionAllocationFailureReleasesQuota) {
  hyle::mem::QuotaAllocator quota(hyle::mem::system_allocator(), 1024 * 1024);
  {
    auto context = vulkan_test::context(&quota).value();
    auto program = v::prepare(context, c::Cast{c::DataType::kBFloat16},
                              dense(c::DataType::kFloat32, {129}));
    // The original IREE pool is acquired on first prepare, before slots exist.
    ASSERT_FALSE(program);
    EXPECT_TRUE(program.error().IsResourceExhausted());
    EXPECT_TRUE(context.close().ok());
  }
  EXPECT_EQ(quota.used(), 0U);
}
} // namespace
