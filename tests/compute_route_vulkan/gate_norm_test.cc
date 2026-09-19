#include <gtest/gtest.h>

#include <algorithm>

#include "gate_norm_test_support.h"
#include "vulkan_internal.h"
#include "vulkan_program_internal.h"

namespace {
using namespace vulkan_gate_norm_test;

TEST(VulkanGateNorm, LayerNormUsesMatureRegionsThroughHighRankStorageAdapters) {
  auto context = vulkan_test::context().value();
  constexpr std::int64_t width = 129;
  constexpr std::size_t elements = 3 * width;
  for (auto type :
       {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16}) {
    const auto size = static_cast<std::int64_t>(c::data_type_size(type));
    const auto description =
        c::TensorDesc::strided_bytes(
            type, std::array<std::int64_t, 8>{1, 1, 1, 1, 1, 3, 1, width},
            std::array<std::int64_t, 8>{0, 0, 0, 0, 0, width * (size + 1) + 3,
                                        0, size + 1})
            .value();
    std::vector<double> data(elements), scale(width), bias(width);
    for (std::size_t i = 0; i < data.size(); ++i)
      data[i] = rounded((static_cast<int>(i % 23) - 11) * 0.125, type);
    for (std::size_t i = 0; i < scale.size(); ++i) {
      scale[i] = static_cast<float>(1.001 + (i % 7) * 0.0625);
      bias[i] = rounded((static_cast<int>(i % 9) - 4) * 0.03125, type);
    }
    for (std::size_t parameters = 0; parameters <= 2; ++parameters) {
      SCOPED_TRACE(::testing::Message()
                   << static_cast<int>(type) << ':' << parameters);
      std::vector<v::Tensor> tensors{make(context, description, data)};
      if (parameters > 0)
        tensors.push_back(
            make(context, dense(c::DataType::kFloat32, {width}), scale));
      if (parameters > 1)
        tensors.push_back(make(context, dense(type, {width}), bias));
      std::vector<v::Argument> arguments;
      for (const auto &tensor : tensors)
        arguments.emplace_back(tensor.description());
      auto program = v::prepare(context, c::Operation(c::LayerNorm{}),
                                std::span<const v::Argument>(arguments));
      ASSERT_TRUE(program) << program.error().message();
      const auto &dispatches = v::detail::Access::program(*program)->dispatches;
      EXPECT_EQ(std::ranges::count_if(dispatches,
                                      [](const auto &dispatch) {
                                        return dispatch.region != nullptr;
                                      }),
                1);
      auto slot = program->create_invocation().value();
      ASSERT_TRUE(slot.bind(tensors).ok());
      auto work = slot.submit().value();
      auto result = work.wait();
      ASSERT_TRUE(result) << result.error().message();
      expect(context, result->get()->tensors()[0],
             norm(data, width, true, 1e-5F,
                  parameters > 0 ? std::span<const double>(scale)
                                 : std::span<const double>{},
                  parameters > 1 ? std::span<const double>(bias)
                                 : std::span<const double>{}));
    }
  }
}

TEST(VulkanGateNorm, MatureLayerNormUsesNativeFallbackOutsideColdDeviceLimits) {
  auto context = vulkan_test::context().value();
  auto &limits = v::detail::Access::runtime(context)->device->properties.limits;
  const auto original = limits;
  for (bool grid : {false, true}) {
    if (grid)
      limits.maxComputeWorkGroupCount[0] = 2;
    else
      limits.maxComputeWorkGroupInvocations = 256;
    auto program = v::prepare(context, c::LayerNorm{},
                              dense(c::DataType::kFloat32, {3, 129}));
    limits = original;
    ASSERT_TRUE(program) << program.error().message();
    const auto &dispatches = v::detail::Access::program(*program)->dispatches;
    EXPECT_TRUE(std::ranges::all_of(dispatches, [](const auto &dispatch) {
      return dispatch.region == nullptr;
    }));
  }
}

TEST(VulkanGateNorm, ThreeGatesSupportBothLayoutsAndOddByteOrigins) {
  auto context = vulkan_test::context().value();
  for (auto type :
       {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16}) {
    const auto size = static_cast<std::int64_t>(c::data_type_size(type));
    for (auto width : {0, 1, 127, 128, 129, 2048}) {
      for (auto kind :
           {c::GatedActivationKind::kSilu, c::GatedActivationKind::kGelu,
            c::GatedActivationKind::kGeluTanh}) {
        for (auto form :
             {c::GatedInputKind::kSeparate, c::GatedInputKind::kPackedHalves}) {
          SCOPED_TRACE(::testing::Message()
                       << static_cast<int>(type) << ':' << width << ':'
                       << static_cast<int>(kind) << ':'
                       << static_cast<int>(form));
          const auto columns =
              form == c::GatedInputKind::kSeparate ? width : 2 * width;
          const auto desc = c::TensorDesc::strided_bytes(
                                type, std::array<std::int64_t, 2>{4, columns},
                                std::array<std::int64_t, 2>{
                                    columns * (size + 1) + 3, size + 1})
                                .value();
          std::vector<double> data(4 * columns), value(3 * width),
              expected(3 * width);
          for (std::size_t i = 0; i < data.size(); ++i)
            data[i] = rounded((static_cast<int>(i % 41) - 20) * 0.25, type);
          for (std::size_t i = 0; i < value.size(); ++i)
            value[i] = rounded((static_cast<int>(i % 13) - 6) * 0.125, type);
          const auto parent = make(context, desc, data);
          const auto input =
              parent.view(c::View::slice(0, 1, 3).value()).value();
          std::vector<v::Tensor> args{input};
          if (form == c::GatedInputKind::kSeparate)
            args.push_back(make(context, dense(type, {3, width}), value));
          for (int row = 0; row < 3; ++row)
            for (int column = 0; column < width; ++column) {
              const auto other =
                  form == c::GatedInputKind::kSeparate
                      ? value[row * width + column]
                      : data[(row + 1) * columns + width + column];
              expected[row * width + column] =
                  activate(kind, data[(row + 1) * columns + column]) * other;
            }
          auto output = execute(
              context, c::Operation(c::GatedActivation{kind, form}), args);
          ASSERT_TRUE(output) << output.error().message();
          expect(context, *output, expected);
        }
      }
    }
  }
  const double inf = std::numeric_limits<double>::infinity();
  const std::array<double, 6> special{-inf, inf, NAN, -100, -0.0, 0};
  const std::array<double, 6> factors{2, -3, 1, 2, 1, 1};
  for (auto type :
       {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16}) {
    const auto a = make(context, dense(type, {6}), special);
    const auto b = make(context, a.description(), factors);
    for (auto kind :
         {c::GatedActivationKind::kSilu, c::GatedActivationKind::kGelu,
          c::GatedActivationKind::kGeluTanh}) {
      // Captured aligned F32 SiLU qualifies the retained ggml GLU shell.
      auto program =
          v::prepare(context,
                     c::GatedActivation{kind, c::GatedInputKind::kSeparate}, a,
                     b)
              .value();
      auto slot = program.create_invocation().value();
      auto work = slot.submit().value();
      auto result = work.wait().value();
      std::array<double, 6> expected{};
      for (std::size_t i = 0; i < expected.size(); ++i)
        expected[i] = activate(kind, special[i]) * factors[i];
      expect(context, result.get()->tensors()[0], expected);
    }
  }
}

TEST(VulkanGateNorm, NormUsesFloatAccumulationAndIndependentParameterStorage) {
  auto context = vulkan_test::context().value();
  constexpr float epsilon = 1e-5F;
  for (auto type :
       {c::DataType::kFloat32, c::DataType::kFloat16, c::DataType::kBFloat16}) {
    const auto size = static_cast<std::int64_t>(c::data_type_size(type));
    for (auto width : {0, 1, 127, 128, 129, 257, 2048}) {
      const auto desc =
          c::TensorDesc::strided_bytes(
              type, std::array<std::int64_t, 2>{4, width},
              std::array<std::int64_t, 2>{width * (size + 1) + 3, size + 1})
              .value();
      std::vector<double> data(4 * width);
      for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = rounded((static_cast<int>(i % 37) - 18) * 0.125, type);
      const auto parent = make(context, desc, data);
      const auto input = parent.view(c::View::slice(0, 1, 3).value()).value();
      const auto host = std::span<const double>(data).subspan(width);
      for (bool layer : {false, true}) {
        for (int parameters = 0; parameters <= (layer ? 2 : 1); ++parameters) {
          for (int mixed = 0; mixed < (parameters == 2   ? 4
                                       : parameters == 1 ? 2
                                                         : 1);
               ++mixed) {
            SCOPED_TRACE(::testing::Message()
                         << static_cast<int>(type) << ':' << width << ':'
                         << layer << ':' << parameters << ':' << mixed);
            std::vector<v::Tensor> args{input};
            std::vector<double> scale, bias;
            for (int port = 0; port < parameters; ++port) {
              const auto parameter_type =
                  (mixed & (1 << port)) != 0 ? c::DataType::kFloat32 : type;
              const auto stride =
                  static_cast<std::int64_t>(c::data_type_size(parameter_type)) +
                  1;
              auto parameter_desc =
                  c::TensorDesc::strided_bytes(
                      parameter_type, std::array<std::int64_t, 1>{width + 1},
                      std::array<std::int64_t, 1>{stride})
                      .value();
              std::vector<double> values(width + 1);
              for (std::size_t i = 0; i < values.size(); ++i)
                values[i] =
                    rounded(port == 0 ? 1.001 + static_cast<int>(i % 7) * 0.0625
                                      : (static_cast<int>(i % 9) - 4) * 0.03125,
                            parameter_type);
              const auto tensor = make(context, parameter_desc, values);
              args.push_back(
                  tensor.view(c::View::slice(0, 1, width).value()).value());
              (port == 0 ? scale : bias)
                  .assign(values.begin() + 1, values.end());
            }
            const auto op = layer ? c::Operation(c::LayerNorm{epsilon})
                                  : c::Operation(c::RmsNorm{epsilon});
            auto output = execute(context, op, args);
            ASSERT_TRUE(output) << output.error().message();
            expect(context, *output,
                   norm(host, width, layer, epsilon, scale, bias));
          }
        }
      }
    }
  }
}

TEST(VulkanGateNorm, SpecialValuesStableVarianceRankEightAndPreciseAffine) {
  auto context = vulkan_test::context().value();
  std::vector<double> data(4 * 128, 0);
  data[128] = NAN;
  data[256] = INFINITY;
  for (std::size_t i = 384; i < data.size(); ++i)
    data[i] = 10000 + (i % 2 == 0 ? -0.25 : 0.25);
  const auto input =
      make(context, dense(c::DataType::kFloat32, {4, 128}), data);
  for (bool layer : {false, true}) {
    const auto op =
        layer ? c::Operation(c::LayerNorm{0}) : c::Operation(c::RmsNorm{0});
    auto output = execute(context, op, std::array{input});
    ASSERT_TRUE(output) << output.error().message();
    expect(context, *output, norm(data, 128, layer, 0));
  }
  for (auto type : {c::DataType::kFloat16, c::DataType::kBFloat16}) {
    const auto desc = dense(type, {1, 1, 1, 1, 1, 1, 1, 2});
    const auto x = make(context, desc, std::array<double, 2>{-1, 1});
    const auto scale = make(context, dense(c::DataType::kFloat32, {2}),
                            std::array<double, 2>{1.0001F, 1.0001F});
    const auto bias =
        make(context, scale.description(), std::array<double, 2>{1, -1});
    auto output = execute(context, c::Operation(c::LayerNorm{0}),
                          std::array{x, scale, bias});
    ASSERT_TRUE(output) << output.error().message();
    const std::array<double, 2> expected{1.0F - 1.0001F, 1.0001F - 1.0F};
    expect(context, *output, expected, true);
    const auto one = make(context, dense(type, {1}), std::array<double, 1>{2});
    auto broadcast =
        one.view(c::View::broadcast_to(
                     c::TensorShape::create(
                         std::array<std::int64_t, 8>{1, 1, 1, 1, 1, 2, 3, 129})
                         .value())
                     .value())
            .value();
    auto normalized =
        execute(context, c::Operation(c::RmsNorm{0}), std::array{broadcast});
    ASSERT_TRUE(normalized) << normalized.error().message();
    expect(context, *normalized, std::vector<double>(6 * 129, 1), true);
    auto packed =
        one.view(c::View::broadcast_to(
                     c::TensorShape::create(
                         std::array<std::int64_t, 8>{1, 1, 1, 1, 1, 2, 3, 258})
                         .value())
                     .value())
            .value();
    auto gated = execute(context, c::Operation(c::GatedActivation{}),
                         std::array{packed});
    ASSERT_TRUE(gated) << gated.error().message();
    expect(context, *gated,
           std::vector<double>(6 * 129,
                               activate(c::GatedActivationKind::kSilu, 2) * 2));
  }
  EXPECT_FALSE(v::prepare(
      context, c::RmsNorm{.accumulation_type = c::DataType::kFloat64},
      input.description()));
  EXPECT_FALSE(v::prepare(context, c::LayerNorm{-1}, input.description()));
  EXPECT_FALSE(v::prepare(context, c::RmsNorm{}, input.description(),
                          dense(c::DataType::kFloat32, {127})));
}

TEST(VulkanGateNorm, OddHalfRowsPreserveNeighborsInFixedRecorder) {
  auto context = vulkan_test::context().value();
  const auto alignment =
      std::max(VkDeviceSize{4},
               v::detail::Access::runtime(context)
                   ->device->properties.limits.minStorageBufferOffsetAlignment);
  for (auto type : {c::DataType::kFloat16, c::DataType::kBFloat16}) {
    for (auto rows : {1, 2, 3}) {
      const auto desc = dense(type, {rows, 129});
      std::vector<double> data(rows * 129);
      for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = (static_cast<int>(i % 23) - 11) * 0.25;
      const auto input = make(context, desc, data);
      const auto capacity = (*desc.required_bytes() + 3) & ~std::size_t{3};
      std::vector<std::byte> guards(alignment + capacity + 8, std::byte{0x5a});
      const auto backing =
          context
              .upload(dense(c::DataType::kBool,
                            {static_cast<std::int64_t>(guards.size())}),
                      guards)
              .value();
      const auto output =
          context
              .borrow_ready(desc, backing.native_buffer(), alignment, capacity,
                            backing.native_owner())
              .value();
      const auto program = v::prepare(context, c::LayerNorm{}, desc).value();
      auto recorder = v::Recorder::create(context).value();
      ASSERT_TRUE(
          recorder.append(program, std::array{input}, std::array{output}).ok());
      auto recording = recorder.finish().value();
      auto work = recording.submit().value();
      ASSERT_TRUE(work.wait().ok());
      expect(context, output, norm(data, 129, true, 1e-5F));
      const auto actual = bytes(context, backing);
      for (std::size_t i = 0; i < actual.size(); ++i)
        if (i < alignment || i >= alignment + capacity)
          EXPECT_EQ(actual[i], guards[i]) << i;
    }
  }
}
} // namespace
