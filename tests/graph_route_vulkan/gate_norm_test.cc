#include <gtest/gtest.h>

#include "../compute_route_vulkan/gate_norm_test_support.h"
#include "hyle/graph/vulkan.h"

namespace {
using namespace vulkan_gate_norm_test;
namespace g = hyle::graph;
namespace gv = g::vulkan;

TEST(VulkanGraph,
     GateNormResidualUsesSharedMixedParametersAndIndependentSlots) {
  auto context = vulkan_test::context().value();
  v::Tensor surviving;
  std::vector<std::uint64_t> surviving_bits;
  for (auto type : {c::DataType::kFloat16, c::DataType::kBFloat16}) {
    for (auto width : {129, 2048}) {
      const auto rows = width == 129 ? 3 : 17;
      const auto input_desc = dense(type, {rows, 2 * width});
      const auto desc = dense(type, {rows, width});
      const auto parameter = dense(c::DataType::kFloat32, {width});
      auto builder = g::GraphBuilder::create().value();
      const auto x = builder.input(input_desc).value(),
                 skip = builder.input(desc).value();
      const auto scale_id = builder.constant(parameter).value(),
                 bias_id = builder.constant(parameter).value();
      const auto gate = builder.add(c::GatedActivation{}, x).value();
      const auto normalized = builder.add(c::RmsNorm{}, gate, scale_id).value();
      const auto residual =
          builder.add(c::Binary{c::BinaryKind::kAdd}, normalized, skip).value();
      const auto end =
          builder.add(c::LayerNorm{}, residual, scale_id, bias_id).value();
      ASSERT_TRUE(builder.publish(normalized).ok());
      ASSERT_TRUE(builder.publish(end).ok());
      auto graph = std::move(builder).finish().value();
      std::vector<double> scale(width), bias(width);
      for (int i = 0; i < width; ++i) {
        scale[i] = static_cast<float>(1.001 + (i % 7) * 0.03125);
        bias[i] = static_cast<float>((i % 11 - 5) * 0.0625);
      }
      std::array constants{make(context, parameter, scale),
                           make(context, parameter, bias)};
      gv::CompileReport report;
      auto program =
          gv::compile(context, graph, constants, {.report = &report});
      ASSERT_TRUE(program) << program.error().message();
      EXPECT_EQ(report.native_dispatch_count, 4U);
      EXPECT_EQ(report.invocation_storage_block_bytes.size(), 3U);
      EXPECT_EQ(report.retained_constant_owner_count, 2U);
      graph = {};
      constants = {};
      auto first = program->create_invocation().value(),
           second = program->create_invocation().value();
      v::Tensor retained;
      std::vector<std::uint64_t> saved;
      for (int phase = 0; phase < 3; ++phase) {
        std::vector<double> data(rows * 2 * width), skip_data(rows * width),
            gated(rows * width);
        for (std::size_t i = 0; i < data.size(); ++i)
          data[i] =
              rounded((static_cast<int>((i + phase) % 31) - 15) * 0.125, type);
        for (std::size_t i = 0; i < skip_data.size(); ++i)
          skip_data[i] = (static_cast<int>(i % 13) - 6) * 0.0625;
        for (int row = 0; row < rows; ++row)
          for (int col = 0; col < width; ++col)
            gated[row * width + col] =
                rounded(activate(c::GatedActivationKind::kSilu,
                                 data[row * 2 * width + col]) *
                            data[row * 2 * width + width + col],
                        type);
        auto rms = norm(gated, width, false, 1e-5F, scale);
        for (auto &value : rms)
          value = rounded(value, type);
        std::vector<double> residual_data(rms.size());
        for (std::size_t i = 0; i < rms.size(); ++i)
          residual_data[i] = rounded(rms[i] + skip_data[i], type);
        const auto expected =
            norm(residual_data, width, true, 1e-5F, scale, bias);
        const auto input = make(context, input_desc, data),
                   bypass = make(context, desc, skip_data);
        auto &slot = phase == 0 ? first : second;
        ASSERT_TRUE(slot.bind(input, bypass).ok());
        auto work = slot.submit().value();
        auto result = work.wait().value();
        expect(context, result.get()->tensors()[0], rms);
        expect(context, result.get()->tensors()[1], expected);
        if (phase == 0) {
          retained = result.get()->tensors()[1];
          saved = bits(context, retained);
        } else
          EXPECT_EQ(bits(context, retained), saved);
      }
      first = {};
      second = {};
      program = {};
      EXPECT_EQ(bits(context, retained), saved);
      surviving = retained;
      surviving_bits = saved;
    }
  }
  EXPECT_TRUE(context.close().ok());
  EXPECT_EQ(bits(context, surviving), surviving_bits);
}
} // namespace
