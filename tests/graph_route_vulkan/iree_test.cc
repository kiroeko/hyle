#include "../compute_route_vulkan/iree_test_support.h"
#include "../compute_route_vulkan/scan_test_support.h"
#include "hyle/graph/vulkan.h"

namespace {
using namespace vulkan_iree_test;
namespace g = hyle::graph;
namespace gv = g::vulkan;
namespace t = vulkan_tensor_test;
namespace s = vulkan_scan_test;

TEST(VulkanGraph, OriginalFloat8RegionContinuesIntoGgmlAndNativeCopy) {
  auto context = vulkan_test::context().value();
  for (auto mode : {c::Float8ScaleMode::kFloat32Block128,
                    c::Float8ScaleMode::kUnsignedExponent8Group32}) {
    Float8Inputs data(3, 128, 128, mode, 1);
    auto builder = g::GraphBuilder::create().value();
    const auto input = builder.input(data.activation).value();
    const auto weight = builder.constant(data.weight).value();
    const auto middle = builder.add(c::MatMul{}, input, weight).value();
    const auto absolute =
        builder.add(c::Unary{c::UnaryKind::kAbs}, middle).value();
    const auto output = builder.add(c::Copy{}, absolute).value();
    const auto low =
        builder.add(c::Cast{c::DataType::kBFloat16}, middle).value();
    ASSERT_TRUE(builder.publish(output).ok());
    ASSERT_TRUE(builder.publish(middle).ok());
    ASSERT_TRUE(builder.publish(low).ok());
    auto graph = std::move(builder).finish().value();
    gv::CompileReport report;
    const auto resident = context.upload(data.weight, data.w).value();
    auto program =
        gv::compile(context, graph, std::array{resident}, {.report = &report});
    ASSERT_TRUE(program) << program.error().message();
    EXPECT_GE(report.managed_region_count, 2U);
    EXPECT_NE(report.numerical_policy.find("FP8"), std::string::npos);
    auto slot = program->create_invocation().value();
    ASSERT_TRUE(
        slot.bind(context.upload(data.activation, data.a).value()).ok());
    const auto expected = data.multiply();
    auto positive = expected;
    for (auto &value : positive)
      value = std::abs(value);
    graph = {};
    *program = {};
    for (int repeat = 0; repeat < 3; ++repeat) {
      auto work = slot.submit().value();
      auto result = work.wait();
      ASSERT_TRUE(result) << result.error().message();
      t::expect(context, result->get()->tensors()[0], positive, true);
      t::expect(context, result->get()->tensors()[1], expected, true);
      t::expect(context, result->get()->tensors()[2], expected, true);
    }
    EXPECT_EQ(t::bytes(context, resident), data.w);
  }
}

TEST(VulkanGraph, CheckedScanChainsTwoOriginalProgramsWithIndependentResults) {
  auto context = vulkan_test::context().value();
  const auto desc = t::dense(c::DataType::kInt32, {2, 257});
  auto builder = g::GraphBuilder::create().value();
  const auto input = builder.input(desc).value();
  const auto middle = builder.add(c::Scan{}, input).value();
  const auto output = builder.add(c::Scan{.exclusive = true}, middle).value();
  ASSERT_TRUE(builder.publish(output).ok());
  ASSERT_TRUE(builder.publish(middle).ok());
  const auto graph = std::move(builder).finish().value();
  gv::CompileReport report;
  auto program = gv::compile(context, graph, {}, {.report = &report}).value();
  EXPECT_EQ(report.managed_region_count, 2U);
  auto first = program.create_invocation().value();
  auto second = program.create_invocation().value();
  v::Tensor retained;
  std::vector<std::int32_t> saved;
  for (int phase = 0; phase < 3; ++phase) {
    const auto data = s::integer_pattern(514, phase);
    auto prefix = data;
    for (std::size_t row = 0; row < 2; ++row) {
      std::int64_t sum = 0;
      for (std::size_t col = 0; col < 257; ++col) {
        sum += data[row * 257 + col];
        prefix[row * 257 + col] = static_cast<std::int32_t>(sum);
      }
    }
    auto &slot = phase == 0 ? first : second;
    ASSERT_TRUE(slot.bind(s::make_integer(context, desc, data)).ok());
    auto work = slot.submit().value();
    auto result = work.wait();
    ASSERT_TRUE(result) << result.error().message();
    s::expect_integer_prefix(context, result->get()->tensors()[0], desc, prefix,
                             c::Scan{.exclusive = true});
    EXPECT_EQ(s::integer_values(context, result->get()->tensors()[1]), prefix);
    if (phase == 0) {
      retained = result->get()->tensors()[0];
      saved = s::integer_values(context, retained);
    } else {
      EXPECT_EQ(s::integer_values(context, retained), saved);
    }
  }
  EXPECT_TRUE(context.close().ok());
  EXPECT_EQ(s::integer_values(context, retained), saved);
}
} // namespace
