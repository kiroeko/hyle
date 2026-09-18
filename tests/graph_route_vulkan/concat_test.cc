#include <gtest/gtest.h>

#include "../compute_route_vulkan/concat_test_support.h"
#include "hyle/graph/vulkan.h"

namespace {
using namespace vulkan_concat_test;
namespace g = hyle::graph;
namespace gv = g::vulkan;

TEST(VulkanGraph,
     FiveViewsReassembleWithoutIntermediateCopiesAndKeepOldResults) {
  auto context = vulkan_test::context().value();
  v::Tensor surviving;
  std::vector<std::uint64_t> saved;
  for (auto width : {1, 129, 2048}) {
    const int rows = width == 2048 ? 17 : 3;
    const auto desc = c::TensorDesc::strided_bytes(
                          c::DataType::kBFloat16,
                          std::array<std::int64_t, 2>{rows, width * 5},
                          std::array<std::int64_t, 2>{width * 10 + 1, 2})
                          .value();
    auto builder = g::GraphBuilder::create().value();
    const auto x = builder.input(desc).value();
    std::array<g::ValueId, 5> views;
    const std::array<int, 5> order{2, 0, 4, 1, 3};
    for (std::size_t port = 0; port < views.size(); ++port)
      views[port] =
          builder.add(c::View::slice(1, order[port] * width, width).value(), x)
              .value();
    const auto joined = builder
                            .add(c::Concat{-1}, views[0], views[1], views[2],
                                 views[3], views[4])
                            .value();
    ASSERT_TRUE(builder.publish(joined).ok());
    auto graph = std::move(builder).finish().value();
    gv::CompileReport report;
    auto program = gv::compile(context, graph, {}, {.report = &report});
    ASSERT_TRUE(program) << program.error().message();
    EXPECT_EQ(report.native_dispatch_count, 2U);
    EXPECT_EQ(report.invocation_storage_block_bytes.size(), 1U);
    graph = {};
    auto first = program->create_invocation().value(),
         second = program->create_invocation().value();
    v::Tensor retained;
    std::vector<std::uint64_t> previous;
    const auto parent_desc =
        c::TensorDesc::strided_bytes(
            c::DataType::kBFloat16,
            std::array<std::int64_t, 2>{rows + 1, width * 5},
            desc.byte_strides())
            .value();
    for (int phase = 0; phase < 3; ++phase) {
      std::vector<std::uint64_t> raw(*parent_desc.element_count());
      for (std::size_t i = 0; i < raw.size(); ++i)
        raw[i] = raw_value(i, phase, c::DataType::kBFloat16);
      const auto parent =
          context.upload(parent_desc, payload(parent_desc, raw)).value();
      const auto input =
          parent.view(c::View::slice(0, phase % 2, rows).value()).value();
      std::vector<std::vector<std::uint64_t>> parts(5);
      for (int row = 0; row < rows; ++row)
        for (std::size_t port = 0; port < parts.size(); ++port)
          for (int column = 0; column < width; ++column)
            parts[port].push_back(raw[(row + phase % 2) * width * 5 +
                                      order[port] * width + column]);
      const std::array<std::int64_t, 5> widths{width, width, width, width,
                                               width};
      auto &slot = phase == 0 ? first : second;
      ASSERT_TRUE(slot.bind(input).ok());
      auto work = slot.submit().value();
      auto result = work.wait().value();
      EXPECT_EQ(bits(context, result.get()->tensors()[0]),
                concatenate(parts, widths, rows, 1));
      if (phase == 0) {
        retained = result.get()->tensors()[0];
        previous = bits(context, retained);
      } else
        EXPECT_EQ(bits(context, retained), previous);
    }
    first = {};
    second = {};
    *program = {};
    surviving = retained;
    saved = previous;
  }
  EXPECT_TRUE(context.close().ok());
  EXPECT_EQ(bits(context, surviving), saved);
}
} // namespace
