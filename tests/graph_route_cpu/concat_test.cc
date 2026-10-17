#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "hyle/compute/backends/cpu.h"
#include "hyle/compute/providers/ggml_cpu.h"
#include "hyle/graph/cpu.h"
#include "hyle/graph/executable.h"

namespace {
#if defined(HYLE_TEST_GGML_CPU)
TEST(CpuGgmlConcatGraph, RepeatedPortsKeepPublishedIntermediateAndDownstream) {
  namespace c = hyle::compute;
  namespace g = hyle::graph;
  auto context = g::cpu::create_context({.native_threads = 1}).value();
  auto queue = context.create_queue().value();
  const auto input_desc =
      c::TensorDesc::dense(c::DataType::kFloat32,
                           std::array<std::int64_t, 3>{2, 1, 7})
          .value();
  const auto output_desc =
      c::TensorDesc::dense(c::DataType::kFloat32,
                           std::array<std::int64_t, 3>{2, 70, 7})
          .value();
  auto buffer = [](const c::TensorDesc &description) {
    return hyle::mem::Buffer::create(hyle::mem::system_resource(),
                                     description.required_bytes().value(), 64)
        .value();
  };
  auto input = buffer(input_desc), joined = buffer(output_desc),
       output = buffer(output_desc);
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(input_desc).value();
  std::array<g::ValueId, 70> ports;
  ports.fill(x);
  const auto concat_node =
      builder.add_node(c::Operation(c::Concat{.axis = 1}), ports).value();
  const auto concat = builder.result(concat_node).value();
  const auto negative =
      builder.add(c::Unary{c::UnaryKind::kNeg}, concat).value();
  ASSERT_TRUE(builder.publish(concat).ok());
  ASSERT_TRUE(builder.publish(negative).ok());
  auto logical = std::move(builder).finish().value();
  auto compiled = g::compile(logical, context);
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 1U);
  EXPECT_EQ(compiled->report.step(0)->provider.id, c::ggml_cpu::kProviderId);
  auto invocation = compiled->executable.create_invocation();
  ASSERT_TRUE(invocation) << invocation.error().message();
  ASSERT_TRUE(
      invocation
          ->bind(x,
                 c::ConstTensorView::create(input_desc, input.view()).value())
          .ok());
  ASSERT_TRUE(
      invocation
          ->bind(concat,
                 c::TensorView::create(output_desc, joined.view()).value())
          .ok());
  ASSERT_TRUE(
      invocation
          ->bind(negative,
                 c::TensorView::create(output_desc, output.view()).value())
          .ok());
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  ASSERT_TRUE(invocation->prepare_replay(queues).ok());
  for (int replay = 0; replay < 2; ++replay) {
    std::array<float, 14> source;
    for (std::size_t i = 0; i < source.size(); ++i)
      source[i] = float(int(i) - 5 + replay * 17);
    ASSERT_TRUE(input.view().write(std::as_bytes(std::span(source))).ok());
    auto done = invocation->submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    std::vector<float> actual(980), negated(980);
    ASSERT_TRUE(
        joined.view().read(std::as_writable_bytes(std::span(actual))).ok());
    ASSERT_TRUE(
        output.view().read(std::as_writable_bytes(std::span(negated))).ok());
    for (std::size_t i = 0; i < actual.size(); ++i) {
      const auto reference = source[(i / 490) * 7 + i % 7];
      EXPECT_EQ(actual[i], reference);
      EXPECT_EQ(negated[i], -reference);
    }
  }
}
#endif
} // namespace
