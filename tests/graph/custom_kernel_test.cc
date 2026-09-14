#include <gtest/gtest.h>

#include <array>
#include <utility>

#include "../compute/custom_kernel_fixture.h"
#include "hyle/graph/executable.h"

namespace {
using namespace custom_test;

TEST(GraphCustomKernel, PublicSchemaAndOwningFactoryUseTheSamePreparedRuntime) {
  Probe probe;
  Tensors tensors;
  ASSERT_TRUE(tensors.fill().ok());
  {
    auto factory = provider(probe).value();
    auto context = Context::create(cpu::device(), factory).value();
    auto queue = context.create_queue().value();
    auto builder = hyle::graph::GraphBuilder::create().value();
    auto input = builder.input(description()).value();
    auto first = builder.add(Operation::custom(Affine{3}), input).value();
    auto output = builder.add(Operation::custom(Affine{1}), first).value();
    ASSERT_TRUE(builder.publish(output).ok());
    auto graph = std::move(builder).finish().value();
    auto compiled = hyle::graph::compile(graph, context);
    ASSERT_TRUE(compiled) << compiled.error().message();
    auto invocation = compiled->executable.create_invocation().value();
    ASSERT_TRUE(invocation.bind(input, tensors.in).ok());
    ASSERT_TRUE(invocation.bind(output, tensors.out).ok());
    factory = {};
    compiled->executable = {};
    const int prepares = probe.preparations, proposals = probe.proposals;
    for (int repeat = 0; repeat < 3; ++repeat) {
      auto done = invocation.submit(queue);
      ASSERT_TRUE(done);
      ASSERT_TRUE(done->wait().ok());
      EXPECT_EQ(tensors.read(), (std::array<float, 4>{11, 15, 19, 23}));
    }
    EXPECT_EQ(probe.preparations, prepares);
    EXPECT_EQ(probe.proposals, proposals);
    EXPECT_EQ(probe.executions, 6);
    EXPECT_EQ(probe.configurations, 1);
    context = {}; // admission closes; existing objects retain implementation
    EXPECT_EQ(probe.configurations, 1);
  }
  EXPECT_EQ(probe.configurations, 0);
  EXPECT_EQ(probe.kernels, 0);
  EXPECT_EQ(probe.invocations, 0);
}

} // namespace
