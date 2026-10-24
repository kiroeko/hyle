#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "hyle/compute/low_precision.h"
#include "hyle/compute/model_ops.h"
#include "hyle/graph/vulkan.h"
#include "tensor_test_support.h"
#include "vulkan_program_internal.h"
#include "../compute_models/gguf_nvfp4_fixture.h"

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace g = hyle::graph;
namespace gv = g::vulkan;
namespace t = vulkan_tensor_test;
using Weights = nvfp4_test::Fixture;

hyle::Result<v::Program> prepare(const v::Context &context, c::Operation operation,
                               std::span<const c::TensorDesc> descriptions,
                               std::size_t constant_port, const v::Tensor &weight,
                               bool graph_route) {
  if (!graph_route) {
    std::vector<v::Argument> arguments;
    for (std::size_t port = 0; port < descriptions.size(); ++port)
      if (port == constant_port) arguments.emplace_back(weight);
      else arguments.emplace_back(descriptions[port]);
    return v::prepare(context, operation, std::span<const v::Argument>(arguments));
  }
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  std::vector<g::ValueId> inputs;
  for (std::size_t port = 0; port < descriptions.size(); ++port) {
    HYLE_ASSIGN_OR_RETURN(auto input, port == constant_port
        ? builder.constant(descriptions[port]) : builder.input(descriptions[port]));
    inputs.push_back(input);
  }
  HYLE_ASSIGN_OR_RETURN(auto node, builder.add_node(operation, inputs));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.result(node));
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  gv::CompileReport report;
  auto result = gv::compile(context, graph, std::array{weight}, {.report = &report});
  if (result) EXPECT_EQ(report.managed_region_count, 1U);
  return result;
}

void expect_native(const v::Program &program) {
  const auto &dispatches = v::detail::Access::program(program)->dispatches;
  EXPECT_EQ(std::ranges::count_if(dispatches, [](const auto &dispatch) {
    return static_cast<bool>(dispatch.region);
  }), 1);
  EXPECT_FALSE(std::ranges::any_of(dispatches, [](const auto &dispatch) {
    return dispatch.iree.has_value();
  }));
}

std::vector<double> activation(const Weights &weights, std::int64_t rows, int pass) {
  const auto original = weights.activations(rows);
  const double scale = pass == 0 ? 1.0 : pass == 1 ? -1.0 : 2.0;
  std::vector<double> result;
  for (auto value : original) result.push_back(value * scale);
  return result;
}

std::vector<double> project(const Weights &weights, std::span<const double> input,
                            std::span<const std::int32_t> experts = {}) {
  const auto rows = input.size() / static_cast<std::size_t>(weights.columns);
  std::vector<double> result(rows * static_cast<std::size_t>(weights.rows));
  for (std::size_t row = 0; row < rows; ++row)
    for (std::int64_t column = 0; column < weights.rows; ++column) {
      const auto expert = experts.empty() ? 0 : experts[row];
      double sum = 0;
      for (std::int64_t k = 0; k < weights.columns; ++k)
        sum += input[row * static_cast<std::size_t>(weights.columns) + k] *
            weights.decoded[static_cast<std::size_t>((expert * weights.rows + column) * weights.columns + k)];
      result[row * static_cast<std::size_t>(weights.rows) + column] = sum;
    }
  return result;
}

std::vector<double> rows(const Weights &weights, std::span<const std::int32_t> ids) {
  std::vector<double> result;
  for (auto id : ids)
    for (std::int64_t k = 0; k < weights.columns; ++k)
      result.push_back(weights.decoded[static_cast<std::size_t>(id * weights.columns + k)]);
  return result;
}

// Every 32-value activation block has one exact +/-127 (or its exact scaled
// value). With this fixture all four active products and their sums are exact
// F32 dyadics, including the original Q8 activation conversion if selected.
// This bit-exact oracle is deliberately limited to these inputs.
void exercise(const v::Context &context, const v::Program &program,
              const std::array<std::vector<v::Tensor>, 3> &inputs,
              const std::array<std::vector<double>, 3> &expected,
              v::Tensor *keep = nullptr) {
  auto first = program.create_invocation().value();
  auto second = program.create_invocation().value();
  ASSERT_TRUE(first.bind(inputs[0]).ok());
  ASSERT_TRUE(second.bind(inputs[1]).ok());
  v::Tensor retained;
  {
    auto a = first.submit(), b = second.submit();
    ASSERT_TRUE(a) << a.error().message();
    ASSERT_TRUE(b) << b.error().message();
    auto ar = a->wait(), br = b->wait();
    ASSERT_TRUE(ar) << ar.error().message();
    ASSERT_TRUE(br) << br.error().message();
    retained = ar->get()->tensors()[0];
    t::expect(context, retained, expected[0], true);
    t::expect(context, br->get()->tensors()[0], expected[1], true);
  }
  EXPECT_FALSE(first.submit());
  ASSERT_TRUE(second.bind(inputs[2]).ok());
  auto replay = second.submit();
  ASSERT_TRUE(replay) << replay.error().message();
  auto result = replay->wait();
  ASSERT_TRUE(result) << result.error().message();
  t::expect(context, result->get()->tensors()[0], expected[2], true);
  t::expect(context, retained, expected[0], true);
  if (keep) *keep = retained;
}

TEST(VulkanNvfp4, DirectAndGraphMatMulKeepPackedSubgroupsAndIndependentSlots) {
  auto context = vulkan_test::context().value();
  const Weights weights;
  for (auto type : {c::DataType::kFloat32, c::DataType::kFloat16})
    for (auto tokens : {1, 9})
      for (bool graph_route : {false, true}) {
        SCOPED_TRACE(::testing::Message() << int(type) << '/' << tokens << '/' << graph_route);
        const auto wd = c::gguf_weight_description(c::GgufQuantizedType::kNvfp4,
            type, weights.columns, weights.rows).value();
        const auto ad = t::dense(type, {tokens, weights.columns});
        auto packed = context.upload(wd, std::as_bytes(std::span(weights.bytes))).value();
        auto prepared = prepare(context, c::Operation(c::MatMul{}), std::array{ad, wd}, 1, packed, graph_route);
        ASSERT_TRUE(prepared) << prepared.error().message();
        expect_native(*prepared);
        std::array<std::vector<v::Tensor>, 3> inputs;
        std::array<std::vector<double>, 3> expected;
        for (int pass = 0; pass < 3; ++pass) {
          const auto values = activation(weights, tokens, pass);
          inputs[pass].push_back(t::make(context, ad, values));
          expected[pass] = project(weights, values);
        }
        exercise(context, *prepared, inputs, expected);
        const auto bytes = std::as_bytes(std::span(weights.bytes));
        EXPECT_EQ(t::bytes(context, packed), std::vector<std::byte>(bytes.begin(), bytes.end()));
      }
}

TEST(VulkanNvfp4, DirectAndGraphGroupedExpertsKeepRepeatedAndReboundIds) {
  auto context = vulkan_test::context().value();
  const Weights weights(5, 128, 3);
  for (auto tokens : {1, 9})
    for (bool graph_route : {false, true}) {
      SCOPED_TRACE(::testing::Message() << tokens << '/' << graph_route);
      constexpr std::int64_t slots = 2;
      const auto wd = c::gguf_grouped_weight_description(c::GgufQuantizedType::kNvfp4,
          c::DataType::kFloat32, weights.experts, weights.rows, weights.columns).value();
      const auto ad = t::dense(c::DataType::kFloat32, {tokens, slots, weights.columns});
      const auto id = t::dense(c::DataType::kInt32, {tokens, slots});
      auto packed = context.upload(wd, std::as_bytes(std::span(weights.bytes))).value();
      auto prepared = prepare(context, c::model_ops::grouped_matmul(),
          std::array{wd, ad, id}, 0, packed, graph_route);
      ASSERT_TRUE(prepared) << prepared.error().message();
      expect_native(*prepared);
      std::array<std::vector<v::Tensor>, 3> inputs;
      std::array<std::vector<double>, 3> expected;
      for (int pass = 0; pass < 3; ++pass) {
        std::vector<std::int32_t> ids(static_cast<std::size_t>(tokens * slots));
        for (std::size_t i = 0; i < ids.size(); ++i)
          ids[i] = static_cast<std::int32_t>((i / 2 + pass) % 3);
        const auto values = activation(weights, tokens * slots, pass);
        inputs[pass] = {t::make(context, ad, values),
            context.upload(id, std::as_bytes(std::span(ids))).value()};
        expected[pass] = project(weights, values, ids);
      }
      exercise(context, *prepared, inputs, expected);
    }
}

TEST(VulkanNvfp4, DirectAndGraphGetRowsCheckPublicationAndRecoverAfterInvalidIds) {
  auto context = vulkan_test::context().value();
  const Weights weights;
  const auto stored = c::gguf_weight_description(c::GgufQuantizedType::kNvfp4,
      c::DataType::kFloat32, weights.columns, weights.rows).value();
  const auto table_desc = stored.transposed().value();
  const auto ids_desc = t::dense(c::DataType::kInt32, {3});
  const auto table = context.upload(table_desc, std::as_bytes(std::span(weights.bytes))).value();
  for (bool graph_route : {false, true}) {
    SCOPED_TRACE(graph_route);
    auto prepared = prepare(context, c::Operation(c::Gather{}), std::array{table_desc, ids_desc},
        0, table, graph_route);
    ASSERT_TRUE(prepared) << prepared.error().message();
    expect_native(*prepared);
    const std::array<std::array<std::int32_t, 3>, 3> selections{{{4, 1, 4}, {0, 3, 1}, {2, 2, 0}}};
    std::array<std::vector<v::Tensor>, 3> inputs;
    std::array<std::vector<double>, 3> expected;
    for (std::size_t pass = 0; pass < selections.size(); ++pass) {
      inputs[pass].push_back(context.upload(ids_desc,
          std::as_bytes(std::span(selections[pass]))).value());
      expected[pass] = rows(weights, selections[pass]);
    }
    v::Tensor retained;
    exercise(context, *prepared, inputs, expected, &retained);
    ASSERT_TRUE(retained.valid());
    const auto before = t::bytes(context, retained);
    for (const auto invalid : {std::array<std::int32_t, 3>{0, -1, 1}, {0, 5, 1}}) {
      auto slot = prepared->create_invocation().value();
      auto ids = context.upload(ids_desc, std::as_bytes(std::span(invalid))).value();
      ASSERT_TRUE(slot.bind(ids).ok());
      auto work = slot.submit();
      ASSERT_TRUE(work) << work.error().message();
      auto failure = work->wait();
      ASSERT_FALSE(failure);
      EXPECT_TRUE(failure.error().IsOutOfRange());
      EXPECT_EQ(t::bytes(context, retained), before);
    }
    auto recovered = prepared->create_invocation().value();
    ASSERT_TRUE(recovered.bind(inputs[0]).ok());
    auto work = recovered.submit();
    ASSERT_TRUE(work) << work.error().message();
    auto result = work->wait();
    ASSERT_TRUE(result) << result.error().message();
    t::expect(context, result->get()->tensors()[0], expected[0], true);
    EXPECT_EQ(t::bytes(context, retained), before);
  }
  EXPECT_FALSE(c::gguf_weight_description(c::GgufQuantizedType::kNvfp4,
      c::DataType::kFloat32, 65, 5));
  EXPECT_FALSE(v::prepare(context, c::Unary{.kind = c::UnaryKind::kAbs}, table_desc));
}
} // namespace
