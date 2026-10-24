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
#include "../compute_models/grouped_shared_activation_fixture.h"

namespace {
namespace c = hyle::compute;
namespace v = c::vulkan;
namespace g = hyle::graph;
namespace gv = g::vulkan;
namespace t = vulkan_tensor_test;
using Fixture = grouped_shared_test::Fixture;

enum class Layout {
  kDenseShared, kBroadcast, kPaddedBroadcast, kOddPaddedBroadcast,
  kSingletonBroadcast, kIndependent
};

bool shared(Layout layout) { return layout != Layout::kIndependent; }
bool broadcast(Layout layout) {
  return layout == Layout::kBroadcast || layout == Layout::kPaddedBroadcast ||
         layout == Layout::kOddPaddedBroadcast || layout == Layout::kSingletonBroadcast;
}

std::vector<double> doubles(std::span<const float> values) {
  return {values.begin(), values.end()};
}

struct Activation final {
  v::Tensor source;
  v::Tensor slots;
};

Activation activation(const v::Context &context, const Fixture &fixture,
                      Layout layout, std::span<const float> values) {
  if (layout == Layout::kSingletonBroadcast) {
    auto scalar_rows = t::make(context,
        t::dense(c::DataType::kFloat32, {fixture.columns}), doubles(values));
    auto source = scalar_rows.view(c::View::broadcast_to(
        c::TensorShape::create(std::array<std::int64_t, 3>{
            1, 1, fixture.columns}).value()).value()).value();
    EXPECT_EQ(source.description().byte_strides()[0], 0);
    EXPECT_EQ(source.description().byte_strides()[1], 0);
    return {source, source};
  }
  const auto a = shared(layout) ? 1 : fixture.slots;
  const auto padding = layout == Layout::kPaddedBroadcast ? 16 :
                       layout == Layout::kOddPaddedBroadcast ? 1 : 0;
  const auto root_desc = t::dense(c::DataType::kFloat32,
      {fixture.tokens, a, fixture.columns + padding});
  std::vector<double> physical(static_cast<std::size_t>(
      fixture.tokens * a * (fixture.columns + padding)), 999.0);
  for (std::int64_t row = 0; row < fixture.tokens * a; ++row)
    std::copy_n(values.begin() + row * fixture.columns, fixture.columns,
        physical.begin() + row * (fixture.columns + padding));
  auto source = t::make(context, root_desc, physical);
  if (padding != 0)
    source = source.view(c::View::slice(2, 0, fixture.columns).value()).value();
  auto slots = source;
  if (broadcast(layout)) {
    const auto shape = c::TensorShape::create(
        std::array{fixture.tokens, fixture.slots, fixture.columns}).value();
    slots = source.view(c::View::broadcast_to(shape).value()).value();
    EXPECT_EQ(slots.description().byte_strides()[1], 0);
    EXPECT_EQ(slots.native_buffer(), source.native_buffer());
    EXPECT_EQ(slots.native_offset(), source.native_offset());
  }
  return {std::move(source), std::move(slots)};
}

hyle::Result<v::Program> prepare(const v::Context &context,
    const v::Tensor &weights, const Activation &input,
    const c::TensorDesc &ids, Layout layout, bool graph_route) {
  const auto operation = c::model_ops::grouped_matmul();
  if (!graph_route) {
    const std::array<v::Argument, 3> arguments{
        weights, input.slots.description(), ids};
    return v::prepare(context, operation, arguments);
  }
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto w, builder.constant(weights.description()));
  HYLE_ASSIGN_OR_RETURN(auto a, builder.input(input.source.description()));
  HYLE_ASSIGN_OR_RETURN(auto i, builder.input(ids));
  if (broadcast(layout)) {
    HYLE_ASSIGN_OR_RETURN(auto shape,
        c::TensorShape::create(input.slots.description().dimensions()));
    HYLE_ASSIGN_OR_RETURN(auto view, c::View::broadcast_to(shape));
    HYLE_ASSIGN_OR_RETURN(auto node,
        builder.add_node(c::Operation(view), std::array{a}));
    HYLE_ASSIGN_OR_RETURN(a, builder.result(node));
  }
  HYLE_ASSIGN_OR_RETURN(auto node,
      builder.add_node(operation, std::array{w, a, i}));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.result(node));
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  gv::CompileReport report;
  auto program = gv::compile(context, graph, std::array{weights},
                             {.report = &report});
  if (program) EXPECT_EQ(report.managed_region_count, 1U);
  return program;
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

// Logical expected values come from the shared independent fixture. Its sparse
// +/-127 activation and small dyadic weights make these sums exact, including
// the native Q8 activation conversion, without asserting general bit parity.
void exercise(const v::Context &context, const Fixture &fixture,
              const v::Tensor &weights, Layout layout, bool graph_route) {
  const auto ids_desc = t::dense(c::DataType::kInt32,
                               {fixture.tokens, fixture.slots});
  std::array<std::array<v::Tensor, 2>, 3> inputs;
  std::array<std::vector<float>, 3> expected;
  Activation first;
  for (std::size_t round = 0; round < inputs.size(); ++round) {
    const auto values = fixture.activations(shared(layout), static_cast<int>(round));
    const auto ids = fixture.ids(round == 1, static_cast<int>(round));
    auto a = activation(context, fixture, layout, values);
    if (round == 0) first = a;
    inputs[round] = {graph_route ? a.source : a.slots,
        context.upload(ids_desc, std::as_bytes(std::span(ids))).value()};
    expected[round] = fixture.expected(shared(layout), values, ids);
  }
  auto program = prepare(context, weights, first, ids_desc, layout, graph_route);
  ASSERT_TRUE(program) << program.error().message();
  expect_native(*program);
  const auto output_desc = t::dense(c::DataType::kFloat32,
      {fixture.tokens, fixture.slots, fixture.rows});
  auto a = program->create_invocation().value();
  auto b = program->create_invocation().value();
  ASSERT_TRUE(a.bind(inputs[0]).ok());
  ASSERT_TRUE(b.bind(inputs[1]).ok());
  v::Tensor retained;
  {
    auto aw = a.submit(), bw = b.submit();
    ASSERT_TRUE(aw) << aw.error().message();
    ASSERT_TRUE(bw) << bw.error().message();
    auto ar = aw->wait(), br = bw->wait();
    ASSERT_TRUE(ar) << ar.error().message();
    ASSERT_TRUE(br) << br.error().message();
    retained = ar->get()->tensors()[0];
    EXPECT_EQ(retained.description(), output_desc);
    t::expect(context, retained, doubles(expected[0]), true);
    t::expect(context, br->get()->tensors()[0], doubles(expected[1]), true);
  }
  ASSERT_TRUE(b.bind(inputs[2]).ok());
  auto work = b.submit();
  ASSERT_TRUE(work) << work.error().message();
  auto result = work->wait();
  ASSERT_TRUE(result) << result.error().message();
  t::expect(context, result->get()->tensors()[0], doubles(expected[2]), true);
  t::expect(context, retained, doubles(expected[0]), true);
}

TEST(VulkanGroupedShared, DenseAndBroadcastViewsKeepSlotsAndFreshBindings) {
  auto context = vulkan_test::context().value();
  struct Case {
    std::int64_t tokens, slots;
    Layout layout;
    c::DataType weights;
    std::int64_t rows{5};
  };
  const std::array cases{
      Case{1, 3, Layout::kDenseShared, c::DataType::kFloat32},
      Case{9, 2, Layout::kBroadcast, c::DataType::kBFloat16},
      Case{32, 3, Layout::kPaddedBroadcast, c::DataType::kFloat32},
      // N>8 reaches the matrix aligned-pipeline selector. The real 260-byte
      // token pitch must use its existing unaligned variant, without copying.
      Case{32, 3, Layout::kOddPaddedBroadcast, c::DataType::kFloat32, 9},
      // T<=8 takes the indexed vector path with the same non-vector pitch.
      Case{3, 2, Layout::kOddPaddedBroadcast, c::DataType::kFloat32},
      Case{1, 1, Layout::kSingletonBroadcast, c::DataType::kFloat32},
      Case{9, 2, Layout::kIndependent, c::DataType::kFloat32}};
  for (const auto &item : cases)
    for (bool graph_route : {false, true}) {
      SCOPED_TRACE(::testing::Message() << item.tokens << '/' << item.slots
          << '/' << int(item.layout) << '/' << graph_route);
      const Fixture fixture(item.tokens, item.slots, 64, item.rows);
      auto weights = t::make(context, t::dense(item.weights,
          {fixture.experts, fixture.rows, fixture.columns}), doubles(fixture.weights));
      exercise(context, fixture, weights, item.layout, graph_route);
    }
}

TEST(VulkanGroupedShared, PackedWeightsKeepRepeatedIdsForSharedActivation) {
  auto context = vulkan_test::context().value();
  const nvfp4_test::Fixture packed(5, 128, 3);
  Fixture fixture(9, 2, packed.columns, packed.rows, packed.experts);
  fixture.weights = packed.decoded;
  const auto description = c::gguf_grouped_weight_description(
      c::GgufQuantizedType::kNvfp4, c::DataType::kFloat32,
      packed.experts, packed.rows, packed.columns).value();
  const auto bytes = std::as_bytes(std::span(packed.bytes));
  auto weights = context.upload(description, bytes).value();
  for (auto layout : {Layout::kDenseShared, Layout::kBroadcast})
    for (bool graph_route : {false, true}) {
      SCOPED_TRACE(::testing::Message() << int(layout) << '/' << graph_route);
      exercise(context, fixture, weights, layout, graph_route);
    }
  EXPECT_EQ(t::bytes(context, weights),
            std::vector<std::byte>(bytes.begin(), bytes.end()));
}

TEST(VulkanGroupedShared, CheckedIdsFailAtCompletionAndOtherStridesDecline) {
  auto context = vulkan_test::context().value();
  const Fixture fixture(9, 2);
  auto weights = t::make(context, t::dense(c::DataType::kFloat32,
      {fixture.experts, fixture.rows, fixture.columns}), doubles(fixture.weights));
  const auto values = fixture.activations(true);
  const auto input = activation(context, fixture, Layout::kBroadcast, values);
  const auto ids_desc = t::dense(c::DataType::kInt32,
                               {fixture.tokens, fixture.slots});
  const auto valid = fixture.ids(false);
  // Strided direct imports cannot encode a scalar offset in the indexed native
  // descriptor. The actual origin remains a per-bind check, even when the
  // same program accepts a separately allocated aligned input.
  auto offset_source = t::make(context, t::dense(c::DataType::kFloat32,
      {fixture.tokens, 1, fixture.columns + 1}),
      std::vector<double>(static_cast<std::size_t>(
          fixture.tokens * (fixture.columns + 1)), 1));
  offset_source = offset_source.view(
      c::View::slice(2, 1, fixture.columns).value()).value();
  const Activation offset_input{offset_source, offset_source.view(
      c::View::broadcast_to(c::TensorShape::create(
          std::array{fixture.tokens, fixture.slots, fixture.columns}).value()).value()).value()};
  auto offset_program = prepare(context, weights, offset_input, ids_desc,
                                Layout::kBroadcast, false);
  ASSERT_TRUE(offset_program) << offset_program.error().message();
  auto offset_slot = offset_program->create_invocation().value();
  const std::array offset_bindings{offset_input.slots,
      context.upload(ids_desc, std::as_bytes(std::span(valid))).value()};
  EXPECT_TRUE(offset_slot.bind(offset_bindings).IsInvalidArgument());
  // Existing dense input ports retain the original byte-addressed alignment
  // adapter. A four-byte-offset dense A1 stays valid in Direct and Graph.
  auto dense_values = doubles(values);
  dense_values.insert(dense_values.begin(), 999.0);
  const auto dense_root = t::make(context, t::dense(c::DataType::kFloat32,
      {static_cast<std::int64_t>(dense_values.size())}), dense_values);
  const auto dense_offset = dense_root.view(c::View::slice(
      0, 1, static_cast<std::int64_t>(values.size())).value()).value().view(
      c::View::reshape(c::TensorShape::create(std::array{
          fixture.tokens, std::int64_t{1}, fixture.columns}).value()).value()).value();
  ASSERT_TRUE(dense_offset.description().is_dense());
  const Activation dense_input{dense_offset, dense_offset};
  for (bool graph_route : {false, true}) {
    auto dense_program = prepare(context, weights, dense_input, ids_desc,
                                 Layout::kDenseShared, graph_route);
    ASSERT_TRUE(dense_program) << dense_program.error().message();
    auto dense_slot = dense_program->create_invocation().value();
    const std::array dense_bindings{dense_offset, offset_bindings[1]};
    ASSERT_TRUE(dense_slot.bind(dense_bindings).ok());
    auto dense_work = dense_slot.submit();
    ASSERT_TRUE(dense_work) << dense_work.error().message();
    auto dense_result = dense_work->wait();
    ASSERT_TRUE(dense_result) << dense_result.error().message();
    t::expect(context, dense_result->get()->tensors()[0],
              doubles(fixture.expected(true, values, valid)), true);
    auto program = prepare(context, weights, input, ids_desc,
                           Layout::kBroadcast, graph_route);
    ASSERT_TRUE(program) << program.error().message();
    for (const auto invalid : {-1, static_cast<int>(fixture.experts)}) {
      auto ids = valid;
      ids[1] = invalid;
      const std::array inputs{graph_route ? input.source : input.slots,
          context.upload(ids_desc, std::as_bytes(std::span(ids))).value()};
      auto slot = program->create_invocation().value();
      ASSERT_TRUE(slot.bind(inputs).ok());
      auto work = slot.submit();
      ASSERT_TRUE(work) << work.error().message();
      auto failure = work->wait();
      ASSERT_FALSE(failure);
      EXPECT_TRUE(failure.error().IsOutOfRange());
    }
    auto recovered = program->create_invocation().value();
    const std::array inputs{graph_route ? input.source : input.slots,
        context.upload(ids_desc, std::as_bytes(std::span(valid))).value()};
    ASSERT_TRUE(recovered.bind(inputs).ok());
    auto work = recovered.submit();
    ASSERT_TRUE(work) << work.error().message();
    auto result = work->wait();
    ASSERT_TRUE(result) << result.error().message();
    t::expect(context, result->get()->tensors()[0],
              doubles(fixture.expected(true, values, valid)), true);
  }
  const auto wide = t::make(context, t::dense(c::DataType::kFloat32,
      {fixture.tokens, 1, fixture.columns * 2}),
      std::vector<double>(static_cast<std::size_t>(fixture.tokens * fixture.columns * 2), 1));
  const auto strided = wide.view(c::View::slice(2, 0, fixture.columns, 2).value()).value()
      .view(c::View::broadcast_to(c::TensorShape::create(
          std::array{fixture.tokens, fixture.slots, fixture.columns}).value()).value()).value();
  std::array<c::TensorDesc, 1> output;
  const std::array descriptions{weights.description(), strided.description(), ids_desc};
  ASSERT_TRUE(c::infer_output_descriptions(c::model_ops::grouped_matmul(),
                                           descriptions, output).ok());
  const std::array<v::Argument, 3> arguments{weights, strided.description(), ids_desc};
  EXPECT_FALSE(v::prepare(context, c::model_ops::grouped_matmul(), arguments));

  // Descriptor-only: logical broadcast elements exceed I32, while the real
  // shared bank is 256 MiB and the ID/output domains are small. No GPU bank is
  // allocated merely to check cold physical capacity qualification.
  const auto shared_root = t::dense(c::DataType::kFloat32, {32768, 1, 2048});
  std::array<c::TensorDesc, 1> broadcast_desc;
  const auto view = c::View::broadcast_to(c::TensorShape::create(
      std::array<std::int64_t, 3>{32768, 40, 2048}).value()).value();
  ASSERT_TRUE(c::infer_output_descriptions(c::Operation(view),
      std::array{shared_root}, broadcast_desc).ok());
  EXPECT_GT(*broadcast_desc[0].element_count(),
            static_cast<std::size_t>(INT32_MAX));
  EXPECT_EQ(*broadcast_desc[0].required_bytes(), std::size_t{256} << 20);
  const std::array large_inputs{
      t::dense(c::DataType::kFloat32, {3, 5, 2048}), broadcast_desc[0],
      t::dense(c::DataType::kInt32, {32768, 40})};
  ASSERT_TRUE(c::infer_output_descriptions(c::model_ops::grouped_matmul(),
                                          large_inputs, output).ok());
  EXPECT_TRUE(c::ggml_vulkan::detail::region_supports(
      c::model_ops::grouped_matmul(), large_inputs, output));
}
} // namespace
