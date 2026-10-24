#include "hyle/graph/providers/ggml_cuda.h"
#include "../model_fragment_test_support.h"
#include "../sparse_fragment_test_support.h"
#include "region_test_support.h"
#include "softmax_test_support.h"
#include "tensor_test_support.h"
#include "../compute_models/d5_gguf_test_support.h"
#include "../compute_models/gdn_region_test.h"
#include "../graph_ggml_constant_test.h"
#include "../compute_models/gather_elements_fixture.h"
#include "../compute_provider_ggml_cuda/bfloat_elementwise_test_support.h"

#include <ggml.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>

#include "hyle/compute/low_precision.h"

TEST(GgmlCudaGraph, BfloatTypedElementwiseMatchesFloatRecipeAndKeepsEveryRoundingBoundary) {
  using namespace ggml_cuda_graph_test;
  using mem::Buffer;
  using bfloat_elementwise_test::BF16;
  auto context = compute::Context::create(cuda::device(0).value(),
                                         graph::ggml_cuda::provider()).value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto description = ggml_cuda_tensor_test::dense(compute::DataType::kBFloat16, {3, 17});
  const auto row_description = ggml_cuda_tensor_test::dense(compute::DataType::kBFloat16, {17});
  const auto column_description = ggml_cuda_tensor_test::dense(compute::DataType::kBFloat16, {3, 1});
  const auto broadcast_row = ggml_cuda_tensor_test::dense(compute::DataType::kBFloat16, {1, 17});
  auto builder = graph::GraphBuilder::create().value();
  const auto x = builder.input(description).value();
  const auto y = builder.input(row_description).value();
  const auto wide_x = builder.add(compute::Cast{compute::DataType::kFloat32}, x).value();
  const auto wide_y = builder.add(compute::Cast{compute::DataType::kFloat32}, y).value();
  const auto operations = bfloat_elementwise_test::operations();
  std::vector<graph::ValueId> typed, original;
  for (std::size_t op = 0; op < operations.size(); ++op) {
    const std::array narrow_inputs{x, y}, wide_inputs{wide_x, wide_y};
    const auto count = op < 4 ? 2u : 1u;
    const auto narrow = builder.result(builder.add_node(
        operations[op], std::span(narrow_inputs).first(count)).value()).value();
    const auto wide = builder.result(builder.add_node(
        operations[op], std::span(wide_inputs).first(count)).value()).value();
    const auto rounded = builder.add(compute::Cast{compute::DataType::kBFloat16}, wide).value();
    ASSERT_TRUE(builder.publish(narrow).ok());
    ASSERT_TRUE(builder.publish(rounded).ok());
    typed.push_back(narrow); original.push_back(rounded);
  }
  const auto one = builder.input(column_description).value();
  const auto delta = builder.input(broadcast_row).value();
  const auto one_half = builder.input(column_description).value();
  const auto factor = builder.input(broadcast_row).value();
  const auto first_add = builder.add_node(compute::Operation(compute::Binary{}),
      std::array{one, delta}).value();
  const auto second_add = builder.add_node(compute::Operation(compute::Binary{}),
      std::array{builder.result(first_add).value(), delta}).value();
  const auto first_mul = builder.add_node(
      compute::Operation(compute::Binary{compute::BinaryKind::kMultiply}),
      std::array{one_half, factor}).value();
  const auto second_mul = builder.add_node(
      compute::Operation(compute::Binary{compute::BinaryKind::kMultiply}),
      std::array{builder.result(first_mul).value(), factor}).value();
  const auto added = builder.result(second_add).value();
  const auto multiplied = builder.result(second_mul).value();
  // The first Add/Mul results are deliberately private: the native fuser
  // could otherwise elide their BF16 stores and change the following result.
  ASSERT_TRUE(builder.publish(added).ok());
  ASSERT_TRUE(builder.publish(multiplied).ok());
  auto logical = std::move(builder).finish().value();
  const std::array domains{graph::StorageDomain{.resource = memory.ref()}};
  const std::array targets{graph::ExecutionTarget{
      .source = graph::ExecutionSource::acquire(context).value(),
      .default_storage = graph::StorageDomainId{0}}};
  const std::array lanes{graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
  auto compiled = graph::compile(logical,
      {.storage_domains = domains, .targets = targets, .lanes = lanes});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 1u);
  EXPECT_EQ(compiled->report.step(0)->provider.id, compute::ggml_cuda::kProviderId);
  EXPECT_EQ(compiled->report.step(0)->kind, graph::CompileStepKind::kProviderRegion);
  std::array slots{compiled->executable.create_invocation().value(),
                   compiled->executable.create_invocation().value()};
  std::array queues{context.create_queue().value(), context.create_queue().value()};
  Buffer old_output;
  std::vector<Buffer> old_inputs;
  std::vector<BF16> old_values;
  for (std::size_t round = 0; round < 2; ++round) {
    SCOPED_TRACE(::testing::Message() << "round=" << round);
    std::array<Buffer, 2> left, right, ones, deltas, halves, factors;
    std::array<std::vector<Buffer>, 2> output;
    std::array<std::vector<BF16>, 2> a, b;
    std::array<compute::Completion, 2> done;
    for (std::size_t slot = 0; slot < 2; ++slot) {
      SCOPED_TRACE(::testing::Message() << "slot=" << slot);
      const auto edge = bfloat_elementwise_test::edge_values();
      a[slot].resize(51); b[slot].resize(17);
      for (std::size_t i = 0; i < a[slot].size(); ++i)
        a[slot][i] = edge[(i + round + slot) % edge.size()];
      for (std::size_t i = 0; i < b[slot].size(); ++i)
        b[slot][i] = edge[(i + round + slot + 4) % edge.size()];
      left[slot] = ggml_cuda_tensor_test::upload<BF16>(memory.ref(), a[slot]);
      right[slot] = ggml_cuda_tensor_test::upload<BF16>(memory.ref(), b[slot]);
      std::array<BF16, 3> one_values, half_values;
      std::array<BF16, 17> delta_values, factor_values;
      one_values.fill(half_test::encode<BF16>(1.0));
      half_values.fill(half_test::encode<BF16>(1.5));
      delta_values.fill(half_test::encode<BF16>(1.0 / 256.0));
      factor_values.fill(half_test::encode<BF16>(1.0 + 1.0 / 128.0));
      ones[slot] = ggml_cuda_tensor_test::upload<BF16>(memory.ref(), one_values);
      halves[slot] = ggml_cuda_tensor_test::upload<BF16>(memory.ref(), half_values);
      deltas[slot] = ggml_cuda_tensor_test::upload<BF16>(memory.ref(), delta_values);
      factors[slot] = ggml_cuda_tensor_test::upload<BF16>(memory.ref(), factor_values);
      auto &run = slots[slot];
      const auto input_status = run.bind(x, compute::ConstTensorView::create(
          description, left[slot].view()).value());
      ASSERT_TRUE(input_status.ok()) << input_status.message();
      ASSERT_TRUE(run.bind(y, compute::ConstTensorView::create(row_description, right[slot].view()).value()).ok());
      ASSERT_TRUE(run.bind(one, compute::ConstTensorView::create(column_description, ones[slot].view()).value()).ok());
      ASSERT_TRUE(run.bind(one_half, compute::ConstTensorView::create(column_description, halves[slot].view()).value()).ok());
      ASSERT_TRUE(run.bind(delta, compute::ConstTensorView::create(broadcast_row, deltas[slot].view()).value()).ok());
      ASSERT_TRUE(run.bind(factor, compute::ConstTensorView::create(broadcast_row, factors[slot].view()).value()).ok());
      const auto bind_output = [&](graph::ValueId id) {
        output[slot].push_back(ggml_cuda_tensor_test::output_buffer(memory.ref(), description));
        return run.bind(id, compute::TensorView::create(description, output[slot].back().view()).value());
      };
      for (std::size_t op = 0; op < operations.size(); ++op) {
        ASSERT_TRUE(bind_output(typed[op]).ok());
        ASSERT_TRUE(bind_output(original[op]).ok());
      }
      ASSERT_TRUE(bind_output(added).ok());
      ASSERT_TRUE(bind_output(multiplied).ok());
      const std::array bindings{graph::QueueBinding{graph::ExecutionLaneId{0}, &queues[slot]}};
      ASSERT_TRUE(run.prepare_replay(bindings).ok());
      auto submitted = run.submit(queues[slot]);
      ASSERT_TRUE(submitted) << submitted.error().message();
      done[slot] = std::move(*submitted);
    }
    for (std::size_t slot = 0; slot < 2; ++slot) {
      SCOPED_TRACE(::testing::Message() << "slot=" << slot);
      ASSERT_TRUE(done[slot].wait().ok());
      done[slot] = {};
      for (std::size_t op = 0; op < operations.size(); ++op) {
        SCOPED_TRACE(op);
        const auto actual = half_test::read<BF16>(output[slot][op * 2]);
        const auto previous = half_test::read<BF16>(output[slot][op * 2 + 1]);
        ASSERT_EQ(actual.size(), 51u); ASSERT_EQ(previous.size(), actual.size());
        for (std::size_t i = 0; i < actual.size(); ++i)
          bfloat_elementwise_test::expect_bits(actual[i], previous[i], i);
      }
      const auto sum = half_test::read<BF16>(output[slot][22]);
      const auto product = half_test::read<BF16>(output[slot][23]);
      for (std::size_t i = 0; i < sum.size(); ++i) {
        EXPECT_EQ(sum[i].bits, half_test::encode<BF16>(1.0).bits) << i;
        EXPECT_EQ(product[i].bits, half_test::encode<BF16>(1.53125).bits) << i;
      }
      auto held = slots[slot].result(typed[0]);
      ASSERT_TRUE(held) << held.error().message();
      EXPECT_TRUE(slots[slot].bind(x, compute::ConstTensorView::create(
          description, left[slot].view()).value()).IsUnavailable());
      held->reset();
      // Detach borrowed tensor bindings while their caller buffers still
      // exist, before the next round can reuse any old output address.
      const auto reset_status = slots[slot].reset();
      ASSERT_TRUE(reset_status.ok()) << reset_status.message();
    }
    if (round == 0) {
      old_values = half_test::read<BF16>(output[0][22]);
      old_output = std::move(output[0][22]);
      for (std::size_t slot = 0; slot < 2; ++slot) {
        old_inputs.push_back(std::move(left[slot]));
        old_inputs.push_back(std::move(right[slot]));
      }
    } else {
      const auto preserved = half_test::read<BF16>(old_output);
      ASSERT_EQ(preserved.size(), old_values.size());
      for (std::size_t i = 0; i < preserved.size(); ++i)
        EXPECT_EQ(preserved[i].bits, old_values[i].bits) << i;
    }
  }
}

TEST(GgmlCudaGraph, BfloatChainsPreserveEveryRoundingAndPublishedIntermediate) {
  using namespace ggml_cuda_graph_test;
  using bfloat_elementwise_test::BF16;
  auto context = compute::Context::create(cuda::device(0).value(),
                                         graph::ggml_cuda::provider()).value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto description = ggml_cuda_tensor_test::dense(compute::DataType::kBFloat16, {3, 17});
  const auto row_description = ggml_cuda_tensor_test::dense(compute::DataType::kBFloat16, {17});
  std::array queues{context.create_queue().value(), context.create_queue().value()};
  const auto round_chain = [](float start, float operand, std::size_t depth, bool multiply) {
    auto result = half_test::encode<BF16>(start);
    for (std::size_t step = 0; step < depth; ++step) {
      const auto value = static_cast<float>(half_test::decode(result));
      result = half_test::encode<BF16>(multiply ? value * operand : value + operand);
    }
    return result;
  };
  for (const std::size_t depth : {2u, 3u, 8u}) {
    for (const bool publish_first : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "depth=" << depth << " published=" << publish_first);
      auto builder = graph::GraphBuilder::create().value();
      const auto add_input = builder.input(description).value();
      const auto multiply_input = builder.input(description).value();
      const auto delta = builder.input(row_description).value();
      const auto factor = builder.input(row_description).value();
      auto sum = add_input;
      auto product = multiply_input;
      graph::ValueId first_sum, first_product;
      for (std::size_t step = 0; step < depth; ++step) {
        sum = builder.result(builder.add_node(compute::Operation(compute::Binary{compute::BinaryKind::kAdd}),
            std::array{sum, delta}).value()).value();
        if (step == 0) first_sum = sum;
      }
      for (std::size_t step = 0; step < depth; ++step) {
        product = builder.result(builder.add_node(compute::Operation(compute::Binary{compute::BinaryKind::kMultiply}),
            std::array{product, factor}).value()).value();
        if (step == 0) first_product = product;
      }
      ASSERT_TRUE(builder.publish(sum).ok());
      ASSERT_TRUE(builder.publish(product).ok());
      if (publish_first) {
        ASSERT_TRUE(builder.publish(first_sum).ok());
        ASSERT_TRUE(builder.publish(first_product).ok());
      }
      auto logical = std::move(builder).finish().value();
      const std::array domains{graph::StorageDomain{.resource = memory.ref()}};
      const std::array targets{graph::ExecutionTarget{
          .source = graph::ExecutionSource::acquire(context).value(),
          .default_storage = graph::StorageDomainId{0}}};
      const std::array lanes{graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
      auto compiled = graph::compile(logical,
          {.storage_domains = domains, .targets = targets, .lanes = lanes});
      ASSERT_TRUE(compiled) << compiled.error().message();
      ASSERT_EQ(compiled->report.step_count(), 1u);
      std::array slots{compiled->executable.create_invocation().value(),
                       compiled->executable.create_invocation().value()};
      mem::Buffer held_output;
      std::vector<mem::Buffer> previous_buffers;
      std::vector<BF16> held_values;
      for (std::size_t round = 0; round < 2; ++round) {
        std::array<std::array<mem::Buffer, 4>, 2> inputs, outputs;
        std::array<std::array<BF16, 4>, 2> expected;
        std::array<compute::Completion, 2> done;
        for (std::size_t slot = 0; slot < slots.size(); ++slot) {
          const auto scale = static_cast<float>(1u << (round + slot));
          const auto initial_sum = scale;
          const auto increment = scale / 256.0F;
          const auto initial_product = 1.5F * scale;
          const auto multiplier = 1.0F + 1.0F / 128.0F;
          expected[slot] = {round_chain(initial_sum, increment, depth, false),
                            round_chain(initial_product, multiplier, depth, true),
                            round_chain(initial_sum, increment, 1, false),
                            round_chain(initial_product, multiplier, 1, true)};
          // These inputs distinguish public per-node RNE from one final store.
          float unrounded_sum = initial_sum, unrounded_product = initial_product;
          for (std::size_t step = 0; step < depth; ++step) {
            unrounded_sum += increment;
            unrounded_product *= multiplier;
          }
          ASSERT_NE(expected[slot][0].bits, half_test::encode<BF16>(unrounded_sum).bits);
          ASSERT_NE(expected[slot][1].bits, half_test::encode<BF16>(unrounded_product).bits);
          std::vector<BF16> sums(51, half_test::encode<BF16>(initial_sum));
          std::vector<BF16> products(51, half_test::encode<BF16>(initial_product));
          std::vector<BF16> increments(17, half_test::encode<BF16>(increment));
          std::vector<BF16> multipliers(17, half_test::encode<BF16>(multiplier));
          inputs[slot] = {ggml_cuda_tensor_test::upload<BF16>(memory.ref(), sums),
                          ggml_cuda_tensor_test::upload<BF16>(memory.ref(), products),
                          ggml_cuda_tensor_test::upload<BF16>(memory.ref(), increments),
                          ggml_cuda_tensor_test::upload<BF16>(memory.ref(), multipliers)};
          const std::array input_values{add_input, multiply_input, delta, factor};
          const std::array output_values{sum, product, first_sum, first_product};
          for (std::size_t port = 0; port < input_values.size(); ++port) {
            const auto &desc = port < 2 ? description : row_description;
            ASSERT_TRUE(slots[slot].bind(input_values[port], compute::ConstTensorView::create(
                desc, inputs[slot][port].view()).value()).ok());
          }
          for (std::size_t port = 0; port < (publish_first ? 4u : 2u); ++port) {
            outputs[slot][port] = ggml_cuda_tensor_test::output_buffer(memory.ref(), description);
            ASSERT_TRUE(slots[slot].bind(output_values[port], compute::TensorView::create(
                description, outputs[slot][port].view()).value()).ok());
          }
          const std::array bindings{graph::QueueBinding{graph::ExecutionLaneId{0}, &queues[slot]}};
          ASSERT_TRUE(slots[slot].prepare_replay(bindings).ok());
          auto submitted = slots[slot].submit(queues[slot]);
          ASSERT_TRUE(submitted) << submitted.error().message();
          done[slot] = std::move(*submitted);
        }
        for (std::size_t slot = 0; slot < slots.size(); ++slot) {
          ASSERT_TRUE(done[slot].wait().ok());
          done[slot] = {};
          for (std::size_t port = 0; port < (publish_first ? 4u : 2u); ++port) {
            const auto actual = half_test::read<BF16>(outputs[slot][port]);
            ASSERT_EQ(actual.size(), 51u);
            for (std::size_t i = 0; i < actual.size(); ++i)
              EXPECT_EQ(actual[i].bits, expected[slot][port].bits) << "slot=" << slot << " port=" << port << " i=" << i;
          }
          // Retire and detach while all borrowed caller buffers remain alive.
          ASSERT_TRUE(slots[slot].reset().ok());
        }
        if (round == 0) {
          held_values = half_test::read<BF16>(outputs[0][0]);
          held_output = std::move(outputs[0][0]);
          for (std::size_t slot = 0; slot < slots.size(); ++slot) {
            for (auto &buffer : inputs[slot]) previous_buffers.push_back(std::move(buffer));
            for (auto &buffer : outputs[slot]) previous_buffers.push_back(std::move(buffer));
          }
        } else {
          const auto preserved = half_test::read<BF16>(held_output);
          ASSERT_EQ(preserved.size(), held_values.size());
          for (std::size_t i = 0; i < preserved.size(); ++i)
            EXPECT_EQ(preserved[i].bits, held_values[i].bits) << i;
        }
      }
    }
  }
}

TEST(GgmlCudaGraph, ConstantDenseRhsPacksWithoutChangingOtherConsumersOrDynamicInputs) {
  auto context = hyle::compute::Context::create(
      hyle::compute::cuda::device(0).value(), hyle::graph::ggml_cuda::provider()).value();
  auto memory = hyle::compute::StorageOwner::adopt(context,
      hyle::compute::cuda::MemoryResource::create(
          context, hyle::compute::cuda::MemoryKind::kDevice).value()).value();
  ggml_constant_test::dense_rhs(context, memory.ref(), memory);
}

namespace {
void bfloat_expert_constants(bool single_node) {
  using namespace ggml_cuda_graph_test;
  using BF16 = hyle::simd::bfloat16_t;
  constexpr std::int64_t experts = 3, columns = 7, reduction = 37;
  constexpr std::int64_t tokens = 3, slots = 2;
  const auto weight_desc = compute::TensorDesc::dense(compute::DataType::kBFloat16,
      std::array{experts, columns, reduction}).value();
  const auto activation_desc = compute::TensorDesc::dense(compute::DataType::kFloat32,
      std::array{tokens, slots, reduction}).value();
  const auto ids_desc = compute::TensorDesc::dense(compute::DataType::kInt32,
      std::array{tokens, slots}).value();
  const auto output_desc = compute::TensorDesc::dense(compute::DataType::kFloat32,
      std::array{tokens, slots, columns}).value();
  for (const bool constant : {true, false}) {
    SCOPED_TRACE(constant);
    auto context = compute::Context::create(cuda::device(0).value(),
                                            graph::ggml_cuda::provider()).value();
    auto memory = compute::StorageOwner::adopt(context,
        cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value()).value();
    auto weights = mem::Buffer::create(memory.ref(), weight_desc.required_bytes().value(), 256).value();
    auto activations = mem::Buffer::create(memory.ref(), activation_desc.required_bytes().value(), 256).value();
    auto ids = mem::Buffer::create(memory.ref(), ids_desc.required_bytes().value(), 256).value();
    std::array results{ggml_cuda_tensor_test::output_buffer(memory.ref(), output_desc),
                       ggml_cuda_tensor_test::output_buffer(memory.ref(), output_desc)};
    std::array negative{ggml_cuda_tensor_test::output_buffer(memory.ref(), output_desc),
                        ggml_cuda_tensor_test::output_buffer(memory.ref(), output_desc)};
    std::vector<BF16> w(experts * columns * reduction);
    const auto write_weights = [&](int phase) {
      for (std::size_t i = 0; i < w.size(); ++i)
        w[i] = half_test::encode<BF16>((int((i * 19 + phase * 13) % 47) - 23) * 0.01371);
      return weights.view().write(std::as_bytes(std::span(w)));
    };
    ASSERT_TRUE(write_weights(0).ok());
    auto builder = graph::GraphBuilder::create().value();
    const auto weight = (constant ? builder.constant(weight_desc) : builder.input(weight_desc)).value();
    const auto x = builder.input(activation_desc).value();
    const auto indices = builder.input(ids_desc).value();
    const std::array inputs{weight, x, indices};
    const auto product = builder.result(builder.add_node(
        compute::model_ops::grouped_matmul(), inputs).value()).value();
    const auto neg = single_node ? product :
        builder.add(compute::Unary{compute::UnaryKind::kNeg}, product).value();
    ASSERT_TRUE(builder.publish(product).ok());
    if (!single_node) ASSERT_TRUE(builder.publish(neg).ok());
    auto logical = std::move(builder).finish().value();
    const std::array domains{graph::StorageDomain{.resource = memory.ref(), .owner = memory}};
    const std::array targets{graph::ExecutionTarget{
        .source = hyle::graph::ExecutionSource::acquire(context).value(), .default_storage = graph::StorageDomainId{0}}};
    const std::array lanes{graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
    compute::ReadLease ready;
    if (constant) ready = memory.adopt_buffer(std::move(weights)).value();
    const auto binding = ready ? ready.tensor(weight_desc).value()
        : compute::ConstTensorView::create(weight_desc, weights.view()).value();
    const std::array constants{graph::ConstantBinding{weight, binding,
        graph::ConstantPolicy::kBorrow,
        ready ? memory : compute::StorageOwner{}, ready}};
    auto compiled = graph::compile(logical,
        {.storage_domains = domains, .targets = targets, .lanes = lanes},
        {.constants = constant ? std::span(constants) : std::span<const graph::ConstantBinding>{}});
    ASSERT_TRUE(compiled) << compiled.error().message();
    ASSERT_EQ(compiled->report.step_count(), 1u);
    // This odd-K fallback keeps the original BF16 bank too; a selected expert
    // may use bounded F32 scratch, but no managed whole-bank widening remains.
    EXPECT_EQ(compiled->report.borrowed_constant_count(), constant ? 1u : 0u);
    EXPECT_EQ(compiled->report.materialized_constant_bytes(), 0u);
    EXPECT_EQ(compiled->report.packed_constant_count(), 0u);
    EXPECT_EQ(compiled->report.packed_constant_bytes(), 0u);
    if (constant) EXPECT_EQ(compiled->report.constant(0)->description, weight_desc);
    std::array invocations{compiled->executable.create_invocation().value(),
                           compiled->executable.create_invocation().value()};
    std::array queues{context.create_queue().value(), context.create_queue().value()};
    for (std::size_t lane = 0; lane < invocations.size(); ++lane) {
      if (!constant)
        ASSERT_TRUE(invocations[lane].bind(weight, compute::ConstTensorView::create(
            weight_desc, weights.view()).value()).ok());
      ASSERT_TRUE(invocations[lane].bind(x, compute::ConstTensorView::create(
          activation_desc, activations.view()).value()).ok());
      ASSERT_TRUE(invocations[lane].bind(indices, compute::ConstTensorView::create(
          ids_desc, ids.view()).value()).ok());
      ASSERT_TRUE(invocations[lane].bind(product, compute::TensorView::create(
          output_desc, results[lane].view()).value()).ok());
      if (!single_node) ASSERT_TRUE(invocations[lane].bind(neg, compute::TensorView::create(
          output_desc, negative[lane].view()).value()).ok());
      const std::array bindings{graph::QueueBinding{
          .lane = graph::ExecutionLaneId{0}, .queue = &queues[lane]}};
      ASSERT_TRUE(invocations[lane].prepare_replay(bindings).ok());
    }
    std::vector<float> old;
    for (int phase = 0; phase < 2; ++phase) {
      if (!constant && phase != 0) ASSERT_TRUE(write_weights(phase).ok());
      std::vector<float> a(tokens * slots * reduction);
      std::vector<std::int32_t> selected(tokens * slots);
      for (std::size_t i = 0; i < a.size(); ++i)
        a[i] = float((int((i * 11 + phase * 7) % 43) - 21) * 0.01913);
      for (std::size_t i = 0; i < selected.size(); ++i)
        selected[i] = std::int32_t((i / 2 + phase) % experts); // repeated experts remain legal
      ASSERT_TRUE(activations.view().write(std::as_bytes(std::span(a))).ok());
      ASSERT_TRUE(ids.view().write(std::as_bytes(std::span(selected))).ok());
      auto done = invocations[phase].submit(queues[phase]);
      ASSERT_TRUE(done) << done.error().message();
      if (phase == 1) {
        compiled->executable = {};
        invocations = {};
        ASSERT_TRUE(context.close().ok());
      }
      ASSERT_TRUE(done->wait().ok());
      const auto actual = half_test::read<float>(results[phase]);
      const auto inverted = single_node ? std::vector<float>{} : half_test::read<float>(negative[phase]);
      for (std::size_t row = 0; row < selected.size(); ++row)
        for (std::size_t column = 0; column < std::size_t(columns); ++column) {
          double expected = 0;
          for (std::size_t k = 0; k < std::size_t(reduction); ++k)
            expected += double(a[row * reduction + k]) * half_test::decode(
                w[(std::size_t(selected[row]) * columns + column) * reduction + k]);
          const auto index = row * columns + column;
          const auto budget = 2e-6 + std::abs(expected) * 6e-6;
          EXPECT_NEAR(actual[index], expected, budget) << index;
          if (!single_node) EXPECT_NEAR(inverted[index], -expected, budget) << index;
        }
      if (phase == 0) old = actual;
      else EXPECT_EQ(half_test::read<float>(results[0]), old);
    }
  }
}

} // namespace

TEST(GgmlCudaGraph, BfloatExpertConstantsRetainOriginalBankAndDynamicWeightsRemainRebindable) {
  bfloat_expert_constants(false);
}
TEST(GgmlCudaGraph, SingleBfloatExpertConstantRetainsOriginalBankAndDynamicWeightRemainsRebindable) {
  bfloat_expert_constants(true);
}

TEST(GgmlCudaGraph, CheckedIdsKeepEveryPublishedOutputAndOtherConsumersIntact) {
  using namespace ggml_cuda_graph_test;
  using namespace ggml_cuda_tensor_test;
  constexpr std::int64_t columns = 2, reduction = 32;
  const auto w3desc = dense(compute::DataType::kFloat32, {3, columns, reduction});
  const auto w2desc = dense(compute::DataType::kFloat32, {2, columns, reduction});
  const auto adesc = dense(compute::DataType::kFloat32, {1, 2, reduction});
  const auto idesc = dense(compute::DataType::kInt32, {1, 2});
  const auto ydesc = dense(compute::DataType::kFloat32, {1, 2, columns});
  auto context = compute::Context::create(cuda::device(0).value(),
      graph::ggml_cuda::provider()).value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const std::vector<float> wa(3 * columns * reduction, 1.0F);
  const std::vector<float> wb(2 * columns * reduction, 2.0F);
  const std::vector<float> activation(2 * reduction, 0.25F);
  auto weights3 = upload<float>(memory.ref(), wa);
  auto weights2 = upload<float>(memory.ref(), wb);
  auto a = upload<float>(memory.ref(), activation);
  auto ids = output_buffer(memory.ref(), idesc);
  auto builder = graph::GraphBuilder::create().value();
  const auto w3 = builder.input(w3desc).value(), w2 = builder.input(w2desc).value();
  const auto x = builder.input(adesc).value(), index = builder.input(idesc).value();
  const auto product = [&](graph::ValueId w) {
    const std::array inputs{w, x, index};
    return builder.result(builder.add_node(compute::model_ops::grouped_matmul(), inputs).value()).value();
  };
  const auto y3 = product(w3), y2 = product(w2);
  const auto neg = builder.add(compute::Unary{compute::UnaryKind::kNeg}, y3).value();
  for (const auto value : {y3, y2, neg}) ASSERT_TRUE(builder.publish(value).ok());
  const auto logical = std::move(builder).finish().value();
  const std::array domains{graph::StorageDomain{.resource = memory.ref()}};
  const std::array targets{graph::ExecutionTarget{
      .source = hyle::graph::ExecutionSource::acquire(context).value(), .default_storage = graph::StorageDomainId{0}}};
  const std::array lanes{graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
  auto compiled = graph::compile(logical,
      {.storage_domains = domains, .targets = targets, .lanes = lanes});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 1U);
  EXPECT_EQ(compiled->report.step(0)->provider.id, compute::ggml_cuda::kProviderId);
  std::array calls{compiled->executable.create_invocation().value(),
                   compiled->executable.create_invocation().value()};
  std::array queues{context.create_queue().value(), context.create_queue().value()};
  std::array buffers{output_buffer(memory.ref(), ydesc), output_buffer(memory.ref(), ydesc),
                     output_buffer(memory.ref(), ydesc), output_buffer(memory.ref(), ydesc),
                     output_buffer(memory.ref(), ydesc), output_buffer(memory.ref(), ydesc)};
  for (std::size_t slot = 0; slot < calls.size(); ++slot) {
    ASSERT_TRUE(calls[slot].bind(w3, compute::ConstTensorView::create(w3desc, weights3.view()).value()).ok());
    ASSERT_TRUE(calls[slot].bind(w2, compute::ConstTensorView::create(w2desc, weights2.view()).value()).ok());
    ASSERT_TRUE(calls[slot].bind(x, compute::ConstTensorView::create(adesc, a.view()).value()).ok());
    ASSERT_TRUE(calls[slot].bind(index, compute::ConstTensorView::create(idesc, ids.view()).value()).ok());
    const std::array values{y3, y2, neg};
    for (std::size_t port = 0; port < values.size(); ++port)
      ASSERT_TRUE(calls[slot].bind(values[port], compute::TensorView::create(
          ydesc, buffers[slot * 3 + port].view()).value()).ok());
    const std::array bindings{graph::QueueBinding{
        .lane = graph::ExecutionLaneId{0}, .queue = &queues[slot]}};
    ASSERT_TRUE(calls[slot].prepare_replay(bindings).ok());
    const std::array<std::int32_t, 2> valid{std::int32_t(slot), 0};
    ASSERT_TRUE(ids.view().write(std::as_bytes(std::span(valid))).ok());
    auto done = calls[slot].submit(queues[slot]);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    for (std::size_t port = 0; port < values.size(); ++port)
      EXPECT_EQ(half_test::read<float>(buffers[slot * 3 + port]),
                std::vector<float>(4, port == 0 ? 8.0F : port == 1 ? 16.0F : -8.0F));
  }
  // ID 2 is valid for one consumer and invalid for the other. Neither partial
  // success nor downstream nodes may publish; the original ID must stay 2.
  const std::array<std::int32_t, 2> invalid{1, 2};
  ASSERT_TRUE(ids.view().write(std::as_bytes(std::span(invalid))).ok());
  auto failed = calls[1].submit(queues[1]);
  ASSERT_TRUE(failed) << failed.error().message();
  calls = {};
  compiled->executable = {};
  ASSERT_TRUE(context.close().ok());
  EXPECT_TRUE(failed->wait().IsOutOfRange());
  for (std::size_t slot = 0; slot < 2; ++slot)
    for (std::size_t port = 0; port < 3; ++port)
      EXPECT_EQ(half_test::read<float>(buffers[slot * 3 + port]),
                std::vector<float>(4, port == 0 ? 8.0F : port == 1 ? 16.0F : -8.0F));
  EXPECT_EQ(half_test::read<std::int32_t>(ids),
            std::vector<std::int32_t>(invalid.begin(), invalid.end()));
}

TEST(GgmlCudaGraph, GatherElementsAndNegativeShareLinearPerRowResults) {
  using namespace ggml_cuda_graph_test;
  using namespace ggml_cuda_tensor_test;
  namespace fixture = gather_elements_test;
  const auto ddesc = dense(compute::DataType::kFloat32, {3, 5});
  const auto idesc = dense(compute::DataType::kInt32, {3, 7});
  const auto ydesc = dense(compute::DataType::kFloat32, {3, 7});
  for (const auto bounds : {compute::IndexBoundsPolicy::kChecked,
                            compute::IndexBoundsPolicy::kUnchecked}) {
    SCOPED_TRACE(static_cast<int>(bounds));
    auto context = compute::Context::create(cuda::device(0).value(),
        graph::ggml_cuda::provider()).value();
    auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
    auto data = output_buffer(memory.ref(), ddesc), ids = output_buffer(memory.ref(), idesc);
    auto output = output_buffer(memory.ref(), ydesc), negative = output_buffer(memory.ref(), ydesc);
    auto builder = graph::GraphBuilder::create().value();
    const auto x = builder.input(ddesc).value(), index = builder.input(idesc).value();
    const std::array inputs{x, index};
    const auto selected = builder.result(builder.add_node(
        compute::model_ops::gather_elements(1, bounds), inputs).value()).value();
    const auto neg = builder.add(compute::Unary{compute::UnaryKind::kNeg}, selected).value();
    ASSERT_TRUE(builder.publish(selected).ok());
    ASSERT_TRUE(builder.publish(neg).ok());
    const auto logical = std::move(builder).finish().value();
    const std::array domains{graph::StorageDomain{.resource = memory.ref()}};
    const std::array targets{graph::ExecutionTarget{
        .source = hyle::graph::ExecutionSource::acquire(context).value(), .default_storage = graph::StorageDomainId{0}}};
    const std::array lanes{graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
    auto compiled = graph::compile(logical,
        {.storage_domains = domains, .targets = targets, .lanes = lanes});
    ASSERT_TRUE(compiled) << compiled.error().message();
    ASSERT_EQ(compiled->report.step_count(), 1U);
    EXPECT_EQ(compiled->report.step(0)->provider.id, compute::ggml_cuda::kProviderId);
    auto call = compiled->executable.create_invocation().value();
    auto queue = context.create_queue().value();
    ASSERT_TRUE(call.bind(x, compute::ConstTensorView::create(ddesc, data.view()).value()).ok());
    ASSERT_TRUE(call.bind(index, compute::ConstTensorView::create(idesc, ids.view()).value()).ok());
    ASSERT_TRUE(call.bind(selected, compute::TensorView::create(ydesc, output.view()).value()).ok());
    ASSERT_TRUE(call.bind(neg, compute::TensorView::create(ydesc, negative.view()).value()).ok());
    const std::array bindings{graph::QueueBinding{
        .lane = graph::ExecutionLaneId{0}, .queue = &queue}};
    ASSERT_TRUE(call.prepare_replay(bindings).ok());
    for (int phase = 0; phase < 2; ++phase) {
      const auto source = fixture::data(phase);
      const auto indices = fixture::indices(phase);
      std::vector<float> stored;
      for (const auto value : source) stored.push_back(static_cast<float>(value));
      ASSERT_TRUE(data.view().write(std::as_bytes(std::span(stored))).ok());
      ASSERT_TRUE(ids.view().write(std::as_bytes(std::span(indices))).ok());
      auto done = call.submit(queue);
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      const auto actual = half_test::read<float>(output), inverted = half_test::read<float>(negative);
      const auto reference = fixture::reference(source, indices);
      for (std::size_t i = 0; i < reference.size(); ++i) {
        EXPECT_EQ(actual[i], reference[i]) << i;
        EXPECT_EQ(inverted[i], -reference[i]) << i;
      }
    }
  }
}

TEST(GgmlCudaGraph, GatedDeltaNetRegionsKeepBothSnapshotOutputs) {
  using namespace ggml_cuda_graph_test;
  auto context = compute::Context::create(cuda::device(0).value(),
                                          graph::ggml_cuda::provider()).value();
  auto memory = cuda::MemoryResource::create(
      context, cuda::MemoryKind::kDevice).value();
  gdn_region_test::exercise(context, memory.ref(), [&](const graph::Graph &logical) {
    const std::array storage{graph::StorageDomain{.resource = memory.ref()}};
    const std::array targets{graph::ExecutionTarget{
        .source = hyle::graph::ExecutionSource::acquire(context).value(), .default_storage = graph::StorageDomainId{0}}};
    const std::array lanes{graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
    return graph::compile(logical,
        {.storage_domains = storage, .targets = targets, .lanes = lanes});
  });
}

TEST(GgmlCudaGraph, NativeQ8GraphImportsGgufBytesAndReplays) {
  using namespace ggml_cuda_graph_test;
  d5_gguf_test::QuantizerCleanup cleanup;
  for (const auto format : {compute::GgufQuantizedType::kQ4_0,
       compute::GgufQuantizedType::kQ4_1, compute::GgufQuantizedType::kQ2K,
       compute::GgufQuantizedType::kIq2Xxs, compute::GgufQuantizedType::kIq2Xs,
       compute::GgufQuantizedType::kIq2S, compute::GgufQuantizedType::kIq1S,
       compute::GgufQuantizedType::kIq1M, compute::GgufQuantizedType::kQ8_0}) {
    SCOPED_TRACE(static_cast<int>(format));
    constexpr std::int64_t reduction = 256;
    constexpr std::int64_t columns = 2;
    const auto left_desc =
        ggml_cuda_tensor_test::dense(compute::DataType::kFloat32, {1, reduction});
    const auto right_desc = compute::gguf_weight_description(
        format, compute::DataType::kFloat32, reduction,
        columns);
    ASSERT_TRUE(right_desc) << right_desc.error().message();
    const auto output_desc =
        ggml_cuda_tensor_test::dense(compute::DataType::kFloat32, {1, columns});
    auto builder = graph::GraphBuilder::create().value();
    const auto left = builder.input(left_desc).value();
    const auto right = builder.input(*right_desc).value();
    const auto output = builder.add(compute::MatMul{}, left, right).value();
    ASSERT_TRUE(builder.publish(output).ok());
    auto context = compute::Context::create(cuda::device(0).value(),
                                            graph::ggml_cuda::provider())
                       .value();
    auto memory =
        cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
    const std::array domains{graph::StorageDomain{.resource = memory.ref()}};
    const std::array targets{graph::ExecutionTarget{
        .source = hyle::graph::ExecutionSource::acquire(context).value(), .default_storage = graph::StorageDomainId{0}}};
    const std::array lanes{
        graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
    auto compiled = graph::compile(
        std::move(builder).finish().value(),
        {.storage_domains = domains, .targets = targets, .lanes = lanes});
    ASSERT_TRUE(compiled) << compiled.error().message();
    ASSERT_EQ(compiled->report.step_count(), 1U);
    EXPECT_EQ(compiled->report.step(0)->provider.id,
              compute::ggml_cuda::kProviderId);

    std::vector<float> left_values(static_cast<std::size_t>(reduction), 0.25F);
    std::vector<float> physical_weights(
        static_cast<std::size_t>(reduction * columns));
    std::fill_n(physical_weights.begin(), reduction, 1.0F);
    std::fill_n(physical_weights.begin() + reduction, reduction, -2.0F);
    const auto packed = d5_gguf_test::pack(format, physical_weights,
        columns, reduction, *right_desc->required_bytes());
    auto left_buffer =
        ggml_cuda_tensor_test::upload<float>(memory.ref(), left_values);
    auto right_buffer =
        ggml_cuda_tensor_test::upload<std::byte>(memory.ref(), packed.bytes);
    auto output_buffer =
        ggml_cuda_tensor_test::output_buffer(memory.ref(), output_desc);
    auto invocation = compiled->executable.create_invocation().value();
    ASSERT_TRUE(invocation
                    .bind(left, compute::ConstTensorView::create(
                                    left_desc, left_buffer.view())
                                    .value())
                    .ok());
    ASSERT_TRUE(invocation
                    .bind(right, compute::ConstTensorView::create(
                                     *right_desc, right_buffer.view())
                                     .value())
                    .ok());
    ASSERT_TRUE(invocation
                    .bind(output, compute::TensorView::create(
                                      output_desc, output_buffer.view())
                                      .value())
                    .ok());
    auto queue = context.create_queue().value();
    const std::array queues{
        graph::QueueBinding{.lane = graph::ExecutionLaneId{0}, .queue = &queue}};
    ASSERT_TRUE(invocation.prepare_replay(queues).ok());
    for (int repeat = 0; repeat < 2; ++repeat) {
      auto done = invocation.submit(queue);
      ASSERT_TRUE(done) << done.error().message();
      ASSERT_TRUE(done->wait().ok());
      const auto actual = half_test::read<float>(output_buffer);
      const auto scale = repeat == 0 ? 1.0F : 2.0F;
      ASSERT_EQ(actual.size(), 2U);
      for (std::size_t column = 0; column < actual.size(); ++column) {
        float expected = 0.0F;
        for (std::int64_t k = 0; k < reduction; ++k)
          expected += packed.decoded[column * reduction + k] * 0.25F * scale;
        EXPECT_NEAR(actual[column], expected, std::abs(expected) * 0.002F + 1.0e-5F);
      }
      if (repeat == 0) {
        std::fill(left_values.begin(), left_values.end(), 0.5F);
        ASSERT_TRUE(
            left_buffer.view().write(std::as_bytes(std::span(left_values))).ok());
      }
    }
  }
}

TEST(GgmlCudaGraph, MatureMoeAndHyperConnectionBatchUsesOrdinaryGraphEntry) {
  using namespace ggml_cuda_graph_test;
  auto context = compute::Context::create(cuda::device(0).value(),
                                          graph::ggml_cuda::provider())
                     .value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  for (const auto &test : model_fragment_test::cases()) {
    SCOPED_TRACE(test.name);
    std::vector<mem::Buffer> storage;
    std::vector<graph::ValueId> inputs;
    auto builder = graph::GraphBuilder::create().value();
    storage.reserve(test.inputs.size());
    inputs.reserve(test.inputs.size());
    for (std::size_t i = 0; i < test.inputs.size(); ++i) {
      storage.push_back(mem::Buffer::create(
                            memory.ref(), test.inputs[i].required_bytes().value(),
                            256)
                            .value());
      ASSERT_TRUE(storage.back().view().write(test.payloads[i]).ok());
      inputs.push_back(builder.input(test.inputs[i]).value());
    }
    const auto node = builder.add_node(test.operation, inputs).value();
    const auto output = builder.result(node).value();
    ASSERT_TRUE(builder.publish(output).ok());
    auto logical = std::move(builder).finish().value();
    const std::array domains{
        graph::StorageDomain{.resource = memory.ref()}};
    const std::array targets{graph::ExecutionTarget{
        .source = hyle::graph::ExecutionSource::acquire(context).value(), .default_storage = graph::StorageDomainId{0}}};
    const std::array lanes{
        graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
    auto compiled = graph::compile(
        logical,
        {.storage_domains = domains, .targets = targets, .lanes = lanes});
    ASSERT_TRUE(compiled) << compiled.error().message();
    ASSERT_EQ(compiled->report.step_count(), 1U);
    EXPECT_EQ(compiled->report.step(0)->provider.id,
              compute::ggml_cuda::kProviderId);
    auto invocation = compiled->executable.create_invocation().value();
    for (std::size_t i = 0; i < inputs.size(); ++i)
      ASSERT_TRUE(invocation
                      .bind(inputs[i], compute::ConstTensorView::create(
                                           test.inputs[i], storage[i].view())
                                           .value())
                      .ok());
    auto result = mem::Buffer::create(memory.ref(),
                                      test.output.required_bytes().value(), 256)
                      .value();
    ASSERT_TRUE(invocation
                    .bind(output, compute::TensorView::create(
                                      test.output, result.view())
                                      .value())
                    .ok());
    auto queue = context.create_queue().value();
    const std::array queues{graph::QueueBinding{
        .lane = graph::ExecutionLaneId{0}, .queue = &queue}};
    ASSERT_TRUE(invocation.prepare_replay(queues).ok());
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<float>(result);
    ASSERT_EQ(actual.size(), test.expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], test.expected[i], 2.0e-5F) << i;
  }
}

TEST(GgmlCudaGraph, MatureLightningIndexerUsesOrdinaryGraphEntry) {
  using namespace ggml_cuda_graph_test;
  const sparse_fragment_test::Fixture test;
  auto context = compute::Context::create(cuda::device(0).value(),
                                          graph::ggml_cuda::provider())
                     .value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  auto builder = graph::GraphBuilder::create().value();
  std::array<graph::ValueId, 4> inputs;
  std::vector<mem::Buffer> storage;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    inputs[i] = builder.input(test.inputs[i]).value();
    storage.push_back(mem::Buffer::create(
                          memory.ref(), test.inputs[i].required_bytes().value(),
                          256)
                          .value());
    ASSERT_TRUE(storage.back().view().write(test.payloads[i]).ok());
  }
  const auto node = builder.add_node(test.operation, inputs).value();
  const auto output = builder.result(node).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto logical = std::move(builder).finish().value();
  const std::array domains{graph::StorageDomain{.resource = memory.ref()}};
  const std::array targets{graph::ExecutionTarget{
      .source = hyle::graph::ExecutionSource::acquire(context).value(), .default_storage = graph::StorageDomainId{0}}};
  const std::array lanes{
      graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
  auto compiled = graph::compile(
      logical,
      {.storage_domains = domains, .targets = targets, .lanes = lanes})
                      .value();
  ASSERT_EQ(compiled.report.step_count(), 1U);
  EXPECT_EQ(compiled.report.step(0)->provider.id,
            compute::ggml_cuda::kProviderId);
  auto invocation = compiled.executable.create_invocation().value();
  for (std::size_t i = 0; i < inputs.size(); ++i)
    ASSERT_TRUE(invocation
                    .bind(inputs[i], compute::ConstTensorView::create(
                                         test.inputs[i], storage[i].view())
                                         .value())
                    .ok());
  auto result = mem::Buffer::create(
                    memory.ref(), test.output.required_bytes().value(), 256)
                    .value();
  ASSERT_TRUE(invocation
                  .bind(output, compute::TensorView::create(
                                    test.output, result.view())
                                    .value())
                  .ok());
  auto queue = context.create_queue().value();
  const std::array queues{graph::QueueBinding{
      .lane = graph::ExecutionLaneId{0}, .queue = &queue}};
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto done = invocation.submit(queue);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    const auto actual = half_test::read<float>(result);
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], test.expected[i], 2.0e-5F) << i;
  }
}

TEST(GgmlCudaGraph, SoftmaxRegionPublishesLogicalAxesAndRetainsOldResults) {
  using namespace ggml_cuda_graph_test;
  auto context = compute::Context::create(cuda::device(0).value(),
                                          graph::ggml_cuda::provider())
                     .value();
  const auto compile = [&](const graph::Graph &logical,
                           mem::MemoryResourceRef resource) {
    const std::array storage{graph::StorageDomain{.resource = resource}};
    const std::array targets{graph::ExecutionTarget{
        .source = hyle::graph::ExecutionSource::acquire(context).value(), .default_storage = graph::StorageDomainId{0}}};
    const std::array lanes{
        graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
    return graph::compile(
        logical,
        {.storage_domains = storage, .targets = targets, .lanes = lanes});
  };
  exercise_softmax_graph<float>(context, compile);
  exercise_softmax_graph<hyle::simd::float16_t>(context, compile);
  exercise_softmax_graph<hyle::simd::bfloat16_t>(context, compile);
}

TEST(GgmlCudaGraph, WholeOriginalRegionUsesIndependentInvocations) {
  using namespace ggml_cuda_graph_test;
  auto context = compute::Context::create(cuda::device(0).value(),
                                          graph::ggml_cuda::provider())
                     .value();
  exercise(
      context, false,
      [&](const graph::Graph &logical, mem::MemoryResourceRef resource) {
        const std::array storage{graph::StorageDomain{.resource = resource}};
        const std::array targets{graph::ExecutionTarget{
            .source = hyle::graph::ExecutionSource::acquire(context).value(), .default_storage = graph::StorageDomainId{0}}};
        const std::array lanes{
            graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
        return graph::compile(
            logical,
            {.storage_domains = storage, .targets = targets, .lanes = lanes});
      });
}

TEST(GgmlCudaGraph,
     TensorRegionPreservesRoundingPublishedIntermediatesAndOldResults) {
  using namespace ggml_cuda_graph_test;
  auto context = compute::Context::create(cuda::device(0).value(),
                                          graph::ggml_cuda::provider())
                     .value();
  const auto compile = [&](const graph::Graph &logical,
                           mem::MemoryResourceRef resource) {
    const std::array storage{graph::StorageDomain{.resource = resource}};
    const std::array targets{graph::ExecutionTarget{
        .source = hyle::graph::ExecutionSource::acquire(context).value(), .default_storage = graph::StorageDomainId{0}}};
    const std::array lanes{
        graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
    return graph::compile(
        logical,
        {.storage_domains = storage, .targets = targets, .lanes = lanes});
  };
  exercise_tensor_graph<float>(context, compile);
  exercise_tensor_graph<hyle::simd::float16_t>(context, compile);
  exercise_tensor_graph<hyle::simd::bfloat16_t>(context, compile);
}

TEST(GgmlCudaGraph, ColdMeanCaptureKeepsOriginalCubSelectionAndDoesNotExecute) {
  using namespace ggml_cuda_graph_test;
  using namespace ggml_cuda_tensor_test;
  auto context = compute::Context::create(cuda::device(0).value(),
                                          graph::ggml_cuda::provider()).value();
  auto memory = cuda::MemoryResource::create(context,
                                             cuda::MemoryKind::kDevice).value();
  constexpr std::size_t width = 50000;
  const auto description = dense(DataType::kFloat32, {1, width});
  const Reduce operation{
      .kind = ReductionKind::kMean,
      .axes = Axes::from(std::array<std::int32_t, 1>{1}).value(),
      .keep_dimensions = true,
      .accumulation_type = DataType::kFloat32};
  const auto output_description = infer_single_output(
      Operation(operation), std::span(&description, 1)).value();
  std::vector<float> values(width);
  double expected = 0;
  for (std::size_t i = 0; i < width; ++i) {
    values[i] = static_cast<float>(
        0.17 + 0.37 * std::sin(static_cast<double>(i) * 0.013) +
        static_cast<double>(static_cast<int>((i * 37) % 997) - 498) * 0.00031 +
        static_cast<double>(i % 7) * 0.0000013);
    expected += static_cast<double>(values[i]);
  }
  expected /= static_cast<double>(width);
  auto input = upload<float>(memory.ref(), values);
  constexpr std::uint32_t sentinel_bits = 0x4b123456U;
  const std::array sentinel{std::bit_cast<float>(sentinel_bits)};
  auto captured_output = upload<float>(memory.ref(), sentinel);
  auto direct_output = output_buffer(memory.ref(), output_description);
  const auto source = ConstTensorView::create(description, input.view()).value();
  const auto destination =
      TensorView::create(output_description, captured_output.view()).value();
  const auto direct_destination =
      TensorView::create(output_description, direct_output.view()).value();
  auto builder = graph::GraphBuilder::create().value();
  const auto x = builder.input(description).value();
  const auto y = builder.add(operation, x).value();
  ASSERT_TRUE(builder.publish(y).ok());
  auto logical = std::move(builder).finish().value();
  const std::array storage{graph::StorageDomain{.resource = memory.ref()}};
  const std::array targets{graph::ExecutionTarget{
      .source = graph::ExecutionSource::acquire(context).value(),
      .default_storage = graph::StorageDomainId{0}}};
  const std::array lanes{
      graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
  auto compiled = graph::compile(
      logical, {.storage_domains = storage, .targets = targets, .lanes = lanes});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 1u);
  EXPECT_EQ(compiled->report.step(0)->provider.id, ggml_cuda::kProviderId);
  ASSERT_TRUE(compiled->report.step(0)->runtime.capture_compatible);
  auto slot = compiled->executable.create_invocation().value();
  auto queue = context.create_queue().value();
  ASSERT_TRUE(slot.bind(x, source).ok());
  ASSERT_TRUE(slot.bind(y, destination).ok());
  const std::array queues{
      graph::QueueBinding{graph::ExecutionLaneId{0}, &queue}};
  // No model math has run: this is the first recording of this native slot.
  // 50000 lies between the original Mean CUB thresholds, so omitting its
  // graph-enabled metadata could select a different reduction algorithm.
  const auto cold = slot.prepare_replay(
      queues, {.mode = graph::ReplayMode::kRequireCapture,
               .host_wait = graph::HostWaitPolicy::kReject});
  ASSERT_TRUE(cold.ok()) << cold.message();
  EXPECT_EQ(slot.replay_route(), graph::ReplayRoute::kCaptured);
  const auto untouched = half_test::read<float>(captured_output);
  ASSERT_EQ(untouched.size(), 1u);
  EXPECT_EQ(std::bit_cast<std::uint32_t>(untouched[0]), sentinel_bits);

  // Only after recording has proved nonexecuting, obtain the ordinary Direct
  // Commands result with the same original provider, input and float storage.
  auto prepared = prepare(context, Operation(operation), std::span(&source, 1),
                           std::span(&direct_destination, 1));
  ASSERT_TRUE(prepared) << prepared.error().message();
  EXPECT_EQ(prepared->provider().id, ggml_cuda::kProviderId);
  auto workspace = ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
  auto direct = prepared->create_invocation(workspace.view()).value();
  auto reference = direct.submit(queue, source, direct_destination);
  ASSERT_TRUE(reference) << reference.error().message();
  ASSERT_TRUE(reference->wait().ok());
  auto replay = slot.submit(queues);
  ASSERT_TRUE(replay) << replay.error().message();
  ASSERT_TRUE(replay->wait().ok());
  const auto commands = half_test::read<float>(direct_output);
  const auto captured = half_test::read<float>(captured_output);
  ASSERT_EQ(commands.size(), 1u);
  ASSERT_EQ(captured.size(), 1u);
  EXPECT_TRUE(std::isfinite(commands[0]));
  EXPECT_TRUE(std::isfinite(captured[0]));
  EXPECT_EQ(std::bit_cast<std::uint32_t>(captured[0]),
            std::bit_cast<std::uint32_t>(commands[0]));
  // Existing F32 reduction tests use this independent FP64 absolute budget.
  EXPECT_NEAR(commands[0], expected, 2.e-6);
  EXPECT_NEAR(captured[0], expected, 2.e-6);
}

TEST(GgmlCudaGraph, NonAttentionChainReusesDeadNativeIntermediates) {
  using namespace ggml_cuda_graph_test;
  constexpr std::size_t mib = std::size_t{1} << 20;
  constexpr std::uint32_t layers = 24;
  auto context = compute::Context::create(cuda::device(0).value(),
                                          graph::ggml_cuda::provider()).value();
  auto memory = cuda::MemoryResource::create(
      context, cuda::MemoryKind::kDevice).value();
  auto queue = context.create_queue().value();
  // Warm the same native provider before measuring allocation, without a
  // matrix operation or its independently lazy BLAS workspace.
  {
    Tensor input(memory.ref()), output(memory.ref());
    input.write(0);
    auto prepared = compute::prepare(context, compute::Unary{compute::UnaryKind::kNeg},
                                     input.description, memory.ref());
    ASSERT_TRUE(prepared) << prepared.error().message();
    auto workspace = workspace_buffer(*prepared, memory.ref());
    auto invocation = prepared->create_invocation(workspace.view());
    ASSERT_TRUE(invocation) << invocation.error().message();
    auto done = invocation->submit(queue, input.view.as_const(), output.view);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
  }
  Tensor input(memory.ref(), 1024, 2048), output(memory.ref(), 1024, 2048);
  ASSERT_EQ(input.storage.size(), 8 * mib);
  const auto values = input.write(0);
  auto builder = graph::GraphBuilder::create().value();
  const auto x = builder.input(input.description).value();
  auto y = x;
  for (std::uint32_t i = 0; i < layers; ++i)
    y = builder.add(compute::Binary{}, y, x).value();
  // Publishing intermediates would intentionally extend their lifetimes.
  ASSERT_TRUE(builder.publish(y).ok());
  const auto logical = std::move(builder).finish().value();
  const std::array storage{graph::StorageDomain{.resource = memory.ref()}};
  const std::array targets{graph::ExecutionTarget{
      .source = graph::ExecutionSource::acquire(context).value(),
      .default_storage = graph::StorageDomainId{0}}};
  const std::array lanes{graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
  auto compiled = graph::compile(logical,
      {.storage_domains = storage, .targets = targets, .lanes = lanes});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 1u);
  ASSERT_EQ(compiled->report.step(0)->provider.id, compute::ggml_cuda::kProviderId);
  ASSERT_EQ(compiled->report.step(0)->node_count, layers);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::size_t before = 0, after = 0, total = 0;
  ASSERT_EQ(cudaMemGetInfo(&before, &total), cudaSuccess);
  if (before < 512 * mib)
    GTEST_SKIP() << "Native liveness comparison requires 512 MiB free";
  auto invocation = compiled->executable.create_invocation();
  ASSERT_TRUE(invocation) << invocation.error().message();
  ASSERT_TRUE(invocation->bind(x, input.view.as_const()).ok());
  ASSERT_TRUE(invocation->bind(y, output.view).ok());
  const std::array queues{graph::QueueBinding{graph::ExecutionLaneId{0}, &queue}};
  ASSERT_TRUE(invocation->prepare_replay(queues).ok());
  auto done = invocation->submit(queue);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemGetInfo(&after, &total), cudaSuccess);
  const auto retained = before > after ? before - after : 0;
  RecordProperty("native_retained_bytes", std::to_string(retained));
  // This is the total retained device delta for the Invocation, including
  // caller workspace preallocated by Graph, not only provider-private memory.
  // The old per-tensor allocator retained 23 dead 8 MiB intermediates. This
  // wide limit permits allocator/driver overhead while requiring reuse; it
  // does not prescribe a driver page size or an exact two-buffer layout.
  EXPECT_LT(retained, 128 * mib);
  const auto actual = output.read();
  ASSERT_EQ(actual.size(), values.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    ASSERT_EQ(actual[i], values[i] * (layers + 1)) << i;
  EXPECT_EQ(input.read(), values);
}

TEST(GgmlCudaGraph, OrderedNativeRegionsShareCallerWorkspace) {
  using namespace ggml_cuda_graph_test;
  constexpr std::uint32_t layers = 24;
  constexpr std::uint32_t split = layers / 2;
  // Two real installations of one family have distinct instance identities.
  // Hard node constraints create two regions without adding an artificial
  // operation, transfer, lane or executor between them.
  const std::array providers{
      compute::ProviderBinding{graph::ggml_cuda::provider(), 100},
      compute::ProviderBinding{graph::ggml_cuda::provider(), 200}};
  auto context = compute::Context::create(
      cuda::device(0).value(), {.providers = providers}).value();
  const auto first_instance = context.provider_instance_id(0).value();
  const auto second_instance = context.provider_instance_id(1).value();
  ASSERT_NE(first_instance, second_instance);
  auto memory = cuda::MemoryResource::create(
      context, cuda::MemoryKind::kDevice).value();
  Tensor input(memory.ref(), 1024, 2048), output(memory.ref(), 1024, 2048);
  const auto values = input.write(0);
  auto builder = graph::GraphBuilder::create().value();
  const auto x = builder.input(input.description).value();
  auto y = x;
  for (std::uint32_t i = 0; i < layers; ++i)
    y = builder.add(compute::Binary{}, y, x).value();
  ASSERT_TRUE(builder.publish(y).ok());
  const auto logical = std::move(builder).finish().value();
  std::array<graph::NodeProviderConstraint, layers> constraints;
  for (std::uint32_t i = 0; i < layers; ++i)
    constraints[i] = {logical.node_id(i).value(), compute::ggml_cuda::kProviderId,
                      i < split ? first_instance : second_instance};
  const std::array storage{graph::StorageDomain{.resource = memory.ref()}};
  const std::array targets{graph::ExecutionTarget{
      .source = graph::ExecutionSource::acquire(context).value(),
      .default_storage = graph::StorageDomainId{0}}};
  const std::array lanes{graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
  auto compiled = graph::compile(logical,
      {.storage_domains = storage, .targets = targets, .lanes = lanes},
      {.node_providers = constraints,
       .operation_fallback = graph::OperationFallbackPolicy::kReject});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 2u);
  ASSERT_EQ(compiled->report.region_count(), 2u);
  const auto first = compiled->report.step(0).value();
  const auto second = compiled->report.step(1).value();
  ASSERT_EQ(first.kind, graph::CompileStepKind::kProviderRegion);
  ASSERT_EQ(second.kind, graph::CompileStepKind::kProviderRegion);
  EXPECT_EQ(first.provider_instance, first_instance);
  EXPECT_EQ(second.provider_instance, second_instance);
  ASSERT_EQ(first.node_count, split);
  ASSERT_EQ(second.node_count, layers - split);
  ASSERT_EQ(first.lane, second.lane);
  ASSERT_EQ(first.runtime.workspace_owner, compute::ProviderWorkspaceOwner::kCaller);
  ASSERT_EQ(second.runtime.workspace_owner, compute::ProviderWorkspaceOwner::kCaller);
  ASSERT_GT(first.workspace.size, 0u);
  ASSERT_GT(second.workspace.size, 0u);
  const auto bridge_bytes = input.storage.size();
  const auto alignment = std::max(first.workspace.alignment, second.workspace.alignment);
  const auto reused_bound = bridge_bytes +
      std::max(first.workspace.size, second.workspace.size) + 2 * alignment;
  const auto separate_bytes = bridge_bytes + first.workspace.size + second.workspace.size;
  const auto actual_bytes = compiled->executable.workspace_bytes();
  // The bridge remains live while the second region executes. The two
  // ordered native scratch intervals must overlap in storage, not be added
  // as independent per-region allocations. Allow only alignment padding.
  EXPECT_GE(actual_bytes, bridge_bytes);
  EXPECT_LE(actual_bytes, reused_bound);
  EXPECT_LT(actual_bytes, separate_bytes);
  EXPECT_EQ(compiled->report.workspace_bytes(), actual_bytes);
  RecordProperty("first_native_workspace_bytes", std::to_string(first.workspace.size));
  RecordProperty("second_native_workspace_bytes", std::to_string(second.workspace.size));
  RecordProperty("graph_workspace_bytes", std::to_string(actual_bytes));
  auto invocation = compiled->executable.create_invocation();
  ASSERT_TRUE(invocation) << invocation.error().message();
  ASSERT_TRUE(invocation->bind(x, input.view.as_const()).ok());
  ASSERT_TRUE(invocation->bind(y, output.view).ok());
  auto queue = context.create_queue().value();
  const std::array queues{graph::QueueBinding{graph::ExecutionLaneId{0}, &queue}};
  ASSERT_TRUE(invocation->prepare_replay(queues).ok());
  auto done = invocation->submit(queue);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  const auto actual = output.read();
  ASSERT_EQ(actual.size(), values.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    ASSERT_EQ(actual[i], values[i] * (layers + 1)) << i;
  EXPECT_EQ(input.read(), values);
}

TEST(GgmlCudaGraph, VariadicPortsReserveMetadataEvenWhenValuesRepeat) {
  using namespace ggml_cuda_graph_test;
  auto context = compute::Context::create(cuda::device(0).value(),
                                          graph::ggml_cuda::provider())
                     .value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  Tensor input(memory.ref(), 1, 7), joined(memory.ref(), 1, 490),
      output(memory.ref(), 1, 490);
  const auto values = input.write(0);
  auto builder = graph::GraphBuilder::create().value();
  const auto x = builder.input(input.description).value();
  std::array<graph::ValueId, 70> ports;
  ports.fill(x);
  auto node = builder.add_node(compute::Operation(compute::Concat{-1}), ports);
  ASSERT_TRUE(node) << node.error().message();
  const auto concat = builder.result(*node).value();
  const auto y =
      builder.add(compute::Unary{compute::UnaryKind::kNeg}, concat).value();
  ASSERT_TRUE(builder.publish(concat).ok());
  ASSERT_TRUE(builder.publish(y).ok());
  auto logical = std::move(builder).finish().value();
  const std::array storage{graph::StorageDomain{.resource = memory.ref()}};
  const std::array targets{graph::ExecutionTarget{
      .source = hyle::graph::ExecutionSource::acquire(context).value(), .default_storage = graph::StorageDomainId{0}}};
  const std::array lanes{
      graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
  auto compiled = graph::compile(
      logical,
      {.storage_domains = storage, .targets = targets, .lanes = lanes});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 1u);
  auto invocation = compiled->executable.create_invocation();
  ASSERT_TRUE(invocation) << invocation.error().message();
  ASSERT_TRUE(invocation->bind(x, input.view.as_const()).ok());
  ASSERT_TRUE(invocation->bind(concat, joined.view).ok());
  ASSERT_TRUE(invocation->bind(y, output.view).ok());
  auto queue = context.create_queue().value();
  const std::array queues{
      graph::QueueBinding{graph::ExecutionLaneId{0}, &queue}};
  ASSERT_TRUE(invocation->prepare_replay(queues).ok());
  auto done = invocation->submit(queue);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  const auto positive = joined.read(), negative = output.read();
  for (std::size_t i = 0; i < positive.size(); ++i) {
    EXPECT_EQ(positive[i], values[i % 7]);
    EXPECT_EQ(negative[i], -values[i % 7]);
  }
}

TEST(GgmlCudaGraph, TypedSliceCopyBindsBeforeAlignedNativeMath) {
  using namespace ggml_cuda_graph_test;
  auto context = compute::Context::create(cuda::device(0).value(),
                                          graph::ggml_cuda::provider())
                     .value();
  auto memory =
      cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  Tensor backing(memory.ref(), 1, 8), copied(memory.ref(), 1, 7),
      output(memory.ref(), 1, 7), unaligned_output(memory.ref(), 1, 8);
  const auto values = backing.write(2);
  auto builder = graph::GraphBuilder::create().value();
  const auto x = builder.input(copied.description).value();
  const auto copy = builder.add(compute::Copy{}, x).value();
  const auto y =
      builder.add(compute::Unary{compute::UnaryKind::kNeg}, copy).value();
  ASSERT_TRUE(builder.publish(copy).ok());
  ASSERT_TRUE(builder.publish(y).ok());
  auto logical = std::move(builder).finish().value();
  const std::array storage{graph::StorageDomain{.resource = memory.ref()}};
  const std::array targets{graph::ExecutionTarget{
      .source = hyle::graph::ExecutionSource::acquire(context).value(), .default_storage = graph::StorageDomainId{0}}};
  const std::array lanes{
      graph::ExecutionLane{.target = graph::ExecutionTargetId{0}}};
  auto compiled = graph::compile(
      logical,
      {.storage_domains = storage, .targets = targets, .lanes = lanes});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 1u);
  auto queue = context.create_queue().value();
  const std::array queues{
      graph::QueueBinding{graph::ExecutionLaneId{0}, &queue}};
  const auto input =
      compute::ConstTensorView::create(
          copied.description, backing.storage.view()
                                  .subview(sizeof(float), 7 * sizeof(float))
                                  .value())
          .value();
  auto invocation = compiled->executable.create_invocation().value();
  ASSERT_TRUE(invocation.bind(x, input).ok());
  ASSERT_TRUE(invocation.bind(copy, copied.view).ok());
  ASSERT_TRUE(invocation.bind(y, output.view).ok());
  ASSERT_TRUE(invocation.prepare_replay(queues).ok());
  auto done = invocation.submit(queue);
  ASSERT_TRUE(done) << done.error().message();
  ASSERT_TRUE(done->wait().ok());
  const auto a = copied.read(), b = output.read();
  for (std::size_t i = 0; i < 7; ++i) {
    EXPECT_EQ(a[i], values[i + 1]);
    EXPECT_EQ(b[i], -values[i + 1]);
  }
  // The published copy is also a math input, so writer alignment alone must
  // not permit binding it at a four-byte offset.
  auto rejected = compiled->executable.create_invocation().value();
  ASSERT_TRUE(rejected.bind(x, input).ok());
  ASSERT_TRUE(rejected.bind(y, output.view).ok());
  const auto bad_output =
      compute::TensorView::create(copied.description,
                                  unaligned_output.storage.view()
                                      .subview(sizeof(float), 7 * sizeof(float))
                                      .value())
          .value();
  const auto bound = rejected.bind(copy, bad_output);
  if (bound.ok()) {
    const auto prepared = rejected.prepare_replay(queues);
    if (!prepared.ok()) {
      EXPECT_TRUE(prepared.IsInvalidArgument()) << prepared.message();
      RecordProperty("unaligned_output_prepare_rejection",
                     std::string(prepared.message()));
    } else {
      auto attempt = rejected.submit(queue);
      EXPECT_TRUE(!attempt || !attempt->wait().ok());
    }
  } else {
    EXPECT_TRUE(bound.IsInvalidArgument());
  }
}
