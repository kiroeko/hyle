#include "../compute_provider_flashinfer/attention_test_support.h"
#include "../compute_provider_native/half_test_support.h"

#include <atomic>
#include <limits>
#include <optional>
#include <type_traits>
#include <c10/cuda/CUDACachingAllocator.h>

#include "hyle/compute/providers/torch_cuda.h"
#include "hyle/compute/providers/ggml_cuda.h"
#include "hyle/graph/executable.h"

namespace {
using namespace flashinfer_test;
namespace graph = hyle::graph;
using Bfloat = __nv_bfloat16;

struct Shape {
  int query = 17;
  int kv = 31;
  int heads = 8;
  int kv_heads = 2;
  int depth = 256;
  float scale = 0.0625F;
  bool causal = true;
  bool wide_exponent = false;
};

bool supports_fused_attention(const compute::Context &context) {
  const auto environment = cuda::environment(context).value();
  const int sm = environment.sm_major * 10 + environment.sm_minor;
  return sm >= 80 && sm <= 121;
}

template <class T>
void exercise_softplus_capture(compute::DataType type) {
  SCOPED_TRACE(static_cast<int>(type));
  constexpr std::size_t count = 34;
  auto context = compute::Context::create(cuda::device(0).value(),
                                           compute::torch_cuda::provider()).value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto description = compute::TensorDesc::dense(type,
      std::array<std::int64_t, 2>{2, 17}).value();
  const auto allocate = [&] {
    return mem::Buffer::create(memory.ref(), count * sizeof(T), 256).value();
  };
  std::array inputs{allocate(), allocate()};
  std::array outputs{allocate(), allocate()};
  auto replacement_input = allocate(), replacement_output = allocate();
  const float infinity = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::array<float, 17> points{-infinity, -1000.0F, -80.0F, -21.0F,
      -20.0F, -1.0F, -0.0F, 0.0F, 1.0F, 20.0F, 21.0F, 80.0F,
      1000.0F, infinity, nan, 0.25F, -0.25F};
  const auto values = [&](std::size_t seed) {
    std::vector<T> result(count);
    for (std::size_t i = 0; i < count; ++i)
      result[i] = half_test::encode<T>(points[(i + seed) % points.size()]);
    return result;
  };
  const auto write = [](mem::Buffer &buffer, const std::vector<T> &data) {
    const auto status = buffer.view().write(std::as_bytes(std::span(data)));
    ASSERT_TRUE(status.ok()) << status.message();
  };
  const auto raw = [](const mem::Buffer &buffer) {
    std::vector<std::byte> result(buffer.size());
    const auto status = buffer.view().read(std::span(result));
    EXPECT_TRUE(status.ok()) << status.message();
    return result;
  };
  const auto check = [&](const mem::Buffer &buffer, const std::vector<T> &input) {
    const auto actual = half_test::read<T>(buffer);
    ASSERT_EQ(actual.size(), input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
      SCOPED_TRACE(i);
      const double x = half_test::decode(input[i]);
      // Original ATen Softplus defaults: beta=1, threshold=20. Evaluate the
      // scalar reference independently, then apply the public output codec.
      const double expected = x > 20.0 ? x : std::log1p(std::exp(x));
      const double got = half_test::decode(actual[i]);
      const double rounded = half_test::decode(half_test::encode<T>(expected));
      if (std::isnan(rounded)) {
        EXPECT_TRUE(std::isnan(got));
      } else if (std::isinf(rounded)) {
        EXPECT_EQ(got, rounded);
      } else if (rounded == 0.0) {
        EXPECT_EQ(got, 0.0);
        EXPECT_FALSE(std::signbit(got));
      } else if constexpr (std::is_same_v<T, float>) {
        EXPECT_NEAR(got, rounded, 8.0 * std::numeric_limits<float>::epsilon() *
            std::max(std::abs(rounded), double(std::numeric_limits<float>::min())));
      } else {
        half_test::expect_near(actual[i], expected);
      }
      if (x > 20.0) {
        if constexpr (std::is_same_v<T, float>)
          EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),
                    std::bit_cast<std::uint32_t>(input[i]));
        else
          EXPECT_EQ(actual[i].bits, input[i].bits);
      }
    }
  };
  const compute::Operation operation(compute::Unary{compute::UnaryKind::kSoftplus});
  const std::array descriptions{description};
  auto prepared = compute::prepare(context, operation, descriptions, descriptions, memory.ref()).value();
  EXPECT_EQ(prepared.provider().id, compute::torch_cuda::kProviderId);
  EXPECT_TRUE(prepared.runtime_properties().capture_compatible);
  EXPECT_FALSE(prepared.runtime_properties().blocks_host);
  EXPECT_EQ(prepared.runtime_properties().workspace_owner, compute::ProviderWorkspaceOwner::kCaller);
  EXPECT_EQ(prepared.workspace_requirement().size, 0u);

  auto builder = graph::GraphBuilder::create().value();
  const auto input = builder.input(description).value();
  const auto output = builder.add(operation, input).value();
  ASSERT_TRUE(builder.publish(output).ok());
  auto logical = std::move(builder).finish().value();
  const std::array domains{graph::StorageDomain{memory.ref()}};
  const std::array targets{graph::ExecutionTarget{
      graph::ExecutionSource::acquire(context).value(), graph::StorageDomainId{0}}};
  auto compiled = graph::compile(logical,
      {.storage_domains = domains, .targets = targets}).value();
  ASSERT_EQ(compiled.report.step_count(), 1u);
  EXPECT_EQ(compiled.report.step(0)->provider.id, compute::torch_cuda::kProviderId);
  EXPECT_TRUE(compiled.report.step(0)->runtime.capture_compatible);
  ASSERT_EQ(compiled.report.command_segment_count(), 1u);
  ASSERT_EQ(compiled.report.capturable_segment_count(), 1u);
  std::array slots{compiled.executable.create_invocation().value(),
                   compiled.executable.create_invocation().value()};
  std::array queues{context.create_queue().value(), context.create_queue().value()};
  const std::vector<T> sentinel(count, half_test::encode<T>(-123.0));
  for (std::size_t slot = 0; slot < slots.size(); ++slot) {
    write(inputs[slot], values(slot));
    write(outputs[slot], sentinel);
    const auto before = raw(outputs[slot]);
    ASSERT_TRUE(slots[slot].bind(input,
        compute::ConstTensorView::create(description, inputs[slot].view()).value()).ok());
    ASSERT_TRUE(slots[slot].bind(output,
        compute::TensorView::create(description, outputs[slot].view()).value()).ok());
    const std::array bindings{graph::QueueBinding{graph::ExecutionLaneId{0}, &queues[slot]}};
    const auto status = slots[slot].prepare_replay(bindings,
        {.mode = graph::ReplayMode::kRequireCapture, .host_wait = graph::HostWaitPolicy::kReject});
    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_EQ(slots[slot].replay_route(), graph::ReplayRoute::kCaptured);
    // Neither Direct submission nor Graph mathematical warmup precedes this
    // first capture; recording must leave caller output bytes untouched.
    EXPECT_EQ(raw(outputs[slot]), before);
  }
  for (std::size_t round = 0; round < 2; ++round) {
    std::array<std::vector<T>, 2> source;
    std::array<compute::Completion, 2> done;
    for (std::size_t slot = 0; slot < slots.size(); ++slot) {
      source[slot] = values(round * 3 + slot);
      write(inputs[slot], source[slot]);
      auto submitted = slots[slot].submit(queues[slot]);
      ASSERT_TRUE(submitted) << submitted.error().message();
      done[slot] = std::move(*submitted);
    }
    for (std::size_t slot = 0; slot < slots.size(); ++slot) {
      ASSERT_TRUE(done[slot].wait().ok());
      done[slot] = {};
      check(outputs[slot], source[slot]);
      EXPECT_EQ(slots[slot].replay_route(), graph::ReplayRoute::kCaptured);
    }
  }
  auto held = slots[0].result(output);
  ASSERT_TRUE(held) << held.error().message();
  const auto preserved = raw(outputs[0]);
  EXPECT_TRUE(slots[0].reset().IsUnavailable());
  EXPECT_TRUE(slots[0].bind(input,
      compute::ConstTensorView::create(description, replacement_input.view()).value()).IsUnavailable());
  const auto other = values(9);
  write(inputs[1], other);
  {
    auto done = slots[1].submit(queues[1]);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
  }
  check(outputs[1], other);
  EXPECT_EQ(raw(outputs[0]), preserved);
  held->reset();
  ASSERT_TRUE(slots[0].reset().ok());
  EXPECT_EQ(slots[0].replay_route(), graph::ReplayRoute::kUnprepared);
  const auto replaced = values(13);
  write(replacement_input, replaced);
  write(replacement_output, sentinel);
  const auto replacement_before = raw(replacement_output);
  ASSERT_TRUE(slots[0].bind(input,
      compute::ConstTensorView::create(description, replacement_input.view()).value()).ok());
  ASSERT_TRUE(slots[0].bind(output,
      compute::TensorView::create(description, replacement_output.view()).value()).ok());
  const std::array binding{graph::QueueBinding{graph::ExecutionLaneId{0}, &queues[0]}};
  const auto status = slots[0].prepare_replay(binding,
      {.mode = graph::ReplayMode::kRequireCapture, .host_wait = graph::HostWaitPolicy::kReject});
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(raw(replacement_output), replacement_before);
  {
    auto done = slots[0].submit(queues[0]);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
  }
  check(replacement_output, replaced);
  EXPECT_EQ(raw(outputs[0]), preserved);
  EXPECT_EQ(slots[1].replay_route(), graph::ReplayRoute::kCaptured);
  check(outputs[1], other);
  for (auto &slot : slots)
    ASSERT_TRUE(slot.reset().ok());
}

TEST(TorchCudaSoftplus, ColdCaptureRetainsTypedTailsTwoSlotsAndReboundAddresses) {
  exercise_softplus_capture<float>(compute::DataType::kFloat32);
  exercise_softplus_capture<hyle::simd::float16_t>(compute::DataType::kFloat16);
  exercise_softplus_capture<hyle::simd::bfloat16_t>(compute::DataType::kBFloat16);
}

TEST(TorchCudaSoftplus, OneCapturedRangeCrossesGgmlAndTorchWithIndependentSlots) {
  constexpr std::size_t count = 34;
  constexpr std::size_t depth = 32, columns = 17, input_count = 2 * depth;
  constexpr std::size_t matrix_count = depth * columns;
  const std::array providers{
      compute::ProviderBinding{compute::ggml_cuda::provider(), 0},
      compute::ProviderBinding{compute::torch_cuda::provider(), 0}};
  auto context = compute::Context::create(cuda::device(0).value(),
                                           {.providers = providers}).value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  // The ordinary ggml carrier supports dense batched MatMul; its established
  // separate two-dimensional float route declines this ordinary request.
  // A real batch of one keeps the same small GEMM and output element count.
  const auto description = compute::TensorDesc::dense(compute::DataType::kFloat32,
      std::array<std::int64_t, 3>{1, 2, 17}).value();
  const auto input_description = compute::TensorDesc::dense(compute::DataType::kFloat32,
      std::array<std::int64_t, 3>{1, 2, 32}).value();
  const auto matrix_description = compute::TensorDesc::dense(compute::DataType::kFloat32,
      std::array<std::int64_t, 3>{1, 32, 17}).value();
  const auto allocate = [&](std::size_t elements) {
    return mem::Buffer::create(memory.ref(), elements * sizeof(float), 256).value();
  };
  std::array inputs{allocate(input_count), allocate(input_count)},
      matrices{allocate(matrix_count), allocate(matrix_count)},
      factors{allocate(count), allocate(count)}, outputs{allocate(count), allocate(count)};
  auto replacement_input = allocate(input_count), replacement_output = allocate(count);
  const auto values = [](std::size_t seed) {
    std::vector<float> result(input_count);
    for (std::size_t i = 0; i < input_count; ++i)
      result[i] = float(int((i + seed) % 27) - 13) * 0.25F;
    return result;
  };
  const auto coefficients = [](std::size_t seed) {
    std::vector<float> result(matrix_count);
    for (std::size_t i = 0; i < matrix_count; ++i)
      result[i] = float(int((i * 7 + seed) % 19) - 9) * 0.0625F;
    return result;
  };
  const auto multipliers = [](std::size_t seed) {
    std::vector<float> result(count);
    for (std::size_t i = 0; i < count; ++i)
      result[i] = float(2 + (i + seed) % 5) * 0.125F;
    return result;
  };
  const auto write = [](mem::Buffer &buffer, const std::vector<float> &data) {
    const auto status = buffer.view().write(std::as_bytes(std::span(data)));
    ASSERT_TRUE(status.ok()) << status.message();
  };
  const auto check = [&](const mem::Buffer &buffer, const std::vector<float> &x,
                          const std::vector<float> &matrix,
                          const std::vector<float> &factor) {
    const auto actual = half_test::read<float>(buffer);
    ASSERT_EQ(actual.size(), count);
    for (std::size_t i = 0; i < count; ++i) {
      // Independent double GEMM, with each public F32 boundary retained.
      double accumulated = 0;
      for (std::size_t k = 0; k < depth; ++k)
        accumulated += double(x[(i / columns) * depth + k]) * matrix[k * columns + i % columns];
      const float sum = float(accumulated);
      const float softplus = float(sum > 20.0F ? double(sum) : std::log1p(std::exp(double(sum))));
      const float expected = softplus * factor[i];
      EXPECT_NEAR(actual[i], expected,
          8.0F * std::numeric_limits<float>::epsilon() * std::max(1.0F, expected)) << i;
    }
  };
  auto builder = graph::GraphBuilder::create().value();
  const auto input = builder.input(input_description).value();
  const auto matrix = builder.input(matrix_description).value();
  const auto factor = builder.input(description).value();
  const auto sum = builder.add(compute::MatMul{}, input, matrix).value();
  const auto activated = builder.add(
      compute::Unary{compute::UnaryKind::kSoftplus}, sum).value();
  const auto output = builder.add(
      compute::Binary{compute::BinaryKind::kMultiply}, activated, factor).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto logical = std::move(builder).finish().value();
  const std::array constraints{
      graph::NodeProviderConstraint{logical.node_id(0).value(), compute::ggml_cuda::kProviderId},
      graph::NodeProviderConstraint{logical.node_id(1).value(), compute::torch_cuda::kProviderId},
      graph::NodeProviderConstraint{logical.node_id(2).value(), compute::ggml_cuda::kProviderId}};
  const std::array domains{graph::StorageDomain{memory.ref()}};
  const std::array targets{graph::ExecutionTarget{
      graph::ExecutionSource::acquire(context).value(), graph::StorageDomainId{0}}};
  auto compiled = graph::compile(logical,
      {.storage_domains = domains, .targets = targets},
      {.node_providers = constraints,
       .region_selection = graph::RegionSelectionPolicy::kDisabled});
  ASSERT_TRUE(compiled) << compiled.error().message();
  ASSERT_EQ(compiled->report.step_count(), 3u);
  for (std::size_t step = 0; step < 3; ++step) {
    EXPECT_EQ(compiled->report.step(step)->kind, graph::CompileStepKind::kOperationFallback);
    EXPECT_EQ(compiled->report.step(step)->provider.id,
        step == 1 ? compute::torch_cuda::kProviderId : compute::ggml_cuda::kProviderId);
    EXPECT_TRUE(compiled->report.step(step)->runtime.capture_compatible);
  }
  ASSERT_EQ(compiled->report.command_segment_count(), 1u);
  std::array slots{compiled->executable.create_invocation().value(),
                   compiled->executable.create_invocation().value()};
  std::array queues{context.create_queue().value(), context.create_queue().value()};
  const std::vector<float> sentinel(count, -123.0F);
  const auto prepare = [&](std::size_t slot, mem::Buffer &x, mem::Buffer &y) {
    ASSERT_TRUE(slots[slot].bind(input,
        compute::ConstTensorView::create(input_description, x.view()).value()).ok());
    ASSERT_TRUE(slots[slot].bind(matrix,
        compute::ConstTensorView::create(matrix_description, matrices[slot].view()).value()).ok());
    ASSERT_TRUE(slots[slot].bind(factor,
        compute::ConstTensorView::create(description, factors[slot].view()).value()).ok());
    ASSERT_TRUE(slots[slot].bind(output,
        compute::TensorView::create(description, y.view()).value()).ok());
    const std::array binding{graph::QueueBinding{graph::ExecutionLaneId{0}, &queues[slot]}};
    const auto status = slots[slot].prepare_replay(binding,
        {.mode = graph::ReplayMode::kRequireCapture, .host_wait = graph::HostWaitPolicy::kReject});
    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_EQ(slots[slot].replay_route(), graph::ReplayRoute::kCaptured);
  };
  const auto capture_counts = [&](std::size_t slot, std::size_t recorded,
                                   std::size_t reused) {
    const auto report = slots[slot].replay_report();
    ASSERT_TRUE(report) << report.error().message();
    EXPECT_EQ(report->recorded_capture_range_count, recorded);
    EXPECT_EQ(report->reused_capture_range_count, reused);
  };
  for (std::size_t slot = 0; slot < slots.size(); ++slot) {
    write(inputs[slot], values(slot));
    write(matrices[slot], coefficients(slot));
    write(factors[slot], multipliers(slot));
    write(outputs[slot], sentinel);
    prepare(slot, inputs[slot], outputs[slot]);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    EXPECT_EQ(half_test::read<float>(outputs[slot]), sentinel);
    capture_counts(slot, 1, 0);
    prepare(slot, inputs[slot], outputs[slot]);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    capture_counts(slot, 0, 1);
    EXPECT_EQ(half_test::read<float>(outputs[slot]), sentinel);
  }
  for (std::size_t round = 0; round < 2; ++round) {
    std::array<std::vector<float>, 2> x, rhs, weights;
    std::array<compute::Completion, 2> done;
    for (std::size_t slot = 0; slot < slots.size(); ++slot) {
      if (round != 0) {
        ASSERT_TRUE(slots[slot].reset().ok());
        write(outputs[slot], sentinel);
      }
      x[slot] = values(7 * round + slot);
      rhs[slot] = coefficients(5 * round + slot);
      weights[slot] = multipliers(3 * round + slot);
      write(inputs[slot], x[slot]);
      write(matrices[slot], rhs[slot]);
      write(factors[slot], weights[slot]);
      if (round != 0) {
        // Reset/rebind changes the public episode, while the exact native
        // ranges and owned workspace addresses remain the same for this slot.
        prepare(slot, inputs[slot], outputs[slot]);
        ASSERT_FALSE(::testing::Test::HasFatalFailure());
        capture_counts(slot, 0, 1);
        EXPECT_EQ(half_test::read<float>(outputs[slot]), sentinel);
      }
      auto submitted = slots[slot].submit(queues[slot]);
      ASSERT_TRUE(submitted) << submitted.error().message();
      done[slot] = std::move(*submitted);
    }
    for (std::size_t slot = 0; slot < slots.size(); ++slot) {
      ASSERT_TRUE(done[slot].wait().ok());
      done[slot] = {};
      check(outputs[slot], x[slot], rhs[slot], weights[slot]);
    }
  }
  auto held = slots[0].result(output);
  ASSERT_TRUE(held) << held.error().message();
  const auto preserved = half_test::read<float>(outputs[0]);
  EXPECT_TRUE(slots[0].reset().IsUnavailable());
  EXPECT_TRUE(slots[0].bind(input,
      compute::ConstTensorView::create(input_description, inputs[0].view()).value()).IsUnavailable());
  EXPECT_TRUE(slots[0].bind(input,
      compute::ConstTensorView::create(input_description, replacement_input.view()).value()).IsUnavailable());
  const auto other = values(19), other_matrix = coefficients(11), other_factor = multipliers(4);
  write(inputs[1], other);
  write(matrices[1], other_matrix);
  write(factors[1], other_factor);
  {
    auto done = slots[1].submit(queues[1]);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
  }
  check(outputs[1], other, other_matrix, other_factor);
  EXPECT_EQ(half_test::read<float>(outputs[0]), preserved);
  held->reset();
  ASSERT_TRUE(slots[0].reset().ok());
  const auto replacement = values(23), replacement_matrix = coefficients(13),
      replacement_factor = multipliers(8);
  write(replacement_input, replacement);
  write(matrices[0], replacement_matrix);
  write(factors[0], replacement_factor);
  write(replacement_output, sentinel);
  prepare(0, replacement_input, replacement_output);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  capture_counts(0, 1, 0);
  EXPECT_EQ(half_test::read<float>(replacement_output), sentinel);
  {
    auto done = slots[0].submit(queues[0]);
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
  }
  check(replacement_output, replacement, replacement_matrix, replacement_factor);
  EXPECT_EQ(half_test::read<float>(outputs[0]), preserved);
  check(outputs[1], other, other_matrix, other_factor);
  for (auto &slot : slots)
    ASSERT_TRUE(slot.reset().ok());
}

hyle::Result<graph::CompiledGraph> compile_attention(
    const compute::Context &context, mem::MemoryResourceRef memory,
    compute::ScaledDotProductAttention operation,
    const std::array<compute::TensorDesc, 3> &descriptions,
    std::array<graph::ValueId, 4> &ids) {
  auto builder = graph::GraphBuilder::create().value();
  for (std::size_t i = 0; i < descriptions.size(); ++i)
    ids[i] = builder.input(descriptions[i]).value();
  ids[3] = builder.add(operation, ids[0], ids[1], ids[2]).value();
  const auto published = builder.publish(ids[3]);
  if (!published.ok()) return std::unexpected(published);
  auto logical = std::move(builder).finish().value();
  const std::array storage{graph::StorageDomain{memory}};
  const std::array targets{
      graph::ExecutionTarget{hyle::graph::ExecutionSource::acquire(context).value(), graph::StorageDomainId{0}}};
  return graph::compile(logical,
                        graph::CompileEnvironment{.storage_domains = storage,
                                                   .targets = targets});
}

void bind_graph(graph::Invocation &invocation,
                const std::array<graph::ValueId, 4> &ids,
                const std::array<compute::ConstTensorView, 3> &inputs,
                compute::TensorView output, compute::Queue &queue) {
  for (std::size_t i = 0; i < inputs.size(); ++i)
    ASSERT_TRUE(invocation.bind(ids[i], inputs[i]).ok());
  ASSERT_TRUE(invocation.bind(ids[3], output).ok());
  const std::array queues{
      graph::QueueBinding{graph::ExecutionLaneId{0}, &queue}};
  ASSERT_TRUE(invocation.prepare_replay(
      queues, {.mode = graph::ReplayMode::kPreferCapture,
               .host_wait = graph::HostWaitPolicy::kReject}).ok());
  EXPECT_EQ(invocation.replay_route(), graph::ReplayRoute::kCommands);
}

void exercise(const Shape &shape) {
  SCOPED_TRACE(::testing::Message() << "Q=" << shape.query << " KV=" << shape.kv
      << " H=" << shape.heads << "/" << shape.kv_heads << " D=" << shape.depth
      << " scale=" << shape.scale << " causal=" << shape.causal);
  auto context = compute::Context::create(cuda::device(0).value(),
                                           compute::torch_cuda::provider()).value();
  if (!supports_fused_attention(context))
    GTEST_SKIP() << "Fused BF16 attention requires SM80..121";
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto q_shape = std::array<std::int64_t, 4>{2, shape.query, shape.heads, shape.depth};
  const auto kv_shape = std::array<std::int64_t, 4>{2, shape.kv, shape.kv_heads, shape.depth};
  constexpr auto type = compute::DataType::kBFloat16;
  Tensor q(type, q_shape, memory.ref()), k(type, kv_shape, memory.ref()),
      v(type, kv_shape, memory.ref()), other_q(type, q_shape, memory.ref()),
      other_k(type, kv_shape, memory.ref()), other_v(type, kv_shape, memory.ref()),
      output(type, q_shape, memory.ref()), other_output(type, q_shape, memory.ref()),
      graph_output(type, q_shape, memory.ref()), other_graph_output(type, q_shape, memory.ref());
  const std::array inputs{q.view.as_const(), k.view.as_const(), v.view.as_const()};
  const std::array other_inputs{other_q.view.as_const(), other_k.view.as_const(), other_v.view.as_const()};
  const std::array outputs{output.view}, other_outputs{other_output.view};
  const compute::ScaledDotProductAttention operation{.scale = shape.scale, .causal = shape.causal};
  auto prepared_result = compute::prepare(context, compute::Operation(operation), inputs, outputs, memory.ref());
  ASSERT_TRUE(prepared_result) << prepared_result.error().message();
  auto prepared = std::move(*prepared_result);
  ASSERT_EQ(prepared.provider().id, compute::torch_cuda::kProviderId);
  EXPECT_EQ(prepared.runtime_properties().mode, compute::ProviderRuntimeMode::kSharedQueue);
  EXPECT_EQ(prepared.runtime_properties().workspace_owner, compute::ProviderWorkspaceOwner::kProvider);
  EXPECT_EQ(prepared.runtime_properties().replay_class, compute::ProviderReplayClass::kManaged);
  EXPECT_FALSE(prepared.runtime_properties().capture_compatible);
  auto first = prepared.create_invocation().value();
  auto second = prepared.create_invocation().value();
  auto first_queue = context.create_queue().value();
  auto second_queue = context.create_queue().value();
  std::array<graph::ValueId, 4> ids;
  auto compiled_result = compile_attention(context, memory.ref(), operation,
      {q.description, k.description, v.description}, ids);
  ASSERT_TRUE(compiled_result) << compiled_result.error().message();
  auto compiled = std::move(*compiled_result);
  ASSERT_EQ(compiled.report.step_count(), 1u);
  ASSERT_EQ(compiled.report.step(0)->provider.id, compute::torch_cuda::kProviderId);
  auto graph_first = compiled.executable.create_invocation().value();
  auto graph_second = compiled.executable.create_invocation().value();

  for (int seed : {0, 3, 9}) {
    auto host_q = payload<Bfloat>(q, seed);
    auto host_k = payload<Bfloat>(k, seed + 5);
    auto host_v = payload<Bfloat>(v, seed + 13);
    if (shape.wide_exponent) {
      std::fill(host_q.begin(), host_q.end(), encode<Bfloat>(0));
      std::fill(host_k.begin(), host_k.end(), encode<Bfloat>(131072));
      for (std::size_t i = 0; i < host_v.size(); ++i)
        host_v[i] = encode<Bfloat>((i % 2 == 0 ? 1.0F : -1.0F) * 131072);
      ASSERT_TRUE(q.storage.view().write(std::as_bytes(std::span(host_q))).ok());
      ASSERT_TRUE(k.storage.view().write(std::as_bytes(std::span(host_k))).ok());
      ASSERT_TRUE(v.storage.view().write(std::as_bytes(std::span(host_v))).ok());
    }
    const auto oq = payload<Bfloat>(other_q, seed + 19);
    const auto ok = payload<Bfloat>(other_k, seed + 23);
    const auto ov = payload<Bfloat>(other_v, seed + 31);
    bind_graph(graph_first, ids, inputs, graph_output.view, first_queue);
    bind_graph(graph_second, ids, other_inputs, other_graph_output.view, second_queue);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    auto first_result = first.submit(first_queue, inputs, outputs);
    ASSERT_TRUE(first_result) << first_result.error().message();
    auto first_done = std::move(*first_result);
    auto second_result = second.submit(second_queue, other_inputs, other_outputs);
    ASSERT_TRUE(second_result) << second_result.error().message();
    auto second_done = std::move(*second_result);
    auto graph_first_result = graph_first.submit(first_queue);
    ASSERT_TRUE(graph_first_result) << graph_first_result.error().message();
    auto graph_first_done = std::move(*graph_first_result);
    auto graph_second_result = graph_second.submit(second_queue);
    ASSERT_TRUE(graph_second_result) << graph_second_result.error().message();
    auto graph_second_done = std::move(*graph_second_result);
    ASSERT_TRUE(graph_second_done.wait().ok());
    ASSERT_TRUE(graph_first_done.wait().ok());
    ASSERT_TRUE(second_done.wait().ok());
    ASSERT_TRUE(first_done.wait().ok());
    for (Tensor *destination : {&output, &graph_output})
      check_output(*destination, host_q, host_k, host_v, shape.query, shape.kv, 0.01,
                    shape.scale, shape.causal);
    for (Tensor *destination : {&other_output, &other_graph_output})
      check_output(*destination, oq, ok, ov, shape.query, shape.kv, 0.01,
                    shape.scale, shape.causal);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    auto rebound = first.submit(first_queue, other_inputs, other_outputs).value();
    bind_graph(graph_first, ids, other_inputs, other_graph_output.view, first_queue);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    auto graph_rebound = graph_first.submit(first_queue).value();
    ASSERT_TRUE(graph_rebound.wait().ok());
    ASSERT_TRUE(rebound.wait().ok());
    for (Tensor *destination : {&other_output, &other_graph_output})
      check_output(*destination, oq, ok, ov, shape.query, shape.kv, 0.01,
                    shape.scale, shape.causal);
    for (Tensor *destination : {&output, &graph_output})
      check_output(*destination, host_q, host_k, host_v, shape.query, shape.kv, 0.01,
                    shape.scale, shape.causal);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
  }
}

TEST(TorchCudaAttention, SameHeadAndGroupedBottomRightPrefillDecodeAndRebind) {
  exercise({.query = 7, .kv = 11, .heads = 4, .kv_heads = 4, .depth = 64});
  exercise({});
  exercise({.query = 1, .kv = 513});
}

TEST(TorchCudaAttention, ZeroAndNegativeScaleKeepBottomRightCausality) {
  for (float scale : {0.0F, -0.0625F}) exercise({.scale = scale});
}

TEST(TorchCudaAttention, NoncausalQueriesCanOutnumberKeys) {
  for (float scale : {0.0625F, 0.0F, -0.0625F})
    exercise({.query = 5, .kv = 3, .scale = scale, .causal = false});
}

TEST(TorchCudaAttention, Bfloat16ValuesOutsideFloat16RangeRemainFinite) {
  exercise({.wide_exponent = true});
  exercise({.query = 1, .kv = 513, .wide_exponent = true});
}

struct HostGate {
  std::atomic<bool> released{false};
  static void CUDART_CB wait(void *raw) {
    auto &gate = *static_cast<HostGate *>(raw);
    while (!gate.released.load(std::memory_order_acquire))
      gate.released.wait(false, std::memory_order_relaxed);
  }
  void open() noexcept {
    released.store(true, std::memory_order_release);
    released.notify_all();
  }
};

class OriginalAllocatorLimit {
public:
  OriginalAllocatorLimit()
      : previous_(c10::cuda::CUDACachingAllocator::getMemoryFraction(0)) {
    c10::cuda::CUDACachingAllocator::emptyCache();
    c10::cuda::CUDACachingAllocator::setMemoryFraction(0, 0);
  }
  ~OriginalAllocatorLimit() {
    c10::cuda::CUDACachingAllocator::setMemoryFraction(previous_, 0);
  }
private:
  double previous_;
};

void pending_external_stream(bool graph_path, bool fail_allocation) {
  auto context = compute::Context::create(cuda::device(0).value(),
                                           compute::torch_cuda::provider()).value();
  if (!supports_fused_attention(context))
    GTEST_SKIP() << "Fused BF16 attention requires SM80..121";
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  auto host = cuda::MemoryResource::create(context, cuda::MemoryKind::kPinnedHost).value();
  cudaStream_t native = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&native, cudaStreamNonBlocking), cudaSuccess);
  struct NativeOwner {
    cudaStream_t stream;
    ~NativeOwner() { EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess); }
  } native_owner{native};
  auto queue = cuda::borrow_stream(context, native).value();
  EXPECT_EQ(cuda::stream(queue).value(), native);
  constexpr auto type = compute::DataType::kBFloat16;
  const std::array<std::int64_t, 4> qs{1, 7, 4, 64}, kvs{1, 11, 4, 64};
  Tensor q(type, qs, memory.ref()), k(type, kvs, memory.ref()),
      v(type, kvs, memory.ref()), output(type, qs, memory.ref()),
      staging(type, qs, host.ref());
  (void)payload<Bfloat>(q, 0);
  const auto kh = payload<Bfloat>(k, 5), vh = payload<Bfloat>(v, 13);
  auto qh = payload<Bfloat>(staging, 19);
  for (auto &value : qh) value = encode<Bfloat>(4 * decode(value));
  ASSERT_TRUE(staging.storage.view().write(std::as_bytes(std::span(qh))).ok());
  const std::array inputs{q.view.as_const(), k.view.as_const(), v.view.as_const()};
  const std::array outputs{output.view};
  const compute::ScaledDotProductAttention operation{.scale = 0.125F, .causal = true};
  auto prepared = compute::prepare(context, compute::Operation(operation), inputs,
                                    outputs, memory.ref()).value();
  ASSERT_EQ(prepared.provider().id, compute::torch_cuda::kProviderId);
  auto direct = prepared.create_invocation().value();
  std::array<graph::ValueId, 4> ids;
  auto compiled = compile_attention(context, memory.ref(), operation,
      {q.description, k.description, v.description}, ids).value();
  ASSERT_EQ(compiled.report.step(0)->provider.id, compute::torch_cuda::kProviderId);
  auto replay = compiled.executable.create_invocation().value();
  bind_graph(replay, ids, inputs, output.view, queue);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  const auto submit = [&]() -> hyle::Result<compute::Completion> {
    return graph_path ? replay.submit(queue) : direct.submit(queue, inputs, outputs);
  };
  {
    // Finish CUDA/Torch first use and prime this exact stream's allocations
    // before a host callback deliberately blocks progress.
    auto warm = submit().value();
    ASSERT_TRUE(warm.wait().ok());
  }
  if (fail_allocation) {
    // A warm Invocation retains its last native result. Release those owners
    // so emptyCache can return whole blocks before the original allocator's
    // limit is applied; a free sub-block could otherwise satisfy this request.
    direct = {};
    replay = {};
    ASSERT_TRUE(context.drain().ok());
    direct = prepared.create_invocation().value();
    replay = compiled.executable.create_invocation().value();
    bind_graph(replay, ids, inputs, output.view, queue);
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
  }
  ASSERT_TRUE(q.storage.view().fill(std::byte{0}).ok());
  auto *destination = cuda::pointer<Bfloat>(q.storage.view()).value();
  auto *source = cuda::pointer<Bfloat>(staging.storage.view()).value();
  const auto bytes = q.storage.size();
  std::optional<OriginalAllocatorLimit> limit;
  if (fail_allocation) limit.emplace();
  HostGate gate;
  struct ReleaseGate {
    HostGate &gate;
    compute::Context &context;
    ~ReleaseGate() { gate.open(); context.drain().IgnoreError(); }
  } release{gate, context};
  auto producer = cuda::submit(queue, [&](cudaStream_t stream) noexcept {
    const auto status = cudaLaunchHostFunc(stream, &HostGate::wait, &gate);
    return status == cudaSuccess
               ? cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream)
               : status;
  }).value();
  auto submitted = submit();
  ASSERT_TRUE(submitted) << submitted.error().message();
  EXPECT_EQ(submitted->poll().value(), compute::CompletionPoll::kPending);
  // Caller tensor/resource owners remain live. Artifact/invocation handles may
  // disappear while the accepted Queue prefix and translated failure are pending.
  replay = {};
  compiled = {};
  direct = {};
  prepared = {};
  gate.open();
  const auto terminal = submitted->wait();
  ASSERT_TRUE(producer.wait().ok());
  if (fail_allocation) {
    EXPECT_TRUE(terminal.IsResourceExhausted()) << terminal.message();
    limit.reset();  // Restore the original process-local allocator policy.
    auto recovered = compute::prepare(context, compute::Operation(operation), inputs,
                                        outputs, memory.ref()).value();
    auto invocation = recovered.create_invocation().value();
    auto done = invocation.submit(queue, inputs, outputs).value();
    ASSERT_TRUE(done.wait().ok());
  } else {
    ASSERT_TRUE(terminal.ok()) << terminal.message();
  }
  // A provider using a foreign/default stream could read the old zero Q before
  // the gated producer. This oracle uses the new Q, not the warmup input.
  check_output(output, qh, kh, vh, 7, 11, 0.01, 0.125F, true);
}

class TorchCudaAttentionAsync : public ::testing::TestWithParam<bool> {};
TEST_P(TorchCudaAttentionAsync, BorrowedStreamProducerAndPendingArtifacts) {
  pending_external_stream(GetParam(), false);
}
TEST_P(TorchCudaAttentionAsync, OriginalOomRetainsPrefixAndQueueRecovers) {
  pending_external_stream(GetParam(), true);
}
INSTANTIATE_TEST_SUITE_P(DirectAndGraph, TorchCudaAttentionAsync,
                         ::testing::Values(false, true));

TEST(TorchCudaAttention, UnsupportedDescriptionsDeclineBeforeNativeExecution) {
  auto context = compute::Context::create(cuda::device(0).value(),
                                           compute::torch_cuda::provider()).value();
  auto memory = cuda::MemoryResource::create(context, cuda::MemoryKind::kDevice).value();
  const auto rejected = [&](compute::DataType type, int depth,
                             compute::ScaledDotProductAttention operation,
                             int query = 7, int kv = 11) {
    const auto q = compute::TensorDesc::dense(type,
        std::array<std::int64_t, 4>{1, query, 4, depth}).value();
    const auto k = compute::TensorDesc::dense(type,
        std::array<std::int64_t, 4>{1, kv, 2, depth}).value();
    const std::array inputs{q, k, k};
    const std::array outputs{q};
    EXPECT_FALSE(compute::prepare(context, compute::Operation(operation), inputs,
                                   outputs, memory.ref()));
  };
  rejected(compute::DataType::kFloat16, 64, {});
  rejected(compute::DataType::kFloat32, 64, {});
  rejected(compute::DataType::kBFloat16, 7, {});
  rejected(compute::DataType::kBFloat16, 264, {});
  rejected(compute::DataType::kBFloat16, 64,
            {.accumulation_type = compute::DataType::kBFloat16});
  rejected(compute::DataType::kBFloat16, 64, {.causal = true}, 11, 7);
  rejected(compute::DataType::kBFloat16, 64,
            {.layout = compute::AttentionLayout::kBatchHeadSequenceDepth}, 4, 2);
  rejected(compute::DataType::kBFloat16, 64,
            {.scale = std::numeric_limits<float>::infinity()});
  const auto q = compute::TensorDesc::dense(compute::DataType::kBFloat16,
      std::array<std::int64_t, 4>{1, 7, 4, 64}).value();
  const auto k = compute::TensorDesc::dense(compute::DataType::kBFloat16,
      std::array<std::int64_t, 4>{1, 11, 2, 64}).value();
  const auto mask = compute::TensorDesc::dense(compute::DataType::kBFloat16,
      std::array<std::int64_t, 4>{1, 4, 7, 11}).value();
  const std::array masked_inputs{q, k, k, mask};
  const std::array outputs{q};
  EXPECT_FALSE(compute::prepare(context,
      compute::Operation(compute::ScaledDotProductAttention{.mask = compute::MaskKind::kAdditive}),
      masked_inputs, outputs, memory.ref()));
}
} // namespace
