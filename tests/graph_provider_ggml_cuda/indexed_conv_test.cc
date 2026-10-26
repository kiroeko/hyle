#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "hyle/compute/model_ops.h"
#include "hyle/graph/cuda.h"
#include "hyle/graph/slots.h"
#include "../compute_provider_ggml_cuda/tensor_test_support.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace s = g::slots;
namespace m = hyle::mem;
using F16 = hyle::simd::float16_t;
using BF16 = hyle::simd::bfloat16_t;
using ggml_cuda_tensor_test::dense;
using ggml_cuda_tensor_test::output_buffer;
using ggml_cuda_tensor_test::upload;
constexpr std::size_t kB = 2, kC = 128, kL = 3, kP = 17;
constexpr std::array<std::uint32_t, kB> kSources{13, 2};
constexpr std::array<std::uint32_t, 6> kDestinations{7, 16, 4, 11, 6, 9};
constexpr c::RoutePolicyId kPolicy{0x48594c4547474355ULL, 7, 1};
template<class T> T take(hyle::Result<T> result, const char *what) {
  if (!result) throw std::runtime_error(std::string(what) + ": " + std::string(result.error().message()));
  return std::move(*result);
}
void check(hyle::Status status, const char *what) {
  if (!status.ok()) throw std::runtime_error(std::string(what) + ": " + std::string(status.message()));
}
template<class T> T sample(std::size_t index) {
  if (index % kC == 0) {
    if constexpr (std::is_same_v<T, float>) return std::bit_cast<float>(std::uint32_t{0x80000000});
    else return T::from_bits(0x8000);
  }
  if (index % kC == 1) {
    if constexpr (std::is_same_v<T, float>) return std::bit_cast<float>(std::uint32_t{0x7fa12345});
    else if constexpr (std::is_same_v<T, BF16>) return T::from_bits(0x7fa5);
    else return T::from_bits(0x7d55);
  }
  return half_test::encode<T>(static_cast<double>(static_cast<int>(index % 37) - 18) / 32.0);
}
void expect_bytes(m::ConstBufferView view, std::span<const std::byte> expected) {
  std::vector<std::byte> actual(expected.size());
  ASSERT_TRUE(view.read(actual).ok());
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) EXPECT_EQ(actual[i], expected[i]) << "byte " << i;
}
template<class T>
std::vector<float> dense_reference(c::Context &context, m::MemoryResourceRef resource,
    std::span<const T> samples, std::span<const float> weights, std::size_t tokens) {
  const auto raw = dense(half_test::data_type<T>(), {kB, static_cast<std::int64_t>(kL + tokens), kC});
  const auto wide = dense(c::DataType::kFloat32, {kB, static_cast<std::int64_t>(kL + tokens), kC});
  const auto output = dense(c::DataType::kFloat32, {kB, static_cast<std::int64_t>(tokens), kC});
  const auto weight = dense(c::DataType::kFloat32, {kC, kL + 1});
  auto raw_buffer = upload<T>(resource, samples);
  auto weight_buffer = upload<float>(resource, weights);
  m::Buffer wide_buffer;
  auto output_storage = output_buffer(resource, output);
  // Low precision follows the original explicit Cast; F32 needs no identity
  // operation before the legacy dense SSM convolution.
  auto samples_view = take(c::ConstTensorView::create(raw, raw_buffer.view()), "reference raw");
  if constexpr (!std::is_same_v<T, float>) {
    wide_buffer = output_buffer(resource, wide);
    const std::array cast_inputs{samples_view};
    check(ggml_cuda_tensor_test::execute(context, c::Cast{c::DataType::kFloat32}, cast_inputs,
        take(c::TensorView::create(wide, wide_buffer.view()), "reference wide")), "reference Cast");
    samples_view = take(c::ConstTensorView::create(wide, wide_buffer.view()), "reference samples");
  }
  const std::array inputs{samples_view,
      take(c::ConstTensorView::create(weight, weight_buffer.view()), "reference weights")};
  check(ggml_cuda_tensor_test::execute(context, c::model_ops::causal_conv1d(), inputs,
      take(c::TensorView::create(output, output_storage.view()), "reference output")), "reference conv");
  return half_test::read<float>(output_storage);
}

template<class T> void exercise(std::size_t tokens, std::size_t prefixes,
    bool padded, bool escape = false) {
  auto context = take(g::cuda::create_context(), "CUDA Context");
  auto memory = take(c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice), "device memory");
  auto staging = take(c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kPinnedHost), "host metadata");
  const auto item = dense(half_test::data_type<T>(), {kL, kC});
  constexpr auto item_bytes = kL * kC * sizeof(T);
  const auto stride = item_bytes + (padded ? 256 : 0);
  const auto schema = take(s::Schema::create(item, kP, stride), "slot schema");
  std::vector<T> initial(kB * kL * kC), chunk(kB * tokens * kC), logical(kB * (kL + tokens) * kC);
  for (std::size_t i = 0; i < initial.size(); ++i) initial[i] = sample<T>(i);
  for (std::size_t i = 0; i < chunk.size(); ++i) chunk[i] = sample<T>(i + 41);
  for (std::size_t b = 0; b < kB; ++b) {
    std::copy_n(initial.begin() + b * kL * kC, kL * kC, logical.begin() + b * (kL + tokens) * kC);
    std::copy_n(chunk.begin() + b * tokens * kC, tokens * kC, logical.begin() + (b * (kL + tokens) + kL) * kC);
  }
  std::vector<float> weights(kC * (kL + 1));
  for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 32.F;
  const auto reference = dense_reference<T>(context, memory.ref(), logical, weights, tokens);
  ASSERT_EQ(reference.size(), kB * tokens * kC);
  std::vector<std::byte> before(kP * stride, std::byte{0xa5});
  for (std::size_t b = 0; b < kB; ++b) {
    const auto bytes = std::as_bytes(std::span(initial).subspan(b * kL * kC, kL * kC));
    std::copy(bytes.begin(), bytes.end(), before.begin() + kSources[b] * stride);
  }
  auto backing = take(m::Buffer::create(memory.ref(), before.size(), 256), "pool backing");
  check(backing.view().write(before), "initialize pool");
  const m::ConstBufferView diagnostic = backing.view();
  auto pool = take(c::LeasePool::adopt(std::move(backing),
      {.max_blocks = static_cast<std::uint32_t>(padded ? 2 * kP : kP)}), "pool");
  const auto layout = take(c::LeaseSlotLayout::create(pool, item, kP, stride), "slot layout");
  std::array<c::WriteLease, kP> blocks;
  std::array<c::ReadLease, kP> gaps, unused;
  for (std::size_t slot = 0; slot < kP; ++slot) {
    blocks[slot] = take(pool.allocate(item_bytes, 256), "slot writer");
    ASSERT_EQ(take(layout.locate(blocks[slot]), "slot index").index, slot);
    if (padded) gaps[slot] = take(take(pool.allocate(256, 256), "gap").publish(), "retain gap");
  }
  std::array<c::ReadLease, kB> sources;
  std::vector<c::WriteLease> destinations(kB * prefixes);
  for (std::size_t b = 0; b < kB; ++b) sources[b] = take(blocks[kSources[b]].publish(), "source");
  for (std::size_t i = 0; i < destinations.size(); ++i) destinations[i] = std::move(blocks[kDestinations[i]]);
  for (std::size_t slot = 0; slot < kP; ++slot)
    if (blocks[slot].valid()) unused[slot] = take(blocks[slot].publish(), "unused slot");

  auto builder = take(g::GraphBuilder::create(), "GraphBuilder");
  const auto bank = take(builder.input(take(schema.capacity_description(), "capacity description")), "bank input");
  const auto src = take(builder.input(dense(c::DataType::kInt32, {kB})), "source IDs");
  const auto dst = take(builder.input(dense(c::DataType::kInt32, {static_cast<std::int64_t>(kB * prefixes)})), "destination IDs");
  const auto chunk_desc = dense(half_test::data_type<T>(), {kB, static_cast<std::int64_t>(tokens), kC});
  const auto weight_desc = dense(c::DataType::kFloat32, {kC, kL + 1});
  const auto chunk_input = take(builder.input(chunk_desc), "chunk input");
  const auto weight_input = take(builder.input(weight_desc), "weight input");
  const auto gathered = take(builder.add(c::Gather{.axis = 0, .bounds = c::IndexBoundsPolicy::kChecked}, bank, src), "Gather");
  const auto reshape = [&](g::ValueId input, std::initializer_list<std::int64_t> dimensions) {
    return take(builder.add(take(c::View::reshape(take(c::TensorShape::create(
        std::span(dimensions.begin(), dimensions.size())), "reshape shape")), "reshape descriptor"), input), "reshape");
  };
  const auto history = reshape(gathered, {kB, kL, kC});
  const auto samples = take(builder.add(c::Concat{.axis = 1}, history, chunk_input), "original samples");
  auto wide = samples;
  if constexpr (!std::is_same_v<T, float>) wide = take(builder.add(c::Cast{c::DataType::kFloat32}, samples), "Cast F32");
  const auto convolution = take(builder.add(c::model_ops::causal_conv1d(), wide, weight_input), "CausalConv");
  std::vector<g::ValueId> windows;
  for (std::size_t p = tokens - prefixes + 1; p <= tokens; ++p) {
    const auto slice = take(builder.add(take(c::View::slice(1, static_cast<std::int64_t>(p), kL), "window descriptor"), samples), "window");
    if (prefixes == 1) {
      // For B2 this slice has a real batch gap. Materialize it as a legal
      // ordinary graph before flattening, rather than claiming dense storage.
      windows.push_back(take(builder.add(c::Copy{}, slice), "materialized final window"));
    } else windows.push_back(reshape(slice, {kB, 1, kL, kC}));
  }
  auto stacked = windows.front();
  if (windows.size() > 1) {
    const auto node = take(builder.add_node(c::Operation(c::Concat{.axis = 1}), windows), "stack windows");
    stacked = take(builder.result(node), "stacked windows");
  }
  const auto rows = reshape(stacked, {static_cast<std::int64_t>(kB * prefixes), kL * kC});
  const auto store = take(builder.add_node(c::Operation::custom(s::Store{schema}), std::array{rows, dst}), "Store");
  check(builder.publish(convolution), "publish conv");
  if (escape) check(builder.publish(gathered), "publish escaped history");
  const auto graph = take(std::move(builder).finish(), "Graph");
  auto compiled = take(g::cuda::compile(graph, context, memory.ref()), "compile");
  if (!escape) {
    ASSERT_EQ(compiled.report.step_count(), 1u);
    const auto step = take(compiled.report.step(0), "indexed step");
    EXPECT_EQ(step.policy, kPolicy); EXPECT_EQ(step.implementation, c::ProviderImplementationId{8});
    EXPECT_EQ(step.dispatch_count, 1u); EXPECT_EQ(step.node_count, graph.node_count());
  } else {
    EXPECT_GT(compiled.report.step_count(), 1u);
    for (std::size_t i = 0; i < compiled.report.step_count(); ++i) EXPECT_NE(compiled.report.step(i)->policy, kPolicy);
  }
  auto invocation = take(compiled.executable.create_invocation(), "invocation");
  auto queue = take(context.create_queue(), "Queue");
  const std::array queues{g::QueueBinding{g::ExecutionLaneId{0}, &queue}};
  auto chunk_storage = upload<T>(memory.ref(), chunk);
  auto weight_storage = upload<float>(memory.ref(), weights);
  const auto output_desc = dense(c::DataType::kFloat32, {kB, static_cast<std::int64_t>(tokens), kC});
  auto output = output_buffer(memory.ref(), output_desc);
  m::Buffer escaped;
  if (escape) escaped = output_buffer(memory.ref(), take(graph.value_description(gathered), "gather description"));
  const auto bind_inputs = [&] {
    check(invocation.bind(chunk_input, take(c::ConstTensorView::create(chunk_desc, chunk_storage.view()), "chunk view")), "bind chunk");
    check(invocation.bind(weight_input, take(c::ConstTensorView::create(weight_desc, weight_storage.view()), "weight view")), "bind weights");
    check(invocation.bind(convolution, take(c::TensorView::create(output_desc, output.view()), "output view")), "bind output");
    if (escape) check(invocation.bind(gathered, take(c::TensorView::create(
        take(graph.value_description(gathered), "escaped description"), escaped.view()), "escaped view")), "bind escaped");
  };
  bind_inputs();
  const std::array associations{s::SourceBinding{store, take(graph.producer(gathered), "original Gather")}};
  check(s::prepare(invocation, graph, queues, associations, {.host_workspace = staging.ref()}), "prepare slots");
  auto transaction = take(s::SlotBindings::create(layout, sources, destinations), "transaction");
  check(s::bind(invocation, graph, store, std::move(transaction)), "bind slots");
  const auto replay_mode = std::is_same_v<T, float> && !escape
      ? g::ReplayMode::kRequireCapture : g::ReplayMode::kCommands;
  check(invocation.prepare_replay(queues, {.mode = replay_mode}), "prepare original episode");
  if constexpr (std::is_same_v<T, float>) {
    if (!escape) {
      EXPECT_EQ(invocation.replay_route(), g::ReplayRoute::kCaptured);
      EXPECT_EQ(invocation.replay_report()->recorded_capture_range_count, 1u);
      EXPECT_EQ(invocation.replay_report()->reused_capture_range_count, 0u);
    }
  }
  auto completion = take(invocation.submit(queues), "submit");
  EXPECT_FALSE(s::take(invocation, store)); // The original Completion is still retained.
  check(completion.wait(), "wait"); completion = {};
  const auto actual = half_test::read<float>(output);
  ASSERT_EQ(actual.size(), reference.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (std::isnan(reference[i])) EXPECT_TRUE(std::isnan(actual[i])) << i;
    else EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]), std::bit_cast<std::uint32_t>(reference[i])) << i;
  }
  if (escape) expect_bytes(escaped.view(), std::as_bytes(std::span(initial)));
  auto returned = take(s::take(invocation, store), "retired take");
  ASSERT_EQ(returned.destination_count(), kB * prefixes);
  std::vector<c::ReadLease> written(kB * prefixes);
  auto expected = before;
  for (std::size_t b = 0; b < kB; ++b) for (std::size_t p = 0; p < prefixes; ++p) {
    const auto row = b * prefixes + p;
    const auto start = tokens - prefixes + 1 + p;
    const auto selected = std::as_bytes(std::span(logical).subspan((b * (kL + tokens) + start) * kC, kL * kC));
    written[row] = take(returned.destinations()[row].publish(), "publish history");
    expect_bytes(take(written[row].view(), "history bytes"), selected);
    std::copy(selected.begin(), selected.end(), expected.begin() + kDestinations[row] * stride);
  }
  // This covers both source slots, every untouched slot, all padding and tail.
  expect_bytes(diagnostic, expected);
  expect_bytes(chunk_storage.view(), std::as_bytes(std::span(chunk)));
  expect_bytes(weight_storage.view(), std::as_bytes(std::span(weights)));
  if constexpr (std::is_same_v<T, float>) {
    if (!escape) {
      std::vector<c::WriteLease> next(written.size());
      for (std::size_t i = 0; i < next.size(); ++i) next[i] = take(written[i].try_write(), "reuse exact history writer");
      ASSERT_EQ(next.size(), 2u);
      std::swap(next[0], next[1]);
      auto second = take(s::SlotBindings::create(layout, sources, next), "second transaction");
      check(s::bind(invocation, graph, store, std::move(second)), "second binding");
      check(invocation.prepare_replay(queues, {.mode = g::ReplayMode::kRequireCapture}), "prepare exchanged destination IDs");
      EXPECT_EQ(invocation.replay_route(), g::ReplayRoute::kCaptured);
      EXPECT_EQ(invocation.replay_report()->recorded_capture_range_count, 0u);
      EXPECT_EQ(invocation.replay_report()->reused_capture_range_count, 1u);
      auto second_completion = take(invocation.submit(queues), "submit exchanged destination IDs");
      EXPECT_FALSE(s::take(invocation, store));
      check(second_completion.wait(), "wait exchanged destination IDs"); second_completion = {};
      const auto second_actual = half_test::read<float>(output);
      ASSERT_EQ(second_actual.size(), reference.size());
      for (std::size_t i = 0; i < second_actual.size(); ++i) {
        if (std::isnan(reference[i])) EXPECT_TRUE(std::isnan(second_actual[i])) << i;
        else EXPECT_EQ(std::bit_cast<std::uint32_t>(second_actual[i]), std::bit_cast<std::uint32_t>(reference[i])) << i;
      }
      auto second_returned = take(s::take(invocation, store), "exchanged destination take");
      ASSERT_EQ(second_returned.destination_count(), 2u);
      for (std::size_t b = 0; b < kB; ++b) {
        const auto destination = kDestinations[1 - b];
        EXPECT_EQ(take(layout.locate(second_returned.destinations()[b]), "exchanged destination address").index, destination);
        const auto selected = std::as_bytes(std::span(logical).subspan((b * (kL + tokens) + tokens) * kC, kL * kC));
        written[b] = take(second_returned.destinations()[b].publish(), "publish exchanged history");
        expect_bytes(take(written[b].view(), "exchanged history bytes"), selected);
        std::copy(selected.begin(), selected.end(), expected.begin() + destination * stride);
      }
      expect_bytes(diagnostic, expected);
      expect_bytes(chunk_storage.view(), std::as_bytes(std::span(chunk)));
      expect_bytes(weight_storage.view(), std::as_bytes(std::span(weights)));
      std::vector<c::WriteLease> third_writers(written.size());
      for (std::size_t i = 0; i < third_writers.size(); ++i)
        third_writers[i] = take(written[i].try_write(), "reuse exchanged history writer");
      auto third = take(s::SlotBindings::create(layout, sources, third_writers), "third transaction");
      check(s::bind(invocation, graph, store, std::move(third)), "third binding");
      check(invocation.prepare_replay(queues, {.mode = g::ReplayMode::kRequireCapture}), "legal third preparation");
      EXPECT_EQ(invocation.replay_route(), g::ReplayRoute::kCaptured);
      check(invocation.reset(), "reset original authority"); bind_inputs();
      EXPECT_FALSE(invocation.prepare_replay(queues, {.mode = g::ReplayMode::kRequireCapture}).ok());
      auto recovered = take(s::take(invocation, store), "reset take");
      EXPECT_EQ(recovered.destination_count(), kB * prefixes);
      for (std::size_t i = 0; i < recovered.destination_count(); ++i)
        EXPECT_EQ(take(layout.locate(recovered.destinations()[i]), "recovered address").index, kDestinations[1 - i]);
      expect_bytes(diagnostic, expected);
    }
  }
}

TEST(GgmlCudaIndexedConvGraph, Float32BatchedFinalWindowUsesOriginalStateAndReset) {
  exercise<float>(1, 1, false);
}
TEST(GgmlCudaIndexedConvGraph, Bfloat16LongChunkWritesLastPrefixesAndKeepsPadding) {
  exercise<BF16>(35, 3, true);
}
TEST(GgmlCudaIndexedConvGraph, Float16HistoryPreservesOriginalBitsForEveryPrefix) {
  exercise<F16>(2, 2, true);
}
TEST(GgmlCudaIndexedConvGraph, EscapingGatherKeepsLegalProjectedFallback) {
  exercise<BF16>(2, 2, true, true);
}
} // namespace
