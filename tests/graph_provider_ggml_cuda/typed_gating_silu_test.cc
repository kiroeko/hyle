#include "hyle/graph/cuda.h"
#include "hyle/graph/providers/ggml_cuda.h"
#include "../compute_provider_ggml_cuda/tensor_test_support.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
using BF16 = hyle::simd::bfloat16_t;
using hyle::mem::Buffer;
using ggml_cuda_tensor_test::dense;
using ggml_cuda_tensor_test::output_buffer;
using ggml_cuda_tensor_test::upload;
constexpr c::RoutePolicyId kPolicy{0x48594c4547474355ULL, 2, 1};

template <class Op>
bool direct(c::Context &context, hyle::mem::MemoryResourceRef resource,
    Op descriptor, std::span<const c::ConstTensorView> inputs, c::TensorView output) {
  auto prepared = c::prepare(context, c::Operation(descriptor), inputs,
      std::span(&output, 1));
  EXPECT_TRUE(prepared) << (prepared ? "" : prepared.error().message());
  if (!prepared) return false;
  EXPECT_EQ(prepared->provider().id, c::ggml_cuda::kProviderId);
  auto workspace = ggml_cuda_test::workspace_buffer(*prepared, resource);
  auto invocation = prepared->create_invocation(workspace.view()).value();
  auto queue = context.create_queue().value();
  auto done = invocation.submit(queue, inputs, std::span(&output, 1));
  EXPECT_TRUE(done) << (done ? "" : done.error().message());
  if (!done) return false;
  const auto status = done->wait();
  EXPECT_TRUE(status.ok()) << status.message();
  return status.ok();
}

struct Corpus final {
  std::vector<BF16> gate;
  std::vector<float> other;
  std::vector<BF16> known_final;
};
Corpus exhaustive() {
  Corpus result;
  for (std::uint32_t bits = 0; bits < 65536; ++bits) {
    result.gate.push_back(BF16::from_bits(static_cast<std::uint16_t>(bits)));
    result.other.push_back(std::bit_cast<float>(0x3f800001U + (bits % 127) * 257U));
  }
  const std::array special{0U, 0x80000000U, 1U, 0x007fffffU, 0x00800000U,
      0x7f7fffffU, 0xff7fffffU, 0x7f800000U, 0xff800000U, 0x7fc01234U};
  for (std::size_t i = 0; i < special.size(); ++i)
    result.other[i * 257] = std::bit_cast<float>(special[i]);
  return result;
}
Corpus rounding_neighbors() {
  Corpus result;
  // SiLU(128) is exactly128 in the original F32 stages. Powers-of-two scaling
  // lets an independent encoder check both sides of final BF16 halfway ties.
  for (const auto base : {0x3e7fU, 0x3f00U,
                          0x3f7fU, 0x4000U, 0x407fU, 0x7f7fU,
                          0xbe7fU, 0xbf00U, 0xbf7fU}) {
    const auto halfway = std::bit_cast<float>((base << 16) | 0x8000U);
    for (const auto target : {std::nextafter(halfway, -INFINITY), halfway,
                              std::nextafter(halfway, INFINITY)}) {
      const float other = target / 128.F;
      result.gate.push_back(BF16::from_bits(0x4300));
      result.other.push_back(other);
      // This independent host check uses only normal F32 products. The full
      // BF16/subnormal domain is compared with the original GPU pair above.
      result.known_final.push_back(half_test::encode<BF16>(float(128.F * other)));
    }
  }
  while (result.gate.size() % 32) {
    result.gate.push_back(BF16::from_bits(0x4300));
    result.other.push_back(0.125F);
    result.known_final.push_back(half_test::encode<BF16>(16.F));
  }
  return result;
}
Corpus small() {
  Corpus result;
  for (std::size_t i = 0; i < 256; ++i) {
    result.gate.push_back(half_test::encode<BF16>((int(i % 33) - 16) / 8.0));
    result.other.push_back(std::bit_cast<float>(0x3f800001U + std::uint32_t(i) * 257U));
  }
  return result;
}

template <class Value>
void check(const Buffer &buffer, const std::vector<Value> &expected) {
  const auto actual = half_test::read<Value>(buffer);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (std::isnan(half_test::decode(expected[i]))) {
      EXPECT_TRUE(std::isnan(half_test::decode(actual[i]))) << i;
    } else if constexpr (std::is_same_v<Value, float>) {
      EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),
          std::bit_cast<std::uint32_t>(expected[i])) << i;
    } else {
      EXPECT_EQ(actual[i].bits, expected[i].bits) << i;
    }
  }
}
template <class Value>
void unchanged(const Buffer &buffer, const std::vector<Value> &expected) {
  const auto actual = half_test::read<Value>(buffer);
  ASSERT_EQ(actual.size(), expected.size());
  EXPECT_EQ(std::memcmp(actual.data(), expected.data(), expected.size() * sizeof(Value)), 0);
}

struct Reference final {
  std::vector<float> widened, activated, product, standalone_product;
  std::vector<BF16> output, standalone_output;
};
Reference reference(const Corpus &corpus, bool reverse) {
  // The middle reference contains ONLY the original F32 SiLU+Mul route2;
  // the new typed four-node candidate cannot match. Its surrounding casts are
  // separate original Direct invocations. This preserves the old physical
  // strict-unary product even if a standalone Binary TU has different FTZ.
  auto context = c::Context::create(c::cuda::device(0).value(), g::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  const auto f32 = dense(c::DataType::kFloat32, {1, std::int64_t(corpus.gate.size())});
  const auto bf16 = dense(c::DataType::kBFloat16, {1, std::int64_t(corpus.gate.size())});
  auto source = upload<BF16>(resource, corpus.gate), other = upload<float>(resource, corpus.other);
  auto widened = output_buffer(resource, f32), activated = output_buffer(resource, f32);
  auto product = output_buffer(resource, f32), separate = output_buffer(resource, f32);
  auto output = output_buffer(resource, bf16), separate_output = output_buffer(resource, bf16);
  const std::array cast_input{c::ConstTensorView::create(bf16, source.view()).value()};
  if (!direct(context, resource, c::Cast{c::DataType::kFloat32}, cast_input,
      c::TensorView::create(f32, widened.view()).value())) return {};
  const auto wide = c::ConstTensorView::create(f32, widened.view()).value();
  const auto rhs = c::ConstTensorView::create(f32, other.view()).value();
  const std::array unary_input{wide};
  if (!direct(context, resource, c::Unary{c::UnaryKind::kSilu}, unary_input,
      c::TensorView::create(f32, activated.view()).value())) return {};
  const auto activation = c::ConstTensorView::create(f32, activated.view()).value();
  const std::array mul_inputs{reverse ? rhs : activation, reverse ? activation : rhs};
  if (!direct(context, resource, c::Binary{c::BinaryKind::kMultiply}, mul_inputs,
      c::TensorView::create(f32, separate.view()).value())) return {};
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(f32).value(), y = builder.input(f32).value();
  const auto silu = builder.add(c::Unary{c::UnaryKind::kSilu}, x).value();
  const auto z = builder.add(c::Binary{c::BinaryKind::kMultiply},
      reverse ? y : silu, reverse ? silu : y).value();
  EXPECT_TRUE(builder.publish(z).ok());
  const auto graph = std::move(builder).finish().value();
  auto compiled = g::cuda::compile(graph, context, resource).value();
  bool original_pair = false;
  for (std::size_t i = 0; i < compiled.report.step_count(); ++i) {
    const auto step = compiled.report.step(i).value();
    original_pair |= step.policy == kPolicy && step.implementation == c::ProviderImplementationId{3}
        && step.node_count == 2;
  }
  EXPECT_TRUE(original_pair);
  auto run = compiled.executable.create_invocation().value();
  EXPECT_TRUE(run.bind(x, wide).ok());
  EXPECT_TRUE(run.bind(y, rhs).ok());
  EXPECT_TRUE(run.bind(z, c::TensorView::create(f32, product.view()).value()).ok());
  auto queue = context.create_queue().value();
  auto done = run.submit(queue);
  EXPECT_TRUE(done) << (done ? "" : done.error().message());
  if (!done || !done->wait().ok()) return {};
  const std::array final_input{c::ConstTensorView::create(f32, product.view()).value()};
  const std::array separate_input{c::ConstTensorView::create(f32, separate.view()).value()};
  if (!direct(context, resource, c::Cast{c::DataType::kBFloat16}, final_input,
      c::TensorView::create(bf16, output.view()).value())) return {};
  if (!direct(context, resource, c::Cast{c::DataType::kBFloat16}, separate_input,
      c::TensorView::create(bf16, separate_output.view()).value())) return {};
  return {half_test::read<float>(widened), half_test::read<float>(activated),
      half_test::read<float>(product), half_test::read<float>(separate),
      half_test::read<BF16>(output), half_test::read<BF16>(separate_output)};
}

enum class Escape { kNone, kWiden, kActivation, kProduct, kViewReader, kControl };
void exercise(Corpus corpus, bool reverse = false, bool metadata_view = false,
    Escape escape = Escape::kNone, bool lifecycle = false, bool precision_witness = false) {
  auto context = g::cuda::create_context().value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  const auto count = std::int64_t(corpus.gate.size());
  const auto f32 = dense(c::DataType::kFloat32, {1, count});
  const auto bf16 = dense(c::DataType::kBFloat16, {1, count});
  const auto source_desc = metadata_view ? dense(c::DataType::kBFloat16, {count}) : bf16;
  auto builder = g::GraphBuilder::create().value();
  const auto source = builder.input(source_desc).value(), other = builder.input(f32).value();
  auto input = source;
  if (metadata_view)
    input = builder.add(c::View::reshape(c::TensorShape::create(bf16.dimensions()).value()).value(), source).value();
  const auto wide = builder.add(c::Cast{c::DataType::kFloat32}, input).value();
  const auto silu_node = builder.add_node(c::Operation(c::Unary{c::UnaryKind::kSilu}),
      std::array{wide}).value();
  const auto activated = builder.result(silu_node).value();
  const auto product = builder.add(c::Binary{c::BinaryKind::kMultiply},
      reverse ? other : activated, reverse ? activated : other).value();
  const auto result = builder.add(c::Cast{c::DataType::kBFloat16}, product).value();
  const auto copy_node = builder.add_node(c::Operation(c::Copy{}), std::array{source}).value();
  const auto copy = builder.result(copy_node).value();
  EXPECT_TRUE(builder.publish(source).ok()); // External source readers are legal.
  EXPECT_TRUE(builder.publish(copy).ok());
  EXPECT_TRUE(builder.publish(result).ok());
  g::ValueId extra;
  if (escape == Escape::kWiden) extra = wide;
  if (escape == Escape::kActivation) extra = activated;
  if (escape == Escape::kProduct) extra = product;
  if (escape == Escape::kViewReader) {
    const auto view = builder.add(c::View::reshape(c::TensorShape::create(
        std::array<std::int64_t, 1>{count}).value()).value(), wide).value();
    extra = builder.add(c::Copy{}, view).value();
  }
  if (extra.valid()) EXPECT_TRUE(builder.publish(extra).ok());
  if (escape == Escape::kControl)
    EXPECT_TRUE(builder.depends_on(copy_node, silu_node).ok());
  const auto graph = std::move(builder).finish().value();
  auto compiled = g::cuda::compile(graph, context, resource);
  ASSERT_TRUE(compiled) << compiled.error().message();
  const std::array math{graph.producer(wide).value(), graph.producer(activated).value(),
      graph.producer(product).value(), graph.producer(result).value()};
  bool joined = false;
  for (std::size_t i = 0; i < compiled->report.step_count(); ++i) {
    const auto step = compiled->report.step(i).value();
    if (step.policy != kPolicy || step.implementation != c::ProviderImplementationId{3}) continue;
    bool all = true;
    for (const auto node : math) {
      bool contains = false;
      for (std::size_t n = 0; n < step.node_count; ++n)
        contains |= compiled->report.step_node(i, n).value() == node;
      all &= contains;
    }
    if (!all) continue;
    joined = true;
    EXPECT_EQ(step.kind, g::CompileStepKind::kProviderRegion);
    EXPECT_EQ(step.policy, kPolicy);
    EXPECT_EQ(step.implementation, c::ProviderImplementationId{3});
    EXPECT_EQ(step.node_count, 4U); // The input metadata View stays outside.
  }
  EXPECT_EQ(joined, escape == Escape::kNone);
  struct Run final { Corpus host; Reference expected; Buffer gate, other, output, copy, extra; };
  std::array<std::array<Run, 2>, 2> runs;
  std::array queues{context.create_queue().value(), context.create_queue().value()};
  std::array slots{compiled->executable.create_invocation().value(),
                   compiled->executable.create_invocation().value()};
  const std::size_t rounds = lifecycle ? 2 : 1, slot_count = lifecycle ? 2 : 1;
  const bool standalone = escape == Escape::kActivation || escape == Escape::kControl;
  const auto check_run = [&](const Run &run) {
    check(run.output, standalone ? run.expected.standalone_output : run.expected.output);
    unchanged(run.gate, run.host.gate); unchanged(run.other, run.host.other);
    unchanged(run.copy, run.host.gate);
    if (extra.valid()) {
      const auto &expected = escape == Escape::kActivation ? run.expected.activated
          : escape == Escape::kProduct ? run.expected.product : run.expected.widened;
      check(run.extra, expected);
    }
  };
  for (std::size_t round = 0; round < rounds; ++round) {
    std::array<c::Completion, 2> done;
    for (std::size_t slot = 0; slot < slot_count; ++slot) {
      auto &run = runs[round][slot]; run.host = corpus;
      const auto shift = (round * 17 + slot * 7) % run.host.gate.size();
      std::rotate(run.host.gate.begin(), run.host.gate.begin() + shift, run.host.gate.end());
      std::rotate(run.host.other.begin(), run.host.other.begin() + shift, run.host.other.end());
      if (!run.host.known_final.empty())
        std::rotate(run.host.known_final.begin(), run.host.known_final.begin() + shift, run.host.known_final.end());
      run.expected = reference(run.host, reverse);
      ASSERT_EQ(run.expected.output.size(), run.host.gate.size());
      if (!run.host.known_final.empty())
        for (std::size_t i = 0; i < run.expected.output.size(); ++i)
          EXPECT_EQ(run.expected.output[i].bits, run.host.known_final[i].bits) << i;
      if (precision_witness) {
        std::size_t witnesses = 0;
        for (std::size_t i = 0; i < run.expected.activated.size(); ++i) {
          const auto value = run.expected.activated[i];
          if (!std::isfinite(value) || std::abs(value) < 1.e-4F || std::abs(value) > 1.e4F) continue;
          const auto early = half_test::encode<BF16>(value);
          const float wrong = float(half_test::decode(early)) * run.host.other[i];
          if (std::isfinite(wrong))
            witnesses += half_test::encode<BF16>(wrong).bits != run.expected.output[i].bits;
        }
        EXPECT_GT(witnesses, 0U); // A BF16 SiLU intermediate is observably wrong.
      }
      run.gate = upload<BF16>(resource, run.host.gate);
      run.other = upload<float>(resource, run.host.other);
      run.output = output_buffer(resource, bf16);
      run.copy = output_buffer(resource, source_desc);
      if (extra.valid()) run.extra = output_buffer(resource, graph.value_description(extra).value());
      if (round) {
        EXPECT_NE(run.gate.native_handle(), runs[0][slot].gate.native_handle());
        EXPECT_NE(run.output.native_handle(), runs[0][slot].output.native_handle());
      }
      ASSERT_TRUE(slots[slot].bind(source, c::ConstTensorView::create(source_desc, run.gate.view()).value()).ok());
      ASSERT_TRUE(slots[slot].bind(other, c::ConstTensorView::create(f32, run.other.view()).value()).ok());
      ASSERT_TRUE(slots[slot].bind(copy, c::TensorView::create(source_desc, run.copy.view()).value()).ok());
      const std::array bindings{g::QueueBinding{g::ExecutionLaneId{0}, &queues[slot]}};
      if (lifecycle && round == 0) {
        EXPECT_FALSE(slots[slot].prepare_replay(bindings, {.mode = g::ReplayMode::kRequireCapture}).ok());
        EXPECT_EQ(slots[slot].replay_route(), g::ReplayRoute::kUnprepared);
        EXPECT_FALSE(slots[slot].submit(queues[slot]));
      }
      ASSERT_TRUE(slots[slot].bind(result, c::TensorView::create(bf16, run.output.view()).value()).ok());
      if (extra.valid()) ASSERT_TRUE(slots[slot].bind(extra, c::TensorView::create(
          graph.value_description(extra).value(), run.extra.view()).value()).ok());
      const auto status = slots[slot].prepare_replay(bindings, {.mode = g::ReplayMode::kRequireCapture});
      ASSERT_TRUE(status.ok()) << status.message();
      EXPECT_EQ(slots[slot].replay_route(), g::ReplayRoute::kCaptured);
      auto completion = slots[slot].submit(queues[slot]);
      ASSERT_TRUE(completion) << completion.error().message(); done[slot] = std::move(*completion);
    }
    for (std::size_t slot = slot_count; slot-- > 0;) ASSERT_TRUE(done[slot].wait().ok());
    for (std::size_t slot = 0; slot < slot_count; ++slot) {
      check_run(runs[round][slot]);
      if (round) check_run(runs[0][slot]);
      auto held = slots[slot].result(result).value(); // Actual GraphOutput lease.
      EXPECT_TRUE(slots[slot].reset().IsUnavailable());
      EXPECT_TRUE(slots[slot].bind(other, c::ConstTensorView::create(f32, runs[round][slot].other.view()).value()).IsUnavailable());
      held.reset();
      ASSERT_TRUE(slots[slot].reset().ok());
      EXPECT_EQ(slots[slot].replay_route(), g::ReplayRoute::kUnprepared);
      EXPECT_FALSE(slots[slot].output(result));
      check_run(runs[round][slot]); // Caller backing remains independently alive.
    }
  }
}

TEST(GgmlCudaTypedGatingSilu, ExhaustiveBfloatInputsKeepF32StagesAndOriginalNonfiniteClasses) {
  exercise(exhaustive(), false, false, Escape::kNone, false, true);
}
TEST(GgmlCudaTypedGatingSilu, ReverseMultiplyAndExternalMetadataViewKeepFinalRoundingNeighbors) {
  exercise(rounding_neighbors(), true, true);
}
TEST(GgmlCudaTypedGatingSilu, CapturedSlotsRebindAfterFailedPreparationAndRespectRealResultLeases) {
  exercise(small(), false, true, Escape::kNone, true);
}
TEST(GgmlCudaTypedGatingSilu, PublishedAndViewReadIntermediatesKeepOriginalDecomposition) {
  for (const auto escape : {Escape::kWiden, Escape::kActivation, Escape::kProduct, Escape::kViewReader}) {
    SCOPED_TRACE(static_cast<int>(escape)); exercise(small(), false, false, escape);
  }
}
TEST(GgmlCudaTypedGatingSilu, ExternalControlReaderKeepsOriginalDecomposition) {
  exercise(small(), false, false, Escape::kControl);
}
} // namespace
