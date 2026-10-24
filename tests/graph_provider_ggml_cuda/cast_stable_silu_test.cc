#include "hyle/graph/cuda.h"
#include "hyle/compute/providers/cuda.h"
#include "../compute_provider_ggml_cuda/tensor_test_support.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
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

// Each original operation has its own Direct preparation and submission. The
// BF16 SiLU must remain the ordinary default SGLang operation, not the new pair.
template <class Op>
bool direct(c::Context &context, hyle::mem::MemoryResourceRef resource,
    Op descriptor, c::ConstTensorView input, c::TensorView output) {
  const c::Operation operation(descriptor);
  auto prepared = c::prepare(context, operation, std::span(&input, 1), std::span(&output, 1));
  EXPECT_TRUE(prepared) << (prepared ? "" : prepared.error().message());
  if (!prepared) return false;
  if (operation.kind() == c::OperationKind::kUnary)
    EXPECT_EQ(prepared->provider().id, c::cuda_native::kSglangProviderId);
  auto workspace = ggml_cuda_test::workspace_buffer(*prepared, resource);
  auto queue = context.create_queue().value();
  auto invocation = prepared->create_invocation(workspace.view()).value();
  auto done = invocation.submit(queue, std::span(&input, 1), std::span(&output, 1));
  EXPECT_TRUE(done) << (done ? "" : done.error().message());
  if (!done) return false;
  const auto status = done->wait();
  EXPECT_TRUE(status.ok()) << status.message();
  return status.ok();
}
struct Reference final { std::vector<BF16> rounded, activated; };
Reference reference(c::Context &context, hyle::mem::MemoryResourceRef resource,
    const std::vector<float> &host, bool require_rounding_witness) {
  const auto f32 = dense(c::DataType::kFloat32, {1, std::int64_t(host.size())});
  const auto bf16 = dense(c::DataType::kBFloat16, {1, std::int64_t(host.size())});
  auto source = upload<float>(resource, host), rounded = output_buffer(resource, bf16);
  auto activated = output_buffer(resource, bf16);
  const auto x = c::ConstTensorView::create(f32, source.view()).value();
  if (!direct(context, resource, c::Cast{c::DataType::kBFloat16}, x,
      c::TensorView::create(bf16, rounded.view()).value())) return {};
  if (!direct(context, resource, c::Unary{c::UnaryKind::kSilu},
      c::ConstTensorView::create(bf16, rounded.view()).value(),
      c::TensorView::create(bf16, activated.view()).value())) return {};
  Reference result{half_test::read<BF16>(rounded), half_test::read<BF16>(activated)};
  if (require_rounding_witness) {
    auto full = output_buffer(resource, f32), once = output_buffer(resource, bf16);
    if (!direct(context, resource, c::Unary{c::UnaryKind::kSilu}, x,
        c::TensorView::create(f32, full.view()).value())) return {};
    if (!direct(context, resource, c::Cast{c::DataType::kBFloat16},
        c::ConstTensorView::create(f32, full.view()).value(),
        c::TensorView::create(bf16, once.view()).value())) return {};
    const auto last_round_only = half_test::read<BF16>(once);
    std::size_t witnesses = 0;
    for (std::size_t i = 0; i < host.size(); ++i)
      if (std::isfinite(half_test::decode(result.activated[i])) &&
          std::isfinite(half_test::decode(last_round_only[i])))
        witnesses += result.activated[i].bits != last_round_only[i].bits;
    EXPECT_GT(witnesses, 0U);
  }
  return result;
}
void check_bfloat(const Buffer &buffer, const std::vector<BF16> &expected) {
  const auto actual = half_test::read<BF16>(buffer);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (std::isnan(half_test::decode(expected[i])))
      EXPECT_TRUE(std::isnan(half_test::decode(actual[i]))) << i;
    else EXPECT_EQ(actual[i].bits, expected[i].bits) << i;
}
void check_float_storage(const Buffer &buffer, const std::vector<float> &expected) {
  const auto actual = half_test::read<float>(buffer);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),
              std::bit_cast<std::uint32_t>(expected[i])) << i;
}
std::vector<float> exhaustive() {
  std::vector<float> result;
  result.reserve(65536);
  for (std::uint32_t bits = 0; bits < 65536; ++bits)
    result.push_back(std::bit_cast<float>(bits << 16));
  return result;
}
std::vector<float> neighbors() {
  std::vector<float> result;
  // Both sides of BF16 halfway values, including carry/overflow and signed
  // underflow, exercise the first RNE separately from the final SiLU RNE.
  for (const auto base : {0x0000U, 0x007fU, 0x0080U, 0x3d00U, 0x3e7fU,
                          0x3f00U, 0x3f7fU, 0x4000U, 0x40ffU, 0x42c7U,
                          0x7f7fU, 0x8000U, 0x807fU, 0xbd00U, 0xbe7fU,
                          0xbf00U, 0xbf7fU, 0xc000U, 0xc0ffU, 0xc2c7U}) {
    const auto middle = std::bit_cast<float>((base << 16) | 0x8000U);
    result.push_back(std::nextafter(middle, -std::numeric_limits<float>::infinity()));
    result.push_back(middle);
    result.push_back(std::nextafter(middle, std::numeric_limits<float>::infinity()));
  }
  for (const auto bits : {0U, 0x80000000U, 1U, 0x007fffffU, 0x00800000U,
                          0x7f7fffffU, 0xff7fffffU, 0x7f800000U, 0xff800000U,
                          0x7fc01234U, 0x7f801234U})
    result.push_back(std::bit_cast<float>(bits));
  for (const float value : {-100.F, -96.F, -92.F, -88.F, -84.F, -80.F})
    result.push_back(value);
  while (result.size() % 32) result.push_back(0.125F);
  return result;
}

// escape: 0 private Cast; 1 published Cast; 2 extra Copy reader;
// 3 metadata View reader; 4 a control consumer outside the pair.
void exercise(const std::vector<float> &corpus, int escape,
              bool rounding_witness = false, bool failed_prepare = false) {
  auto context = g::cuda::create_context().value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  const auto resource = memory.ref();
  const auto f32 = dense(c::DataType::kFloat32, {1, std::int64_t(corpus.size())});
  const auto bf16 = dense(c::DataType::kBFloat16, {1, std::int64_t(corpus.size())});
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(f32).value();
  const auto cast_node = builder.add_node(c::Operation(c::Cast{c::DataType::kBFloat16}),
      std::array{x}).value();
  const auto rounded = builder.result(cast_node).value();
  const auto y = builder.add(c::Unary{c::UnaryKind::kSilu}, rounded).value();
  const auto copy_node = builder.add_node(c::Operation(c::Copy{}), std::array{x}).value();
  const auto source_copy = builder.result(copy_node).value();
  ASSERT_TRUE(builder.publish(x).ok()); // A source escape is legal for the pair.
  ASSERT_TRUE(builder.publish(y).ok());
  ASSERT_TRUE(builder.publish(source_copy).ok());
  g::ValueId extra;
  if (escape == 1) extra = rounded;
  if (escape == 2) extra = builder.add(c::Copy{}, rounded).value();
  if (escape == 3) {
    const auto shape = c::TensorShape::create(std::array<std::int64_t, 1>{
        std::int64_t(corpus.size())}).value();
    const auto view = builder.add(c::View::reshape(shape).value(), rounded).value();
    extra = builder.add(c::Copy{}, view).value();
  }
  if (escape > 0 && escape < 4) ASSERT_TRUE(builder.publish(extra).ok());
  if (escape == 4)
    ASSERT_TRUE(builder.depends_on(copy_node, cast_node).ok());
  const auto graph = std::move(builder).finish().value();
  auto compiled = g::cuda::compile(graph, context, resource);
  ASSERT_TRUE(compiled) << compiled.error().message();
  bool pair = false;
  for (std::size_t i = 0; i < compiled->report.step_count(); ++i) {
    const auto step = compiled->report.step(i);
    ASSERT_TRUE(step);
    if (step->implementation != c::ProviderImplementationId{5}) continue;
    pair = true;
    EXPECT_EQ(step->kind, g::CompileStepKind::kProviderRegion);
    EXPECT_EQ(step->policy, (c::RoutePolicyId{0x48594c4547474355ULL, 4, 1}));
    ASSERT_EQ(step->node_count, 2U);
    EXPECT_EQ(compiled->report.step_node(i, 0).value(), graph.producer(rounded).value());
    EXPECT_EQ(compiled->report.step_node(i, 1).value(), graph.producer(y).value());
  }
  EXPECT_EQ(pair, escape == 0);
  struct Run final {
    std::vector<float> host;
    Reference expected;
    Buffer input, output, copy, extra;
  };
  // Backings outlive both Queues, Invocations, captures and old-result checks.
  std::array<std::array<Run, 2>, 2> runs;
  std::array queues{context.create_queue().value(), context.create_queue().value()};
  std::array slots{compiled->executable.create_invocation().value(),
                   compiled->executable.create_invocation().value()};
  const auto check = [&](const Run &run) {
    check_bfloat(run.output, run.expected.activated);
    check_float_storage(run.input, run.host);
    check_float_storage(run.copy, run.host);
    if (escape > 0 && escape < 4) check_bfloat(run.extra, run.expected.rounded);
  };
  for (std::size_t round = 0; round < 2; ++round) {
    std::array<c::Completion, 2> done;
    for (std::size_t slot = 0; slot < slots.size(); ++slot) {
      auto &run = runs[round][slot];
      run.host = corpus;
      std::rotate(run.host.begin(), run.host.begin() +
          (round * 7 + slot * 11) % run.host.size(), run.host.end());
      run.expected = reference(context, resource, run.host, rounding_witness);
      ASSERT_EQ(run.expected.activated.size(), run.host.size());
      run.input = upload<float>(resource, run.host);
      run.output = output_buffer(resource, bf16);
      run.copy = output_buffer(resource, f32);
      if (escape > 0 && escape < 4) run.extra = output_buffer(resource, bf16);
      if (round) {
        ASSERT_TRUE(slots[slot].reset().ok());
        EXPECT_NE(run.input.native_handle(), runs[0][slot].input.native_handle());
        EXPECT_NE(run.output.native_handle(), runs[0][slot].output.native_handle());
      }
      ASSERT_TRUE(slots[slot].bind(x, c::ConstTensorView::create(f32, run.input.view()).value()).ok());
      ASSERT_TRUE(slots[slot].bind(source_copy, c::TensorView::create(f32, run.copy.view()).value()).ok());
      const std::array bindings{g::QueueBinding{g::ExecutionLaneId{0}, &queues[slot]}};
      if (failed_prepare && !round) {
        EXPECT_FALSE(slots[slot].prepare_replay(bindings,
            {.mode = g::ReplayMode::kRequireCapture}).ok());
        EXPECT_EQ(slots[slot].replay_route(), g::ReplayRoute::kUnprepared);
        EXPECT_FALSE(slots[slot].replay_report());
        EXPECT_FALSE(slots[slot].submit(queues[slot]));
      }
      ASSERT_TRUE(slots[slot].bind(y, c::TensorView::create(bf16, run.output.view()).value()).ok());
      if (escape > 0 && escape < 4) {
        const auto desc = graph.value_description(extra).value();
        ASSERT_TRUE(slots[slot].bind(extra, c::TensorView::create(desc, run.extra.view()).value()).ok());
      }
      const auto status = slots[slot].prepare_replay(bindings, {.mode = g::ReplayMode::kRequireCapture});
      ASSERT_TRUE(status.ok()) << status.message();
      EXPECT_EQ(slots[slot].replay_route(), g::ReplayRoute::kCaptured);
    }
    for (std::size_t slot = 0; slot < slots.size(); ++slot) {
      auto result = slots[slot].submit(queues[slot]);
      ASSERT_TRUE(result) << result.error().message(); done[slot] = std::move(*result);
    }
    ASSERT_TRUE(done[1].wait().ok()); ASSERT_TRUE(done[0].wait().ok());
    for (std::size_t slot = 0; slot < slots.size(); ++slot) {
      check(runs[round][slot]);
      if (round) check(runs[0][slot]);
      {
        auto old_result = slots[slot].output(y);
        ASSERT_TRUE(old_result) << old_result.error().message();
        // output() is a borrowed view of this caller-owned backing, not an
        // Invocation workspace lease. Reset revokes readiness while the
        // caller keeps the old bytes and view valid through the next bind.
        ASSERT_TRUE(slots[slot].reset().ok());
        EXPECT_EQ(slots[slot].replay_route(), g::ReplayRoute::kUnprepared);
        EXPECT_FALSE(slots[slot].output(y));
        std::vector<BF16> old_bytes(runs[round][slot].expected.activated.size());
        ASSERT_TRUE(old_result->storage().read(std::as_writable_bytes(std::span(old_bytes))).ok());
        const auto &expected = runs[round][slot].expected.activated;
        for (std::size_t i = 0; i < old_bytes.size(); ++i)
          if (std::isnan(half_test::decode(expected[i])))
            EXPECT_TRUE(std::isnan(half_test::decode(old_bytes[i]))) << i;
          else EXPECT_EQ(old_bytes[i].bits, expected[i].bits) << i;
      }
    }
  }
}

TEST(GgmlCudaCastStableSilu, ExhaustiveBfloatDomainMatchesOriginalDirectStages) {
  exercise(exhaustive(), 0);
}
TEST(GgmlCudaCastStableSilu, RoundingNeighborsRetainBothBfloatBoundaries) {
  exercise(neighbors(), 0, true);
}
TEST(GgmlCudaCastStableSilu, PublishedCastAndExtraReadersKeepIntermediate) {
  exercise(neighbors(), 1); exercise(neighbors(), 2);
}
TEST(GgmlCudaCastStableSilu, ViewAndControlEscapesRejectPrivatePair) {
  exercise(neighbors(), 3); exercise(neighbors(), 4);
}
TEST(GgmlCudaCastStableSilu, FailedPreparationCanRecoverWithoutRevivingOldReadiness) {
  exercise(neighbors(), 0, false, true);
}
} // namespace
