#include "hyle/graph/providers/ggml_cuda.h"
#include "hyle/graph/executable.h"
#include "hyle/compute/model_ops.h"
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
using ggml_cuda_tensor_test::execute;
using ggml_cuda_tensor_test::output_buffer;
using ggml_cuda_tensor_test::upload;

struct Shape final {
  std::int64_t tokens, channels, kernel;
  bool padded = false;
  std::int64_t window() const { return tokens + kernel - 1; }
  std::int64_t pitch() const { return channels + (padded ? 8 : 0); }
  std::int64_t rows() const { return window() + (padded ? 1 : 0); }
};
struct Samples final {
  std::vector<BF16> history, chunk, logical, physical;
  std::vector<float> weights;
};
Samples samples(Shape s, int seed) {
  Samples h;
  h.physical.assign(std::size_t(2 * s.rows() * s.pitch()), BF16::from_bits(0x7fc1));
  h.logical.resize(std::size_t(2 * s.window() * s.channels));
  h.weights.resize(std::size_t(s.channels * s.kernel));
  for (std::int64_t batch = 0; batch < 2; ++batch)
    for (std::int64_t time = 0; time < s.window(); ++time)
      for (std::int64_t channel = 0; channel < s.channels; ++channel) {
        auto x = half_test::encode<BF16>(
            double((time * 17 + channel * 5 + batch * 7 + seed) % 31 - 15) / 32);
        // Signed zeros, BF16 subnormals and nonfinite classifications. The
        // independent FP32 oracle models the original fast-math FMA/FTZ.
        if (channel == 0) x = BF16::from_bits(time % 2 ? 0x8000 : 0);
        if (channel == 1) x = BF16::from_bits(time % 2 ? 0x8001 : 0x0001);
        if (channel == 2) x = BF16::from_bits(0x7f80);
        if (channel == 3) x = BF16::from_bits(0x7fc1);
        if (channel == 4) x = BF16::from_bits(0xff80);
        h.logical[std::size_t((batch * s.window() + time) * s.channels + channel)] = x;
        h.physical[std::size_t((batch * s.rows() + time) * s.pitch() + channel)] = x;
      }
  for (std::int64_t channel = 0; channel < s.channels; ++channel)
    for (std::int64_t tap = 0; tap < s.kernel; ++tap)
      h.weights[std::size_t(channel * s.kernel + tap)] = channel < 5
          ? float(tap + 1) / 8.F
          : float((channel * 3 + tap * 7 + seed) % 19 - 9) / 64.F;
  for (std::int64_t batch = 0; batch < 2; ++batch) {
    const auto begin = h.physical.begin() + batch * s.rows() * s.pitch();
    h.history.insert(h.history.end(), begin, begin + (s.kernel - 1) * s.pitch());
    h.chunk.insert(h.chunk.end(), begin + (s.kernel - 1) * s.pitch(), begin + s.rows() * s.pitch());
  }
  return h;
}

float ftz(float x) {
  return std::fpclassify(x) == FP_SUBNORMAL ? std::copysign(0.F, x) : x;
}
struct Reference final {
  std::vector<float> cast, convolution;
  std::vector<BF16> rounded, activated, history;
};
Reference reference(c::Context &context, hyle::mem::MemoryResourceRef resource,
    Shape s, const Samples &h) {
  Reference result;
  const auto source = c::TensorDesc::strided_bytes(c::DataType::kBFloat16,
      std::array<std::int64_t, 3>{2, s.window(), s.channels},
      std::array<std::int64_t, 3>{s.rows() * s.pitch() * 2, s.pitch() * 2, 2}).value();
  const auto wide = dense(c::DataType::kFloat32, {2, s.window(), s.channels});
  const auto weights = dense(c::DataType::kFloat32, {s.channels, s.kernel});
  const auto output = dense(c::DataType::kFloat32, {2, s.tokens, s.channels});
  const auto low = dense(c::DataType::kBFloat16, {2, s.tokens, s.channels});
  auto physical = upload<BF16>(resource, h.physical);
  auto weight_buffer = upload<float>(resource, h.weights);
  auto cast_buffer = output_buffer(resource, wide);
  auto convolution = output_buffer(resource, output);
  auto rounded = output_buffer(resource, low);
  auto activated = output_buffer(resource, low);
  const auto run = [&](c::Operation operation, std::span<const c::ConstTensorView> inputs,
                       c::TensorView destination) {
    const auto status = execute(context, operation, inputs, destination);
    EXPECT_TRUE(status.ok()) << status.message();
    return status.ok();
  };
  const std::array raw{c::ConstTensorView::create(source, physical.view()).value()};
  if (!run(c::Operation(c::Cast{c::DataType::kFloat32}), raw,
      c::TensorView::create(wide, cast_buffer.view()).value())) return result;
  const std::array inputs{c::ConstTensorView::create(wide, cast_buffer.view()).value(),
      c::ConstTensorView::create(weights, weight_buffer.view()).value()};
  if (!run(c::model_ops::causal_conv1d(), inputs,
      c::TensorView::create(output, convolution.view()).value())) return result;
  const std::array full{c::ConstTensorView::create(output, convolution.view()).value()};
  if (!run(c::Operation(c::Cast{c::DataType::kBFloat16}), full,
      c::TensorView::create(low, rounded.view()).value())) return result;
  const std::array narrow{c::ConstTensorView::create(low, rounded.view()).value()};
  if (!run(c::Operation(c::Unary{c::UnaryKind::kSilu}), narrow,
      c::TensorView::create(low, activated.view()).value())) return result;
  result.cast = half_test::read<float>(cast_buffer);
  result.convolution = half_test::read<float>(convolution);
  result.rounded = half_test::read<BF16>(rounded);
  result.activated = half_test::read<BF16>(activated);
  for (std::int64_t batch = 0; batch < 2; ++batch) {
    const auto begin = h.logical.begin() + (batch * s.window() + s.tokens) * s.channels;
    result.history.insert(result.history.end(), begin, begin + (s.kernel - 1) * s.channels);
  }
  return result;
}
void check_float(const Buffer &buffer, const std::vector<float> &expected) {
  const auto actual = half_test::read<float>(buffer);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (std::isnan(expected[i])) EXPECT_TRUE(std::isnan(actual[i])) << i;
    else EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),
                   std::bit_cast<std::uint32_t>(expected[i])) << i;
}
void check_bfloat(const Buffer &buffer, const std::vector<BF16> &expected,
                  bool exact_storage = false) {
  const auto actual = half_test::read<BF16>(buffer);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (!exact_storage && std::isnan(half_test::decode(expected[i])))
      EXPECT_TRUE(std::isnan(half_test::decode(actual[i]))) << i;
    else EXPECT_EQ(actual[i].bits, expected[i].bits) << i;
}
void check_oracle(Shape s, const Samples &h, const std::vector<float> &actual) {
  ASSERT_EQ(actual.size(), std::size_t(2 * s.tokens * s.channels));
  for (std::int64_t batch = 0; batch < 2; ++batch)
    for (std::int64_t time = 0; time < s.tokens; ++time)
      for (std::int64_t channel = 0; channel < s.channels; ++channel) {
        float sum = 0.F;
        double crosscheck = 0;
        for (std::int64_t tap = 0; tap < s.kernel; ++tap) {
          const auto x = float(half_test::decode(h.logical[std::size_t(
              (batch * s.window() + time + tap) * s.channels + channel)]));
          const auto w = h.weights[std::size_t(channel * s.kernel + tap)];
          sum = ftz(std::fma(ftz(x), ftz(w), ftz(sum)));
          crosscheck += double(x) * double(w);
        }
        sum = ftz(std::fma(1.F, ftz(sum), 0.F)); // Original final +bias/zero.
        const auto value = actual[std::size_t((batch * s.tokens + time) * s.channels + channel)];
        if (std::isnan(sum)) EXPECT_TRUE(std::isnan(value));
        else {
          EXPECT_EQ(std::bit_cast<std::uint32_t>(value), std::bit_cast<std::uint32_t>(sum));
          if (std::isfinite(crosscheck)) EXPECT_NEAR(value, crosscheck, 3.e-6);
        }
      }
}

void exercise(c::Context &context, hyle::mem::MemoryResourceRef resource,
    Shape s, int escape) {
  const auto history_desc = dense(c::DataType::kBFloat16, {2, s.kernel - 1, s.pitch()});
  const auto chunk_desc = dense(c::DataType::kBFloat16,
      {2, s.tokens + (s.padded ? 1 : 0), s.pitch()});
  const auto weight_desc = dense(c::DataType::kFloat32, {s.channels, s.kernel});
  const auto wide_desc = dense(c::DataType::kFloat32, {2, s.window(), s.channels});
  const std::array descriptions{dense(c::DataType::kFloat32, {2, s.tokens, s.channels}),
      dense(c::DataType::kBFloat16, {2, s.tokens, s.channels}),
      dense(c::DataType::kBFloat16, {2, s.tokens, s.channels}),
      dense(c::DataType::kBFloat16, {2, s.kernel - 1, s.channels})};
  auto builder = g::GraphBuilder::create().value();
  const std::array inputs{builder.input(history_desc).value(), builder.input(chunk_desc).value(),
                           builder.input(weight_desc).value()};
  auto source = builder.add(c::Concat{.axis = 1}, inputs[0], inputs[1]).value();
  if (s.padded) {
    source = builder.add(c::View::slice(1, 0, s.window()).value(), source).value();
    source = builder.add(c::View::slice(2, 0, s.channels).value(), source).value();
  }
  source = builder.add(c::View::reshape(c::TensorShape::create(
      std::array<std::int64_t, 3>{2, s.window(), s.channels}).value()).value(), source).value();
  const auto wide = builder.add(c::Cast{c::DataType::kFloat32}, source).value();
  // A genuinely different producer frontier reproduces cold weight inputs;
  // the pair must be selected without swallowing that external preparation.
  auto weight = builder.add(c::Copy{}, inputs[2]).value();
  weight = builder.add(c::View::reshape(c::TensorShape::create(
      weight_desc.dimensions()).value()).value(), weight).value();
  const auto convolution = builder.add(c::model_ops::causal_conv1d(), wide, weight).value();
  const auto rounded = builder.add(c::Cast{c::DataType::kBFloat16}, convolution).value();
  const auto activated = builder.add(c::Unary{c::UnaryKind::kSilu}, rounded).value();
  const auto tail = builder.add(c::View::slice(1, s.tokens, s.kernel - 1).value(), source).value();
  const auto history = builder.add(c::Copy{}, tail).value();
  const std::array outputs{convolution, rounded, activated, history};
  for (const auto id : outputs) ASSERT_TRUE(builder.publish(id).ok());
  g::ValueId escaped;
  if (escape) {
    escaped = escape == 1 ? wide : builder.add(c::Copy{}, wide).value();
    ASSERT_TRUE(builder.publish(escaped).ok());
  }
  const auto graph = std::move(builder).finish().value();
  const std::array domains{g::StorageDomain{.resource = resource}};
  const std::array targets{g::ExecutionTarget{
      .source = g::ExecutionSource::acquire(context).value(), .default_storage = g::StorageDomainId{0}}};
  const std::array lanes{g::ExecutionLane{.target = g::ExecutionTargetId{0}}};
  auto compiled = g::compile(graph, {.storage_domains = domains, .targets = targets, .lanes = lanes});
  ASSERT_TRUE(compiled) << compiled.error().message();
  const auto step_for = [&](g::NodeId target) {
    for (std::size_t step = 0; step < compiled->report.step_count(); ++step)
      for (std::size_t n = 0; n < compiled->report.step(step)->node_count; ++n)
        if (compiled->report.step_node(step, n).value() == target) return step;
    return SIZE_MAX;
  };
  const auto cast_step = step_for(graph.producer(wide).value());
  const auto conv_step = step_for(graph.producer(convolution).value());
  ASSERT_NE(cast_step, SIZE_MAX); ASSERT_NE(conv_step, SIZE_MAX);
  EXPECT_EQ(cast_step == conv_step, escape == 0);
  if (!escape) {
    EXPECT_EQ(compiled->report.step(conv_step)->kind, g::CompileStepKind::kProviderRegion);
    EXPECT_EQ(compiled->report.step(conv_step)->implementation, c::ProviderImplementationId{2});
    EXPECT_EQ(compiled->report.step(conv_step)->node_count, 2U);
  }
  struct Run final {
    Samples data;
    Reference expected;
    std::array<Buffer, 3> inputs;
    std::array<Buffer, 4> outputs;
    Buffer escaped;
  };
  // Every old backing remains live through both queues and all rebindings.
  std::array<std::array<Run, 2>, 2> runs;
  std::array queues{context.create_queue().value(), context.create_queue().value()};
  std::array slots{compiled->executable.create_invocation().value(),
                   compiled->executable.create_invocation().value()};
  const auto check = [&](const Run &run) {
    check_float(run.outputs[0], run.expected.convolution);
    check_bfloat(run.outputs[1], run.expected.rounded);
    check_bfloat(run.outputs[2], run.expected.activated);
    check_bfloat(run.outputs[3], run.expected.history, true);
    if (escape) check_float(run.escaped, run.expected.cast);
    const std::array<const std::vector<BF16> *, 2> original{&run.data.history, &run.data.chunk};
    for (std::size_t port = 0; port < original.size(); ++port) {
      const auto actual = half_test::read<BF16>(run.inputs[port]);
      ASSERT_EQ(actual.size(), original[port]->size());
      for (std::size_t i = 0; i < actual.size(); ++i) EXPECT_EQ(actual[i].bits, (*original[port])[i].bits);
    }
    EXPECT_EQ(half_test::read<float>(run.inputs[2]), run.data.weights);
  };
  for (std::size_t round = 0; round < 2; ++round) {
    std::array<c::Completion, 2> done;
    for (std::size_t slot = 0; slot < slots.size(); ++slot) {
      auto &run = runs[round][slot];
      run.data = samples(s, int(7 * round + 3 * slot));
      run.expected = reference(context, resource, s, run.data);
      check_oracle(s, run.data, run.expected.convolution);
      run.inputs = {upload<BF16>(resource, run.data.history), upload<BF16>(resource, run.data.chunk),
                    upload<float>(resource, run.data.weights)};
      for (std::size_t port = 0; port < outputs.size(); ++port)
        run.outputs[port] = output_buffer(resource, descriptions[port]);
      if (escape) run.escaped = output_buffer(resource, wide_desc);
      if (round) {
        ASSERT_TRUE(slots[slot].reset().ok());
        for (std::size_t port = 0; port < run.inputs.size(); ++port)
          EXPECT_NE(run.inputs[port].native_handle(), runs[0][slot].inputs[port].native_handle());
        for (std::size_t port = 0; port < run.outputs.size(); ++port)
          EXPECT_NE(run.outputs[port].native_handle(), runs[0][slot].outputs[port].native_handle());
      }
      const std::array input_desc{history_desc, chunk_desc, weight_desc};
      for (std::size_t port = 0; port < inputs.size(); ++port)
        ASSERT_TRUE(slots[slot].bind(inputs[port], c::ConstTensorView::create(
            input_desc[port], run.inputs[port].view()).value()).ok());
      for (std::size_t port = 0; port < outputs.size(); ++port)
        ASSERT_TRUE(slots[slot].bind(outputs[port], c::TensorView::create(
            descriptions[port], run.outputs[port].view()).value()).ok());
      if (escape)
        ASSERT_TRUE(slots[slot].bind(escaped, c::TensorView::create(
            wide_desc, run.escaped.view()).value()).ok());
      const std::array bindings{g::QueueBinding{g::ExecutionLaneId{0}, &queues[slot]}};
      const auto status = slots[slot].prepare_replay(bindings, {.mode = g::ReplayMode::kRequireCapture});
      ASSERT_TRUE(status.ok()) << status.message();
    }
    for (std::size_t slot = 0; slot < slots.size(); ++slot) {
      auto submitted = slots[slot].submit(queues[slot]);
      ASSERT_TRUE(submitted) << submitted.error().message();
      done[slot] = std::move(*submitted);
    }
    ASSERT_TRUE(done[1].wait().ok()); ASSERT_TRUE(done[0].wait().ok());
    for (std::size_t slot = 0; slot < slots.size(); ++slot) {
      check(runs[round][slot]);
      if (round) check(runs[0][slot]);
    }
  }
}

TEST(GgmlCudaConvTypedLoad, OriginalMathAndTypedBoundariesAcrossSlotsAndLongTiles) {
  auto context = c::Context::create(c::cuda::device(0).value(), g::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  for (const auto shape : {Shape{1, 128, 3}, Shape{33, 256, 4, true}, Shape{128, 128, 4}}) {
    SCOPED_TRACE(::testing::Message() << "T=" << shape.tokens << " C=" << shape.channels);
    exercise(context, memory.ref(), shape, 0);
  }
}
TEST(GgmlCudaConvTypedLoad, PublishedSampleCastKeepsF32Output) {
  auto context = c::Context::create(c::cuda::device(0).value(), g::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  exercise(context, memory.ref(), Shape{1, 128, 3}, 1);
}
TEST(GgmlCudaConvTypedLoad, ExtraSampleCastReaderKeepsF32Value) {
  auto context = c::Context::create(c::cuda::device(0).value(), g::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(context, c::cuda::MemoryKind::kDevice).value();
  exercise(context, memory.ref(), Shape{33, 128, 4}, 2);
}
} // namespace
