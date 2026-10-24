#include "hyle/graph/providers/ggml_cuda.h"
#include "hyle/graph/executable.h"
#include "hyle/compute/storage_owner.h"
#include "../compute_provider_ggml_cuda/tensor_test_support.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
using hyle::mem::Buffer;
using BF16 = hyle::simd::bfloat16_t;
using ggml_cuda_tensor_test::dense;
using ggml_cuda_tensor_test::execute;
using ggml_cuda_tensor_test::output_buffer;
using ggml_cuda_tensor_test::upload;

c::Context context() {
  return c::Context::create(c::cuda::device(0).value(), g::ggml_cuda::provider()).value();
}
hyle::Result<g::CompiledGraph> compile_graph(const g::Graph &graph, c::Context &ctx,
                                           hyle::mem::MemoryResourceRef resource) {
  const std::array storage{g::StorageDomain{.resource = resource}};
  const std::array targets{g::ExecutionTarget{.source = g::ExecutionSource::acquire(ctx).value(),
      .default_storage = g::StorageDomainId{0}}};
  const std::array lanes{g::ExecutionLane{.target = g::ExecutionTargetId{0}}};
  return g::compile(graph, {.storage_domains = storage, .targets = targets, .lanes = lanes});
}
std::vector<float> widen(const std::vector<BF16> &values) {
  std::vector<float> result;
  for (const auto value : values)
    result.push_back(std::bit_cast<float>(std::uint32_t(value.bits) << 16));
  return result;
}
template <class T> void check_bits(const Buffer &buffer, const std::vector<T> &expected) {
  const auto actual = half_test::read<T>(buffer);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (std::isnan(half_test::decode(expected[i]))) {
      EXPECT_TRUE(std::isnan(half_test::decode(actual[i]))) << i;
    } else if constexpr (std::is_same_v<T, BF16>) {
      EXPECT_EQ(actual[i].bits, expected[i].bits) << i;
    } else {
      EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),
                std::bit_cast<std::uint32_t>(expected[i])) << i;
    }
  }
}

// Every reference operation is a separate Direct submission with an original
// F32 input boundary. No BF16 Cast->RMS pattern exists in its native graph.
std::vector<float> native_normal(c::Context &ctx, hyle::mem::MemoryResourceRef resource,
    const c::TensorDesc &description, const std::vector<float> &host, float epsilon) {
  auto source = upload<float>(resource, host), result = output_buffer(resource, description);
  const std::array inputs{c::ConstTensorView::create(description, source.view()).value()};
  const auto status = execute(ctx, c::RmsNorm{.epsilon = epsilon}, inputs,
      c::TensorView::create(description, result.view()).value());
  EXPECT_TRUE(status.ok()) << status.message();
  return status.ok() ? half_test::read<float>(result) : std::vector<float>{};
}
std::vector<BF16> native_narrow(c::Context &ctx, hyle::mem::MemoryResourceRef resource,
    const c::TensorDesc &description, const std::vector<float> &host) {
  auto source = upload<float>(resource, host);
  const auto output = c::TensorDesc::dense(c::DataType::kBFloat16, description.dimensions()).value();
  auto result = output_buffer(resource, output);
  const std::array inputs{c::ConstTensorView::create(description, source.view()).value()};
  const auto status = execute(ctx, c::Cast{c::DataType::kBFloat16}, inputs,
      c::TensorView::create(output, result.view()).value());
  EXPECT_TRUE(status.ok()) << status.message();
  return status.ok() ? half_test::read<BF16>(result) : std::vector<BF16>{};
}
std::vector<BF16> native_add(c::Context &ctx, hyle::mem::MemoryResourceRef resource,
    const c::TensorDesc &description, const std::vector<BF16> &left,
    const std::vector<BF16> &right) {
  auto a = upload<BF16>(resource, left), b = upload<BF16>(resource, right);
  auto result = output_buffer(resource, description);
  const std::array inputs{c::ConstTensorView::create(description, a.view()).value(),
      c::ConstTensorView::create(description, b.view()).value()};
  const auto status = execute(ctx, c::Binary{c::BinaryKind::kAdd}, inputs,
      c::TensorView::create(description, result.view()).value());
  EXPECT_TRUE(status.ok()) << status.message();
  return status.ok() ? half_test::read<BF16>(result) : std::vector<BF16>{};
}
struct Reference final {
  std::vector<float> normal, scaled;
  std::vector<BF16> rounded_normal, output;
};
Reference reference(c::Context &ctx, hyle::mem::MemoryResourceRef resource,
    const c::TensorDesc &description, const std::vector<BF16> &host,
    const std::vector<float> &gamma, float epsilon, bool round_normal) {
  Reference result;
  result.normal = native_normal(ctx, resource, description, widen(host), epsilon);
  if (result.normal.size() != host.size()) return result;
  if (round_normal) {
    result.rounded_normal = native_narrow(ctx, resource, description, result.normal);
    if (result.rounded_normal.size() != host.size()) return result;
  }
  const auto normal = round_normal ? widen(result.rounded_normal) : result.normal;
  auto source = upload<float>(resource, normal), weight = upload<float>(resource, gamma);
  auto product = output_buffer(resource, description);
  const auto parameter = dense(c::DataType::kFloat32, {std::int64_t(gamma.size())});
  const std::array inputs{c::ConstTensorView::create(description, source.view()).value(),
      c::ConstTensorView::create(parameter, weight.view()).value()};
  const auto status = execute(ctx, c::Binary{c::BinaryKind::kMultiply}, inputs,
      c::TensorView::create(description, product.view()).value());
  EXPECT_TRUE(status.ok()) << status.message();
  if (status.ok()) {
    result.scaled = half_test::read<float>(product);
    result.output = native_narrow(ctx, resource, description, result.scaled);
  }
  return result;
}

std::vector<BF16> input_values(std::size_t count, std::size_t width, int seed) {
  std::vector<BF16> result(count);
  for (std::size_t i = 0; i < count; ++i) {
    const int code = int((i * 37 + std::size_t(seed) * 13) % 257) - 128;
    const double magnitude = i % 29 == 0 ? 32.0 : i % 31 == 0 ? 1.0 / 128 : 1.0;
    result[i] = half_test::encode<BF16>(code * magnitude / 32);
    if (i % 53 == 0) result[i] = BF16::from_bits(static_cast<std::uint16_t>(i % 2 ? 0x8000 : 0));
  }
  // Original GPU math, not a CPU sqrt, decides DAZ/FTZ and non-finite behavior.
  // Keep ordinary rows as well, so NaNs cannot substitute for an RNE witness.
  constexpr std::array<std::uint16_t, 8> small{0x0001, 0x007f, 0x0080, 0x0081,
                                              0x8001, 0x807f, 0x8080, 0x1f80};
  for (std::size_t i = 0; i < small.size(); ++i) result[i] = BF16::from_bits(small[i]);
  if (count / width > 1)
    for (std::size_t col = 0; col < width; ++col)
      result[width + col] = BF16::from_bits(small[col % small.size()]);
  if (count / width > 2) result[2 * width] = BF16::from_bits(0x7f80);
  if (count / width > 3) result[3 * width] = BF16::from_bits(0x7fc1);
  return result;
}
std::vector<float> scale_values(std::size_t width, int seed, bool boundaries = true) {
  std::vector<float> result(width);
  for (std::size_t i = 0; i < width; ++i)
    result[i] = 1.F + float(int(i % 13) - 6) / 8 +
        std::ldexp(float(2 + seed + int(i % 11)), -17);
  if (boundaries) {
    result[0] = std::bit_cast<float>(std::uint32_t{0x007fffff});
    result[1] = std::numeric_limits<float>::min();
    result[2] = std::ldexp(1.F, -100);
    result[3] = std::ldexp(1.F, 100);
    result[4] = std::numeric_limits<float>::infinity();
    result[5] = -std::numeric_limits<float>::infinity();
    result[6] = std::numeric_limits<float>::quiet_NaN();
  }
  return result;
}
Buffer source_buffer(hyle::mem::MemoryResourceRef resource,
    const c::TensorDesc &description, const std::vector<BF16> &host) {
  if (description.is_dense()) return upload<BF16>(resource, host);
  // One real padded-row control. The original matrix()/CONT path owns any
  // materialization; this test does not grant a typed kernel arbitrary strides.
  const auto width = std::size_t(description.dimensions()[1]);
  const auto stride = std::size_t(description.byte_strides()[0]) / sizeof(BF16);
  std::vector<BF16> physical(description.required_bytes().value() / sizeof(BF16), BF16::from_bits(0x7fc1));
  for (std::size_t i = 0; i < host.size(); ++i) physical[(i / width) * stride + i % width] = host[i];
  return upload<BF16>(resource, physical);
}
struct Ports final {
  g::ValueId x, gamma;
  std::array<g::ValueId, 2> outputs;
  g::ValueId intermediate;
};

void exercise(c::Context &ctx, hyle::mem::MemoryResourceRef resource,
    g::CompiledGraph &compiled, const c::TensorDesc &input, Ports ports,
    float epsilon, bool rounded_intermediate) {
  const auto normal = c::TensorDesc::dense(c::DataType::kFloat32, input.dimensions()).value();
  const auto output = c::TensorDesc::dense(c::DataType::kBFloat16, input.dimensions()).value();
  const auto middle = rounded_intermediate ? output : normal;
  const auto width = std::size_t(input.dimensions().back());
  const auto parameter = dense(c::DataType::kFloat32, {std::int64_t(width)});
  std::vector<Buffer> retained;
  std::array queues{ctx.create_queue().value(), ctx.create_queue().value()};
  std::array slots{compiled.executable.create_invocation().value(),
                   compiled.executable.create_invocation().value()};
  std::array<Reference, 2> old;
  for (int round = 0; round < 2; ++round) {
    std::array<std::size_t, 2> base;
    std::array<Reference, 2> expected;
    std::array<c::Completion, 2> done;
    for (std::size_t slot = 0; slot < 2; ++slot) {
      const int seed = 1 + round * 3 + int(slot);
      const auto values = input_values(input.element_count().value(), width, seed);
      const auto gamma = scale_values(width, seed);
      expected[slot] = reference(ctx, resource, normal, values, gamma, epsilon, rounded_intermediate);
      ASSERT_EQ(expected[slot].output.size(), values.size());
      if (rounded_intermediate) {
        std::size_t witnesses = 0;
        for (std::size_t i = 0; i < values.size(); ++i) {
          const float n = expected[slot].normal[i], scale = gamma[i % width];
          if (!std::isfinite(n) || !std::isfinite(scale) ||
              std::abs(n) < 1.e-6F || std::abs(n) > 10.F ||
              std::abs(scale) < 0.25F || std::abs(scale) > 2.F)
            continue;
          witnesses += expected[slot].output[i].bits != half_test::encode<BF16>(n * scale).bits;
        }
        EXPECT_GT(witnesses, 0U);
      }
      base[slot] = retained.size();
      retained.push_back(source_buffer(resource, input, values));
      retained.push_back(upload<float>(resource, gamma));
      retained.push_back(output_buffer(resource, output));
      retained.push_back(output_buffer(resource, output));
      retained.push_back(output_buffer(resource, middle));
      if (round)
        for (std::size_t port = 0; port < 5; ++port)
          EXPECT_NE(retained[base[slot] + port].native_handle(), retained[slot * 5 + port].native_handle());
      ASSERT_TRUE(slots[slot].bind(ports.x, c::ConstTensorView::create(input, retained[base[slot]].view()).value()).ok());
      ASSERT_TRUE(slots[slot].bind(ports.gamma, c::ConstTensorView::create(parameter, retained[base[slot] + 1].view()).value()).ok());
      for (std::size_t port = 0; port < 2; ++port)
        ASSERT_TRUE(slots[slot].bind(ports.outputs[port], c::TensorView::create(output, retained[base[slot] + 2 + port].view()).value()).ok());
      ASSERT_TRUE(slots[slot].bind(ports.intermediate, c::TensorView::create(middle, retained[base[slot] + 4].view()).value()).ok());
      const std::array bindings{g::QueueBinding{g::ExecutionLaneId{0}, &queues[slot]}};
      auto status = slots[slot].prepare_replay(bindings, {.mode = g::ReplayMode::kRequireCapture});
      ASSERT_TRUE(status.ok()) << status.message();
    }
    for (std::size_t slot = 0; slot < 2; ++slot) {
      auto submitted = slots[slot].submit(queues[slot]);
      ASSERT_TRUE(submitted) << submitted.error().message();
      done[slot] = std::move(*submitted);
    }
    ASSERT_TRUE(done[1].wait().ok()); ASSERT_TRUE(done[0].wait().ok());
    for (std::size_t slot = 0; slot < 2; ++slot) {
      for (std::size_t port = 0; port < 2; ++port)
        check_bits(retained[base[slot] + 2 + port], expected[slot].output);
      if (rounded_intermediate)
        check_bits(retained[base[slot] + 4], expected[slot].rounded_normal);
      else
        check_bits(retained[base[slot] + 4], expected[slot].normal);
      if (!round) old[slot] = expected[slot];
      else {
        for (std::size_t port = 0; port < 2; ++port) check_bits(retained[slot * 5 + 2 + port], old[slot].output);
        if (rounded_intermediate) check_bits(retained[slot * 5 + 4], old[slot].rounded_normal);
        else check_bits(retained[slot * 5 + 4], old[slot].normal);
      }
    }
  }
}

TEST(GgmlCudaRmsCastFusion, WeightedChainMatchesIndependentFloatNativeAndPublishedIntermediate) {
  auto ctx = context();
  auto memory = c::cuda::MemoryResource::create(ctx, c::cuda::MemoryKind::kDevice).value();
  struct Shape final { std::int64_t rows, width; float epsilon; bool padded; };
  for (const auto shape : {Shape{1, 128, 0.F, false}, Shape{3, 256, 1.e-6F, true},
                          Shape{128, 2048, 1.e-5F, false}}) {
    SCOPED_TRACE(::testing::Message() << shape.rows << '/' << shape.width << '/' << shape.padded);
    auto input = dense(c::DataType::kBFloat16, {shape.rows, shape.width});
    if (shape.padded) input = c::TensorDesc::strided_bytes(c::DataType::kBFloat16,
        input.dimensions(), std::array{(shape.width + 4) * std::int64_t(sizeof(BF16)), std::int64_t(sizeof(BF16))}).value();
    const auto parameter = dense(c::DataType::kFloat32, {shape.width});
    auto builder = g::GraphBuilder::create().value();
    Ports ports{.x = builder.input(input).value(), .gamma = builder.input(parameter).value()};
    for (std::size_t branch = 0; branch < 2; ++branch) {
      const auto wide = builder.add(c::Cast{c::DataType::kFloat32}, ports.x).value();
      const auto normal = builder.add(c::RmsNorm{.epsilon = shape.epsilon}, wide).value();
      const auto scaled = builder.add(c::Binary{c::BinaryKind::kMultiply}, normal, ports.gamma).value();
      ports.outputs[branch] = builder.add(c::Cast{c::DataType::kBFloat16}, scaled).value();
      ASSERT_TRUE(builder.publish(ports.outputs[branch]).ok());
      if (branch) { ports.intermediate = normal; ASSERT_TRUE(builder.publish(normal).ok()); }
    }
    const auto graph = std::move(builder).finish().value();
    auto compiled = compile_graph(graph, ctx, memory.ref()); ASSERT_TRUE(compiled) << compiled.error().message();
    exercise(ctx, memory.ref(), *compiled, input, ports, shape.epsilon, false);
  }
}

TEST(GgmlCudaRmsCastFusion, BfloatNormMultiplyPreservesIntermediateRounding) {
  auto ctx = context();
  auto memory = c::cuda::MemoryResource::create(ctx, c::cuda::MemoryKind::kDevice).value();
  const auto input = dense(c::DataType::kBFloat16, {1, 2, 3, 128});
  const auto parameter = dense(c::DataType::kFloat32, {128});
  auto builder = g::GraphBuilder::create().value();
  Ports ports{.x = builder.input(input).value(), .gamma = builder.input(parameter).value()};
  for (std::size_t branch = 0; branch < 2; ++branch) {
    const auto normal = builder.add(c::RmsNorm{.epsilon = 1.e-6F}, ports.x).value();
    const auto wide = builder.add(c::Cast{c::DataType::kFloat32}, normal).value();
    const auto scaled = builder.add(c::Binary{c::BinaryKind::kMultiply}, wide, ports.gamma).value();
    ports.outputs[branch] = builder.add(c::Cast{c::DataType::kBFloat16}, scaled).value();
    ASSERT_TRUE(builder.publish(ports.outputs[branch]).ok());
    if (branch) { ports.intermediate = normal; ASSERT_TRUE(builder.publish(normal).ok()); }
  }
  const auto graph = std::move(builder).finish().value();
  auto compiled = compile_graph(graph, ctx, memory.ref()); ASSERT_TRUE(compiled) << compiled.error().message();
  exercise(ctx, memory.ref(), *compiled, input, ports, 1.e-6F, true);
}

TEST(GgmlCudaRmsCastFusion, InternalBfloatProducerPreservesRoundedNormAcrossReaders) {
  auto ctx = context();
  auto memory = c::cuda::MemoryResource::create(ctx, c::cuda::MemoryKind::kDevice).value();
  const std::array descriptions{dense(c::DataType::kBFloat16, {1, 1, 16, 128}),
      dense(c::DataType::kBFloat16, {1, 128, 16, 128}), dense(c::DataType::kBFloat16, {2, 2048})};
  for (const auto &input : descriptions) for (const bool published_wide : {false, true}) {
    SCOPED_TRACE(::testing::Message() << input.element_count().value() << '/' << input.dimensions().back()
                                    << "/published-wide=" << published_wide);
    const auto wide_desc = c::TensorDesc::dense(c::DataType::kFloat32, input.dimensions()).value();
    auto builder = g::GraphBuilder::create().value();
    const auto a = builder.input(input).value(), b = builder.input(input).value();
    const auto producer = builder.add(c::Binary{c::BinaryKind::kAdd}, a, b).value();
    const auto wide = builder.add(c::Cast{c::DataType::kFloat32}, producer).value();
    const auto normal = builder.add(c::RmsNorm{.epsilon = 1.e-6F}, wide).value();
    const auto rounded = builder.add(c::Cast{c::DataType::kBFloat16}, normal).value();
    // Keep the rounded norm private, as in GDN. Its later F32 reader must
    // observe BF16 RNE even if native storage reuses the dead producer hole.
    const auto reader = builder.add(c::Cast{c::DataType::kFloat32}, rounded).value();
    const auto output = builder.add(c::Copy{}, reader).value();
    ASSERT_TRUE(builder.publish(output).ok());
    if (published_wide) ASSERT_TRUE(builder.publish(wide).ok());
    const auto graph = std::move(builder).finish().value();
    auto compiled = compile_graph(graph, ctx, memory.ref());
    ASSERT_TRUE(compiled) << compiled.error().message();
    std::vector<Buffer> retained;
    std::array queues{ctx.create_queue().value(), ctx.create_queue().value()};
    std::array slots{compiled->executable.create_invocation().value(),
                    compiled->executable.create_invocation().value()};
    std::array<std::vector<float>, 2> old, old_wide;
    for (int round = 0; round < 2; ++round) {
      std::array<std::size_t, 2> base;
      std::array<std::vector<float>, 2> expected, expected_wide;
      std::array<c::Completion, 2> done;
      for (std::size_t slot = 0; slot < 2; ++slot) {
        const auto count = input.element_count().value(), width = std::size_t(input.dimensions().back());
        const int seed = 1 + round * 3 + int(slot);
        const auto left = input_values(count, width, seed), right = input_values(count, width, seed + 7);
        const auto produced = native_add(ctx, memory.ref(), input, left, right);
        ASSERT_EQ(produced.size(), count);
        expected_wide[slot] = widen(produced); // Bit widening, including subnormals/NaNs.
        const auto normalized = native_normal(ctx, memory.ref(), wide_desc, expected_wide[slot], 1.e-6F);
        ASSERT_EQ(normalized.size(), count);
        expected[slot] = widen(native_narrow(ctx, memory.ref(), wide_desc, normalized));
        ASSERT_EQ(expected[slot].size(), count);
        std::size_t witnesses = 0;
        for (std::size_t i = 0; i < count; ++i)
          witnesses += std::isfinite(normalized[i]) && normalized[i] != expected[slot][i];
        EXPECT_GT(witnesses, 0U);
        base[slot] = retained.size();
        retained.push_back(upload<BF16>(memory.ref(), left));
        retained.push_back(upload<BF16>(memory.ref(), right));
        retained.push_back(output_buffer(memory.ref(), wide_desc));
        retained.push_back(output_buffer(memory.ref(), wide_desc));
        if (round) for (std::size_t port = 0; port < 4; ++port)
          EXPECT_NE(retained[base[slot] + port].native_handle(), retained[slot * 4 + port].native_handle());
        ASSERT_TRUE(slots[slot].bind(a, c::ConstTensorView::create(input, retained[base[slot]].view()).value()).ok());
        ASSERT_TRUE(slots[slot].bind(b, c::ConstTensorView::create(input, retained[base[slot] + 1].view()).value()).ok());
        ASSERT_TRUE(slots[slot].bind(output, c::TensorView::create(wide_desc, retained[base[slot] + 2].view()).value()).ok());
        if (published_wide)
          ASSERT_TRUE(slots[slot].bind(wide, c::TensorView::create(wide_desc, retained[base[slot] + 3].view()).value()).ok());
        const std::array bindings{g::QueueBinding{g::ExecutionLaneId{0}, &queues[slot]}};
        const auto prepared = slots[slot].prepare_replay(bindings, {.mode = g::ReplayMode::kRequireCapture});
        ASSERT_TRUE(prepared.ok()) << prepared.message();
      }
      for (std::size_t slot = 0; slot < 2; ++slot) {
        auto submitted = slots[slot].submit(queues[slot]);
        ASSERT_TRUE(submitted) << submitted.error().message();
        done[slot] = std::move(*submitted);
      }
      ASSERT_TRUE(done[1].wait().ok()); ASSERT_TRUE(done[0].wait().ok());
      for (std::size_t slot = 0; slot < 2; ++slot) {
        check_bits(retained[base[slot] + 2], expected[slot]);
        if (published_wide) check_bits(retained[base[slot] + 3], expected_wide[slot]);
        if (!round) { old[slot] = expected[slot]; old_wide[slot] = expected_wide[slot]; }
        else {
          check_bits(retained[slot * 4 + 2], old[slot]);
          if (published_wide) check_bits(retained[slot * 4 + 3], old_wide[slot]);
        }
      }
    }
  }
}

TEST(GgmlCudaRmsCastFusion, FloatRmsMultiplyAddRetainsOriginalPrecision) {
  auto ctx = context();
  auto memory = c::cuda::MemoryResource::create(ctx, c::cuda::MemoryKind::kDevice).value();
  constexpr std::size_t width = 2048;
  const auto description = dense(c::DataType::kFloat32, {2, std::int64_t(width)});
  const auto parameter = dense(c::DataType::kFloat32, {std::int64_t(width)});
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(description).value(), gamma = builder.input(parameter).value();
  const auto bias = builder.input(parameter).value();
  const auto normal = builder.add(c::RmsNorm{.epsilon = 0.F}, x).value();
  const auto scaled = builder.add(c::Binary{c::BinaryKind::kMultiply}, normal, gamma).value();
  const auto output = builder.add(c::Binary{}, scaled, bias).value();
  ASSERT_TRUE(builder.publish(output).ok());
  const auto graph = std::move(builder).finish().value();
  auto compiled = compile_graph(graph, ctx, memory.ref()); ASSERT_TRUE(compiled) << compiled.error().message();
  std::vector<Buffer> retained;
  std::array queues{ctx.create_queue().value(), ctx.create_queue().value()};
  std::array slots{compiled->executable.create_invocation().value(),
                   compiled->executable.create_invocation().value()};
  std::array<std::vector<float>, 2> old;
  for (int round = 0; round < 2; ++round) {
    std::array<std::size_t, 2> base;
    std::array<std::vector<float>, 2> reference_norm, weights, biases;
    std::array<c::Completion, 2> done;
    for (std::size_t slot = 0; slot < 2; ++slot) {
      std::vector<float> values(2 * width);
      for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = (i % 2 ? -1.F : 1.F) * (1.F + std::ldexp(float(1 + (i + slot + std::size_t(round)) % 17), -14));
      weights[slot] = scale_values(width, 1 + round + int(slot), false);
      biases[slot].resize(width);
      for (std::size_t i = 0; i < width; ++i) biases[slot][i] = std::ldexp(float(int(i % 13) - 6), -24);
      reference_norm[slot] = native_normal(ctx, memory.ref(), description, values, 0.F);
      ASSERT_EQ(reference_norm[slot].size(), values.size());
      base[slot] = retained.size();
      retained.push_back(upload<float>(memory.ref(), values));
      retained.push_back(upload<float>(memory.ref(), weights[slot]));
      retained.push_back(upload<float>(memory.ref(), biases[slot]));
      retained.push_back(output_buffer(memory.ref(), description));
      if (round)
        for (std::size_t port = 0; port < 4; ++port)
          EXPECT_NE(retained[base[slot] + port].native_handle(), retained[slot * 4 + port].native_handle());
      for (const auto binding : {std::pair{x, description}, std::pair{gamma, parameter}, std::pair{bias, parameter}}) {
        const auto port = binding.first == x ? 0U : binding.first == gamma ? 1U : 2U;
        ASSERT_TRUE(slots[slot].bind(binding.first, c::ConstTensorView::create(binding.second, retained[base[slot] + port].view()).value()).ok());
      }
      ASSERT_TRUE(slots[slot].bind(output, c::TensorView::create(description, retained[base[slot] + 3].view()).value()).ok());
      const std::array bindings{g::QueueBinding{g::ExecutionLaneId{0}, &queues[slot]}};
      const auto prepared = slots[slot].prepare_replay(bindings, {.mode = g::ReplayMode::kRequireCapture});
      ASSERT_TRUE(prepared.ok()) << prepared.message();
    }
    for (std::size_t slot = 0; slot < 2; ++slot) {
      auto submitted = slots[slot].submit(queues[slot]); ASSERT_TRUE(submitted) << submitted.error().message();
      done[slot] = std::move(*submitted);
    }
    ASSERT_TRUE(done[1].wait().ok()); ASSERT_TRUE(done[0].wait().ok());
    for (std::size_t slot = 0; slot < 2; ++slot) {
      const auto actual = half_test::read<float>(retained[base[slot] + 3]);
      std::size_t precision_witnesses = 0;
      for (std::size_t i = 0; i < actual.size(); ++i) {
        const float a = reference_norm[slot][i], w = weights[slot][i % width], b = biases[slot][i % width];
        const float separate = float(a * w) + b, contracted = std::fma(a, w, b);
        const auto bits = std::bit_cast<std::uint32_t>(actual[i]);
        // The original F32 kernel may contract its final MUL+ADD. Both exact
        // F32 outcomes are legitimate; no tolerance or BF16 conversion is used.
        EXPECT_TRUE(bits == std::bit_cast<std::uint32_t>(separate) || bits == std::bit_cast<std::uint32_t>(contracted)) << i;
        precision_witnesses += actual[i] != static_cast<float>(half_test::decode(half_test::encode<BF16>(actual[i])));
      }
      EXPECT_GT(precision_witnesses, 0U);
      if (!round) old[slot] = actual;
      else check_bits(retained[slot * 4 + 3], old[slot]);
    }
  }
}

// A real external metadata producer and earlier cold scale force the old
// new_frontier splitter to emit activation Cast alone. Region membership is
// the proposal regression; the unchanged native math still has its GPU oracle.
TEST(GgmlCudaRmsCastFusion, ExternalFrontierKeepsRmsCastChainAndColdScale) {
  auto ctx = context();
  auto owner = c::StorageOwner::adopt(ctx, c::cuda::MemoryResource::create(
      ctx, c::cuda::MemoryKind::kDevice).value()).value();
  constexpr std::int64_t width = 256;
  const auto input_desc = dense(c::DataType::kBFloat16, {1, 2, width});
  const auto output_desc = dense(c::DataType::kBFloat16, {2, width});
  const auto float_desc = dense(c::DataType::kFloat32, {2, width});
  const auto weight_desc = dense(c::DataType::kBFloat16, {width});
  const auto one_desc = dense(c::DataType::kFloat32, {1});
  std::vector<BF16> weights(width);
  std::vector<float> gamma(width);
  for (std::size_t i = 0; i < weights.size(); ++i) {
    weights[i] = half_test::encode<BF16>(double(int(i % 13) - 6) / 32);
    gamma[i] = 1.F + float(half_test::decode(weights[i])); // Exact dyadic F32 add.
  }
  const auto constant = [&]<class T>(g::ValueId id, const c::TensorDesc &desc,
                                    const std::vector<T> &values) {
    auto backing = upload<T>(owner.ref(), values);
    auto ready = owner.adopt_buffer(std::move(backing)).value();
    return g::ConstantBinding{id, ready.tensor(desc).value(),
        g::ConstantPolicy::kBorrow, owner, std::move(ready)};
  };
  std::size_t packed_count = 0, packed_bytes = 0;
  // 0: private chain; 1/2: published F32 intermediate; 3: external control reader.
  for (int mode = 0; mode < 4; ++mode) {
    SCOPED_TRACE(mode);
    auto builder = g::GraphBuilder::create().value();
    const auto x = builder.input(input_desc).value();
    const auto external = builder.add(c::View::reshape(
        c::TensorShape::create(output_desc.dimensions()).value()).value(), x).value();
    const auto w = builder.constant(weight_desc).value(), one = builder.constant(one_desc).value();
    const auto scale_cast = builder.add(c::Cast{c::DataType::kFloat32}, w).value();
    const auto scale = builder.add(c::Binary{}, scale_cast, one).value();
    const auto widen_node = builder.add_node(c::Operation(c::Cast{c::DataType::kFloat32}),
        std::array{external}).value();
    const auto wide = builder.result(widen_node).value();
    g::ValueId control;
    g::NodeId control_node;
    if (mode == 3) {
      control_node = builder.add_node(c::Operation(c::View::reshape(
          c::TensorShape::create(output_desc.dimensions()).value()).value()), std::array{x}).value();
      ASSERT_TRUE(builder.depends_on(control_node, widen_node).ok());
      control = builder.result(control_node).value();
      ASSERT_TRUE(builder.publish(control).ok());
    }
    const auto normal = builder.add(c::RmsNorm{.epsilon = 1.e-6F}, wide, scale).value();
    const auto y = builder.add(c::Cast{c::DataType::kBFloat16}, normal).value();
    ASSERT_TRUE(builder.publish(y).ok());
    if (mode == 1) ASSERT_TRUE(builder.publish(wide).ok());
    if (mode == 2) ASSERT_TRUE(builder.publish(normal).ok());
    const auto graph = std::move(builder).finish().value();
    std::vector<g::ConstantBinding> constants{constant(w, weight_desc, weights),
        constant(one, one_desc, std::vector{1.F})};
    const std::array domains{g::StorageDomain{.resource = owner.ref(), .owner = owner}};
    const std::array targets{g::ExecutionTarget{.source = g::ExecutionSource::acquire(ctx).value(),
        .default_storage = g::StorageDomainId{0}}};
    const std::array lanes{g::ExecutionLane{.target = g::ExecutionTargetId{0}}};
    auto compiled = g::compile(graph, {.storage_domains = domains, .targets = targets, .lanes = lanes},
        {.constants = constants});
    ASSERT_TRUE(compiled) << compiled.error().message();
    const auto step_for = [&](g::NodeId node) {
      for (std::size_t step = 0; step < compiled->report.step_count(); ++step)
        for (std::size_t n = 0; n < compiled->report.step(step)->node_count; ++n)
          if (compiled->report.step_node(step, n).value() == node) return step;
      return SIZE_MAX;
    };
    const auto wide_step = step_for(widen_node);
    const auto rms_node = graph.producer(normal).value(), output_node = graph.producer(y).value();
    const auto rms_step = step_for(rms_node), output_step = step_for(output_node);
    ASSERT_NE(wide_step, SIZE_MAX); ASSERT_NE(rms_step, SIZE_MAX); ASSERT_NE(output_step, SIZE_MAX);
    const bool joined = wide_step == rms_step && rms_step == output_step;
    EXPECT_EQ(joined, mode == 0);
    if (mode == 0) {
      const auto region = compiled->report.step(wide_step).value();
      EXPECT_EQ(region.kind, g::CompileStepKind::kProviderRegion);
      EXPECT_EQ(region.implementation, c::ProviderImplementationId{2});
      EXPECT_EQ(region.node_count, 3U);
      EXPECT_EQ(compiled->report.step_node(wide_step, 0).value(), widen_node);
      EXPECT_EQ(compiled->report.step_node(wide_step, 1).value(), rms_node);
      EXPECT_EQ(compiled->report.step_node(wide_step, 2).value(), output_node);
    }
    const auto cold_step = step_for(graph.producer(scale_cast).value());
    ASSERT_NE(cold_step, SIZE_MAX);
    EXPECT_EQ(cold_step, step_for(graph.producer(scale).value()));
    EXPECT_NE(cold_step, rms_step);
    EXPECT_EQ(compiled->report.step(cold_step)->dispatch_count, 0U);
    EXPECT_EQ(compiled->report.step(cold_step)->output_count, 0U);
    EXPECT_EQ(compiled->report.step(step_for(graph.producer(external).value()))->kind,
        g::CompileStepKind::kView);
    EXPECT_GT(compiled->report.packed_constant_count(), 0U);
    if (!mode) {
      packed_count = compiled->report.packed_constant_count();
      packed_bytes = compiled->report.packed_constant_bytes();
    } else {
      EXPECT_EQ(compiled->report.packed_constant_count(), packed_count);
      EXPECT_EQ(compiled->report.packed_constant_bytes(), packed_bytes);
    }
    if (mode == 3) {
      EXPECT_EQ(graph.node_dependency_count(control_node).value(), 1U);
      EXPECT_EQ(graph.node_dependency(control_node, 0).value(), widen_node);
    }
    constants.clear(); // Prepared scale/leases must retain the original cold backing.
    std::vector<Buffer> retained;
    std::array queues{ctx.create_queue().value(), ctx.create_queue().value()};
    std::array slots{compiled->executable.create_invocation().value(),
                    compiled->executable.create_invocation().value()};
    std::array<Reference, 2> old;
    std::array<std::vector<float>, 2> old_wide;
    for (int round = 0; round < 2; ++round) {
      std::array<std::size_t, 2> base;
      std::array<Reference, 2> expected;
      std::array<std::vector<BF16>, 2> values;
      std::array<c::Completion, 2> done;
      for (std::size_t slot = 0; slot < 2; ++slot) {
        values[slot] = input_values(input_desc.element_count().value(), width, 1 + 3 * round + int(slot));
        expected[slot] = reference(ctx, owner.ref(), float_desc, values[slot], gamma, 1.e-6F, false);
        ASSERT_EQ(expected[slot].output.size(), values[slot].size());
        ASSERT_EQ(expected[slot].scaled.size(), values[slot].size());
        base[slot] = retained.size();
        retained.push_back(upload<BF16>(owner.ref(), values[slot]));
        retained.push_back(output_buffer(owner.ref(), output_desc));
        retained.push_back(output_buffer(owner.ref(), float_desc));
        if (round) for (std::size_t port = 0; port < 3; ++port)
          EXPECT_NE(retained[base[slot] + port].native_handle(), retained[slot * 3 + port].native_handle());
        ASSERT_TRUE(slots[slot].bind(x, c::ConstTensorView::create(input_desc,
            retained[base[slot]].view()).value()).ok());
        ASSERT_TRUE(slots[slot].bind(y, c::TensorView::create(output_desc,
            retained[base[slot] + 1].view()).value()).ok());
        if (mode == 1 || mode == 2)
          ASSERT_TRUE(slots[slot].bind(mode == 1 ? wide : normal, c::TensorView::create(float_desc,
              retained[base[slot] + 2].view()).value()).ok());
        const std::array bindings{g::QueueBinding{g::ExecutionLaneId{0}, &queues[slot]}};
        const auto prepared = slots[slot].prepare_replay(bindings, {.mode = g::ReplayMode::kRequireCapture});
        ASSERT_TRUE(prepared.ok()) << prepared.message();
      }
      for (std::size_t slot = 0; slot < 2; ++slot) {
        auto submitted = slots[slot].submit(queues[slot]);
        ASSERT_TRUE(submitted) << submitted.error().message(); done[slot] = std::move(*submitted);
      }
      ASSERT_TRUE(done[1].wait().ok()); ASSERT_TRUE(done[0].wait().ok());
      for (std::size_t slot = 0; slot < 2; ++slot) {
        check_bits(retained[base[slot] + 1], expected[slot].output);
        if (mode == 1) check_bits(retained[base[slot] + 2], widen(values[slot]));
        if (mode == 2) check_bits(retained[base[slot] + 2], expected[slot].scaled);
        if (mode == 3) {
          auto alias = slots[slot].output(control); ASSERT_TRUE(alias) << alias.error().message();
          std::vector<BF16> actual(values[slot].size());
          ASSERT_TRUE(alias->storage().read(std::as_writable_bytes(std::span(actual))).ok());
          for (std::size_t i = 0; i < actual.size(); ++i) EXPECT_EQ(actual[i].bits, values[slot][i].bits);
        }
        if (!round) { old[slot] = expected[slot]; old_wide[slot] = widen(values[slot]); }
        else {
          check_bits(retained[slot * 3 + 1], old[slot].output);
          if (mode == 1) check_bits(retained[slot * 3 + 2], old_wide[slot]);
          if (mode == 2) check_bits(retained[slot * 3 + 2], old[slot].scaled);
        }
      }
    }
  }
}

} // namespace
