#include "hyle/graph/providers/ggml_cuda.h"
#include "hyle/graph/executable.h"
#include "../compute_provider_ggml_cuda/tensor_test_support.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>
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

c::Reduce sum_last_axis() {
  return {.kind = c::ReductionKind::kSum,
      .axes = c::Axes::from(std::array<std::int32_t, 1>{-1}).value(),
      .keep_dimensions = true, .accumulation_type = c::DataType::kFloat32};
}
hyle::Result<g::CompiledGraph> compile_graph(const g::Graph &graph, c::Context &ctx,
    hyle::mem::MemoryResourceRef resource) {
  const std::array storage{g::StorageDomain{.resource = resource}};
  const std::array targets{g::ExecutionTarget{
      .source = g::ExecutionSource::acquire(ctx).value(),
      .default_storage = g::StorageDomainId{0}}};
  const std::array lanes{g::ExecutionLane{.target = g::ExecutionTargetId{0}}};
  return g::compile(graph, {.storage_domains = storage, .targets = targets, .lanes = lanes});
}
std::array<c::TensorDesc, 5> stage_descriptions(const c::TensorDesc &input) {
  const auto full = c::TensorDesc::dense(c::DataType::kBFloat16, input.dimensions()).value();
  std::vector<std::int64_t> shape(input.dimensions().begin(), input.dimensions().end());
  shape.back() = 1;
  const auto reduced = c::TensorDesc::dense(c::DataType::kBFloat16, shape).value();
  return {full, reduced, reduced, reduced, full};
}
void check_bits(const Buffer &buffer, const std::vector<BF16> &expected) {
  const auto actual = half_test::read<BF16>(buffer);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (std::isnan(half_test::decode(expected[i])))
      EXPECT_TRUE(std::isnan(half_test::decode(actual[i]))) << i;
    else
      EXPECT_EQ(actual[i].bits, expected[i].bits) << i;
  }
}
struct Reference final {
  // Square, sum, sum+epsilon, rsqrt and final product each have BF16 RNE.
  std::array<std::vector<BF16>, 5> values;
};

// Each public operation is a separate original Direct GPU submission. A new
// complete L2 matcher cannot match across these invocation boundaries. Keep
// rank/row count unchanged so SUM_ROWS selects the original reduction tree;
// Rsqrt retains its native strict SQRT, fast DIV and final BF16 conversion.
Reference reference(c::Context &ctx, hyle::mem::MemoryResourceRef resource,
    const std::array<c::TensorDesc, 5> &descriptions,
    const std::vector<BF16> &host, BF16 epsilon) {
  Reference result;
  auto source = upload<BF16>(resource, host);
  auto eps = upload<BF16>(resource, std::span(&epsilon, 1));
  std::array outputs{output_buffer(resource, descriptions[0]),
      output_buffer(resource, descriptions[1]), output_buffer(resource, descriptions[2]),
      output_buffer(resource, descriptions[3]), output_buffer(resource, descriptions[4])};
  const auto input = c::ConstTensorView::create(descriptions[0], source.view()).value();
  const auto scalar = c::ConstTensorView::create(
      dense(c::DataType::kBFloat16, {1}), eps.view()).value();
  const auto view = [&](std::size_t i) {
    return c::ConstTensorView::create(descriptions[i], outputs[i].view()).value();
  };
  const auto run = [&](c::Operation operation, std::span<const c::ConstTensorView> inputs,
      std::size_t stage) {
    const auto status = execute(ctx, operation, inputs,
        c::TensorView::create(descriptions[stage], outputs[stage].view()).value());
    EXPECT_TRUE(status.ok()) << status.message();
    if (status.ok()) result.values[stage] = half_test::read<BF16>(outputs[stage]);
    return status.ok();
  };
  const std::array square_inputs{input, input};
  if (!run(c::Operation(c::Binary{c::BinaryKind::kMultiply}), square_inputs, 0)) return result;
  const std::array sum_inputs{view(0)};
  if (!run(c::Operation(sum_last_axis()), sum_inputs, 1)) return result;
  const std::array add_inputs{view(1), scalar};
  if (!run(c::Operation(c::Binary{c::BinaryKind::kAdd}), add_inputs, 2)) return result;
  const std::array rsqrt_inputs{view(2)};
  if (!run(c::Operation(c::Unary{c::UnaryKind::kRsqrt}), rsqrt_inputs, 3)) return result;
  const std::array product_inputs{input, view(3)};
  (void)run(c::Operation(c::Binary{c::BinaryKind::kMultiply}), product_inputs, 4);
  return result;
}

std::vector<BF16> input_values(std::size_t count, std::size_t width, int seed) {
  std::vector<BF16> values(count);
  for (std::size_t i = 0; i < count; ++i) {
    const int code = int((i * 37 + std::size_t(seed) * 13) % 509) - 254;
    values[i] = half_test::encode<BF16>(double(code) / 256);
  }
  // Include small/FTZ and nonfinite rows without losing ordinary finite
  // rows whose nonexact squares witness the first intermediate rounding.
  constexpr std::array<std::uint16_t, 8> small{
      0x0001, 0x007f, 0x0080, 0x0081, 0x8001, 0x807f, 0x8080, 0x1f80};
  for (std::size_t i = 0; i < small.size(); ++i) values[i] = BF16::from_bits(small[i]);
  for (std::size_t col = 0; col < width; ++col) {
    values[width + col] = BF16::from_bits(small[col % small.size()]);
    values[2 * width + col] = BF16::from_bits(col % 2 ? 0x8000 : 0);
  }
  values[3 * width] = BF16::from_bits(0x7f80);
  values[4 * width] = BF16::from_bits(0x7fc1);
  return values;
}
Buffer source_buffer(hyle::mem::MemoryResourceRef resource,
    const c::TensorDesc &description, const std::vector<BF16> &values) {
  if (description.is_dense()) return upload<BF16>(resource, values);
  const auto width = std::size_t(description.dimensions().back());
  const auto stride = std::size_t(description.byte_strides()[description.rank() - 2]) / sizeof(BF16);
  std::vector<BF16> physical(description.required_bytes().value() / sizeof(BF16), BF16::from_bits(0x7fc1));
  for (std::size_t i = 0; i < values.size(); ++i)
    physical[(i / width) * stride + i % width] = values[i];
  return upload<BF16>(resource, physical);
}
struct Chain final {
  g::ValueId x, epsilon;
  std::array<g::ValueId, 5> stages;
  g::ValueId reader;
};
Chain add_chain(g::GraphBuilder &builder, const c::TensorDesc &input, bool publish_all) {
  Chain chain{.x = builder.input(input).value(),
      .epsilon = builder.input(dense(c::DataType::kBFloat16, {1})).value()};
  chain.stages[0] = builder.add(c::Binary{c::BinaryKind::kMultiply}, chain.x, chain.x).value();
  chain.stages[1] = builder.add(sum_last_axis(), chain.stages[0]).value();
  chain.stages[2] = builder.add(c::Binary{c::BinaryKind::kAdd}, chain.stages[1], chain.epsilon).value();
  chain.stages[3] = builder.add(c::Unary{c::UnaryKind::kRsqrt}, chain.stages[2]).value();
  chain.stages[4] = builder.add(c::Binary{c::BinaryKind::kMultiply}, chain.x, chain.stages[3]).value();
  EXPECT_TRUE(builder.publish(chain.stages[4]).ok());
  if (publish_all) {
    for (std::size_t i = 0; i < 4; ++i) EXPECT_TRUE(builder.publish(chain.stages[i]).ok());
    chain.reader = builder.add(c::Copy{}, chain.stages[1]).value();
    EXPECT_TRUE(builder.publish(chain.reader).ok());
  }
  return chain;
}

void exercise(c::Context &ctx, hyle::mem::MemoryResourceRef resource,
    g::CompiledGraph &compiled, const c::TensorDesc &input, Chain chain, bool publish_all) {
  const auto descriptions = stage_descriptions(input);
  struct Export final { g::ValueId value; std::size_t stage; };
  std::vector<Export> exports{{chain.stages[4], 4}};
  if (publish_all) {
    for (std::size_t i = 0; i < 4; ++i) exports.push_back({chain.stages[i], i});
    exports.push_back({chain.reader, 1});
  }
  // Buffers outlive both queues and every invocation borrowing them.
  std::vector<Buffer> retained;
  std::array queues{ctx.create_queue().value(), ctx.create_queue().value()};
  std::array slots{compiled.executable.create_invocation().value(),
                  compiled.executable.create_invocation().value()};
  std::array<Reference, 2> old;
  const auto port_count = exports.size() + 2;
  for (int round = 0; round < 2; ++round) {
    std::array<std::size_t, 2> base;
    std::array<Reference, 2> expected;
    std::array<c::Completion, 2> done;
    for (std::size_t slot = 0; slot < 2; ++slot) {
      const auto count = input.element_count().value();
      const auto values = input_values(count, std::size_t(input.dimensions().back()),
          1 + 3 * round + int(slot));
      const auto epsilon = slot == 0 ? half_test::encode<BF16>(round ? -1.e-6 : 1.e-6)
          : (round ? BF16::from_bits(0x0080) : half_test::encode<BF16>(0.0));
      expected[slot] = reference(ctx, resource, descriptions, values, epsilon);
      for (std::size_t i = 0; i < 5; ++i)
        ASSERT_EQ(expected[slot].values[i].size(), descriptions[i].element_count().value());
      std::size_t rounded_squares = 0;
      for (std::size_t i = 0; i < count; ++i) {
        const auto x = half_test::decode(values[i]);
        if (std::isfinite(x) && std::abs(x) >= 1.0 / 128 && std::abs(x) < 2)
          rounded_squares += half_test::decode(expected[slot].values[0][i]) != x * x;
      }
      EXPECT_GT(rounded_squares, 0U);
      base[slot] = retained.size();
      retained.push_back(source_buffer(resource, input, values));
      retained.push_back(upload<BF16>(resource, std::span(&epsilon, 1)));
      for (const auto output : exports)
        retained.push_back(output_buffer(resource, descriptions[output.stage]));
      if (round)
        for (std::size_t port = 0; port < port_count; ++port)
          EXPECT_NE(retained[base[slot] + port].native_handle(), retained[slot * port_count + port].native_handle());
      ASSERT_TRUE(slots[slot].bind(chain.x,
          c::ConstTensorView::create(input, retained[base[slot]].view()).value()).ok());
      ASSERT_TRUE(slots[slot].bind(chain.epsilon,
          c::ConstTensorView::create(dense(c::DataType::kBFloat16, {1}),
              retained[base[slot] + 1].view()).value()).ok());
      for (std::size_t i = 0; i < exports.size(); ++i)
        ASSERT_TRUE(slots[slot].bind(exports[i].value,
            c::TensorView::create(descriptions[exports[i].stage],
                retained[base[slot] + 2 + i].view()).value()).ok());
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
      for (std::size_t i = 0; i < exports.size(); ++i)
        check_bits(retained[base[slot] + 2 + i], expected[slot].values[exports[i].stage]);
      if (!round) old[slot] = expected[slot];
      else
        for (std::size_t i = 0; i < exports.size(); ++i)
          check_bits(retained[slot * port_count + 2 + i], old[slot].values[exports[i].stage]);
    }
  }
}

TEST(GgmlCudaGdnL2Fusion, ExactConsumerChainMatchesStagedNativeAcrossSlots) {
  auto ctx = c::Context::create(c::cuda::device(0).value(), g::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(ctx, c::cuda::MemoryKind::kDevice).value();
  // Also exercise SUM_ROWS' >=1024-column launch branch at enough rows.
  for (const auto [tokens, width] : {std::pair{1, 128}, std::pair{128, 128},
                                    std::pair{2, 128}, std::pair{16, 1024}}) {
    SCOPED_TRACE(::testing::Message() << "tokens=" << tokens);
    const auto input = dense(c::DataType::kBFloat16, {1, tokens, 16, width});
    auto builder = g::GraphBuilder::create().value();
    const auto chain = add_chain(builder, input, false);
    const auto graph = std::move(builder).finish().value();
    auto compiled = compile_graph(graph, ctx, memory.ref());
    ASSERT_TRUE(compiled) << compiled.error().message();
    exercise(ctx, memory.ref(), *compiled, input, chain, false);
  }
}

TEST(GgmlCudaGdnL2Fusion, PublishedIntermediatesAndPaddedRowsRetainEveryBoundary) {
  auto ctx = c::Context::create(c::cuda::device(0).value(), g::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(ctx, c::cuda::MemoryKind::kDevice).value();
  constexpr std::int64_t stride = 132 * sizeof(BF16);
  const auto input = c::TensorDesc::strided_bytes(c::DataType::kBFloat16,
      std::array<std::int64_t, 4>{1, 2, 16, 128},
      std::array<std::int64_t, 4>{2 * 16 * stride, 16 * stride, stride, sizeof(BF16)}).value();
  auto builder = g::GraphBuilder::create().value();
  const auto chain = add_chain(builder, input, true);
  const auto graph = std::move(builder).finish().value();
  auto compiled = compile_graph(graph, ctx, memory.ref());
  ASSERT_TRUE(compiled) << compiled.error().message();
  // Published boundaries and the extra sum reader must remain observable;
  // a generic region/CONT fallback is legal. Numerical checks do not claim
  // that every region disappears or that a particular kernel was launched.
  exercise(ctx, memory.ref(), *compiled, input, chain, true);
}
} // namespace

namespace {
namespace l2_epilogue_test {

template <class T> std::uint32_t bits(T value) {
  if constexpr (std::is_same_v<T, float>) return std::bit_cast<std::uint32_t>(value);
  else return value.bits;
}
template <class T>
void check(const Buffer &buffer, const std::vector<T> &expected, bool classify_nan = true) {
  const auto actual = half_test::read<T>(buffer);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (classify_nan && std::isnan(half_test::decode(expected[i])))
      EXPECT_TRUE(std::isnan(half_test::decode(actual[i]))) << i;
    else EXPECT_EQ(bits(actual[i]), bits(expected[i])) << i;
  }
}

struct Case {
  std::int64_t tokens;
  bool row_numerator = false, float_output = true, published_core = false;
};
struct Oracle {
  Reference rounded;
  std::vector<float> wide;
  std::vector<BF16> reader;
};

// The complete candidate pattern never occurs in one reference invocation.
// In particular the reference DIV consumes a real dense per-row numerator,
// expanded by bit-copy on the host, so it does not share the new scalar VIEW
// lowering. SUM_ROWS keeps the full original rank/row count and reduction tree.
Oracle reference_with_numerator(c::Context &ctx, hyle::mem::MemoryResourceRef resource,
    const std::array<c::TensorDesc, 5> &bf16, const std::vector<BF16> &values,
    BF16 epsilon, const std::vector<float> &row_numerators, bool reader) {
  Oracle result;
  const auto reduced_float = c::TensorDesc::dense(c::DataType::kFloat32, bf16[1].dimensions()).value();
  const auto full_float = c::TensorDesc::dense(c::DataType::kFloat32, bf16[4].dimensions()).value();
  const std::array descriptions{bf16[0], bf16[1], bf16[2], reduced_float,
      reduced_float, reduced_float, bf16[3], bf16[4], full_float, bf16[4]};
  auto source = upload<BF16>(resource, values);
  auto eps = upload<BF16>(resource, std::span(&epsilon, 1));
  auto numerators = upload<float>(resource, row_numerators);
  const auto x = c::ConstTensorView::create(bf16[0], source.view()).value();
  const auto e = c::ConstTensorView::create(dense(c::DataType::kBFloat16, {1}), eps.view()).value();
  const auto n = c::ConstTensorView::create(reduced_float, numerators.view()).value();
  std::array<Buffer, 10> outputs;
  const auto view = [&](std::size_t i) {
    return c::ConstTensorView::create(descriptions[i], outputs[i].view()).value();
  };
  const auto run = [&](c::Operation operation,
      std::initializer_list<c::ConstTensorView> inputs, std::size_t stage) {
    outputs[stage] = output_buffer(resource, descriptions[stage]);
    const auto status = execute(ctx, operation,
        std::span<const c::ConstTensorView>(inputs.begin(), inputs.size()),
        c::TensorView::create(descriptions[stage], outputs[stage].view()).value());
    EXPECT_TRUE(status.ok()) << status.message();
    return status.ok();
  };
  if (!run(c::Operation(c::Binary{c::BinaryKind::kMultiply}), {x, x}, 0) ||
      !run(c::Operation(sum_last_axis()), {view(0)}, 1) ||
      !run(c::Operation(c::Binary{c::BinaryKind::kAdd}), {view(1), e}, 2) ||
      !run(c::Operation(c::Cast{c::DataType::kFloat32}), {view(2)}, 3) ||
      !run(c::Operation(c::Unary{c::UnaryKind::kSqrt}), {view(3)}, 4) ||
      !run(c::Operation(c::Binary{c::BinaryKind::kDivide}), {n, view(4)}, 5) ||
      !run(c::Operation(c::Cast{c::DataType::kBFloat16}), {view(5)}, 6) ||
      !run(c::Operation(c::Binary{c::BinaryKind::kMultiply}), {x, view(6)}, 7) ||
      !run(c::Operation(c::Cast{c::DataType::kFloat32}), {view(7)}, 8))
    return result;
  constexpr std::array<std::size_t, 5> rounded_stages{0, 1, 2, 6, 7};
  for (std::size_t i = 0; i < rounded_stages.size(); ++i)
    result.rounded.values[i] = half_test::read<BF16>(outputs[rounded_stages[i]]);
  result.wide = half_test::read<float>(outputs[8]);
  if (reader && run(c::Operation(c::Unary{c::UnaryKind::kNeg}), {view(7)}, 9))
    result.reader = half_test::read<BF16>(outputs[9]);
  return result;
}

void exercise(Case test) {
  SCOPED_TRACE(::testing::Message() << "tokens=" << test.tokens << " row="
      << test.row_numerator << " f32=" << test.float_output << " published=" << test.published_core);
  auto ctx = c::Context::create(c::cuda::device(0).value(), g::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(ctx, c::cuda::MemoryKind::kDevice).value();
  const auto input = dense(c::DataType::kBFloat16, {1, test.tokens, 16, 128});
  const auto descriptions = stage_descriptions(input);
  const auto full_float = c::TensorDesc::dense(c::DataType::kFloat32, input.dimensions()).value();
  const auto reduced_float = c::TensorDesc::dense(c::DataType::kFloat32, descriptions[1].dimensions()).value();
  const auto numerator_description = test.row_numerator ? reduced_float : dense(c::DataType::kFloat32, {1});
  auto builder = g::GraphBuilder::create().value();
  const auto x = builder.input(input).value();
  const auto epsilon = builder.input(dense(c::DataType::kBFloat16, {1})).value();
  const auto numerator = builder.input(numerator_description).value();
  const auto square = builder.add(c::Binary{c::BinaryKind::kMultiply}, x, x).value();
  const auto sum = builder.add(sum_last_axis(), square).value();
  const auto shifted = builder.add(c::Binary{c::BinaryKind::kAdd}, sum, epsilon).value();
  const auto wide_shift = builder.add(c::Cast{c::DataType::kFloat32}, shifted).value();
  const auto root = builder.add(c::Unary{c::UnaryKind::kSqrt}, wide_shift).value();
  const auto quotient = builder.add(c::Binary{c::BinaryKind::kDivide}, numerator, root).value();
  const auto inverse = builder.add(c::Cast{c::DataType::kBFloat16}, quotient).value();
  const auto core = builder.add(c::Binary{c::BinaryKind::kMultiply}, x, inverse).value();
  const auto output = test.float_output ? builder.add(c::Cast{c::DataType::kFloat32}, core).value() : core;
  struct Export { g::ValueId value; c::TensorDesc description; bool wide = false, reader = false; };
  std::vector<Export> exports{{output, test.float_output ? full_float : descriptions[4], test.float_output}};
  ASSERT_TRUE(builder.publish(output).ok());
  if (test.published_core) {
    ASSERT_TRUE(test.float_output);
    const auto reader = builder.add(c::Unary{c::UnaryKind::kNeg}, core).value();
    ASSERT_TRUE(builder.publish(core).ok());
    ASSERT_TRUE(builder.publish(reader).ok());
    exports.push_back({core, descriptions[4]});
    exports.push_back({reader, descriptions[4], false, true});
  }
  auto compiled = compile_graph(std::move(builder).finish().value(), ctx, memory.ref());
  ASSERT_TRUE(compiled) << compiled.error().message();
  // No route/kernel-count assertion: a generic region is a legal fallback.
  // The public BF16 output and its extra reader must survive any cast fusion.
  std::vector<Buffer> retained;
  std::array queues{ctx.create_queue().value(), ctx.create_queue().value()};
  std::array slots{compiled->executable.create_invocation().value(),
                  compiled->executable.create_invocation().value()};
  std::array<Oracle, 2> old;
  const std::size_t ports = 3 + exports.size();
  for (int round = 0; round < 2; ++round) {
    std::array<Oracle, 2> expected;
    std::array<std::size_t, 2> base;
    std::array<std::vector<BF16>, 2> source_values;
    std::array<std::vector<float>, 2> numerator_values;
    for (std::size_t slot = 0; slot < 2; ++slot) {
      source_values[slot] = input_values(input.element_count().value(), 128,
          1 + 3 * round + int(slot));
      const auto e = slot == 0 ? half_test::encode<BF16>(round ? -1.e-6 : 1.e-6)
          : (round ? BF16::from_bits(0x0080) : half_test::encode<BF16>(0));
      const std::array<float, 4> scalar_values{-0.75F, 2.5F,
          std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
      const std::array<float, 5> row_values{-0.75F, 2.5F, 0.0F,
          -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
      const auto rows = descriptions[1].element_count().value();
      std::vector<float> expanded(rows);
      if (test.row_numerator) {
        for (std::size_t row = 0; row < rows; ++row)
          expanded[row] = row_values[(row + std::size_t(round) + slot) % row_values.size()];
        numerator_values[slot] = expanded;
      } else {
        const float value = scalar_values[std::size_t(round) * 2 + slot];
        numerator_values[slot] = {value};
        std::fill(expanded.begin(), expanded.end(), value);
      }
      expected[slot] = reference_with_numerator(ctx, memory.ref(), descriptions,
          source_values[slot], e, expanded, test.published_core);
      ASSERT_EQ(expected[slot].wide.size(), input.element_count().value());
      for (std::size_t i = 0; i < descriptions.size(); ++i)
        ASSERT_EQ(expected[slot].rounded.values[i].size(), descriptions[i].element_count().value());
      if (test.published_core) ASSERT_EQ(expected[slot].reader.size(), input.element_count().value());
      bool finite_numerator = false;
      for (float value : expanded) finite_numerator |= std::isfinite(value) && value != 0;
      std::size_t final_rounding_witnesses = 0;
      for (std::size_t i = 0; i < source_values[slot].size(); ++i) {
        const float a = static_cast<float>(half_test::decode(source_values[slot][i]));
        const float b = static_cast<float>(half_test::decode(expected[slot].rounded.values[3][i / 128]));
        // Products of two BF16 values are exact F32 in this ordinary range.
        // A direct F32 store here would delete the final BF16 RNE boundary.
        if (std::isfinite(a) && std::isfinite(b) && std::abs(a) >= 1.0F / 128 &&
            std::abs(a) < 2 && std::abs(b) >= 1.0F / 128 && std::abs(b) < 1024)
          final_rounding_witnesses += bits(expected[slot].wide[i]) != bits(a * b);
      }
      if (finite_numerator) EXPECT_GT(final_rounding_witnesses, 0U);
      base[slot] = retained.size();
      retained.push_back(upload<BF16>(memory.ref(), source_values[slot]));
      retained.push_back(upload<BF16>(memory.ref(), std::span(&e, 1)));
      retained.push_back(upload<float>(memory.ref(), numerator_values[slot]));
      for (const auto &entry : exports) retained.push_back(output_buffer(memory.ref(), entry.description));
      if (round)
        for (std::size_t port = 0; port < ports; ++port)
          EXPECT_NE(retained[base[slot] + port].native_handle(), retained[slot * ports + port].native_handle());
      ASSERT_TRUE(slots[slot].bind(x, c::ConstTensorView::create(input, retained[base[slot]].view()).value()).ok());
      ASSERT_TRUE(slots[slot].bind(epsilon, c::ConstTensorView::create(
          dense(c::DataType::kBFloat16, {1}), retained[base[slot] + 1].view()).value()).ok());
      ASSERT_TRUE(slots[slot].bind(numerator, c::ConstTensorView::create(
          numerator_description, retained[base[slot] + 2].view()).value()).ok());
      for (std::size_t i = 0; i < exports.size(); ++i)
        ASSERT_TRUE(slots[slot].bind(exports[i].value, c::TensorView::create(
            exports[i].description, retained[base[slot] + 3 + i].view()).value()).ok());
    }
    for (const auto mode : {g::ReplayMode::kCommands, g::ReplayMode::kRequireCapture}) {
      std::array<c::Completion, 2> done;
      for (std::size_t slot = 0; slot < 2; ++slot) {
        const std::array bindings{g::QueueBinding{g::ExecutionLaneId{0}, &queues[slot]}};
        const auto prepared = slots[slot].prepare_replay(bindings, {.mode = mode});
        ASSERT_TRUE(prepared.ok()) << prepared.message();
        EXPECT_EQ(slots[slot].replay_route(), mode == g::ReplayMode::kCommands
            ? g::ReplayRoute::kCommands : g::ReplayRoute::kCaptured);
      }
      for (std::size_t slot = 0; slot < 2; ++slot) {
        auto result = slots[slot].submit(queues[slot]);
        ASSERT_TRUE(result) << result.error().message();
        done[slot] = std::move(*result);
      }
      ASSERT_TRUE(done[1].wait().ok()); ASSERT_TRUE(done[0].wait().ok());
      for (std::size_t slot = 0; slot < 2; ++slot) {
        for (std::size_t i = 0; i < exports.size(); ++i) {
          const auto &entry = exports[i];
          const auto &buffer = retained[base[slot] + 3 + i];
          if (entry.wide) check(buffer, expected[slot].wide);
          else check(buffer, entry.reader ? expected[slot].reader : expected[slot].rounded.values[4]);
        }
        check(retained[base[slot]], source_values[slot], false);
        check(retained[base[slot] + 2], numerator_values[slot], false);
      }
    }
    for (std::size_t slot = 0; slot < 2; ++slot) {
      if (!round) {
        // Snapshot the actual result, including its NaN payload. Agreement
        // with the independent oracle intentionally classifies NaNs only.
        for (std::size_t i = 0; i < exports.size(); ++i) {
          const auto &entry = exports[i];
          const auto &buffer = retained[base[slot] + 3 + i];
          if (entry.wide) old[slot].wide = half_test::read<float>(buffer);
          else if (entry.reader) old[slot].reader = half_test::read<BF16>(buffer);
          else old[slot].rounded.values[4] = half_test::read<BF16>(buffer);
        }
      } else for (std::size_t i = 0; i < exports.size(); ++i) {
        const auto &entry = exports[i];
        const auto &buffer = retained[slot * ports + 3 + i];
        if (entry.wide) check(buffer, old[slot].wide, false);
        else check(buffer, entry.reader ? old[slot].reader : old[slot].rounded.values[4], false);
      }
    }
  }
}

template <class T> void generic_scalar_divide(std::initializer_list<std::int64_t> shape) {
  auto ctx = c::Context::create(c::cuda::device(0).value(), g::ggml_cuda::provider()).value();
  auto memory = c::cuda::MemoryResource::create(ctx, c::cuda::MemoryKind::kDevice).value();
  const auto type = half_test::data_type<T>();
  const auto scalar = dense(type, {1});
  const auto description = dense(type, shape);
  auto prepared = c::prepare(ctx, c::Binary{c::BinaryKind::kDivide},
      scalar, description, memory.ref());
  ASSERT_TRUE(prepared) << prepared.error().message();
  std::vector<Buffer> retained;
  auto workspace = ggml_cuda_test::workspace_buffer(*prepared, memory.ref());
  auto queue = ctx.create_queue().value();
  auto run = prepared->create_invocation(workspace.view()).value();
  const std::array<double, 6> divisors{0.5, -2.0, 0.0, -0.0,
      std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()};
  std::vector<T> denominator;
  for (std::size_t i = 0; i < description.element_count().value(); ++i)
    denominator.push_back(half_test::encode<T>(divisors[i % divisors.size()]));
  std::vector<T> old;
  // Independent full-shape numerator avoids the scalar VIEW in the oracle.
  // Reuse one real prepared Invocation, changing the scalar and all addresses.
  for (const double value : {-0.75, 2.5, std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
    const T n = half_test::encode<T>(value);
    std::vector<T> expanded(denominator.size(), n);
    auto numerator = upload<T>(memory.ref(), std::span(&n, 1));
    auto divisor = upload<T>(memory.ref(), denominator);
    auto output = output_buffer(memory.ref(), description);
    auto dense_numerator = upload<T>(memory.ref(), expanded);
    auto oracle = output_buffer(memory.ref(), description);
    const std::array oracle_inputs{
        c::ConstTensorView::create(description, dense_numerator.view()).value(),
        c::ConstTensorView::create(description, divisor.view()).value()};
    const auto status = execute(ctx, c::Binary{c::BinaryKind::kDivide}, oracle_inputs,
        c::TensorView::create(description, oracle.view()).value());
    ASSERT_TRUE(status.ok()) << status.message();
    auto done = run.submit(queue,
        c::ConstTensorView::create(scalar, numerator.view()).value(),
        c::ConstTensorView::create(description, divisor.view()).value(),
        c::TensorView::create(description, output.view()).value());
    ASSERT_TRUE(done) << done.error().message();
    ASSERT_TRUE(done->wait().ok());
    check(output, half_test::read<T>(oracle));
    if (retained.empty()) old = half_test::read<T>(output);
    retained.push_back(std::move(numerator)); retained.push_back(std::move(divisor));
    retained.push_back(std::move(output)); retained.push_back(std::move(dense_numerator));
    retained.push_back(std::move(oracle));
    check(retained[2], old, false);
  }
}

TEST(GgmlCudaGdnL2Fusion, DynamicNumeratorAndFloatOutputMatchStagedNativeAcrossSlots) {
  exercise({1});
  exercise({128});
}

TEST(GgmlCudaGdnL2Fusion, ReadersAndPerRowNumeratorsKeepObservableBoundaries) {
  exercise({2, true, true, false});
  exercise({2, false, true, true});
  exercise({1, false, false, false});
  generic_scalar_divide<float>({2, 3});
  generic_scalar_divide<hyle::simd::float16_t>({2, 2, 3});
}
} // namespace l2_epilogue_test
} // namespace
