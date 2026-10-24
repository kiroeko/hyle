#pragma once

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
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "hyle/compute/resource_leases.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/result.h"

namespace fp8_channel_token_qdq_test {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
using Type = c::DataType;
inline constexpr std::int64_t batch = 1, tokens = 3, depth = 16, channels = 4;

template<class T> T take(hyle::Result<T> result, std::string_view where) {
  if (!result) throw std::runtime_error(std::string(where) + ": " + std::string(result.error().message()));
  return std::move(*result);
}
inline void check(hyle::Status status, std::string_view where) {
  if (!status.ok()) throw std::runtime_error(std::string(where) + ": " + std::string(status.message()));
}
inline c::TensorDesc dense(Type type, std::initializer_list<std::int64_t> shape) {
  return take(c::TensorDesc::dense(type, {shape.begin(), shape.size()}), "dense description");
}
// Independent finite storage oracle: no product decode, SIMD pack or Torch.
inline std::uint16_t bf16(float value) {
  auto bits = std::bit_cast<std::uint32_t>(value);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<std::uint16_t>(bits >> 16);
}
inline float wide(std::uint16_t bits) {
  return std::bit_cast<float>(std::uint32_t(bits) << 16);
}
inline float rounded(float value) { return wide(bf16(value)); }
inline float e4m3(std::uint8_t code) {
  const auto magnitude = code & 0x7f;
  const int exponent = magnitude >> 3, fraction = magnitude & 7;
  const float value = exponent == 0 ? std::ldexp(float(fraction), -9)
      : std::ldexp(1.F + float(fraction) / 8.F, exponent - 7);
  return code & 0x80 ? -value : value;
}
inline std::uint8_t encode_e4m3(float value) {
  const double magnitude = std::min(std::abs(double(value)), 448.);
  unsigned best = 0;
  double distance = magnitude;
  for (unsigned code = 1; code <= 0x7e; ++code) {
    const double next = std::abs(magnitude - double(e4m3(static_cast<std::uint8_t>(code))));
    if (next < distance || (next == distance && (code & 1u) == 0)) {
      best = code; distance = next;
    }
  }
  return static_cast<std::uint8_t>(best | (std::signbit(value) ? 0x80u : 0u));
}
template<class T> std::vector<std::byte> bytes(std::span<const T> source) {
  const auto raw = std::as_bytes(source);
  return {raw.begin(), raw.end()};
}

struct Case final {
  c::TensorDesc activation_description = dense(Type::kBFloat16, {batch, tokens, depth});
  c::TensorDesc output_description = dense(Type::kBFloat16, {batch * tokens, channels});
  std::array<c::TensorDesc, 5> descriptions{
      dense(Type::kFloat8E4M3FN, {channels, depth}), dense(Type::kBFloat16, {channels, 1}),
      dense(Type::kBFloat16, {}), dense(Type::kBFloat16, {}), dense(Type::kBFloat16, {})};
  std::array<std::uint8_t, std::size_t(channels * depth)> weights{};
  const std::array<std::uint16_t, std::size_t(channels)> scales{0x3f81, 0x3ef1, 0x3f03, 0x3fdd};
  std::array<std::vector<std::byte>, 5> payloads;
  g::Graph graph;
  g::ValueId activation{}, output{};
  std::array<g::ValueId, 5> constants{};

  Case() {
    for (std::int64_t n = 0; n < channels; ++n) for (std::int64_t k = 0; k < depth; ++k)
      weights[std::size_t(n * depth + k)] = static_cast<std::uint8_t>(
          0x38 + (k * 3 + n * 5) % 8 + ((k + n) % 5 == 0 ? 0x80 : 0));
    const std::array<std::uint16_t, 1> maximum{bf16(448.F)}, minimum{bf16(-448.F)}, epsilon{bf16(1.F / 128.F)};
    payloads = {bytes<std::uint8_t>(weights), bytes<std::uint16_t>(scales),
        bytes<std::uint16_t>(maximum), bytes<std::uint16_t>(minimum), bytes<std::uint16_t>(epsilon)};
    auto b = take(g::GraphBuilder::create(), "GraphBuilder create");
    activation = take(b.input(activation_description), "rank3 BF16 activation input");
    for (std::size_t i = 0; i < constants.size(); ++i)
      constants[i] = take(b.constant(descriptions[i]), "immutable CT code/scale/scalar constant");
    const auto absolute = take(b.add(c::Unary{c::UnaryKind::kAbs}, activation), "BF16 abs");
    const auto maximum_value = take(b.add(c::Reduce{.kind = c::ReductionKind::kMaximum,
        .axes = c::Axes::from(std::array<std::int32_t, 1>{2}).value(), .keep_dimensions = true, .accumulation_type = Type::kFloat32}, absolute), "BF16 last-axis absmax");
    const auto raw_scale = take(b.add(c::Binary{c::BinaryKind::kDivide}, maximum_value, constants[2]), "BF16 absmax/448 scale");
    const auto nonzero = take(b.add(c::Cast{Type::kBool}, raw_scale), "scale nonzero via Cast Bool");
    const auto scale = take(b.add(c::Select{}, nonzero, raw_scale, constants[4]), "zero-only BF16 epsilon Select");
    const auto normalized = take(b.add(c::Binary{c::BinaryKind::kDivide}, activation, scale), "BF16 activation/scale");
    const auto lower = take(b.add(c::Binary{c::BinaryKind::kMaximum}, normalized, constants[3]), "BF16 clamp lower");
    const auto clipped = take(b.add(c::Binary{c::BinaryKind::kMinimum}, lower, constants[2]), "BF16 clamp upper");
    const auto encoded = take(b.add(c::Cast{Type::kFloat8E4M3FN}, clipped), "activation FP8 RNE");
    const auto widened = take(b.add(c::Cast{Type::kBFloat16}, encoded), "activation codes to BF16");
    const auto qdq = take(b.add(c::Binary{c::BinaryKind::kMultiply}, widened, scale), "BF16 activation dequant round");
    const auto flat_shape = c::TensorShape::create(std::array<std::int64_t, 2>{batch * tokens, depth}).value();
    const auto flat = take(b.add(c::View::reshape(flat_shape).value(), qdq), "flatten only after rank3 QDQ");
    const auto codes = take(b.add(c::Cast{Type::kBFloat16}, constants[0]), "weight codes to BF16");
    const auto weight = take(b.add(c::Binary{c::BinaryKind::kMultiply}, codes, constants[1]), "BF16 channel weight dequant round");
    const auto transpose = take(b.add(c::View::permute(std::array<std::int32_t, 2>{1, 0}).value(), weight), "dequant weight transpose");
    output = take(b.add(c::MatMul{.accumulation_type = Type::kFloat32}, flat, transpose), "ordinary BF16 MatMul/F32 accumulation");
    check(b.publish(output), "publish only final CT QDQ result");
    graph = take(std::move(b).finish(), "finish ordinary CT QDQ graph");
  }
  std::array<std::uint16_t, std::size_t(batch * tokens * depth)> activations(unsigned phase) const {
    std::array<std::uint16_t, std::size_t(batch * tokens * depth)> a{};
    for (std::int64_t k = 0; k < depth; ++k) {
      // Token0 all zero. Token1 has nonzero scale far below BF16 epsilon.
      a[std::size_t(depth + k)] = bf16(float((k * 5 + phase * 2) % 9 - 4) / 256.F);
      a[std::size_t(2 * depth + k)] = bf16(float((k * 3 + phase * 4) % 15 - 7) * 2.F);
    }
    return a;
  }
  std::vector<std::uint16_t> oracle(unsigned phase, bool wrong_floor = false, bool omit_weight_round = false) const {
    const auto a = activations(phase);
    std::vector<std::uint16_t> result(std::size_t(batch * tokens * channels));
    for (std::int64_t t = 0; t < tokens; ++t) {
      float amax = 0;
      for (std::int64_t k = 0; k < depth; ++k) amax = std::max(amax, std::abs(wide(a[std::size_t(t * depth + k)])));
      float scale = rounded(static_cast<float>(double(amax) / 448.));
      if (wrong_floor) scale = std::max(scale, 1.F / 128.F);
      else if (scale == 0) scale = 1.F / 128.F;
      for (std::int64_t n = 0; n < channels; ++n) {
        double dot = 0;
        for (std::int64_t k = 0; k < depth; ++k) {
          const float divided = rounded(static_cast<float>(double(wide(a[std::size_t(t * depth + k)])) / double(scale)));
          const auto code = encode_e4m3(std::clamp(divided, -448.F, 448.F));
          const float x = rounded(static_cast<float>(double(e4m3(code)) * double(scale)));
          const float w = static_cast<float>(double(e4m3(weights[std::size_t(n * depth + k)])) * double(wide(scales[std::size_t(n)])));
          dot += double(x) * double(omit_weight_round ? w : rounded(w));
        }
        // Short BF16 products sum exactly in F32 here; final narrowing is RNE.
        result[std::size_t(t * channels + n)] = bf16(static_cast<float>(dot));
      }
    }
    return result;
  }
};

template<class Read> void expect_result(const g::GraphOutput &output, std::span<const std::uint16_t> expected, Read read) {
  auto tensor = output.tensor();
  ASSERT_TRUE(tensor) << tensor.error().message();
  std::vector<std::uint16_t> actual(expected.size());
  const auto status = read(tensor->storage(), std::as_writable_bytes(std::span(actual)));
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(actual, std::vector<std::uint16_t>(expected.begin(), expected.end()));
}

template<class Compile, class Write, class Read, class Prepare> void run(m::MemoryResourceRef resource,
    std::span<const g::QueueBinding> queues, Compile compile, Write write, Read read, Prepare prepare, c::StorageOwner constant_owner = {}) {
  Case f;
  bool floor_witness = false, weight_witness = false;
  for (unsigned phase = 0; phase < 4; ++phase) {
    const auto expected = f.oracle(phase);
    floor_witness |= expected != f.oracle(phase, true);
    weight_witness |= expected != f.oracle(phase, false, true);
    for (std::int64_t n = 0; n < channels; ++n) EXPECT_EQ(expected[std::size_t(n)], 0u);
  }
  ASSERT_TRUE(floor_witness) << "payload must reject replacing only zero scales by an epsilon floor";
  ASSERT_TRUE(weight_witness) << "payload must expose BF16 weight dequant rounding before MatMul";
  std::array<c::ReadLease, 5> ready;
  std::array<g::ConstantBinding, 5> constants;
  std::size_t constant_bytes = 0;
  for (std::size_t i = 0; i < constants.size(); ++i) {
    auto source = take(m::Buffer::create(resource, f.payloads[i].size(), 256), "constant source buffer");
    check(write(source.view(), f.payloads[i]), "constant payload write");
    ready[i] = constant_owner ? take(constant_owner.adopt_buffer(std::move(source)), "owned constant ready lease")
        : take(c::ReadLease::adopt(std::move(source)), "constant source ready lease");
    constants[i] = {f.constants[i], take(ready[i].tensor(f.descriptions[i]), "constant typed source"), g::ConstantPolicy::kMaterialize,
        constant_owner, constant_owner ? ready[i] : c::ReadLease{}};
    constant_bytes += f.payloads[i].size();
  }
  auto compiled = compile(f.graph, g::CompileOptions{.constants = constants});
  ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->executable.input_count(), 1u);
  EXPECT_EQ(compiled->executable.output_count(), 1u);
  EXPECT_EQ(compiled->report.materialized_constant_count(), constants.size());
  EXPECT_EQ(compiled->report.materialized_constant_bytes(), constant_bytes);
  const auto ownership = take(compiled->executable.output_ownership(), "actual output ownership");
  const bool caller_output = ownership == g::OutputOwnership::kCaller;
  if (!caller_output) ASSERT_EQ(ownership, g::OutputOwnership::kInvocation);
  std::array<g::Invocation, 2> invocations{
      take(compiled->executable.create_invocation({.resources = g::ResourceOptions{}}), "first original invocation"),
      take(compiled->executable.create_invocation({.resources = g::ResourceOptions{}}), "second original invocation")};
  constants = {}; ready = {}; compiled = g::CompiledGraph{};
  const auto output_bytes = f.output_description.required_bytes().value();
  c::LeasePool outputs;
  // LeasePool::allocate defaults to 256-byte alignment; each logical 24-byte
  // output needs its own aligned origin, without changing tensor extents.
  const auto output_stride = (output_bytes + 255) / 256 * 256;
  if (caller_output) outputs = take(c::LeasePool::create(resource, 2 * output_stride, {.max_blocks = 2}), "two original output blocks");
  std::array<m::Buffer, 2> inputs;
  for (std::size_t slot = 0; slot < invocations.size(); ++slot) {
    inputs[slot] = take(m::Buffer::create(resource, f.activation_description.required_bytes().value(), 256), "activation buffer");
    check(invocations[slot].bind(f.activation, take(c::ConstTensorView::create(f.activation_description, inputs[slot].view()), "typed activation")), "bind original activation");
    if (caller_output) check(invocations[slot].bind(f.output, f.output_description,
        take(outputs.allocate(output_bytes), "original output writer")), "bind original owned output");
    check(prepare(invocations[slot], queues), "ordinary QDQ prepare_replay");
  }
  std::array<g::GraphOutput, 2> retained;
  std::array<std::vector<std::uint16_t>, 2> old;
  for (unsigned repeat = 0; repeat < 2; ++repeat) for (std::size_t slot = 0; slot < invocations.size(); ++slot) {
    const unsigned phase = repeat * 2 + static_cast<unsigned>(slot);
    SCOPED_TRACE(phase);
    if (repeat) {
      expect_result(retained[slot], old[slot], read);
      EXPECT_FALSE(invocations[slot].submit(queues));
      retained[slot].reset();
    }
    const auto a = f.activations(phase);
    check(write(inputs[slot].view(), std::as_bytes(std::span(a))), "fresh BF16 activation content");
    auto accepted = invocations[slot].submit(queues);
    ASSERT_TRUE(accepted) << accepted.error().message();
    const auto status = accepted->wait();
    ASSERT_TRUE(status.ok()) << status.message();
    retained[slot] = take(invocations[slot].result(f.output), "final QDQ result");
    old[slot] = f.oracle(phase);
    expect_result(retained[slot], old[slot], read);
    if (retained[1 - slot].valid()) expect_result(retained[1 - slot], old[1 - slot], read);
  }
  EXPECT_NE(f.oracle(0), f.oracle(2));
  EXPECT_NE(old[0], old[1]);
  invocations = {};
  for (std::size_t slot = 0; slot < retained.size(); ++slot) expect_result(retained[slot], old[slot], read);
  retained = {};
  if (caller_output) EXPECT_EQ(outputs.stats().live_blocks, 0u);
  ::testing::Test::RecordProperty("actual_submissions", 4);
}

// These are storage operations: inspect and compare IEEE bits without using
// floating arithmetic, which may flush subnormals or canonicalize NaN payloads.
template<class Compile, class Write, class Read, class Prepare>
void run_boolean_selection(m::MemoryResourceRef resource,
    std::span<const g::QueueBinding> queues, Compile compile, Write write, Read read, Prepare prepare,
    c::StorageOwner = {}) {
  for (const auto type : {Type::kFloat32, Type::kFloat16, Type::kBFloat16}) {
    SCOPED_TRACE(static_cast<int>(type));
    const bool f32 = type == Type::kFloat32;
    const std::uint32_t sign = f32 ? 0x80000000u : 0x8000u;
    const std::uint32_t infinity = f32 ? 0x7f800000u : type == Type::kFloat16 ? 0x7c00u : 0x7f80u;
    const std::uint32_t nan = f32 ? 0x7fc12345u : type == Type::kFloat16 ? 0x7e35u : 0x7fc5u;
    const std::uint32_t one = f32 ? 0x3f800000u : type == Type::kFloat16 ? 0x3c00u : 0x3f80u;
    const std::array<std::uint32_t, 7> patterns{0, sign, 1, sign | 1, infinity, sign | infinity, nan};
    const auto packed = [f32](std::span<const std::uint32_t> values) {
      if (f32) return bytes<std::uint32_t>(values);
      std::vector<std::uint16_t> narrow(values.begin(), values.end());
      return bytes<std::uint16_t>(narrow);
    };
    const auto column = dense(type, {7, 1});
    const auto mask = dense(Type::kBool, {7, 1});
    const auto row = dense(type, {1, 3});
    const auto matrix = dense(type, {7, 3});
    auto builder = take(g::GraphBuilder::create(), "Boolean storage GraphBuilder");
    const std::array inputs{
        take(builder.input(column), "IEEE floating column"),
        take(builder.input(mask), "noncanonical Boolean bytes"),
        take(builder.input(row), "true branch row"),
        take(builder.input(column), "false branch column")};
    const auto boolean = take(builder.add(c::Cast{Type::kBool}, inputs[0]), "IEEE nonzero to Boolean");
    const std::array outputs{boolean,
        take(builder.add(c::Cast{type}, boolean), "Boolean roundtrip to floating"),
        take(builder.add(c::Cast{type}, inputs[1]), "noncanonical Boolean to floating"),
        take(builder.add(c::Select{}, inputs[1], inputs[2], inputs[3]), "broadcast bit-preserving Select")};
    const std::array output_descriptions{mask, column, column, matrix};
    for (const auto output : outputs) check(builder.publish(output), "publish Boolean storage result");
    const auto graph = take(std::move(builder).finish(), "finish Boolean storage graph");
    auto compiled = compile(graph, g::CompileOptions{});
    ASSERT_TRUE(compiled) << compiled.error().message();
    const auto ownership = take(compiled->executable.output_ownership(), "Boolean output ownership");
    const bool caller_output = ownership == g::OutputOwnership::kCaller;
    if (!caller_output) ASSERT_EQ(ownership, g::OutputOwnership::kInvocation);
    auto invocation = take(compiled->executable.create_invocation({.resources = g::ResourceOptions{}}), "Boolean storage invocation");
    compiled = g::CompiledGraph{};
    c::LeasePool pool;
    if (caller_output) pool = take(c::LeasePool::create(resource, outputs.size() * 256,
        {.max_blocks = 4}), "four aligned Boolean result writers");
    const std::array input_descriptions{column, mask, row, column};
    std::array<m::Buffer, 4> buffers;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      buffers[i] = take(m::Buffer::create(resource, input_descriptions[i].required_bytes().value(), 256), "Boolean source buffer");
      check(invocation.bind(inputs[i], take(c::ConstTensorView::create(input_descriptions[i], buffers[i].view()), "Boolean typed source")), "bind Boolean source");
      if (caller_output) check(invocation.bind(outputs[i], output_descriptions[i],
          take(pool.allocate(output_descriptions[i].required_bytes().value()), "Boolean result writer")), "bind Boolean output");
    }
    check(prepare(invocation, queues), "Boolean storage prepare replay");
    std::array<g::GraphOutput, 4> retained;
    std::array<std::vector<std::byte>, 4> expected;
    const auto inspect = [&] {
      for (std::size_t i = 0; i < outputs.size(); ++i) {
        const auto tensor = take(retained[i].tensor(), "retained Boolean result");
        std::vector<std::byte> actual(expected[i].size());
        check(read(tensor.storage(), actual), "read exact Boolean result bytes");
        EXPECT_EQ(actual, expected[i]) << "output " << i;
      }
    };
    for (unsigned phase = 0; phase < 2; ++phase) {
      SCOPED_TRACE(phase);
      if (phase) {
        inspect();
        EXPECT_FALSE(invocation.submit(queues));
        retained = {};
      }
      const std::array<std::uint8_t, 7> bool_bytes = phase
          ? std::array<std::uint8_t, 7>{255, 0, 0, 128, 0, 2, 1}
          : std::array<std::uint8_t, 7>{0, 2, 255, 0, 1, 128, 0};
      const std::array<std::uint32_t, 3> true_bits{sign, nan | (phase ? sign : 0), 1u | (phase ? sign : 0)};
      std::array<std::uint32_t, 7> source{}, false_bits{}, normalized{}, raw_normalized{};
      std::array<std::uint8_t, 7> nonzero{};
      std::array<std::uint32_t, 21> selected{};
      for (std::size_t i = 0; i < source.size(); ++i) {
        source[i] = patterns[(i + phase) % patterns.size()];
        false_bits[i] = patterns[(i + phase + 2) % patterns.size()];
        nonzero[i] = (source[i] & ~sign) != 0 ? 1 : 0;
        normalized[i] = nonzero[i] ? one : 0;
        raw_normalized[i] = bool_bytes[i] ? one : 0;
        for (std::size_t j = 0; j < true_bits.size(); ++j)
          selected[i * true_bits.size() + j] = bool_bytes[i] ? true_bits[j] : false_bits[i];
      }
      const std::array payloads{packed(source), bytes<std::uint8_t>(bool_bytes), packed(true_bits), packed(false_bits)};
      expected = {bytes<std::uint8_t>(nonzero), packed(normalized), packed(raw_normalized), packed(selected)};
      for (std::size_t i = 0; i < inputs.size(); ++i) check(write(buffers[i].view(), payloads[i]), "update Boolean source content");
      auto accepted = invocation.submit(queues);
      ASSERT_TRUE(accepted) << accepted.error().message();
      check(accepted->wait(), "Boolean storage completion");
      for (std::size_t i = 0; i < outputs.size(); ++i) retained[i] = take(invocation.result(outputs[i]), "Boolean result lease");
      inspect();
    }
    invocation = {};
    inspect();
    retained = {};
    if (caller_output) EXPECT_EQ(pool.stats().live_blocks, 0u);
  }
  ::testing::Test::RecordProperty("actual_submissions", 6);
}
} // namespace fp8_channel_token_qdq_test
