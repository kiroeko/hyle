#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "hyle/compute/model_ops.h"
#include "hyle/compute/resource_leases.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/result.h"

namespace nvfp4_reference_test {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
using Type = c::DataType;

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

// Independent storage oracle. No product codec, NVFP4 helper, SIMD or Torch.
// The volatile store makes every named source F32 operation a rounding boundary.
inline float f32(double value) { volatile float rounded = static_cast<float>(value); return rounded; }
inline float multiply(float a, float b) { return f32(double(a) * double(b)); }
inline std::uint16_t bf16(float value) {
  auto bits = std::bit_cast<std::uint32_t>(value);
  if ((bits & 0x7f800000u) == 0x7f800000u) {
    const auto high = static_cast<std::uint16_t>(bits >> 16);
    return (bits & 0x007fffffu) ? static_cast<std::uint16_t>(high | 0x0040u) : high;
  }
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<std::uint16_t>(bits >> 16);
}
inline float wide(std::uint16_t bits) { return std::bit_cast<float>(std::uint32_t(bits) << 16); }
inline float rounded_bf16(float value) { return wide(bf16(value)); }
inline float e4m3(std::uint8_t code) {
  const unsigned magnitude = code & 0x7fu;
  if (magnitude == 0x7fu) return std::numeric_limits<float>::quiet_NaN();
  const int exponent = static_cast<int>(magnitude >> 3), fraction = static_cast<int>(magnitude & 7u);
  const float value = exponent == 0 ? std::ldexp(float(fraction), -9)
      : std::ldexp(1.F + float(fraction) / 8.F, exponent - 7);
  return code & 0x80u ? -value : value;
}
inline std::uint8_t encode_scale(float value) {
  const double magnitude = std::min(double(value), 448.);
  unsigned best = 0;
  double distance = magnitude;
  for (unsigned code = 1; code <= 0x7e; ++code) {
    const double next = std::abs(magnitude - double(e4m3(static_cast<std::uint8_t>(code))));
    if (next < distance || (next == distance && (code & 1u) == 0)) {
      best = code; distance = next;
    }
  }
  return static_cast<std::uint8_t>(best);
}
inline constexpr std::array<float, 16> e2m1{
    0.F, .5F, 1.F, 1.5F, 2.F, 3.F, 4.F, 6.F,
    0.F, -.5F, -1.F, -1.5F, -2.F, -3.F, -4.F, -6.F};
inline float nearest_e2m1(float value) {
  // Enumerate the independent positive code table; equal distances choose an
  // even low code bit. This does not repeat the producer's branch ladder.
  if (value >= 6.F) return 6.F;
  unsigned best = 0;
  double distance = double(value);
  for (unsigned code = 1; code != 8; ++code) {
    const double next = std::abs(double(value) - double(e2m1[code]));
    if (next < distance || (next == distance && (code & 1u) == 0)) {
      best = code; distance = next;
    }
  }
  return e2m1[best];
}
inline float activation_value(float value, float maximum, float global,
                              bool omit_round = false, bool omit_guard = false) {
  const float normalized = std::min(f32(double(maximum) / double(multiply(6.F, global))), 448.F);
  float scale = multiply(e4m3(encode_scale(normalized)), global);
  if (!omit_guard && !(scale >= 1.e-5F)) scale = 1.F;
  const float magnitude = f32(std::abs(double(value)) / double(scale));
  const float restored = multiply(nearest_e2m1(magnitude), scale);
  const float signed_value = value >= 0.F ? restored : -restored;
  return omit_round ? signed_value : rounded_bf16(signed_value);
}
inline float weight_value(unsigned code, std::uint8_t scale, float global,
                          bool omit_round = false, bool fast_tree = false) {
  const float value = fast_tree ? multiply(multiply(e2m1[code], e4m3(scale)), global)
      : multiply(e2m1[code], multiply(e4m3(scale), global));
  return omit_round ? value : rounded_bf16(value);
}
template<class T> std::vector<std::byte> bytes(std::span<const T> source) {
  const auto raw = std::as_bytes(source); return {raw.begin(), raw.end()};
}

struct Data final {
  std::vector<std::uint16_t> a;
  std::vector<std::uint8_t> w, scales;
  float weight_global = 1.00390625F, input_global = .13F;
  std::array<std::vector<std::byte>, 5> payloads() const {
    return {bytes<std::uint16_t>(a), bytes<std::uint8_t>(w), bytes<std::uint8_t>(scales),
        bytes<float>(std::span(&weight_global, 1)), bytes<float>(std::span(&input_global, 1))};
  }
};

struct Case final {
  std::int64_t rows = 2, columns, depth;
  std::array<c::TensorDesc, 5> descriptions;
  c::TensorDesc output_description;
  std::array<g::ValueId, 5> ports{};
  g::ValueId output{};
  g::Graph graph;

  Case(std::int64_t n, std::int64_t k, bool dynamic = false)
      : columns(n), depth(k), descriptions{dense(Type::kBFloat16, {rows, depth}),
          dense(Type::kUInt8, {columns, depth / 2}),
          dense(Type::kFloat8E4M3FN, {columns, depth / 16}),
          dense(Type::kFloat32, {}), dense(Type::kFloat32, {})},
        output_description(dense(Type::kBFloat16, {rows, columns})) {
    auto b = take(g::GraphBuilder::create(), "GraphBuilder create");
    for (std::size_t i = 0; i != ports.size(); ++i)
      ports[i] = take(i == 0 || dynamic ? b.input(descriptions[i]) : b.constant(descriptions[i]), "NVFP4 original port");
    output = take(b.add(c::model_ops::nvfp4_linear(), ports[0], ports[1], ports[2], ports[3], ports[4]), "NVFP4 reference linear");
    check(b.publish(output), "publish only final NVFP4 result");
    graph = take(std::move(b).finish(), "finish NVFP4 reference graph");
  }
  Data data(unsigned phase) const {
    Data result;
    result.a.resize(std::size_t(rows * depth));
    result.w.resize(std::size_t(columns * (depth / 2)));
    result.scales.resize(std::size_t(columns * (depth / 16)));
    if (columns == 16) result.input_global = 1.F;
    constexpr std::array<float, 16> ties{.25F,.75F,1.25F,1.75F,2.5F,3.5F,5.F,6.F,
        -.25F,-.75F,-1.25F,-1.75F,-2.5F,-3.5F,-5.F,-6.F};
    for (std::int64_t row = 0; row != rows; ++row) {
      for (std::int64_t lane = 0; lane != 16; ++lane)
        result.a[std::size_t(row * depth + lane)] = bf16(ties[(std::size_t(lane) + phase * 2 + std::size_t(row) * 3) % ties.size()]);
      result.a[std::size_t(row * depth + 16)] = bf16(1.e-6F);
      result.a[std::size_t(row * depth + 17)] = bf16(-1.e-6F);
      result.a[std::size_t(row * depth + 18)] = 0x8000;
    }
    for (std::int64_t n = 0; n != columns; ++n) {
      for (std::int64_t group = 0; group != depth / 16; ++group)
        result.scales[std::size_t(n * (depth / 16) + group)] = static_cast<std::uint8_t>(0x39 + (n + group) % 5);
      // One nonzero code per output isolates both BF16 operand boundaries from
      // backend dot-reduction order. N16 covers every nibble, including code8.
      const auto lane = columns == 16 ? n : (n == 2 ? 17 : n);
      const unsigned code = columns == 16 ? static_cast<unsigned>(n) : (n == 0 ? 1u : n == 1 ? 7u : 3u);
      result.w[std::size_t(n * (depth / 2) + lane / 2)] = static_cast<std::uint8_t>(code << (4 * (lane % 2)));
    }
    return result;
  }
  std::vector<std::uint16_t> oracle(const Data &data, bool omit_a_round = false,
      bool omit_w_round = false, bool omit_guard = false) const {
    std::vector<float> activation(std::size_t(rows * depth));
    for (std::int64_t row = 0; row != rows; ++row) for (std::int64_t group = 0; group != depth / 16; ++group) {
      float maximum = 0;
      for (std::int64_t lane = 0; lane != 16; ++lane)
        maximum = std::max(maximum, std::abs(wide(data.a[std::size_t(row * depth + group * 16 + lane)])));
      for (std::int64_t lane = 0; lane != 16; ++lane) {
        const auto index = std::size_t(row * depth + group * 16 + lane);
        activation[index] = activation_value(wide(data.a[index]), maximum, data.input_global, omit_a_round, omit_guard);
      }
    }
    std::vector<std::uint16_t> result(std::size_t(rows * columns));
    for (std::int64_t row = 0; row != rows; ++row) for (std::int64_t n = 0; n != columns; ++n) {
      float sum = 0;
      for (std::int64_t k = 0; k != depth; ++k) {
        const auto packed = data.w[std::size_t(n * (depth / 2) + k / 2)];
        const unsigned code = (packed >> (4 * (k % 2))) & 15u;
        const float weight = weight_value(code, data.scales[std::size_t(n * (depth / 16) + k / 16)], data.weight_global, omit_w_round);
        sum = f32(double(sum) + double(activation[std::size_t(row * depth + k)]) * double(weight));
      }
      result[std::size_t(row * columns + n)] = bf16(sum);
    }
    return result;
  }
};

inline void oracle_checks() {
  EXPECT_EQ(bf16(1.F + 1.F / 256.F), 0x3f80u);
  EXPECT_EQ(bf16(1.F + 3.F / 256.F), 0x3f82u);
  EXPECT_EQ(bf16(-0.F), 0x8000u);
  EXPECT_EQ(encode_scale(1.0625F), 0x38u);
  EXPECT_EQ(encode_scale(1.1875F), 0x3au);
  constexpr std::array<float, 7> ties{.25F,.75F,1.25F,1.75F,2.5F,3.5F,5.F};
  constexpr std::array<float, 7> expected{0.F,1.F,1.F,2.F,2.F,4.F,4.F};
  for (std::size_t i = 0; i != ties.size(); ++i) EXPECT_EQ(nearest_e2m1(ties[i]), expected[i]);
  EXPECT_EQ(bf16(activation_value(-0.F, 6.F, 1.F)), 0u);
  EXPECT_EQ(bf16(activation_value(-1.e-6F, 1.e-6F, 1.F)), 0x8000u);
  EXPECT_NE(bf16(activation_value(1.e-6F, 1.e-6F, 1.e-5F)),
      bf16(activation_value(1.e-6F, 1.e-6F, 1.e-5F, false, true)));
  EXPECT_EQ(std::bit_cast<std::uint32_t>(e2m1[8]), 0u);
  const auto largest = std::numeric_limits<float>::max();
  EXPECT_TRUE(std::isnan(weight_value(0, 0x40, largest)));
  EXPECT_EQ(weight_value(0, 0x40, largest, false, true), 0.F);
}

template<class Read> void expect_result(const g::GraphOutput &output, std::span<const std::uint16_t> expected, Read read) {
  auto tensor = output.tensor(); ASSERT_TRUE(tensor) << tensor.error().message();
  std::vector<std::uint16_t> actual(expected.size());
  const auto status = read(tensor->storage(), std::as_writable_bytes(std::span(actual)));
  ASSERT_TRUE(status.ok()) << status.message();
  for (std::size_t i = 0; i != expected.size(); ++i) {
    if (std::isnan(wide(expected[i]))) EXPECT_TRUE(std::isnan(wide(actual[i]))) << i;
    else EXPECT_EQ(actual[i], expected[i]) << i;
  }
}

template<class Compile, class Write, class Read, class Prepare> void run_lifetimes(m::MemoryResourceRef resource,
    std::span<const g::QueueBinding> queues, Compile compile, Write write, Read read, Prepare prepare, c::StorageOwner owner = {}) {
  oracle_checks();
  unsigned submissions = 0;
  bool activation_boundary = false, weight_boundary = false;
  for (const auto geometry : std::array{std::pair{3LL,32LL}, std::pair{16LL,2048LL}}) {
    SCOPED_TRACE(::testing::Message() << "N=" << geometry.first << " K=" << geometry.second);
    Case f(geometry.first, geometry.second);
    const auto initial = f.data(0);
    auto payloads = initial.payloads();
    std::array<c::ReadLease, 4> ready;
    std::array<g::ConstantBinding, 4> constants;
    std::size_t constant_bytes = 0;
    for (std::size_t i = 0; i != constants.size(); ++i) {
      auto buffer = take(m::Buffer::create(resource, payloads[i + 1].size(), 256), "original constant buffer");
      check(write(buffer.view(), payloads[i + 1]), "constant payload write");
      ready[i] = owner ? take(owner.adopt_buffer(std::move(buffer)), "owned bank lease") : take(c::ReadLease::adopt(std::move(buffer)), "bank lease");
      constants[i] = {f.ports[i + 1], take(ready[i].tensor(f.descriptions[i + 1]), "typed original constant"),
          g::ConstantPolicy::kMaterialize, owner, owner ? ready[i] : c::ReadLease{}};
      constant_bytes += payloads[i + 1].size();
    }
    auto compiled = compile(f.graph, g::CompileOptions{.constants = constants});
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_EQ(compiled->executable.input_count(), 1u);
    EXPECT_EQ(compiled->report.materialized_constant_count(), 4u);
    EXPECT_EQ(compiled->report.materialized_constant_bytes(), constant_bytes);
    const auto ownership = take(compiled->executable.output_ownership(), "output ownership");
    const bool caller = ownership == g::OutputOwnership::kCaller;
    if (!caller) ASSERT_EQ(ownership, g::OutputOwnership::kInvocation);
    std::array<g::Invocation, 2> slots{
        take(compiled->executable.create_invocation({.resources = g::ResourceOptions{}}), "first slot"),
        take(compiled->executable.create_invocation({.resources = g::ResourceOptions{}}), "second slot")};
    constants = {}; ready = {}; compiled = g::CompiledGraph{}; f.graph = g::Graph{};
    const auto output_bytes = f.output_description.required_bytes().value();
    const auto output_stride = (output_bytes + 255) / 256 * 256;
    c::LeasePool outputs;
    if (caller) outputs = take(c::LeasePool::create(resource, 2 * output_stride, {.max_blocks = 2}), "two output blocks");
    std::array<m::Buffer, 2> inputs;
    for (std::size_t slot = 0; slot != slots.size(); ++slot) {
      inputs[slot] = take(m::Buffer::create(resource, f.descriptions[0].required_bytes().value(), 256), "activation buffer");
      check(write(inputs[slot].view(), payloads[0]), "initial finite activation");
      check(slots[slot].bind(f.ports[0], take(c::ConstTensorView::create(f.descriptions[0], inputs[slot].view()), "activation view")), "bind activation once");
      if (caller) check(slots[slot].bind(f.output, f.output_description, take(outputs.allocate(output_bytes), "output writer")), "bind output once");
      check(prepare(slots[slot], queues), "prepare reusable reference execution");
    }
    std::array<g::GraphOutput, 2> retained;
    std::array<std::vector<std::uint16_t>, 2> old;
    for (unsigned repeat = 0; repeat != 2; ++repeat) for (std::size_t slot = 0; slot != slots.size(); ++slot) {
      const unsigned phase = repeat * 2 + static_cast<unsigned>(slot);
      SCOPED_TRACE(phase);
      if (repeat) {
        expect_result(retained[slot], old[slot], read);
        EXPECT_FALSE(slots[slot].submit(queues));
        retained[slot].reset();
      }
      const auto data = f.data(phase);
      check(write(inputs[slot].view(), std::as_bytes(std::span(data.a))), "update original activation contents");
      auto done = slots[slot].submit(queues); ASSERT_TRUE(done) << done.error().message();
      const auto status = done->wait(); ASSERT_TRUE(status.ok()) << status.message(); ++submissions;
      retained[slot] = take(slots[slot].result(f.output), "reference final result");
      old[slot] = f.oracle(data);
      expect_result(retained[slot], old[slot], read);
      activation_boundary |= old[slot] != f.oracle(data, true);
      weight_boundary |= old[slot] != f.oracle(data, false, true);
      if (retained[1 - slot].valid()) expect_result(retained[1 - slot], old[1 - slot], read);
    }
    EXPECT_NE(f.oracle(f.data(0)), f.oracle(f.data(2)));
    slots = {};
    for (std::size_t slot = 0; slot != retained.size(); ++slot) expect_result(retained[slot], old[slot], read);
    retained = {};
    if (caller) EXPECT_EQ(outputs.stats().live_blocks, 0u);
  }
  EXPECT_TRUE(activation_boundary) << "chosen payload must expose BF16 activation QDQ storage";
  EXPECT_TRUE(weight_boundary) << "chosen payload must expose BF16 decoded-weight storage";
  ::testing::Test::RecordProperty("actual_submissions", submissions);
}

template<class Compile, class Write, class Read, class Prepare> void run_runtime_domain(m::MemoryResourceRef resource,
    std::span<const g::QueueBinding> queues, Compile compile, Write write, Read read, Prepare prepare) {
  Case f(3, 32, true);
  auto compiled = compile(f.graph, g::CompileOptions{}); ASSERT_TRUE(compiled) << compiled.error().message();
  EXPECT_EQ(compiled->executable.input_count(), 5u);
  auto executable = compiled->executable;
  auto invocation = take(executable.create_invocation({.resources = g::ResourceOptions{}}), "dynamic reference slot");
  const auto ownership = take(compiled->executable.output_ownership(), "dynamic output ownership");
  if (ownership != g::OutputOwnership::kCaller) ASSERT_EQ(ownership, g::OutputOwnership::kInvocation);
  std::array<m::Buffer, 5> input;
  const auto valid = f.data(0);
  const auto upload = [&](const Data &data) {
    const auto payloads = data.payloads();
    for (std::size_t i = 0; i != input.size(); ++i) check(write(input[i].view(), payloads[i]), "update dynamic reference port");
  };
  for (std::size_t i = 0; i != input.size(); ++i) {
    input[i] = take(m::Buffer::create(resource, f.descriptions[i].required_bytes().value(), 256), "dynamic input buffer");
  }
  c::LeasePool outputs;
  if (ownership == g::OutputOwnership::kCaller) {
    const auto size = f.output_description.required_bytes().value();
    outputs = take(c::LeasePool::create(resource, (size + 255) / 256 * 256, {.max_blocks = 1}), "dynamic output pool");
  }
  const auto bind_current = [&] {
    for (std::size_t i = 0; i != input.size(); ++i)
      check(invocation.bind(f.ports[i], take(c::ConstTensorView::create(f.descriptions[i], input[i].view()), "dynamic input view")), "bind original dynamic port");
    if (ownership == g::OutputOwnership::kCaller)
      check(invocation.bind(f.output, f.output_description, take(outputs.allocate(f.output_description.required_bytes().value()), "dynamic output writer")), "bind dynamic output");
  };
  upload(valid); bind_current(); check(prepare(invocation, queues), "prepare dynamic reference replay");
  compiled = g::CompiledGraph{}; f.graph = g::Graph{};
  unsigned successes = 0, failures = 0;
  const auto success = [&](const Data &data) {
    upload(data);
    auto done = take(invocation.submit(queues), "accepted reference submit");
    check(done.wait(), "finite-domain reference Completion");
    auto result = take(invocation.result(f.output), "successful reference result");
    expect_result(result, f.oracle(data), read); ++successes;
  };
  success(valid);
  {
    auto data = valid; data.input_global = 1.e-5F;
    ASSERT_NE(f.oracle(data), f.oracle(data, false, false, true));
    success(data); // The second activation group exercises the tiny-scale guard.
  }
  std::vector<std::pair<std::string, Data>> invalid;
  for (const auto bits : std::array<std::uint16_t,3>{0x7fc1,0x7f80,0xff80}) {
    auto data = valid; data.a[0] = bits; invalid.emplace_back("nonfinite activation", std::move(data));
  }
  for (const auto code : std::array<std::uint8_t,3>{0x7f,0xff,0xb8}) {
    auto data = valid; data.scales[0] = code; invalid.emplace_back("NaN or negative block scale", std::move(data));
  }
  for (const auto value : std::array<float,6>{0.F,-0.F,-1.F,std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
    auto weight = valid; weight.weight_global = value; invalid.emplace_back("invalid weight global", std::move(weight));
    auto activation = valid; activation.input_global = value; invalid.emplace_back("invalid activation global", std::move(activation));
  }
  for (const auto &[label, data] : invalid) {
    SCOPED_TRACE(label);
    upload(data);
    {
      auto done = invocation.submit(queues); ASSERT_TRUE(done) << done.error().message();
      EXPECT_FALSE(done->wait().ok()) << "runtime contract violation must fail Completion";
      EXPECT_FALSE(invocation.result(f.output)) << "failed output must be unusable; no rollback assertion";
    }
    EXPECT_FALSE(invocation.submit(queues)) << "a failed Graph Invocation is terminal";
    ++failures;
    invocation = g::Invocation{};
    invocation = take(executable.create_invocation({.resources = g::ResourceOptions{}}), "replacement slot from the same Executable");
    upload(valid); bind_current();
    check(prepare(invocation, queues), "prepare replacement reference slot");
    success(valid); // Recovery uses a fresh slot; the immutable plan is reused.
  }
  for (const bool weight : {false, true}) {
    auto data = valid;
    if (weight) data.weight_global = std::numeric_limits<float>::denorm_min();
    else data.input_global = std::numeric_limits<float>::denorm_min();
    success(data);
  }
  {
    // This legal subnormal weight must survive multiplication by a normal A.
    // Testing denorm_min alone would leave every decoded BF16 weight zero
    // and could miss a GPU backend that flushes nonzero BF16 operands.
    auto data = valid;
    std::fill(data.a.begin(), data.a.end(), std::uint16_t{0});
    for (std::int64_t row = 0; row != f.rows; ++row)
      data.a[std::size_t(row * f.depth)] = 0x4380; // BF16 256 -> QDQ 264.
    std::fill(data.w.begin(), data.w.end(), std::uint8_t{0});
    for (std::int64_t column = 0; column != f.columns; ++column)
      data.w[std::size_t(column * (f.depth / 2))] = 2; // E2M1 +1.
    std::fill(data.scales.begin(), data.scales.end(), std::uint8_t{0x38});
    data.weight_global = std::bit_cast<float>(std::uint32_t{0x00080000}); // 2^-130.
    data.input_global = 1.F;
    for (const auto value : f.oracle(data)) ASSERT_EQ(value, 0x0284U);
    success(data);
  }
  {
    auto data = valid;
    for (std::size_t i = 0; i != data.scales.size(); ++i) data.scales[i] = i % 2 ? 0x80 : 0;
    success(data); // Both signs of the finite scale zero are admitted.
  }
  {
    auto data = valid;
    std::fill(data.w.begin(), data.w.end(), 0);
    std::fill(data.scales.begin(), data.scales.end(), 0x40); // E4M3 2.
    data.weight_global = std::numeric_limits<float>::max();
    const auto expected = f.oracle(data);
    for (const auto value : expected) ASSERT_TRUE(std::isnan(wide(value)));
    success(data); // Finite source values need not produce finite results.
  }
  ::testing::Test::RecordProperty("successful_submissions", successes);
  ::testing::Test::RecordProperty("failed_completions", failures);
}
} // namespace nvfp4_reference_test
