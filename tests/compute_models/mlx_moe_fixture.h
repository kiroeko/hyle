#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "hyle/compute/model_ops.h"
#include "hyle/compute/resource_leases.h"
#include "hyle/compute/storage_owner.h"
#include "hyle/graph/executable.h"
#include "hyle/graph/result.h"
#include "hyle/simd/dtype.h"
#include "mlx_affine_bitstream.h"

namespace mlx_moe_test {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
inline constexpr std::int64_t tokens = 2, slots = 3, hidden = 64, intermediate = 64;

template<class T> inline std::vector<std::byte> bytes(std::span<const T> values) {
  const auto raw = std::as_bytes(values);
  return {raw.begin(), raw.end()};
}
inline c::TensorDesc dense(c::DataType type, std::initializer_list<std::int64_t> shape) {
  return c::TensorDesc::dense(type, {shape.begin(), shape.size()}).value();
}

struct Bank final {
  unsigned bits;
  std::int64_t experts;
  std::vector<std::uint32_t> codes;
  std::vector<hyle::simd::bfloat16_t> scales, biases;
  std::array<c::TensorDesc, 3> descriptions;
  std::array<std::vector<std::byte>, 3> payloads;

  Bank(unsigned width, std::int64_t count, unsigned stage) : bits(width), experts(count),
      codes(std::size_t(count * hidden * intermediate)),
      scales(std::size_t(count * hidden * 2)), biases(scales.size()) {
    for (std::int64_t e = 0; e < count; ++e) for (std::int64_t n = 0; n < hidden; ++n) {
      for (std::int64_t k = 0; k < hidden; ++k)
        codes[std::size_t((e * hidden + n) * hidden + k)] =
            static_cast<unsigned>(e * 7 + n * 3 + k * 5 + stage * 11) & ((1u << bits) - 1);
      for (std::int64_t group = 0; group < 2; ++group) {
        const auto index = std::size_t((e * hidden + n) * 2 + group);
        // BF16 metadata is deliberately independent of the F32 activation.
        scales[index] = hyle::simd::f32_to_bf16(.0023F * float(1 + e % 3 + n % 2 + group));
        biases[index] = hyle::simd::f32_to_bf16(-.021F * float(1 + n % 3 + group));
      }
    }
    descriptions = {dense(c::DataType::kUInt32, {count, hidden, 2 * bits}),
        dense(c::DataType::kBFloat16, {count, hidden, 2}),
        dense(c::DataType::kBFloat16, {count, hidden, 2})};
    const auto packed = mlx_affine_test::pack(codes, std::size_t(count * hidden), hidden, bits);
    payloads = {bytes<std::uint32_t>(packed), bytes<hyle::simd::bfloat16_t>(scales),
        bytes<hyle::simd::bfloat16_t>(biases)};
  }
  float value(std::int32_t expert, std::int64_t channel, std::int64_t k) const {
    const auto row = std::size_t(std::int64_t(expert) * hidden + channel);
    const auto metadata = row * 2 + std::size_t(k / 32);
    const float scale = hyle::simd::bf16_to_f32(scales[metadata]);
    const float bias = hyle::simd::bf16_to_f32(biases[metadata]);
    // Stage both F32 roundings independently; never call product decoding.
    const float product = static_cast<float>(double(codes[row * hidden + std::size_t(k)]) * double(scale));
    return static_cast<float>(double(product) + double(bias));
  }
  std::array<float, hidden> project(std::int32_t expert, std::span<const float> a) const {
    std::array<float, hidden> result{};
    for (std::int64_t n = 0; n < hidden; ++n) {
      float sum = 0;
      for (std::int64_t group = 0; group < 2; ++group) {
        double dot = 0;
        for (std::int64_t k = group * 32; k < (group + 1) * 32; ++k)
          dot += double(a[std::size_t(k)]) * double(value(expert, n, k));
        sum = static_cast<float>(double(sum) + double(static_cast<float>(dot)));
      }
      result[std::size_t(n)] = sum;
    }
    return result;
  }
};

struct Case final {
  std::int64_t experts;
  std::array<Bank, 3> banks;
  c::TensorDesc activation_description = dense(c::DataType::kFloat32, {tokens, 1, hidden});
  c::TensorDesc logits_description;
  c::TensorDesc output_description = dense(c::DataType::kFloat32, {tokens, hidden});
  g::Graph graph;
  g::ValueId activation{}, logits{}, output{};
  std::array<g::ValueId, 9> constants{};

  explicit Case(std::int64_t count) : experts(count),
      banks{Bank(5, count, 0), Bank(6, count, 1), Bank(4, count, 2)},
      logits_description(dense(c::DataType::kFloat32, {tokens, count})) {
    auto builder = g::GraphBuilder::create().value();
    activation = builder.input(activation_description).value();
    logits = builder.input(logits_description).value();
    for (std::size_t bank = 0; bank < banks.size(); ++bank)
      for (std::size_t port = 0; port < 3; ++port)
        constants[bank * 3 + port] = builder.constant(banks[bank].descriptions[port]).value();
    const auto selected = builder.add_node(c::Operation(c::TopK{.count = std::size_t(slots)}),
        std::array{logits}).value();
    const auto ids = builder.result(selected, 1).value();
    const auto scores = builder.result(selected, 0).value();
    const auto weights = builder.add(c::Softmax{}, scores).value();
    const auto projection = [&](std::size_t bank, g::ValueId a) {
      return builder.add(c::model_ops::mlx_grouped_matmul(c::model_ops::MlxWeightFormat::kAffine,
          banks[bank].bits, 32), a, constants[bank * 3], constants[bank * 3 + 1],
          constants[bank * 3 + 2], ids).value();
    };
    const auto gate = projection(0, activation), up = projection(1, activation);
    const auto silu = builder.add(c::Unary{c::UnaryKind::kSilu}, gate).value();
    const auto activated = builder.add(c::Binary{c::BinaryKind::kMultiply}, silu, up).value();
    const auto down = projection(2, activated);
    const auto route_shape = c::TensorShape::create(std::array<std::int64_t, 3>{tokens, slots, 1}).value();
    const auto route = builder.add(c::View::reshape(route_shape).value(), weights).value();
    const auto weighted = builder.add(c::Binary{c::BinaryKind::kMultiply}, down, route).value();
    output = builder.add(c::Reduce{.kind = c::ReductionKind::kSum,
        .axes = c::Axes::from(std::array<std::int32_t, 1>{1}).value()}, weighted).value();
    // The only published value is the real complete MoE result.
    const auto published = builder.publish(output);
    EXPECT_TRUE(published.ok()) << published.message();
    graph = std::move(builder).finish().value();
  }
  std::vector<float> activations(unsigned phase) const {
    std::vector<float> result(std::size_t(tokens * hidden));
    for (std::int64_t t = 0; t < tokens; ++t) for (std::int64_t h = 0; h < hidden; ++h)
      result[std::size_t(t * hidden + h)] = float((h * 3 + t * 5 + phase * 7) % 19 - 7) / 32.F;
    return result;
  }
  std::vector<float> routing_logits(unsigned phase) const {
    std::vector<float> result(std::size_t(tokens * experts), -8.F);
    for (std::int64_t t = 0; t < tokens; ++t) for (std::int64_t e = 0; e < slots; ++e) {
      const auto rank = (e + t + phase) % slots;
      result[std::size_t(t * experts + e)] = float(3 - rank) * .375F + float(phase) * .03125F * float(e + 1);
    }
    return result;
  }
  std::vector<float> oracle(unsigned phase) const {
    const auto a = activations(phase), routing = routing_logits(phase);
    std::vector<float> result(std::size_t(tokens * hidden));
    for (std::int64_t t = 0; t < tokens; ++t) {
      std::vector<std::int32_t> selected(std::size_t(experts), 0);
      std::iota(selected.begin(), selected.end(), 0);
      std::stable_sort(selected.begin(), selected.end(), [&](auto left, auto right) {
        return routing[std::size_t(t * experts + left)] > routing[std::size_t(t * experts + right)];
      });
      std::array<double, slots> exponents{};
      double total = 0;
      const auto maximum = routing[std::size_t(t * experts + selected[0])];
      for (std::int64_t s = 0; s < slots; ++s) {
        exponents[std::size_t(s)] = std::exp(double(routing[std::size_t(t * experts + selected[std::size_t(s)])] - maximum));
        total += exponents[std::size_t(s)];
      }
      std::array<std::array<float, hidden>, slots> weighted{};
      for (std::int64_t s = 0; s < slots; ++s) {
        const auto expert = selected[std::size_t(s)];
        const auto token = std::span(a).subspan(std::size_t(t * hidden), std::size_t(hidden));
        const auto gate = banks[0].project(expert, token), up = banks[1].project(expert, token);
        std::array<float, intermediate> activated{};
        for (std::int64_t f = 0; f < intermediate; ++f) {
          const auto index = std::size_t(f);
          const auto silu = static_cast<float>(double(gate[index]) / (1 + std::exp(-double(gate[index]))));
          activated[index] = static_cast<float>(double(silu) * double(up[index]));
        }
        const auto down = banks[2].project(expert, activated);
        const float weight = static_cast<float>(exponents[std::size_t(s)] / total);
        for (std::int64_t h = 0; h < hidden; ++h)
          weighted[std::size_t(s)][std::size_t(h)] = static_cast<float>(double(down[std::size_t(h)]) * double(weight));
      }
      for (std::int64_t h = 0; h < hidden; ++h) {
        double sum = 0;
        for (std::int64_t s = 0; s < slots; ++s) sum += weighted[std::size_t(s)][std::size_t(h)];
        result[std::size_t(t * hidden + h)] = static_cast<float>(sum);
      }
    }
    return result;
  }
};

template<class Read> void expect_result(const g::GraphOutput &result, std::span<const float> expected, Read read) {
  auto tensor = result.tensor();
  ASSERT_TRUE(tensor) << tensor.error().message();
  std::vector<float> actual(expected.size());
  const auto downloaded = read(tensor->storage(), std::as_writable_bytes(std::span(actual)));
  ASSERT_TRUE(downloaded.ok()) << downloaded.message();
  for (std::size_t i = 0; i < expected.size(); ++i) {
    ASSERT_TRUE(std::isfinite(actual[i]));
    // Six short F32 reduction/pointwise boundaries; independent double dots
    // do not require matching a provider's permitted 32-lane association.
    EXPECT_NEAR(actual[i], expected[i], 256.F * std::numeric_limits<float>::epsilon() * (1.F + std::abs(expected[i]))) << i;
  }
}

// Compile is the existing platform Graph entry point, not a test executor.
template<class Compile, class Write, class Read> void run(m::MemoryResourceRef resource,
    std::span<const g::QueueBinding> queues, Compile compile, Write write, Read read,
    c::StorageOwner constant_owner = {}) {
  std::array<std::vector<float>, 4> reference;
  for (const std::int64_t experts : {3, 17}) {
    SCOPED_TRACE(experts);
    Case fixture(experts);
    std::array<m::Buffer, 9> source;
    std::array<c::ReadLease, 9> ready;
    std::array<g::ConstantBinding, 9> constants;
    for (std::size_t bank = 0; bank < 3; ++bank) for (std::size_t port = 0; port < 3; ++port) {
      const auto index = bank * 3 + port;
      const auto &payload = fixture.banks[bank].payloads[port];
      source[index] = m::Buffer::create(resource, payload.size(), 256).value();
      const auto uploaded = write(source[index].view(), std::span<const std::byte>(payload));
      ASSERT_TRUE(uploaded.ok()) << uploaded.message();
      ready[index] = constant_owner ? constant_owner.adopt_buffer(std::move(source[index])).value()
          : c::ReadLease::adopt(std::move(source[index])).value();
      constants[index] = {fixture.constants[index], ready[index].tensor(
          fixture.banks[bank].descriptions[port]).value(), g::ConstantPolicy::kMaterialize,
          constant_owner, constant_owner ? ready[index] : c::ReadLease{}};
    }
    auto compiled = compile(fixture.graph, g::CompileOptions{.constants = constants});
    ASSERT_TRUE(compiled) << compiled.error().message();
    EXPECT_EQ(compiled->executable.input_count(), 2u);
    EXPECT_EQ(compiled->executable.output_count(), 1u);
    const auto ownership = compiled->executable.output_ownership();
    ASSERT_TRUE(ownership) << ownership.error().message();
    const bool caller_output = *ownership == g::OutputOwnership::kCaller;
    if (!caller_output) EXPECT_EQ(*ownership, g::OutputOwnership::kInvocation);
    ::testing::Test::RecordProperty("moe_output_ownership_E" + std::to_string(experts),
        static_cast<int>(*ownership));
    std::array<g::Invocation, 2> invocations{
        compiled->executable.create_invocation({.resources = g::ResourceOptions{}}).value(),
        compiled->executable.create_invocation({.resources = g::ResourceOptions{}}).value()};
    // The executable/slots own the canonical immutable copies from now on.
    constants = {};
    ready = {};
    compiled->executable = {};
    const auto output_bytes = fixture.output_description.required_bytes().value();
    c::LeasePool outputs;
    if (caller_output)
      outputs = c::LeasePool::create(resource, 2 * output_bytes, {.max_blocks = 2}).value();
    std::array<m::Buffer, 2> a, logits;
    for (std::size_t slot = 0; slot < invocations.size(); ++slot) {
      a[slot] = m::Buffer::create(resource, fixture.activation_description.required_bytes().value(), 256).value();
      logits[slot] = m::Buffer::create(resource, fixture.logits_description.required_bytes().value(), 256).value();
      ASSERT_TRUE(invocations[slot].bind(fixture.activation, c::ConstTensorView::create(fixture.activation_description, a[slot].view()).value()).ok());
      ASSERT_TRUE(invocations[slot].bind(fixture.logits, c::ConstTensorView::create(fixture.logits_description, logits[slot].view()).value()).ok());
      if (caller_output) {
        auto destination = outputs.allocate(output_bytes).value();
        const auto bound = invocations[slot].bind(fixture.output, fixture.output_description, std::move(destination));
        ASSERT_TRUE(bound.ok()) << bound.message();
      }
      const auto prepared = invocations[slot].prepare_replay(queues);
      ASSERT_TRUE(prepared.ok()) << prepared.message();
    }
    std::array<g::GraphOutput, 2> retained;
    std::array<std::vector<float>, 2> old;
    for (unsigned repeat = 0; repeat < 2; ++repeat) for (std::size_t slot = 0; slot < invocations.size(); ++slot) {
      const auto phase = repeat * 2 + static_cast<unsigned>(slot);
      SCOPED_TRACE(phase);
      auto expected = fixture.oracle(phase);
      ASSERT_TRUE(std::any_of(expected.begin(), expected.end(), [](float x) { return std::abs(x) > 1.e-4F; }));
      if (experts == 3) reference[phase] = expected;
      else EXPECT_EQ(expected, reference[phase]); // unused experts cannot influence the selected calculation
      if (repeat) {
        expect_result(retained[slot], old[slot], read);
        EXPECT_FALSE(invocations[slot].submit(queues));
        retained[slot].reset();
      }
      const auto activation = fixture.activations(phase), routing = fixture.routing_logits(phase);
      ASSERT_TRUE(write(a[slot].view(), std::as_bytes(std::span(activation))).ok());
      ASSERT_TRUE(write(logits[slot].view(), std::as_bytes(std::span(routing))).ok());
      auto done = invocations[slot].submit(queues);
      ASSERT_TRUE(done) << done.error().message();
      const auto status = done->wait();
      ASSERT_TRUE(status.ok()) << status.message();
      retained[slot] = invocations[slot].result(fixture.output).value();
      expect_result(retained[slot], expected, read);
      old[slot] = std::move(expected);
      if (retained[1 - slot].valid()) expect_result(retained[1 - slot], old[1 - slot], read);
    }
    EXPECT_NE(old[0], old[1]);
    if (experts == 3) EXPECT_NE(reference[0], reference[2]);
    invocations = {};
    compiled = g::CompiledGraph{};
    for (std::size_t slot = 0; slot < retained.size(); ++slot) expect_result(retained[slot], old[slot], read);
    if (caller_output) EXPECT_EQ(outputs.stats().live_blocks, 2u);
    retained = {};
    if (caller_output) EXPECT_EQ(outputs.stats().live_blocks, 0u);
  }
}
} // namespace mlx_moe_test
