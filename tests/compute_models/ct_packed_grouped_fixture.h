#pragma once

#include "ct_packed_graph_fixture.h"
#include "hyle/graph/result.h"

namespace ct_packed_grouped_test {
namespace f = ct_packed_graph_test;
namespace c = f::c;
namespace g = f::g;
namespace m = f::m;
using Type = f::Type;
using f::take;
using f::check;
using f::dense;

// A finite set of actual mathematical/layout distinctions, not a model or
// provider matrix. EN is the original public decoder's row axis, including Z.
struct Profile final {
  bool asymmetric;
  Type source, decode, operand;
  std::int64_t group, experts, columns, depth, activation_slots;
  bool padded = false, cast_after = false;
};
inline constexpr std::array profiles{
    Profile{false, Type::kBFloat16, Type::kFloat16, Type::kFloat32, 32, 3, 5, 64, 1},
    Profile{true, Type::kFloat16, Type::kBFloat16, Type::kBFloat16, 128, 3, 5, 256, 3, false, true},
    Profile{true, Type::kBFloat16, Type::kFloat32, Type::kBFloat16, 512, 17, 3, 1024, 3, true},
    // The selected Kimi CT sources actually declare symmetric g128/g512.
    // Exercise those two roles directly, including the no-Cast recipe, rather
    // than borrowing their qualification from the asymmetric branch above.
    Profile{false, Type::kBFloat16, Type::kFloat32, Type::kFloat32, 128, 3, 5, 256, 1},
    Profile{false, Type::kFloat32, Type::kBFloat16, Type::kBFloat16, 512, 3, 5, 1024, 3}};
inline constexpr std::int64_t tokens = 2, slots = 3;

struct Case final {
  Profile p;
  std::vector<c::TensorDesc> descriptions;
  std::vector<g::ValueId> inputs;
  g::Graph graph;
  g::ValueId output{}, decoded{};
  bool escape;
  explicit Case(Profile profile, bool published_decode = false) : p(profile), escape(published_decode) {
    auto activation = dense(Type::kFloat32, {tokens, p.activation_slots, p.depth});
    if (p.padded) activation = take(c::TensorDesc::strided_bytes(Type::kFloat32,
        std::array{tokens, p.activation_slots, p.depth},
        std::array<std::int64_t, 3>{(p.depth * 2 + 7) * 4, 0, 8}), "real token padding/zero slot/depth stride");
    descriptions = {activation, dense(Type::kInt32, {p.experts * p.columns, p.depth / 8})};
    if (p.asymmetric) descriptions.push_back(dense(Type::kInt32,
        {(p.experts * p.columns + 7) / 8, p.depth / p.group}));
    descriptions.push_back(dense(p.source, {p.experts * p.columns, p.depth / p.group}));
    descriptions.push_back(dense(Type::kInt32, {tokens, slots}));
    auto b = take(g::GraphBuilder::create(), "grouped CT GraphBuilder");
    for (const auto &desc : descriptions) inputs.push_back(take(b.input(desc), "original CT input"));
    const auto decode = c::model_ops::artifact_weight_decode(p.asymmetric
        ? f::Format::kCompressedTensorsAsymmetric4 : f::Format::kCompressedTensorsSymmetric4,
        static_cast<std::uint32_t>(p.group), p.decode);
    const auto node = take(b.add_node(decode, std::span(inputs).subspan(1, inputs.size() - 2)), "original flat CT decode");
    decoded = take(b.result(node), "flat EN*K result");
    auto weight = decoded;
    if (p.decode != p.operand && !p.cast_after) weight = take(b.add(c::Cast{p.operand}, weight), "cast before expert reshape");
    weight = take(b.add(take(c::View::reshape(take(c::TensorShape::create(
        std::array{p.experts, p.columns, p.depth}), "expert bank shape")), "expert reshape"), weight), "original expert reshape");
    if (p.decode != p.operand && p.cast_after) weight = take(b.add(c::Cast{p.operand}, weight), "cast after expert reshape");
    output = take(b.add(c::model_ops::grouped_matmul(), weight, inputs[0], inputs.back()), "original GroupedMatMul");
    check(b.publish(output), "publish grouped dot");
    if (escape) check(b.publish(decoded), "published original decoder prevents contraction");
    graph = take(std::move(b).finish(), "finish ordinary grouped CT graph");
  }
};

struct Payload final {
  f::t::Fixture weights;
  std::vector<std::int32_t> ids;
  std::vector<float> activation;
  std::vector<std::vector<std::byte>> inputs;
  std::vector<float> expected;
  Payload(const Case &f, unsigned phase) : weights(f.p.asymmetric, phase,
      f.p.experts * f.p.columns, f.p.depth, f.p.group) {
    const auto &p = f.p;
    ids.resize(static_cast<std::size_t>(tokens * slots));
    for (std::int64_t t = 0; t < tokens; ++t) {
      ids[static_cast<std::size_t>(t * slots)] = static_cast<std::int32_t>((t + phase) % p.experts);
      ids[static_cast<std::size_t>(t * slots + 1)] = static_cast<std::int32_t>(p.experts - 1);
      ids[static_cast<std::size_t>(t * slots + 2)] = ids[static_cast<std::size_t>(t * slots)];
    }
    const auto &desc = f.descriptions[0];
    activation.assign(desc.required_bytes().value() / sizeof(float), std::numeric_limits<float>::quiet_NaN());
    const auto strides = desc.byte_strides();
    for (std::int64_t t = 0; t < tokens; ++t)
      for (std::int64_t a = 0; a < (p.padded ? 1 : p.activation_slots); ++a) {
        const auto hot = ((t + a + phase) % (p.depth / p.group)) * p.group + (t * 3 + a * 5 + phase) % 8;
        for (std::int64_t k = 0; k < p.depth; ++k) {
          const auto offset = static_cast<std::size_t>((t * strides[0] + a * strides[1] + k * strides[2]) / 4);
          activation[offset] = k == hot ? 1.F : 0.F;
        }
      }
    inputs = {f::bytes(activation), f::bytes(weights.codes)};
    if (p.asymmetric) inputs.push_back(f::bytes(weights.zeros));
    inputs.push_back(f::floating(p.source, weights.scales));
    inputs.push_back(f::bytes(ids));
    const auto decoded = weights.decoded(p.source, p.decode, p.operand);
    expected.resize(static_cast<std::size_t>(tokens * slots * p.columns));
    for (std::int64_t t = 0; t < tokens; ++t) for (std::int64_t s = 0; s < slots; ++s)
      for (std::int64_t n = 0; n < p.columns; ++n) {
        double sum = 0;
        const auto a = p.activation_slots == 1 ? 0 : s;
        const auto expert = ids[static_cast<std::size_t>(t * slots + s)];
        for (std::int64_t k = 0; k < p.depth; ++k) {
          const auto offset = static_cast<std::size_t>((t * strides[0] + a * strides[1] + k * strides[2]) / 4);
          sum += activation[offset] * decoded[static_cast<std::size_t>((expert * p.columns + n) * p.depth + k)];
        }
        expected[static_cast<std::size_t>((t * slots + s) * p.columns + n)] = static_cast<float>(sum);
      }
  }
};

inline void selected(const Case &f, const g::CompileReport &report) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < report.step_count(); ++i) {
    const auto step = take(report.step(i), "grouped compile step");
    if (step.policy != f::kPackedPolicy) continue;
    ++count;
    EXPECT_EQ(step.kind, g::CompileStepKind::kProviderRegion);
    EXPECT_EQ(step.node_count, f.graph.node_count());
    // Known public workspace must not be a full decoded float bank. Opaque
    // native allocations are established separately by the actual leaf code.
    EXPECT_LT(step.workspace.size, static_cast<std::size_t>(f.p.experts * f.p.columns * f.p.depth * 2));
  }
  EXPECT_EQ(count, f.escape ? 0u : 1u);
}
template<class Read> void expect_output(const g::GraphOutput &result, const std::vector<float> &expected, Read read) {
  const auto tensor = take(result.tensor(), "owned grouped result tensor");
  std::vector<float> actual(expected.size());
  check(read(tensor.storage(), std::as_writable_bytes(std::span(actual))), "read grouped F32 result");
  EXPECT_EQ(actual, expected); // One-hot A isolates every declared W RNE boundary.
}

// compile(Graph,CompileOptions,packed), prepare(Invocation,queues,packed).
// The Boolean only selects the test's original/packed preparation policy.
template<class Compile, class Write, class Read, class Prepare>
void run(m::MemoryResourceRef resource, std::span<const g::QueueBinding> queues,
    Compile compile, Write write, Read read, Prepare prepare) {
  for (std::size_t index = 0; index < profiles.size(); ++index) {
    SCOPED_TRACE(index);
    Case f(profiles[index]);
    auto compiled = compile(f.graph, g::CompileOptions{}, true);
    ASSERT_TRUE(compiled) << compiled.error().message();
    selected(f, compiled->report);
    const auto ownership = take(compiled->executable.output_ownership(), "grouped result ownership");
    const bool caller = ownership == g::OutputOwnership::kCaller;
    if (!caller) ASSERT_EQ(ownership, g::OutputOwnership::kInvocation);
    std::array<g::Invocation, 2> invocations{
        take(compiled->executable.create_invocation({.resources = g::ResourceOptions{}}), "first grouped slot"),
        take(compiled->executable.create_invocation({.resources = g::ResourceOptions{}}), "second grouped slot")};
    compiled = g::CompiledGraph{};
    const auto output_desc = take(f.graph.value_description(f.output), "grouped output description");
    const auto output_bytes = output_desc.required_bytes().value();
    c::LeasePool pool;
    if (caller) pool = take(c::LeasePool::create(resource, 512, {.max_blocks = 2}), "two aligned grouped result blocks");
    std::array<std::vector<m::Buffer>, 2> buffers;
    for (std::size_t slot = 0; slot < invocations.size(); ++slot) {
      for (std::size_t port = 0; port < f.inputs.size(); ++port) {
        buffers[slot].push_back(take(m::Buffer::create(resource, f.descriptions[port].required_bytes().value(), 256), "original grouped source"));
        check(invocations[slot].bind(f.inputs[port], take(c::ConstTensorView::create(f.descriptions[port], buffers[slot].back().view()), "typed grouped source")), "bind original grouped source");
      }
      if (caller) check(invocations[slot].bind(f.output, output_desc, take(pool.allocate(output_bytes), "grouped result writer")), "bind grouped result");
      check(prepare(invocations[slot], queues, true), "prepare packed grouped replay");
    }
    std::array<g::GraphOutput, 2> old;
    std::array<std::vector<float>, 2> expected;
    for (unsigned repeat = 0; repeat < 2; ++repeat) for (std::size_t slot = 0; slot < invocations.size(); ++slot) {
      const auto phase = repeat * 2 + static_cast<unsigned>(slot);
      SCOPED_TRACE(phase);
      if (repeat) {
        expect_output(old[slot], expected[slot], read);
        EXPECT_FALSE(invocations[slot].submit(queues));
        old[slot].reset();
      }
      Payload payload(f, phase);
      for (std::size_t port = 0; port < f.inputs.size(); ++port) check(write(buffers[slot][port].view(), payload.inputs[port]), "update packed bank/activation/IDs");
      auto accepted = invocations[slot].submit(queues);
      ASSERT_TRUE(accepted) << accepted.error().message();
      check(accepted->wait(), "grouped packed completion");
      old[slot] = take(invocations[slot].result(f.output), "grouped owned result");
      expected[slot] = std::move(payload.expected);
      expect_output(old[slot], expected[slot], read);
      if (old[1-slot].valid()) expect_output(old[1-slot], expected[1-slot], read);
    }
    EXPECT_NE(expected[0], expected[1]);
    invocations = {};
    for (std::size_t slot = 0; slot < old.size(); ++slot) expect_output(old[slot], expected[slot], read);
    old = {};
    if (caller) EXPECT_EQ(pool.stats().live_blocks, 0u);
  }
  ::testing::Test::RecordProperty("actual_submissions", static_cast<int>(profiles.size() * 4));
}

template<class Compile, class Write, class Read, class Prepare>
void run_domain(m::MemoryResourceRef resource, std::span<const g::QueueBinding> queues,
    Compile compile, Write write, Read read, Prepare prepare) {
  Case f(profiles[1]);
  auto compiled = compile(f.graph, g::CompileOptions{}, true);
  ASSERT_TRUE(compiled) << compiled.error().message();
  selected(f, compiled->report);
  const bool caller = take(compiled->executable.output_ownership(), "domain ownership") == g::OutputOwnership::kCaller;
  Payload payload(f, 0);
  std::vector<m::Buffer> buffers;
  for (std::size_t port = 0; port < f.inputs.size(); ++port) {
    buffers.push_back(take(m::Buffer::create(resource, payload.inputs[port].size(), 256), "domain original buffer"));
    check(write(buffers.back().view(), payload.inputs[port]), "domain original content");
  }
  for (unsigned attempt = 0; attempt < 5; ++attempt) {
    SCOPED_TRACE(attempt);
    auto invocation = take(compiled->executable.create_invocation({.resources = g::ResourceOptions{}}), "fresh checked slot");
    for (std::size_t port = 0; port < f.inputs.size(); ++port)
      check(invocation.bind(f.inputs[port], take(c::ConstTensorView::create(f.descriptions[port], buffers[port].view()), "domain typed input")), "domain bind");
    c::LeasePool pool;
    const auto desc = take(f.graph.value_description(f.output), "domain output");
    if (caller) {
      pool = take(c::LeasePool::create(resource, 256, {.max_blocks = 1}), "domain result pool");
      check(invocation.bind(f.output, desc, take(pool.allocate(desc.required_bytes().value()), "domain writer")), "domain result bind");
    }
    auto ids = payload.ids;
    const bool invalid = attempt == 1 || attempt == 3;
    if (invalid) ids.back() = attempt == 1 ? -1 : static_cast<std::int32_t>(f.p.experts);
    check(write(buffers.back().view(), f::bytes(ids)), "last ID payload update");
    check(prepare(invocation, queues, true), "checked grouped prepare");
    auto accepted = invocation.submit(queues);
    ASSERT_TRUE(accepted) << accepted.error().message();
    const auto status = accepted->wait();
    if (invalid) {
      EXPECT_FALSE(status.ok());
      EXPECT_FALSE(invocation.result(f.output));
    } else {
      ASSERT_TRUE(status.ok()) << status.message();
      auto result = take(invocation.result(f.output), "checked successful result");
      expect_output(result, payload.expected, read);
    }
  }
  ::testing::Test::RecordProperty("successful_submissions", 3);
  ::testing::Test::RecordProperty("expected_failed_completions", 2);
}

template<class Compile, class Write, class Read, class Prepare>
void run_escape(m::MemoryResourceRef resource, std::span<const g::QueueBinding> queues,
    Compile compile, Write write, Read read, Prepare prepare) {
  Case f(profiles[0], true);
  auto compiled = compile(f.graph, g::CompileOptions{}, false);
  ASSERT_TRUE(compiled) << compiled.error().message();
  selected(f, compiled->report);
  const bool caller = take(compiled->executable.output_ownership(), "escape ownership") == g::OutputOwnership::kCaller;
  auto invocation = take(compiled->executable.create_invocation({.resources = g::ResourceOptions{}}), "original decomposition slot");
  Payload payload(f, 0);
  std::vector<m::Buffer> buffers;
  for (std::size_t port = 0; port < f.inputs.size(); ++port) {
    buffers.push_back(take(m::Buffer::create(resource, payload.inputs[port].size(), 256), "escape input buffer"));
    check(write(buffers.back().view(), payload.inputs[port]), "escape input write");
    check(invocation.bind(f.inputs[port], take(c::ConstTensorView::create(f.descriptions[port], buffers[port].view()), "escape input")), "escape bind");
  }
  c::LeasePool pool;
  if (caller) {
    const auto bank_bytes = take(f.graph.value_description(f.decoded), "escape bank desc").required_bytes().value();
    pool = take(c::LeasePool::create(resource, bank_bytes + 512, {.max_blocks = 2}), "escaped decoder plus dot writers");
    for (const auto value : {f.output, f.decoded}) {
      const auto desc = take(f.graph.value_description(value), "escape published description");
      check(invocation.bind(value, desc, take(pool.allocate(desc.required_bytes().value()), "escape published writer")), "escape output bind");
    }
  }
  check(prepare(invocation, queues, false), "prepare original decomposition");
  auto accepted = invocation.submit(queues);
  ASSERT_TRUE(accepted) << accepted.error().message();
  check(accepted->wait(), "ordinary escaped decode/grouped completion");
  auto result = take(invocation.result(f.output), "escaped dot result");
  expect_output(result, payload.expected, read);
  const auto decoded = take(invocation.result(f.decoded), "observable decoder result");
  const auto tensor = take(decoded.tensor(), "observable decoder tensor");
  const auto values = payload.weights.decoded(f.p.source, f.p.decode, f.p.decode);
  const auto expected = f::floating(f.p.decode, values);
  std::vector<std::byte> actual(expected.size());
  check(read(tensor.storage(), actual), "read observable decoder payload");
  EXPECT_EQ(actual, expected);
  ::testing::Test::RecordProperty("actual_submissions", 1);
}
} // namespace ct_packed_grouped_test
