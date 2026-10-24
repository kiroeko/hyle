#pragma once

#include "ct_packed_linear_fixture.h"
#include "hyle/graph/executable.h"
#include "hyle/compute/model_ops.h"
#include "hyle/compute/prepared_operation.h"
#include "hyle/compute/storage_owner.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ct_packed_graph_test {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace t = ct_packed_test;
using Type = c::DataType;
using Format = c::model_ops::WeightArtifactFormat;
constexpr c::RoutePolicyId kPackedPolicy{0x48594c454354504cULL, 1, 1};
enum class Escape { kNone, kPublishedDecode, kExtraView, kControl };

template <class T> T take(hyle::Result<T> result, std::string_view where) {
  if (!result) throw std::runtime_error(std::string(where) + ": " + std::string(result.error().message()));
  return std::move(*result);
}
inline void check(hyle::Status status, std::string_view where) {
  if (!status.ok()) throw std::runtime_error(std::string(where) + ": " + std::string(status.message()));
}

inline c::TensorDesc dense(Type type, std::initializer_list<std::int64_t> dimensions) {
  return c::TensorDesc::dense(type, {dimensions.begin(), dimensions.size()}).value();
}
struct Profile final {
  bool asymmetric;
  Type scale, decode, operand;
  std::int64_t m;
  bool cast_before = true;
};
struct Logical final {
  Profile profile;
  g::Graph graph;
  std::vector<g::ValueId> ports;
  std::vector<c::TensorDesc> descriptions;
  std::vector<g::NodeId> nodes;
  g::ValueId decoded{}, transposed{}, cast{}, output{}, extra{};
  c::Operation decode;
  Escape escape = Escape::kNone;
  bool constants = false;
};
inline Logical build(Profile p, std::int64_t n = 9, std::int64_t k = 64,
              Escape escape = Escape::kNone, bool constants = false,
              bool publish_activation = false) {
  Logical f;
  f.profile = p; f.escape = escape; f.constants = constants;
  f.descriptions = {dense(p.operand, {p.m, k}), dense(Type::kInt32, {n, k / 8})};
  if (p.asymmetric) f.descriptions.push_back(dense(Type::kInt32, {(n + 7) / 8, k / 32}));
  f.descriptions.push_back(dense(p.scale, {n, k / 32}));
  auto b = g::GraphBuilder::create().value();
  for (std::size_t i = 0; i < f.descriptions.size(); ++i)
    f.ports.push_back(constants && i != 0 ? b.constant(f.descriptions[i]).value()
                                         : b.input(f.descriptions[i]).value());
  f.decode = c::model_ops::artifact_weight_decode(p.asymmetric
      ? Format::kCompressedTensorsAsymmetric4 : Format::kCompressedTensorsSymmetric4,
      32, p.decode);
  const auto decode_node = b.add_node(f.decode, std::span(f.ports).subspan(1)).value();
  f.decoded = b.result(decode_node).value();
  f.nodes.push_back(decode_node);
  auto weight = f.decoded;
  if (p.decode != p.operand && p.cast_before) {
    f.cast = b.add(c::Cast{p.operand}, weight).value();
    weight = f.cast;
  }
  f.transposed = b.add(c::View::permute(std::array<std::int32_t, 2>{1, 0}).value(), weight).value();
  weight = f.transposed;
  if (p.decode != p.operand && !p.cast_before) {
    f.cast = b.add(c::Cast{p.operand}, weight).value();
    weight = f.cast;
  }
  f.output = b.add(c::MatMul{.accumulation_type = Type::kFloat32}, f.ports[0], weight).value();
  if (escape == Escape::kPublishedDecode) f.extra = f.decoded;
  if (escape == Escape::kExtraView) {
    const auto flattened = b.add(c::View::reshape(c::TensorShape::create(
        std::array{n * k}).value()).value(), f.decoded).value();
    f.extra = b.add(c::Copy{}, flattened).value();
  }
  if (escape == Escape::kControl) {
    const auto observer = b.add_node(c::Operation(c::Copy{}), std::array{f.ports[0]}).value();
    EXPECT_TRUE(b.depends_on(observer, decode_node).ok());
    f.extra = b.result(observer).value();
  }
  if (publish_activation) { EXPECT_TRUE(b.publish(f.ports[0]).ok()); }
  if (f.extra.valid()) { EXPECT_TRUE(b.publish(f.extra).ok()); }
  EXPECT_TRUE(b.publish(f.output).ok());
  f.graph = std::move(b).finish().value();
  f.nodes.push_back(f.graph.producer(f.transposed).value());
  if (f.cast.valid()) f.nodes.push_back(f.graph.producer(f.cast).value());
  f.nodes.push_back(f.graph.producer(f.output).value());
  return f;
}

template <class T> std::vector<std::byte> bytes(const std::vector<T> &values) {
  const auto v = std::as_bytes(std::span(values)); return {v.begin(), v.end()};
}
inline std::vector<std::byte> floating(Type type, std::span<const double> values) {
  if (type == Type::kFloat16) {
    std::vector<t::F16> v; for (const auto x : values) v.push_back(half_test::encode<t::F16>(x));
    return bytes(v);
  }
  if (type == Type::kBFloat16) {
    std::vector<t::BF16> v; for (const auto x : values) v.push_back(half_test::encode<t::BF16>(x));
    return bytes(v);
  }
  std::vector<float> v; for (const auto x : values) v.push_back(static_cast<float>(x));
  return bytes(v);
}
inline std::vector<double> read(c::ConstTensorView tensor) {
  std::vector<double> result;
  const auto type = tensor.description().data_type();
  if (type == Type::kFloat32) {
    std::vector<float> v(tensor.storage().size() / sizeof(float));
    EXPECT_TRUE(tensor.storage().read(std::as_writable_bytes(std::span(v))).ok());
    for (auto x : v) result.push_back(x);
  } else {
    std::vector<std::uint16_t> v(tensor.storage().size() / sizeof(std::uint16_t));
    EXPECT_TRUE(tensor.storage().read(std::as_writable_bytes(std::span(v))).ok());
    for (auto x : v) result.push_back(type == Type::kFloat16
        ? half_test::decode_bits<t::F16>(x) : half_test::decode_bits<t::BF16>(x));
  }
  return result;
}
inline void exact_values(std::span<const double> actual, std::span<const double> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (std::isnan(expected[i])) EXPECT_TRUE(std::isnan(actual[i])) << i;
    else {
      EXPECT_EQ(actual[i], expected[i]) << i;
      if (expected[i] == 0) { EXPECT_EQ(std::signbit(actual[i]), std::signbit(expected[i])) << i; }
    }
  }
}
inline c::ReadLease upload(m::MemoryResourceRef resource, std::span<const std::byte> payload,
                          const c::StorageOwner &owner = {}) {
  auto b = take(m::Buffer::create(resource, payload.size(), 256), "input Buffer::create");
  check(b.view().write(payload), "input upload");
  if (owner.valid()) return take(owner.adopt_buffer(std::move(b)), "input StorageOwner::adopt_buffer");
  return take(c::ReadLease::adopt(std::move(b)), "input ReadLease::adopt");
}
inline c::ReadLease upload(const c::StorageOwner &owner, std::span<const std::byte> payload) {
  return upload(owner.ref(), payload, owner);
}
struct Bindings final {
  t::Fixture fixture;
  std::vector<double> activation, weight;
  std::vector<c::ReadLease> inputs;
  // The public mutable pool is kept open through actual submission. Its
  // resource owner is the caller's output_owner, which outlives all outputs.
  std::vector<c::LeasePool> outputs;
  Bindings(const Logical &f, const c::StorageOwner &owner, unsigned seed,
           bool witness = false, bool nonfinite = false)
      : Bindings(f, owner.ref(), seed, witness, nonfinite, owner) {}
  // Native Graph facades own their actual resource directly, rather than a
  // generic Compute Context. Their wrapper retains that resource through use.
  Bindings(const Logical &f, m::MemoryResourceRef resource, unsigned seed,
           bool witness = false, bool nonfinite = false)
      : Bindings(f, resource, seed, witness, nonfinite, {}) {}
  Bindings(const Logical &f, m::MemoryResourceRef resource, unsigned seed,
           bool witness, bool nonfinite, const c::StorageOwner &owner)
      : fixture(f.profile.asymmetric, seed, f.descriptions.back().dimensions()[0],
                f.descriptions[0].dimensions()[1]) {
    if (nonfinite) {
      fixture.scales[2] = std::numeric_limits<double>::infinity();
      fixture.scales[4] = std::numeric_limits<double>::quiet_NaN();
    }
    activation = fixture.activation(f.profile.m, f.profile.operand, seed, witness);
    weight = fixture.decoded(f.profile.scale, f.profile.decode, f.profile.operand);
    inputs.push_back(upload(resource, floating(f.profile.operand, activation), owner));
    inputs.push_back(upload(resource, bytes(fixture.codes), owner));
    if (f.profile.asymmetric) inputs.push_back(upload(resource, bytes(fixture.zeros), owner));
    inputs.push_back(upload(resource, floating(f.profile.scale, fixture.scales), owner));
  }
};
inline void bind(const Logical &f, Bindings &r, g::Invocation &slot,
          m::MemoryResourceRef output_resource, bool caller_outputs = true) {
  for (std::size_t i = 0; i < f.ports.size(); ++i)
    if (!f.constants || i == 0)
      check(slot.bind(f.ports[i], f.descriptions[i], r.inputs[i]),
          "Invocation::bind input port " + std::to_string(i));
  for (auto value : {f.output, f.extra}) if (caller_outputs && value.valid()) {
    const auto desc = f.graph.value_description(value).value();
    const auto size = desc.required_bytes().value();
    r.outputs.push_back(take(c::LeasePool::create(output_resource, size,
        {.max_blocks = 1, .backing_alignment = 256}), "output LeasePool::create"));
    auto writer = take(r.outputs.back().allocate(size, 256), "output LeasePool::allocate");
    check(slot.bind(value, desc, std::move(writer)),
        "Invocation::bind output ordinal " + std::to_string(value.ordinal()));
  }
}
inline void bind(const Logical &f, Bindings &r, g::Invocation &slot,
                 const c::StorageOwner &output_owner, bool caller_outputs = true) {
  bind(f, r, slot, output_owner.ref(), caller_outputs);
}
template <class Backend>
void selected(const Logical &f, const g::CompileReport &report, bool expected) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < report.step_count(); ++i) {
    const auto step = report.step(i).value();
    if (step.policy != kPackedPolicy) continue;
    ++count;
    EXPECT_EQ(step.kind, g::CompileStepKind::kProviderRegion);
    EXPECT_EQ(step.provider.id, Backend::provider_id);
    EXPECT_EQ(step.implementation, c::ProviderImplementationId{1});
    ASSERT_TRUE(step.persistent_bytes.has_value());
    EXPECT_EQ(step.workspace.size, 0U);
    EXPECT_EQ(step.copy_bytes, 0U);
    EXPECT_EQ(step.node_count, f.nodes.size());
    for (auto node : f.nodes) {
      bool found = false;
      for (std::size_t j = 0; j < step.node_count; ++j)
        found |= report.step_node(i, j).value() == node;
      EXPECT_TRUE(found) << node.ordinal();
    }
  }
  EXPECT_EQ(count, expected ? 1U : 0U);
}
template <class Backend>
void submit(g::Invocation &slot, c::Queue &queue, bool capture) {
  const auto prepared = Backend::prepare(slot, queue, capture);
  check(prepared, "Graph prepare_replay");
  if (capture) { EXPECT_EQ(slot.replay_route(), g::ReplayRoute::kCaptured); }
  auto work = take(slot.submit(queue), "Graph Invocation::submit");
  check(work.wait(), "Graph Completion::wait");
}
inline c::ConstTensorView output(g::Invocation &slot, g::ValueId value) {
  return take(slot.output(value), "Invocation::output ordinal " + std::to_string(value.ordinal()));
}
inline g::Invocation invocation(const g::Executable &program) {
  return take(program.create_invocation({.resources = g::ResourceOptions{}}),
      "Executable::create_invocation with owned resources");
}

// Finite F64 dot reference + conservative F32 reduction error, followed by the
// declared output RNE. The exact one-hot witnesses below use no error budget.
inline void expect_dot(const Logical &f, const Bindings &r, std::span<const double> actual,
                bool witness = false) {
  const auto n = r.fixture.rows, k = r.fixture.depth;
  ASSERT_EQ(actual.size(), static_cast<std::size_t>(f.profile.m * n));
  constexpr double u = std::numeric_limits<float>::epsilon() / 2;
  const double terms = double(2 * k + 32), gamma = terms * u / (1 - terms * u);
  for (std::int64_t m0 = 0; m0 < f.profile.m; ++m0)
    for (std::int64_t n0 = 0; n0 < n; ++n0) {
      double sum = 0, sumabs = 0;
      for (std::int64_t k0 = 0; k0 < k; ++k0) {
        const auto product = r.activation[static_cast<std::size_t>(m0 * k + k0)] *
            r.weight[static_cast<std::size_t>(n0 * k + k0)];
        sum += product; sumabs += std::abs(product);
      }
      const auto i = static_cast<std::size_t>(m0 * n + n0);
      SCOPED_TRACE(i);
      if (std::isnan(sum)) { EXPECT_TRUE(std::isnan(actual[i])); continue; }
      if (std::isinf(sum)) { EXPECT_EQ(actual[i], sum); continue; }
      ASSERT_TRUE(std::isfinite(actual[i]));
      if (witness) EXPECT_EQ(actual[i], t::round(f.profile.operand, sum));
      else {
        const auto bound = gamma * sumabs + double(k) * std::numeric_limits<float>::denorm_min();
        EXPECT_GE(actual[i], t::round(f.profile.operand, sum - bound));
        EXPECT_LE(actual[i], t::round(f.profile.operand, sum + bound));
      }
    }
}
struct Tensor final {
  m::Buffer buffer;
  c::TensorView view;
  Tensor(c::TensorDesc desc, m::MemoryResourceRef memory)
      : buffer(take(m::Buffer::create(memory, desc.required_bytes().value(), 256), "control Buffer::create")),
        view(take(c::TensorView::create(desc, buffer.view()), "control TensorView::create")) {}
};
inline void stage(c::Context &context, c::Queue &queue, c::Operation op,
           std::span<const c::ConstTensorView> inputs, c::TensorView output) {
  const auto label = "ordinary " + std::string(c::operation_name(op));
  auto prepared = take(c::prepare(context, op, inputs, std::span(&output, 1),
      output.storage().resource()), label + " prepare");
  const auto need = prepared.workspace_requirement();
  m::Buffer workspace;
  if (prepared.runtime_properties().workspace_owner == c::ProviderWorkspaceOwner::kCaller && need.size != 0)
    workspace = take(m::Buffer::create(output.storage().resource(), need.size, need.alignment), label + " workspace");
  // The original ATen decoder owns its workspace and rejects even a valid
  // zero-sized caller Buffer. Only caller-owned routes receive a workspace.
  auto call = take(prepared.create_invocation(workspace.view()), label + " create_invocation");
  auto done = take(call.submit(queue, inputs, std::span(&output, 1)), label + " submit");
  check(done.wait(), label + " Completion::wait");
}
inline void ordinary_control(const Logical &f, const Bindings &r, c::Context &context,
                      c::Queue &queue, m::MemoryResourceRef memory, bool witness) {
  std::vector<c::ConstTensorView> packed;
  for (std::size_t i = 1; i < f.ports.size(); ++i)
    packed.push_back(take(r.inputs[i].tensor(f.descriptions[i]), "control packed input " + std::to_string(i)));
  Tensor decoded(f.graph.value_description(f.decoded).value(), memory);
  stage(context, queue, f.decode, packed, decoded.view);
  // Independently verify the original ATen decoder, before the mature MatMul.
  exact_values(read(decoded.view.as_const()), r.fixture.decoded(f.profile.scale,
      f.profile.decode, f.profile.decode));
  c::ConstTensorView weight = decoded.view.as_const();
  std::optional<Tensor> cast;
  if (f.cast.valid() && f.profile.cast_before) {
    cast.emplace(f.graph.value_description(f.cast).value(), memory);
    stage(context, queue, c::Operation(c::Cast{f.profile.operand}), std::array{weight}, cast->view);
    weight = cast->view.as_const();
  }
  weight = take(c::ConstTensorView::create(f.graph.value_description(f.transposed).value(), weight.storage()),
      "control exact transpose view");
  if (f.cast.valid() && !f.profile.cast_before) {
    cast.emplace(f.graph.value_description(f.cast).value(), memory);
    stage(context, queue, c::Operation(c::Cast{f.profile.operand}), std::array{weight}, cast->view);
    weight = cast->view.as_const();
  }
  Tensor result(f.graph.value_description(f.output).value(), memory);
  const std::array operands{take(r.inputs[0].tensor(f.descriptions[0]), "control activation"), weight};
  stage(context, queue, c::Operation(c::MatMul{.accumulation_type = Type::kFloat32}), operands, result.view);
  expect_dot(f, r, read(result.view.as_const()), witness);
}

template <class Backend>
void PublicChainsKeepSourceRoundsAndTypedCastWitnesses() {
  auto context = Backend::create_context();
  auto owner = Backend::create_owner(context);
  auto queue = context.create_queue().value();
  constexpr std::array profiles{
      Profile{false, Type::kFloat32, Type::kFloat32, Type::kFloat32, 1},
      Profile{true, Type::kBFloat16, Type::kBFloat16, Type::kBFloat16, 3},
      Profile{true, Type::kFloat16, Type::kFloat32, Type::kFloat32, 3},
      Profile{false, Type::kBFloat16, Type::kFloat16, Type::kFloat32, 3},
      Profile{true, Type::kFloat32, Type::kBFloat16, Type::kFloat16, 3, false},
      Profile{false, Type::kFloat32, Type::kFloat16, Type::kBFloat16, 3}};
  for (const auto p : profiles) {
    SCOPED_TRACE(::testing::Message() << p.asymmetric << '/' << int(p.scale) << '/'
        << int(p.decode) << '/' << int(p.operand) << '/' << p.cast_before);
    const auto f = build(p);
    auto program = Backend::compile(f.graph, context, owner);
    ASSERT_TRUE(program) << program.error().message();
    selected<Backend>(f, program->report, true);
    Bindings inputs(f, owner, 0, true);
    // Three times the scale differs from the required source-type product.
    if (p.scale == Type::kBFloat16 && p.operand == Type::kFloat32) {
      EXPECT_NE(inputs.weight[0], t::round(p.operand, 3 * t::round(p.scale, inputs.fixture.scales[0])));
    }
    if (p.scale == Type::kFloat16) {
      EXPECT_NE(inputs.weight[32], 3 * t::round(p.scale, inputs.fixture.scales[1]));
    }
    if (p.scale == Type::kFloat32 && p.decode == Type::kBFloat16 && p.operand == Type::kFloat16) {
      EXPECT_NE(inputs.weight[0], t::round(p.operand, 3 * inputs.fixture.scales[0]));
    }
    ordinary_control(f, inputs, context, queue, owner.ref(), true);
    auto slot = invocation(program->executable);
    bind(f, inputs, slot, owner); submit<Backend>(slot, queue, Backend::supports_capture);
    expect_dot(f, inputs, read(output(slot, f.output)), true);
  }
  // Nonfinite payload is legal data; neither route may turn NaN into a pass
  // through fabs/EXPECT_NEAR. Payload classification is checked explicitly.
  const auto f = build({true, Type::kFloat32, Type::kFloat32, Type::kFloat32, 1});
  auto program = take(Backend::compile(f.graph, context, owner), "nonfinite Graph compile");
  selected<Backend>(f, program.report, true);
  Bindings special(f, owner, 0, false, true);
  ordinary_control(f, special, context, queue, owner.ref(), false);
  auto slot = invocation(program.executable);
  bind(f, special, slot, owner); submit<Backend>(slot, queue, Backend::supports_capture);
  expect_dot(f, special, read(output(slot, f.output)));
}

template <class Backend>
void CapturedDynamicInputsRebindAcrossOwnedSlotsAndRetainResults() {
  auto context = Backend::create_context();
  auto owner = Backend::create_owner(context);
  const auto f = build({true, Type::kBFloat16, Type::kFloat16, Type::kFloat32, 3},
      9, 64, Escape::kNone, false, true);
  auto program = take(Backend::compile(f.graph, context, owner), "dynamic Graph compile");
  selected<Backend>(f, program.report, true);
  std::array queues{context.create_queue().value(), context.create_queue().value()};
  std::array slots{invocation(program.executable), invocation(program.executable)};
  std::array<std::vector<double>, 2> old_values;
  std::array<c::ReadLease, 2> old_activation;
  g::GraphOutput old;
  c::ReadLease exported;
  for (unsigned round = 0; round < 2; ++round)
    for (const unsigned lane : {1U, 0U}) {
      if (round) {
        if (lane == 0) old.reset();
        check(slots[lane].reset(), "dynamic Invocation::reset round " + std::to_string(round));
      }
      // The selected compile domain fixes the exact resource object, rather
      // than accepting every allocator on the same GPU. Fresh Buffers use
      // this original domain; the temporary caller owner is dropped below.
      auto input_owner = owner;
      Bindings inputs(f, input_owner, round * 2 + lane);
      const auto source = take(inputs.inputs[0].view(), "fresh activation lease view");
      if (round) { EXPECT_NE(source.native_handle(), take(old_activation[lane].view(), "old activation lease view").native_handle()); }
      bind(f, inputs, slots[lane], owner);
      // Only the original slot's ReadLeases own these fresh input allocations.
      // No raw payload/owner is retained by this caller across submission.
      inputs.inputs.clear(); input_owner = {};
      submit<Backend>(slots[lane], queues[lane], Backend::supports_capture);
      auto result = take(slots[lane].result(f.output), "dynamic Invocation::result output");
      const auto actual = read(take(result.tensor(), "dynamic GraphOutput::tensor"));
      expect_dot(f, inputs, actual);
      if (round == 0) old_values[lane] = actual;
      else EXPECT_NE(actual, old_values[lane]); // Changed codes/Z/scales/A, no stale packed cache.
      EXPECT_TRUE(slots[lane].reset().IsUnavailable());
      EXPECT_TRUE(slots[lane].bind(f.ports[1], f.descriptions[1], c::ReadLease{}).IsUnavailable());
      if (round == 0 && lane == 0) old = std::move(result);
      if (round == 0) {
        auto source_result = take(slots[lane].result(f.ports[0]), "published activation GraphOutput");
        old_activation[lane] = take(source_result.storage_lease(), "published activation storage lease");
      }
      if (round == 1 && lane == 1) {
        exact_values(read(take(old.tensor(), "retained GraphOutput::tensor")), old_values[0]);
        auto first_export = take(old.storage_lease(), "retained GraphOutput::storage_lease");
        EXPECT_NE(take(result.tensor(), "other slot GraphOutput::tensor").storage().native_handle(),
            take(first_export.view(), "retained storage view").native_handle());
      }
      if (round == 1 && lane == 0) exported = take(result.storage_lease(), "final independent output lease");
    }
  const auto exported_before = read(take(exported.tensor(f.graph.value_description(f.output).value()), "export before owner teardown"));
  old.reset(); old_activation = {}; slots = {}; queues = {};
  program.executable = {}; context = {};
  // The independent lease owns its Buffer; the original typed resource owner
  // remains explicit, as required by the public LeasePool borrowing contract.
  exact_values(read(take(exported.tensor(f.graph.value_description(f.output).value()), "export after Context handle close")), exported_before);
  exported.reset(); owner = {};
}

template <class Backend>
void PublishedViewAndControlEscapesExecuteTheOriginalDecomposition() {
  auto context = Backend::create_context();
  auto owner = Backend::create_owner(context);
  auto queue = context.create_queue().value();
  for (auto escape : {Escape::kPublishedDecode, Escape::kExtraView, Escape::kControl}) {
    const auto f = build({true, Type::kBFloat16, Type::kFloat32, Type::kFloat32, 3}, 9, 64, escape);
    auto program = Backend::compile_unrestricted(f.graph, context, owner);
    ASSERT_TRUE(program) << program.error().message(); // A decline is not this boundary's expected success.
    selected<Backend>(f, program->report, false);
    Bindings inputs(f, owner, 2);
    ordinary_control(f, inputs, context, queue, owner.ref(), false);
    auto slot = invocation(program->executable);
    bind(f, inputs, slot, owner); submit<Backend>(slot, queue, false);
    expect_dot(f, inputs, read(output(slot, f.output)));
    exact_values(read(output(slot, f.extra)), escape == Escape::kControl
        ? inputs.activation : inputs.fixture.decoded(f.profile.scale, f.profile.decode, f.profile.decode));
  }
}

template <class Backend>
void OriginalPackedConstantsDoNotRetainOrWorkspaceADenseWeightBank() {
  auto context = Backend::create_context();
  auto owner = Backend::create_owner(context);
  const auto f = build({true, Type::kBFloat16, Type::kFloat32, Type::kFloat32, 3},
      257, 1024, Escape::kNone, true);
  Bindings inputs(f, owner, 1);
  std::vector<g::ConstantBinding> constants;
  for (std::size_t i = 1; i < f.ports.size(); ++i)
    constants.push_back({f.ports[i], take(inputs.inputs[i].tensor(f.descriptions[i]), "constant packed input"),
        g::ConstantPolicy::kBorrow, owner, inputs.inputs[i]});
  auto program = Backend::compile(f.graph, context, owner, {.constants = constants});
  ASSERT_TRUE(program) << program.error().message();
  selected<Backend>(f, program->report, true);
  const auto storage = program->report.compiled_storage();
  ASSERT_TRUE(storage.total_bytes().has_value());
  constexpr std::size_t dense_bank_bytes = 257 * 1024 * sizeof(float);
  constexpr std::size_t minimum_float_bank_bytes = 257 * 1024 * sizeof(std::uint16_t);
  EXPECT_LT(*storage.total_bytes(), minimum_float_bank_bytes);
  EXPECT_LT(program->report.workspace_bytes(), minimum_float_bank_bytes);
  ::testing::Test::RecordProperty("compiled_known_bytes", std::to_string(*storage.total_bytes()));
  ::testing::Test::RecordProperty("workspace_bytes", std::to_string(program->report.workspace_bytes()));
  ::testing::Test::RecordProperty("excluded_dense_f32_bank_bytes", std::to_string(dense_bank_bytes));
  auto slot = invocation(program->executable);
  auto queue = context.create_queue().value();
  bind(f, inputs, slot, owner); submit<Backend>(slot, queue, Backend::supports_capture);
  expect_dot(f, inputs, read(output(slot, f.output)));
}
} // namespace ct_packed_graph_test
