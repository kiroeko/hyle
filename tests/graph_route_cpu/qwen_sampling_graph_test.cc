#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "../../engine/inference/sampling_graph.h"
#include "../../engine/inference/mtp_acceptance.h"
#include "../../engine/models/qwen35/session_state.h"
#include "../compute_models/sampling_fixture.h"
#if defined(HYLE_TEST_QWEN_SAMPLING_CUDA)
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#else
#include "hyle/graph/cpu.h"
#endif
#include "hyle/mem/buffer.h"

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace m = hyle::mem;
namespace s = hyle::inference::sampling;
using V = g::ValueId;
using D = g::Dimension;
using Type = c::DataType;
using sampling_test::take;
using sampling_test::require;
using Identity = sampling_test::Words;

c::TensorDesc dense(Type type, std::initializer_list<std::int64_t> dims) {
  return take(c::TensorDesc::dense(type, {dims.begin(), dims.size()}));
}

// Only graph construction conveniences are mirrored from the engine Recipe.
// Execution, scalar validation, RNG and all arithmetic use real public nodes.
struct Recipe final {
  g::GraphBuilder builder = take(g::GraphBuilder::create());
  std::vector<g::TensorSpec> specs;
  std::vector<m::Buffer> literals;
  std::vector<g::ConstantBinding> constants;
  m::MemoryResourceRef resource = m::system_resource();
  const g::TensorSpec &spec(V value) const { return specs.at(value.ordinal()); }
  V remember(V value, const c::TensorDesc &desc) {
    if (specs.size() <= value.ordinal()) specs.resize(value.ordinal() + 1);
    specs[value.ordinal()] = take(g::TensorSpec::concrete(desc));
    return value;
  }
  V input(c::TensorDesc desc) { return remember(take(builder.input(desc)), desc); }
  std::vector<V> op(c::Operation operation, std::initializer_list<V> arguments) {
    std::vector<c::TensorDesc> in;
    for (const auto value : arguments) in.push_back(take(spec(value).concrete_description()));
    std::vector<c::TensorDesc> out(c::operation_traits(operation).outputs);
    require(c::infer_output_descriptions(operation, in, out));
    const auto node = take(builder.add_node(operation, {arguments.begin(), arguments.size()}));
    std::vector<V> values;
    for (std::size_t i = 0; i < out.size(); ++i) values.push_back(remember(take(builder.result(node, i)), out[i]));
    return values;
  }
  template<class O> V one(O operation, std::initializer_list<V> arguments) {
    return op(c::Operation(operation), arguments).at(0);
  }
  V scalar(float value) {
    const auto desc = dense(Type::kFloat32, {1});
    literals.push_back(take(m::Buffer::create(resource, sizeof(float), 64)));
    require(literals.back().view().write(std::as_bytes(std::span(&value, 1))));
    const auto result = remember(take(builder.constant(desc)), desc);
    constants.push_back({result, take(c::ConstTensorView::create(desc, literals.back().view())),
                         g::ConstantPolicy::kBorrow});
    return result;
  }
  V cast(V value, Type type) {
    return spec(value).data_type() == type ? value : one(c::Cast{type}, {value});
  }
  V unary(c::UnaryKind kind, V value) { return one(c::Unary{kind}, {value}); }
  V copy(V value) { return one(c::Copy{}, {value}); }
  V binary(c::BinaryKind kind, V a, V b) { return one(c::Binary{kind}, {a, b}); }
  V reshape(V value, std::initializer_list<D> dims) {
    std::vector<std::int64_t> shape;
    for (const auto dim : dims) shape.push_back(dim.fixed_value());
    return one(take(c::View::reshape(take(c::TensorShape::create(shape)))), {value});
  }
  V slice(V value, int axis, D first, D count) {
    const auto source = spec(value);
    std::vector<std::int64_t> dimensions;
    for (const auto dim : source.dimensions()) dimensions.push_back(dim.fixed_value());
    dimensions.at(static_cast<std::size_t>(axis)) = count.fixed_value();
    const auto node = take(builder.add_node(take(g::ViewSpec::slice(axis,first,count)),value));
    return remember(take(builder.result(node)),take(c::TensorDesc::dense(source.data_type(),dimensions)));
  }
};

struct Answer final {
  bool accepted = false, valid = false;
  std::int32_t final_id = -1;
  bool p_valid = false, q_valid = false, bonus_valid = false;
  bool proposal_valid = false, residual_positive = false;
};
class Session final {
public:
  explicit Session(std::size_t vocabulary, std::optional<s::Policy> policy = {}) {
#if defined(HYLE_TEST_QWEN_SAMPLING_CUDA)
    context_ = take(g::cuda::create_context());
    device_ = take(c::cuda::MemoryResource::create(context_, c::cuda::MemoryKind::kDevice));
    resource_ = device_.ref();
#else
    context_ = take(g::cpu::create_context());
#endif
    queue_ = take(context_.create_queue());
    Recipe r;
    r.resource = resource_;
    const auto floats = dense(Type::kFloat32, {1, static_cast<std::int64_t>(vocabulary)});
    const auto id = dense(Type::kInt32, {1}), key = dense(Type::kUInt32, {1, 4});
    const std::array descriptions{floats, floats, floats, id, key, key};
    for (const auto &desc : descriptions) {
      input_ids_.push_back(r.input(desc));
      input_desc_.push_back(desc);
      inputs_.push_back(take(m::Buffer::create(resource_, take(desc.required_bytes()), 64)));
    }
    auto p = input_ids_[0], q = input_ids_[1], bonus = input_ids_[2];
    std::vector<V> policy_valid;
    if (policy) {
      const auto target = s::distribution(r, p, *policy), draft = s::distribution(r, q, *policy),
          next = s::distribution(r, bonus, *policy);
      p = target.log_probabilities; q = draft.log_probabilities; bonus = next.log_probabilities;
      policy_valid = {target.valid, draft.valid, next.valid};
    }
    auto result = s::decision(r, p, q, bonus,
        input_ids_[3], input_ids_[4], input_ids_[5]);
    for (const auto valid : policy_valid)
      result.valid = r.one(c::Select{}, {valid, result.valid, r.cast(r.scalar(0), Type::kBool)});
    output_ids_ = {result.accepted, result.final_id, result.valid, result.p_valid,
        result.q_valid, result.bonus_valid, result.proposal_valid, result.residual_positive};
    if (policy) output_ids_.push_back(p); // Small toy oracle only, never a production full-V readback.
    for (const auto value : output_ids_) {
      require(r.builder.publish(value));
      const auto desc = take(r.spec(value).concrete_description());
      output_desc_.push_back(desc);
      outputs_.push_back(take(m::Buffer::create(resource_, take(desc.required_bytes()), 64)));
    }
    const auto graph = take(std::move(r.builder).finish());
#if defined(HYLE_TEST_QWEN_SAMPLING_CUDA)
    auto compiled = take(g::cuda::compile(graph, context_, resource_, {.constants = r.constants}));
#else
    auto compiled = take(g::cpu::compile(graph, context_, {.constants = r.constants}));
#endif
    program_ = std::move(compiled.executable);
    // Borrowed immutable literals outlive the compiled graph, invocation and
    // every completion. CUDA therefore needs no unowned queued materialization.
    literals_ = std::move(r.literals);
    slot_ = take(program_.create_invocation());
  }
  Answer run(std::span<const float> p, std::span<const float> q, std::span<const float> bonus,
             std::int32_t proposal, Identity accept, Identity final) {
    const std::array<std::span<const std::byte>, 6> payloads{std::as_bytes(p), std::as_bytes(q), std::as_bytes(bonus),
        std::as_bytes(std::span(&proposal, 1)), std::as_bytes(std::span(accept)), std::as_bytes(std::span(final))};
    require(slot_.reset());
    for (std::size_t i = 0; i < inputs_.size(); ++i) {
      require(inputs_[i].view().write(payloads[i]));
      require(slot_.bind(input_ids_[i], take(c::ConstTensorView::create(input_desc_[i], inputs_[i].view()))));
    }
    for (std::size_t i = 0; i < outputs_.size(); ++i)
      require(slot_.bind(output_ids_[i], take(c::TensorView::create(output_desc_[i], outputs_[i].view()))));
    const std::array queues{g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue_}};
    require(slot_.prepare_replay(queues, {.mode = g::ReplayMode::kCommands, .host_wait = g::HostWaitPolicy::kReject}));
    const auto report = take(slot_.replay_report());
    sampling_test::check(report.recorded_capture_range_count == 0 && report.reused_capture_range_count == 0,
        "commands sampling unexpectedly entered a capture route");
    auto done = take(slot_.submit(queue_));
    require(done.wait());
    const auto boolean = [&](std::size_t port) {
      std::byte value;
      require(outputs_[port].view().read(std::span(&value, 1)));
      return value != std::byte{};
    };
    std::int32_t final_id = -1;
    require(outputs_[1].view().read(std::as_writable_bytes(std::span(&final_id, 1))));
    return {boolean(0), boolean(2), final_id, boolean(3), boolean(4), boolean(5), boolean(6), boolean(7)};
  }
  std::vector<float> log_probabilities() const {
    if (outputs_.size() != 9) throw std::runtime_error("this toy session has no policy output");
    std::vector<float> result(outputs_[8].size() / sizeof(float));
    require(outputs_[8].view().read(std::as_writable_bytes(std::span(result))));
    return result;
  }
#if defined(HYLE_TEST_QWEN_SAMPLING_CUDA)
  hyle::Status require_capture() {
    const std::array queues{g::QueueBinding{.lane = g::ExecutionLaneId{0}, .queue = &queue_}};
    return slot_.prepare_replay(queues, {.mode = g::ReplayMode::kRequireCapture,
        .host_wait = g::HostWaitPolicy::kReject});
  }
#endif
private:
  c::Context context_;
#if defined(HYLE_TEST_QWEN_SAMPLING_CUDA)
  c::cuda::MemoryResource device_;
#endif
  m::MemoryResourceRef resource_ = m::system_resource();
  c::Queue queue_;
  std::vector<m::Buffer> inputs_, outputs_, literals_;
  std::vector<c::TensorDesc> input_desc_, output_desc_;
  std::vector<V> input_ids_, output_ids_;
  g::Executable program_;
  g::Invocation slot_;
};

std::vector<float> logs(std::initializer_list<double> values) {
  std::vector<float> out;
  for (const auto value : values) out.push_back(static_cast<float>(std::log(value)));
  return out;
}
Identity key(std::uint64_t draw, bool final = false) {
  return {final ? 0xb7e15162u : 0x243f6a88u, final ? 0x8aed2a6bu : 0x85a308d3u,
      static_cast<std::uint32_t>(draw), static_cast<std::uint32_t>(draw >> 32)};
}
// Independent scalar reference uses the documented route's draw equations,
// upstream-KAT-qualified test Philox and F64 arithmetic, never production RNG.
[[maybe_unused]] double uniform(Identity identity) {
  const auto words = sampling_test::philox({0, 0, identity[2], identity[3]}, {identity[0], identity[1]});
  return (static_cast<double>(words[0]) * 0x1p20 + static_cast<double>(words[1] >> 12) + .5) * 0x1p-52;
}
[[maybe_unused]] std::int32_t categorical(std::span<const double> probabilities, double u) {
  double mass = 0;
  for (const auto value : probabilities) mass += value;
  double cumulative = 0;
  for (std::size_t i = 0; i < probabilities.size(); ++i) {
    cumulative += probabilities[i];
    if (cumulative > u * mass) return static_cast<std::int32_t>(i);
  }
  throw std::runtime_error("reference categorical has no positive mass");
}
std::int32_t reference_draw(std::span<const double> probabilities, Identity identity) {
#if defined(HYLE_TEST_QWEN_SAMPLING_CUDA)
  double maximum = -std::numeric_limits<double>::infinity();
  std::int32_t selected = -1;
  for (std::size_t i = 0; i < probabilities.size(); ++i) {
    const auto words = sampling_test::philox({static_cast<std::uint32_t>(i / 2), 0, identity[2], identity[3]},
        {identity[0], identity[1]});
    auto hi = words[(i % 2) * 2], lo = words[(i % 2) * 2 + 1];
    const bool upper = (hi & 0x80000000u) != 0;
    if (upper) { hi = ~hi; lo = ~lo; }
    const double small = std::ldexp(static_cast<double>(hi), -32) + std::ldexp(static_cast<double>(lo) + .5, -64);
    const auto gumbel = -std::log(upper ? -std::log1p(-small) : -std::log(small));
    const auto score = std::log(probabilities[i]) + gumbel;
    if (score > maximum) { maximum = score; selected = static_cast<std::int32_t>(i); }
  }
  if (selected < 0) throw std::runtime_error("reference Gumbel has no positive mass");
  return selected;
#else
  return categorical(probabilities, uniform(identity));
#endif
}
bool accept_draw(double alpha, Identity identity) {
  return reference_draw(std::array{alpha, 1. - alpha}, identity) == 0;
}

TEST(QwenSamplingGraph, AcceptanceResidualAndBonusMatchIndependentRouteDraw) {
  Session session(3);
  const auto p = logs({.125, .375, .5}), q = logs({.5, .25, .25}), bonus = logs({.25, .5, .25});
  const std::array<double, 3> residual{0, .125, .25}, bonus_prob{.25, .5, .25};
  std::size_t accepts = 0, rejects = 0;
  for (std::uint64_t draw = 0; draw < 64; ++draw) {
    const auto y = static_cast<std::int32_t>(draw % 3);
    const auto accept_key = key(draw), final_key = key(draw, true);
    const auto actual = session.run(p, q, bonus, y, accept_key, final_key);
    ASSERT_TRUE(actual.valid);
    const auto alpha = (std::min)(1., std::exp(static_cast<double>(p[static_cast<std::size_t>(y)]) - q[static_cast<std::size_t>(y)]));
    const auto accepted = accept_draw(alpha, accept_key);
    EXPECT_EQ(actual.accepted, accepted) << draw;
    EXPECT_EQ(actual.final_id, reference_draw(accepted ? bonus_prob : residual, final_key)) << draw;
    EXPECT_TRUE(actual.residual_positive);
    actual.accepted ? ++accepts : ++rejects;
  }
  EXPECT_GT(accepts, 0u);
  EXPECT_GT(rejects, 0u);
  // The first published token has target marginal p, including rejection mass.
  const std::array<double, 3> pp{.125, .375, .5}, qq{.5, .25, .25};
  double rejected_mass = 0, residual_mass = 0;
  for (std::size_t i = 0; i < 3; ++i) {
    rejected_mass += qq[i] - (std::min)(pp[i], qq[i]);
    residual_mass += residual[i];
  }
  for (std::size_t i = 0; i < 3; ++i)
    EXPECT_NEAR((std::min)(pp[i], qq[i]) + rejected_mass * residual[i] / residual_mass, pp[i], 1e-14);
  // Same identities repeat exactly after unrelated invocation/input changes.
  const auto repeated = session.run(p, q, bonus, 0, key(0), key(0, true));
  EXPECT_EQ(repeated.accepted, accept_draw(.25, key(0)));
  EXPECT_EQ(repeated.final_id, reference_draw(repeated.accepted ? bonus_prob : residual, key(0, true)));
}

TEST(QwenSamplingGraph, EqualAndDisjointSupportsKeepEagerBranchesSafe) {
  Session session(3);
  const auto p = logs({0, .25, .75}), bonus = logs({1, 0, 0});
  for (std::uint64_t draw = 0; draw < 8; ++draw) {
    const auto same = session.run(p, p, bonus, 1, key(draw), key(draw, true));
    EXPECT_TRUE(same.valid);
    EXPECT_TRUE(same.accepted);
    EXPECT_FALSE(same.residual_positive);
    EXPECT_EQ(same.final_id, 0);
  }
  const auto disjoint = session.run(logs({0, 1, 0}), logs({1, 0, 0}), logs({0, 0, 1}), 0, key(0), key(0, true));
  EXPECT_TRUE(disjoint.valid);
  EXPECT_FALSE(disjoint.accepted);
  EXPECT_TRUE(disjoint.residual_positive);
  EXPECT_EQ(disjoint.final_id, 1);
}

TEST(QwenSamplingGraph, LogResidualRetainsMassLostByRoundedExpSubtraction) {
  Session session(2);
  const std::vector<float> p{static_cast<float>(std::log1p(-std::exp(-20.))), -20.F};
  const std::vector<float> q{static_cast<float>(std::log1p(-std::exp(-18.))), -18.F};
  ASSERT_GT(p[0], q[0]);
  ASSERT_EQ(std::exp(p[0]), std::exp(q[0])); // Ordinary F32 subtraction loses this residual.
  std::size_t rejects = 0;
  for (std::uint64_t draw = 0; draw < 16; ++draw) {
    const auto actual = session.run(p, q, logs({0, 1}), 1, key(draw), key(draw, true));
    ASSERT_TRUE(actual.valid);
    EXPECT_TRUE(actual.residual_positive);
    const auto accepted = accept_draw(std::exp(-2.), key(draw));
    EXPECT_EQ(actual.accepted, accepted);
    EXPECT_EQ(actual.final_id, accepted ? 1 : 0);
    if (!actual.accepted) ++rejects;
  }
  EXPECT_GT(rejects, 0u);
}

TEST(QwenSamplingGraph, EveryColumnIsValidatedBeforeOutputsCanBePublished) {
  Session session(3);
  const auto original = logs({.25, .25, .5}), bonus = logs({1, 0, 0});
  for (const auto bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
    auto damaged = original;
    damaged[2] = bad; // Not the selected proposal.
    auto actual = session.run(damaged, original, bonus, 0, key(0), key(0, true));
    EXPECT_FALSE(actual.valid);
    EXPECT_FALSE(actual.p_valid);
    actual = session.run(original, damaged, bonus, 0, key(0), key(0, true));
    EXPECT_FALSE(actual.valid);
    EXPECT_FALSE(actual.q_valid);
    actual = session.run(original, original, damaged, 0, key(0), key(0, true));
    EXPECT_FALSE(actual.valid);
    EXPECT_FALSE(actual.bonus_valid);
  }
  const auto empty = logs({0, 0, 0});
  EXPECT_FALSE(session.run(empty, original, bonus, 0, key(0), key(0, true)).p_valid);
  EXPECT_FALSE(session.run(original, empty, bonus, 0, key(0), key(0, true)).q_valid);
  EXPECT_FALSE(session.run(original, original, empty, 0, key(0), key(0, true)).bonus_valid);
  const auto zero_q = session.run(original, logs({0, .5, .5}), bonus, 0, key(0), key(0, true));
  EXPECT_FALSE(zero_q.valid);
  EXPECT_FALSE(zero_q.proposal_valid);
  EXPECT_TRUE(zero_q.q_valid);
}

TEST(QwenSamplingGraph, ProposalBoundsUseOriginalCheckedCompletion) {
  for (const auto y : {-1, 3}) {
    Session session(3);
    const auto p = logs({.25, .25, .5});
    EXPECT_THROW((void)session.run(p, p, p, y, key(0), key(0, true)), std::runtime_error);
  }
}

TEST(QwenSamplingGraph, InvalidColdPortsFailBeforeBuildingTheRecipe) {
  Recipe r;
  const auto p = r.input(dense(Type::kFloat32, {1, 3}));
  const auto wider = r.input(dense(Type::kFloat32, {2, 3}));
  const auto id = r.input(dense(Type::kInt32, {1}));
  const auto key_port = r.input(dense(Type::kUInt32, {1, 4}));
  EXPECT_THROW((void)s::decision(r, wider, wider, wider, id, key_port, key_port), std::runtime_error);
  EXPECT_THROW((void)s::decision(r, p, wider, p, id, key_port, key_port), std::runtime_error);
  EXPECT_THROW((void)s::decision(r, p, p, p, p, key_port, key_port), std::runtime_error);
  EXPECT_THROW((void)s::decision(r, p, p, p, id, p, key_port), std::runtime_error);
}

void expect_probabilities(std::span<const float> actual, std::span<const double> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  double total = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (!expected[i]) EXPECT_EQ(actual[i], -std::numeric_limits<float>::infinity());
    else EXPECT_NEAR(std::exp(static_cast<double>(actual[i])), expected[i], 3e-6) << i;
    total += std::exp(static_cast<double>(actual[i]));
  }
  EXPECT_NEAR(total, 1., 3e-6);
}

TEST(QwenSamplingGraph, PolicyHasExplicitTemperatureThenTopKThenNucleusOrder) {
  Session session(3, s::Policy{.temperature = 2, .top_k = 2, .top_p = .6F});
  const auto logits = logs({1, 2, 4});
  const auto actual = session.run(logits, logits, logits, 2, key(0), key(0, true));
  ASSERT_TRUE(actual.valid);
  EXPECT_TRUE(actual.accepted);
  // After temperature, highest probability among the top two is .5858 < .6,
  // so both survive. Applying temperature after top-p would retain only ID2.
  const double small = std::sqrt(2.) / (2. + std::sqrt(2.));
  expect_probabilities(session.log_probabilities(), std::array{0., small, 1. - small});

  Session all(3, s::Policy{.top_p = .6F});
  ASSERT_TRUE(all.run(logits, logits, logits, 2, key(0), key(0, true)).valid);
  expect_probabilities(all.log_probabilities(), std::array{0., 1. / 3, 2. / 3});
}

TEST(QwenSamplingGraph, PolicyStableTiesAndExactCdfBoundariesRetainFirst) {
  const std::array<float, 4> logits{0, 0, 0, 0};
  for (const auto policy : {s::Policy{.top_p = .5F}, s::Policy{.top_k = 2},
                            s::Policy{.top_k = 2, .top_p = 1}}) {
    Session session(4, policy);
    ASSERT_TRUE(session.run(logits, logits, logits, 0, key(0), key(0, true)).valid);
    expect_probabilities(session.log_probabilities(), std::array{.5, .5, 0., 0.});
  }
  for (const auto top_p : {.25F, (std::numeric_limits<float>::denorm_min)()}) {
    Session session(4, s::Policy{.top_p = top_p});
    ASSERT_TRUE(session.run(logits, logits, logits, 0, key(0), key(0, true)).valid);
    expect_probabilities(session.log_probabilities(), std::array{1., 0., 0., 0.});
  }
}

TEST(QwenSamplingGraph, PolicyCannotHideBadUnselectedColumnsAndCentersLargeLogits) {
  Session session(3, s::Policy{.temperature = .5F, .top_k = 1, .top_p = .5F});
  for (const auto bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
    const std::array logits{3.F, 2.F, bad};
    EXPECT_FALSE(session.run(logits, logits, logits, 0, key(0), key(0, true)).valid);
  }
  const auto empty = logs({0, 0, 0});
  EXPECT_FALSE(session.run(empty, empty, empty, 0, key(0), key(0, true)).valid);
  Session large(3, s::Policy{.temperature = .5F});
  const std::array logits{3e38F, 3e38F, -3e38F};
  ASSERT_TRUE(large.run(logits, logits, logits, 0, key(0), key(0, true)).valid);
  expect_probabilities(large.log_probabilities(), std::array{.5, .5, 0.});
}

TEST(QwenSamplingGraph, PolicyParametersFailClearlyAtColdConstruction) {
  Recipe r;
  const auto p = r.input(dense(Type::kFloat32, {1, 3}));
  for (const auto policy : {s::Policy{.temperature = 0}, s::Policy{.temperature = -1},
      s::Policy{.temperature = std::numeric_limits<float>::infinity()}, s::Policy{.top_p = 0},
      s::Policy{.top_p = 1.1F}, s::Policy{.top_p = std::numeric_limits<float>::quiet_NaN()}, s::Policy{.top_k = 4}})
    EXPECT_THROW((void)s::distribution(r, p, policy), std::runtime_error);
}

// This fixture invokes the very same finite row-assembly recipe as the real
// engine. It retains tiny distributions only for an independent scalar oracle;
// the production path still reads back only scalar validity/decisions/IDs.
class PrefixSession final {
public:
  struct Answer final {std::vector<std::uint8_t> accepted,valid;std::vector<std::int32_t> final_ids;};
  explicit PrefixSession(std::size_t count):count_(count) {
#if defined(HYLE_TEST_QWEN_SAMPLING_CUDA)
    context_=take(g::cuda::create_context());device_=take(c::cuda::MemoryResource::create(context_,c::cuda::MemoryKind::kDevice));resource_=device_.ref();
#else
    context_=take(g::cpu::create_context());
#endif
    queue_=take(context_.create_queue());Recipe r;r.resource=resource_;
    const auto rows=static_cast<std::int64_t>(count+1);
    logits_=r.input(dense(Type::kFloat32,{rows,3}));keys_=r.input(dense(Type::kUInt32,{rows,4}));
    if(count)ids_=r.input(dense(Type::kInt32,{static_cast<std::int64_t>(count)}));
    result_=s::speculative_prefix(r,logits_,keys_,ids_);
    inputs_={logits_,keys_};if(count)inputs_.push_back(ids_);
    inputs_.insert(inputs_.end(),result_.q.begin(),result_.q.end());
    for(std::size_t row=0;row<result_.final_ids.size();++row){
      if(count)outputs_.push_back(result_.accepted[row]);
      outputs_.push_back(result_.final_ids[row]);outputs_.push_back(result_.valid[row]);
    }
    for(auto value:outputs_)require(r.builder.publish(value));
    const auto allocate=[&](auto ports,auto &descriptions,auto &buffers){
      for(auto value:ports){const auto desc=take(r.spec(value).concrete_description());descriptions.push_back(desc);buffers.push_back(take(m::Buffer::create(resource_,take(desc.required_bytes()),64)));}
    };
    allocate(inputs_,input_desc_,input_buffers_);allocate(outputs_,output_desc_,output_buffers_);
    auto graph=take(std::move(r.builder).finish());
#if defined(HYLE_TEST_QWEN_SAMPLING_CUDA)
    auto compiled=take(g::cuda::compile(graph,context_,resource_,{.constants=r.constants}));
#else
    auto compiled=take(g::cpu::compile(graph,context_,{.constants=r.constants}));
#endif
    program_=std::move(compiled.executable);literals_=std::move(r.literals);run_=take(program_.create_invocation());
  }
  Answer run(std::span<const float> logits,std::span<const float> q,
             std::span<const std::int32_t> ids,qwen35::cache::SamplingCursor cursor) {
    if(logits.size()!=(count_+1)*3||q.size()!=count_*3||ids.size()!=count_)throw std::runtime_error("tiny prefix fixture input extent differs");
    std::vector<std::uint32_t> keys;
    for(std::size_t row=0;row<=count_;++row){const auto key=cursor.identity(row==count_?2*count_:2*row+1);keys.insert(keys.end(),key.begin(),key.end());}
    std::vector<std::span<const std::byte>> payloads{std::as_bytes(logits),std::as_bytes(std::span(keys))};
    if(count_)payloads.push_back(std::as_bytes(ids));
    for(std::size_t row=0;row<count_;++row)payloads.push_back(std::as_bytes(q.subspan(row*3,3)));
    require(run_.reset());
    for(std::size_t i=0;i<inputs_.size();++i){require(input_buffers_[i].view().write(payloads[i]));require(run_.bind(inputs_[i],take(c::ConstTensorView::create(input_desc_[i],input_buffers_[i].view()))));}
    for(std::size_t i=0;i<outputs_.size();++i)require(run_.bind(outputs_[i],take(c::TensorView::create(output_desc_[i],output_buffers_[i].view()))));
    const std::array queues{g::QueueBinding{.lane=g::ExecutionLaneId{0},.queue=&queue_}};
    require(run_.prepare_replay(queues,{.mode=g::ReplayMode::kCommands,.host_wait=g::HostWaitPolicy::kReject}));auto done=take(run_.submit(queues));require(done.wait());
    Answer answer;answer.accepted.resize(count_);answer.final_ids.resize(count_?count_:1);answer.valid.resize(answer.final_ids.size());
    std::size_t index=0;
    for(std::size_t row=0;row<answer.final_ids.size();++row){
      if(count_)require(output_buffers_[index++].view().read(std::as_writable_bytes(std::span(&answer.accepted[row],1))));
      require(output_buffers_[index++].view().read(std::as_writable_bytes(std::span(&answer.final_ids[row],1))));
      require(output_buffers_[index++].view().read(std::as_writable_bytes(std::span(&answer.valid[row],1))));
    }
    return answer;
  }
private:
  std::size_t count_;
  c::Context context_;
#if defined(HYLE_TEST_QWEN_SAMPLING_CUDA)
  c::cuda::MemoryResource device_;
#endif
  m::MemoryResourceRef resource_=m::system_resource();c::Queue queue_;
  V logits_,keys_,ids_;s::PrefixDecisions result_;
  std::vector<V>inputs_,outputs_;std::vector<c::TensorDesc>input_desc_,output_desc_;
  std::vector<m::Buffer>input_buffers_,output_buffers_,literals_;g::Executable program_;g::Invocation run_;
};

TEST(QwenSamplingGraph, MultiRowPrefixKeepsConditionalRowsAndOneFinalAddress) {
  namespace policy=hyle::inference::mtp;
  const std::array<std::array<double,3>,5> target{{{.125,.375,.5},{.75,.125,.125},{.25,.5,.25},{.5,.25,.25},{.125,.125,.75}}};
  const std::array<std::array<double,3>,4> draft{{{.5,.25,.25},{.125,.5,.375},{.5,.125,.375},{.75,.125,.125}}};
  const std::array<std::int32_t,4> proposal{0,1,2,0};
  for(const auto count:{std::size_t{0},std::size_t{2},std::size_t{4}}){
    PrefixSession session(count);std::vector<float> logits,q;
    for(std::size_t row=0;row<=count;++row)for(auto p:target[row])logits.push_back(static_cast<float>(std::log(p)));
    for(std::size_t row=0;row<count;++row)for(auto p:draft[row])q.push_back(static_cast<float>(std::log(p)));
    std::vector<std::int32_t> inputs{1};inputs.insert(inputs.end(),proposal.begin(),proposal.begin()+static_cast<std::ptrdiff_t>(count));
    // Choose a finite deterministic set that witnesses first/middle rejection
    // and all accepted. This is route-draw conformance, not a Monte Carlo test.
    std::vector<bool>witness(count+1);std::size_t covered=0;
    for(std::uint64_t draw=0;draw<4096&&covered<witness.size();++draw){
      const qwen35::cache::SamplingCursor cursor{123,7,draw};std::vector<std::uint8_t> accepts(count);
      std::vector<std::int32_t> finals(count?count:1);std::size_t first_reject=count;
      for(std::size_t row=0;row<count;++row){
        const auto y=static_cast<std::size_t>(proposal[row]);const double alpha=(std::min)(1.,target[row][y]/draft[row][y]);
        accepts[row]=accept_draw(alpha,cursor.identity(2*row+1));
        if(!accepts[row]&&first_reject==count)first_reject=row;
        std::array<double,3> residual;for(std::size_t v=0;v<3;++v)residual[v]=(std::max)(0.,target[row][v]-draft[row][v]);
        finals[row]=reference_draw(accepts[row]?target[row+1]:residual,cursor.identity(2*count));
      }
      if(!count)finals[0]=reference_draw(target[0],cursor.identity());
      if(witness[first_reject])continue;witness[first_reject]=true;++covered;
      const auto actual=session.run(logits,q,std::span(proposal).first(count),cursor);
      EXPECT_EQ(actual.accepted,accepts)<<"m="<<count<<" draw="<<draw;
      EXPECT_EQ(actual.final_ids,finals)<<"m="<<count<<" draw="<<draw;
      for(auto valid:actual.valid)ASSERT_EQ(valid,1U);
      for(const auto quota:{std::size_t{1},std::size_t{2},count+1})for(bool with_eos:{false,true}){
        const std::array<std::int32_t,1> eos{2};std::vector<std::int32_t> emitted,expected;
        const auto selected=policy::select_sampled_prefix(inputs,actual.accepted,actual.final_ids,quota,with_eos?std::span(eos):std::span<const std::int32_t>{},3,emitted);
        std::size_t consumed=0;bool stopped=false;
        for(std::size_t row=0;row<=count;++row){
          const bool accepted=row<count&&accepts[row];const auto id=accepted?proposal[row]:finals[(std::min)(row,finals.size()-1)];
          expected.push_back(id);consumed=row+1;stopped=with_eos&&id==2;
          if(!accepted||stopped||expected.size()==quota)break;
        }
        EXPECT_EQ(emitted,expected);EXPECT_EQ(selected.consumed,consumed);EXPECT_EQ(selected.next_id,expected.back());EXPECT_EQ(selected.stopped,stopped);
        EXPECT_EQ(cursor.advanced(2*count+1).draw,draw+2*count+1);
      }
    }
    EXPECT_EQ(covered,count+1)<<"finite route keys missed a requested rejection/bonus witness";
  }
}

TEST(QwenSamplingGraph, MultiRowPrefixRejectsInvalidLaterQAndFinalTarget) {
  PrefixSession session(2);std::vector<float> logits,q;
  const auto row=logs({.25,.25,.5});for(int i=0;i<3;++i)logits.insert(logits.end(),row.begin(),row.end());
  for(int i=0;i<2;++i)q.insert(q.end(),row.begin(),row.end());
  const std::array<std::int32_t,2> ids{0,1};const qwen35::cache::SamplingCursor cursor{123,7,0};
  auto invalid=q;invalid[5]=std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(session.run(logits,invalid,ids,cursor).valid.back(),0U);
  invalid=logits;invalid[8]=std::numeric_limits<float>::infinity();
  EXPECT_EQ(session.run(invalid,q,ids,cursor).valid.back(),0U);
  EXPECT_EQ(session.run(logits,q,ids,cursor).valid,(std::vector<std::uint8_t>{1,1}));
}

TEST(QwenSamplingGraph, MultiRowPrefixRejectsMalformedRowPortsAtColdConstruction) {
  Recipe r;const auto logits=r.input(dense(Type::kFloat32,{3,3})),keys=r.input(dense(Type::kUInt32,{2,4})),ids=r.input(dense(Type::kInt32,{2}));
  EXPECT_THROW((void)s::speculative_prefix(r,logits,keys,ids),std::runtime_error);
  const auto good_keys=r.input(dense(Type::kUInt32,{3,4})),short_ids=r.input(dense(Type::kInt32,{1}));
  EXPECT_THROW((void)s::speculative_prefix(r,logits,good_keys,short_ids),std::runtime_error);
}
#if defined(HYLE_TEST_QWEN_SAMPLING_CUDA)
TEST(QwenSamplingGraph, CategoricalRequiresCommandsAndRejectsRequiredCapture) {
  Session session(3);
  const auto p = logs({.25, .25, .5});
  ASSERT_TRUE(session.run(p, p, p, 0, key(0), key(0, true)).valid);
  const auto capture = session.require_capture();
  EXPECT_FALSE(capture.ok());
  EXPECT_NE(capture.message().find("capture"), std::string_view::npos) << capture.message();
  EXPECT_TRUE(session.run(p, p, p, 0, key(1), key(1, true)).valid);
}
#endif
} // namespace
