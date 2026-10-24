#include "../compute_models/ct_packed_graph_fixture.h"
#include "hyle/compute/apple_memory.h"
#include "hyle/graph/apple.h"
#include "hyle/graph/detail/region_adapter.h"
#include "../../src/graph/execution_source_internal.h"

namespace {
namespace f = ct_packed_graph_test;
namespace a = hyle::compute::apple;

struct Session final {
  a::Context context = f::take(a::Context::create(), "Apple Context");
  a::MemoryResource memory = f::take(a::MemoryResource::create(context), "Apple memory");
  std::array<f::g::StorageDomain, 1> domains{{{memory.ref()}}};
  std::array<f::g::ExecutionTarget, 1> targets{{{
      f::take(f::g::apple::execution_source(context), "Apple execution source"),
      f::g::StorageDomainId{0}}}};
  Session() { ::testing::Test::RecordProperty("device", context.environment()->device_name); }
  hyle::Result<f::g::CompiledGraph> compile(const f::Logical &logical,
                                          f::g::CompileOptions options = {}) {
    if (logical.escape != f::Escape::kNone)
      return f::g::compile(logical.graph, {domains, targets}, options);
    return compile_route(logical, options, 4);
  }
  hyle::Result<f::g::CompiledGraph> compile_original(const f::Logical &logical,
                                                   f::g::CompileOptions options = {}) {
    return compile_route(logical, options, 1);
  }
  hyle::Result<f::g::CompiledGraph> compile_route(const f::Logical &logical,
      f::g::CompileOptions options, std::uint64_t route) {
    auto capability = f::take(f::g::detail::ExecutionSourceAccess::region(targets[0].source),
        "original Apple capability");
    struct Original final {
      const f::g::RegionProviderVTable *native;
      void *context;
      std::span<const f::g::NodeId> nodes;
      std::uint64_t route;
    } original{static_cast<const f::g::RegionProviderVTable *>(capability.native_interface()),
        f::take(f::g::detail::ExecutionSourceAccess::native_context(targets[0].source),
            "original Apple native context"), logical.nodes, route};
    const f::g::RegionProviderVTable original_only{
      +[](void *opaque, const f::g::RegionDiscoveryRequest &, f::m::AllocatorRef,
          f::g::RegionProposalWriter &writer) noexcept -> hyle::Status {
        const auto &choice = *static_cast<const Original *>(opaque);
        HYLE_ASSIGN_OR_RETURN(auto more, writer.try_add({.nodes = choice.nodes,
            .route = choice.route, .replay_class = f::c::ProviderReplayClass::kManaged,
            .implementation = f::c::ProviderImplementationId{1},
            .policy = choice.route == 4 ? f::kPackedPolicy : f::c::RoutePolicyId{},
            .preference = 0}));
        (void)more;
        return hyle::Status::Ok();
      },
      +[](void *opaque, const f::g::RegionCompileRequest &request, f::m::AllocatorRef allocator,
          std::span<f::g::RegionInputRequirement> inputs,
          std::span<f::g::RegionOutputRequirement> outputs,
          f::g::RegionResourceProposal *proposal) noexcept -> hyle::Status {
        const auto &choice = *static_cast<const Original *>(opaque);
        return choice.native->propose_resources(choice.context, request, allocator, inputs, outputs, proposal);
      },
      +[](void *opaque, const f::g::RegionCompileRequest &request, f::m::AllocatorRef allocator,
          f::g::ProviderRegionPrepared *prepared) noexcept -> f::c::ProviderPrepareOutcome {
        const auto &choice = *static_cast<const Original *>(opaque);
        return choice.native->compile(choice.context, request, allocator, prepared);
      }};
    const std::array adapters{f::g::detail::RegionAdapter{.source = std::move(capability),
        .target = f::g::ExecutionTargetId{0}, .interface = &original_only, .context = &original}};
    return f::g::detail::compile_with_adapters(logical.graph, {domains, targets}, options, adapters);
  }
  void bind(const f::Logical &logical, f::Bindings &inputs, f::g::Invocation &slot,
            const f::g::Executable &program) {
    f::bind(logical, inputs, slot, memory.ref(),
        program.output_ownership() == f::g::OutputOwnership::kCaller);
  }
  void submit(f::g::Invocation &slot) {
    const std::span<const f::g::QueueBinding> queues;
    f::check(slot.prepare_replay(queues), "Apple prepare_replay");
    auto work = f::take(slot.submit(queues), "Apple submit");
    f::check(work.wait(), "Apple Completion::wait");
  }
};
void selected(const f::g::CompileReport &report, bool expected) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < report.step_count(); ++i) {
    const auto step = f::take(report.step(i), "Apple compile step");
    if (step.policy != f::kPackedPolicy) continue;
    ++count;
    EXPECT_EQ(step.kind, f::g::CompileStepKind::kProviderRegion);
  }
  EXPECT_EQ(count, expected ? 1U : 0U);
  // MLX-managed storage is opaque; no zero-workspace or dense-byte claim is
  // inferred from an unknown native allocation report.
}

TEST(AppleCtPackedLinear, PublicChainsKeepSourceRoundsAndTypedCastWitnesses) {
  Session session;
  constexpr std::array profiles{
      f::Profile{false, f::Type::kFloat32, f::Type::kFloat32, f::Type::kFloat32, 1},
      f::Profile{true, f::Type::kBFloat16, f::Type::kBFloat16, f::Type::kBFloat16, 3},
      f::Profile{true, f::Type::kFloat16, f::Type::kFloat32, f::Type::kFloat32, 3},
      f::Profile{false, f::Type::kBFloat16, f::Type::kFloat16, f::Type::kFloat32, 3},
      f::Profile{true, f::Type::kFloat32, f::Type::kBFloat16, f::Type::kFloat16, 3, false},
      f::Profile{false, f::Type::kFloat32, f::Type::kFloat16, f::Type::kBFloat16, 3}};
  for (const auto profile : profiles) {
    SCOPED_TRACE(::testing::Message() << profile.asymmetric << '/' << int(profile.scale)
        << '/' << int(profile.decode) << '/' << int(profile.operand) << '/' << profile.cast_before);
    const auto logical = f::build(profile);
    auto ordinary = f::take(f::g::compile(logical.graph, {session.domains, session.targets}),
        "Apple default source-round compile");
    selected(ordinary.report, false);
    auto program = f::take(session.compile(logical), "Apple packed compile");
    selected(program.report, true);
    f::Bindings inputs(logical, session.memory.ref(), 0, true);
    auto slot = f::invocation(program.executable);
    session.bind(logical, inputs, slot, program.executable);
    session.submit(slot);
    f::expect_dot(logical, inputs, f::read(f::output(slot, logical.output)), true);
    auto original = f::take(session.compile_original(logical), "Apple original source-round compile");
    selected(original.report, false);
    auto original_slot = f::invocation(original.executable);
    session.bind(logical, inputs, original_slot, original.executable);
    session.submit(original_slot);
    f::expect_dot(logical, inputs, f::read(f::output(original_slot, logical.output)), true);
  }
  const auto logical = f::build({true, f::Type::kFloat32, f::Type::kFloat32,
                                  f::Type::kFloat32, 1});
  auto program = f::take(session.compile(logical), "Apple nonfinite compile");
  selected(program.report, true);
  f::Bindings special(logical, session.memory.ref(), 0, false, true);
  auto slot = f::invocation(program.executable);
  session.bind(logical, special, slot, program.executable);
  session.submit(slot);
  f::expect_dot(logical, special, f::read(f::output(slot, logical.output)));
}

TEST(AppleCtPackedLinear, DynamicInputsRebindAcrossOwnedSlotsAndRetainResults) {
  Session session;
  const auto logical = f::build({true, f::Type::kBFloat16, f::Type::kFloat16,
                                  f::Type::kFloat32, 3});
  auto program = f::take(session.compile(logical), "Apple dynamic compile");
  selected(program.report, true);
  std::array slots{f::invocation(program.executable), f::invocation(program.executable)};
  std::array<std::vector<double>, 2> previous;
  f::g::GraphOutput held, exported;
  for (unsigned round = 0; round < 2; ++round) {
    for (const unsigned lane : {1U, 0U}) {
      if (round) {
        if (lane == 0) held.reset();
        f::check(slots[lane].reset(), "Apple dynamic reset");
      }
      f::Bindings inputs(logical, session.memory.ref(), round * 2 + lane);
      session.bind(logical, inputs, slots[lane], program.executable);
      inputs.inputs.clear(); // Native slots, rather than the caller, retain the original bytes.
      session.submit(slots[lane]);
      auto result = f::take(slots[lane].result(logical.output), "Apple result");
      const auto actual = f::read(f::take(result.tensor(), "Apple result tensor"));
      f::expect_dot(logical, inputs, actual);
      EXPECT_TRUE(slots[lane].reset().IsUnavailable());
      if (!round) previous[lane] = actual;
      else EXPECT_NE(previous[lane], actual);
      if (!round && lane == 0) held = std::move(result);
      if (round && lane == 1)
        f::exact_values(f::read(f::take(held.tensor(), "Apple other-slot old result")), previous[0]);
      if (round && lane == 0) exported = std::move(result);
    }
  }
  const auto values = f::read(f::take(exported.tensor(), "Apple retained before close"));
  held.reset(); slots = {}; program.executable = {};
  f::exact_values(f::read(f::take(exported.tensor(), "Apple retained after invocation close")), values);
  ::testing::Test::RecordProperty("actual_submissions", 4);
}

TEST(AppleCtPackedLinear, PublishedViewAndControlEscapesExecuteOriginalDecomposition) {
  Session session;
  for (auto escape : {f::Escape::kPublishedDecode, f::Escape::kExtraView, f::Escape::kControl}) {
    const auto logical = f::build({true, f::Type::kBFloat16, f::Type::kFloat32,
                                    f::Type::kFloat32, 3}, 9, 64, escape);
    auto program = f::take(session.compile(logical), "Apple escaped compile");
    selected(program.report, false);
    f::Bindings inputs(logical, session.memory.ref(), 2);
    auto slot = f::invocation(program.executable);
    session.bind(logical, inputs, slot, program.executable);
    session.submit(slot);
    f::expect_dot(logical, inputs, f::read(f::output(slot, logical.output)));
    f::exact_values(f::read(f::output(slot, logical.extra)), escape == f::Escape::kControl
        ? inputs.activation : inputs.fixture.decoded(logical.profile.scale,
            logical.profile.decode, logical.profile.decode));
  }
}

TEST(AppleCtPackedLinear, OriginalPackedConstantsKeepTheirExplicitCallerLifetime) {
  Session session;
  const auto logical = f::build({true, f::Type::kBFloat16, f::Type::kFloat32,
                                  f::Type::kFloat32, 3}, 257, 1024, f::Escape::kNone, true);
  f::Bindings inputs(logical, session.memory.ref(), 1);
  std::vector<f::g::ConstantBinding> constants;
  for (std::size_t i = 1; i < logical.ports.size(); ++i)
    constants.push_back({logical.ports[i], f::take(inputs.inputs[i].tensor(logical.descriptions[i]),
        "Apple original packed constant"), f::g::ConstantPolicy::kBorrow, {}, {}});
  auto program = f::take(session.compile(logical, {.constants = constants}), "Apple constant compile");
  selected(program.report, true);
  auto slot = f::invocation(program.executable);
  session.bind(logical, inputs, slot, program.executable);
  // Native Apple resources have no generic Compute StorageOwner. This is the
  // public raw-borrow contract: the original ready leases and real resource
  // remain caller-owned for the complete compiled-program lifetime.
  const auto retained_banks = inputs.inputs;
  constants.clear(); inputs.inputs.clear();
  session.submit(slot);
  f::expect_dot(logical, inputs, f::read(f::output(slot, logical.output)));
  EXPECT_EQ(retained_banks.size(), logical.ports.size());
  ::testing::Test::RecordProperty("logical_dense_weight_elements", 257 * 1024);
}
} // namespace
