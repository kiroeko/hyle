#include "../compute_models/ct_packed_graph_fixture.h"
#include "hyle/compute/providers/native.h"
#include "hyle/graph/cpu.h"

namespace {
namespace f = ct_packed_graph_test;
// The actual process-lifetime system resource needs no separate native owner.
// StorageOwner still retains this original CPU Context and allocator contract.
struct SystemResource final {
  f::m::MemoryResourceRef ref() noexcept { return f::m::system_resource(); }
};
struct Backend final {
  static constexpr bool supports_capture = false;
  static constexpr auto provider_id = f::c::native::kProviderId;
  static f::c::Context create_context() {
    return f::take(f::g::cpu::create_context(), "CPU Context");
  }
  static f::c::StorageOwner create_owner(const f::c::Context &context) {
    return f::take(f::c::StorageOwner::adopt(context, SystemResource{}), "CPU StorageOwner");
  }
  static hyle::Result<f::g::CompiledGraph> compile(const f::g::Graph &graph,
      const f::c::Context &context, const f::c::StorageOwner &owner,
      f::g::CompileOptions options = {}) noexcept {
    options.provider_replay = f::g::ProviderReplayPolicy::kRequirePrepared;
    return compile_unrestricted(graph, context, owner, options);
  }
  static hyle::Result<f::g::CompiledGraph> compile_unrestricted(const f::g::Graph &graph,
      const f::c::Context &context, const f::c::StorageOwner &,
      f::g::CompileOptions options = {}) noexcept {
    return f::g::cpu::compile(graph, context, options);
  }
  static hyle::Status prepare(f::g::Invocation &slot, f::c::Queue &queue,
      bool capture) noexcept {
    if (capture) return hyle::Status::InvalidArgument("CPU fixture has no capture backend");
    const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0}, &queue}};
    return slot.prepare_replay(queues,
        {.mode = f::g::ReplayMode::kCommands, .host_wait = f::g::HostWaitPolicy::kAllow});
  }
};

TEST(CpuCtPackedLinear, PublicChainsKeepSourceRoundsAndTypedCastWitnesses) {
  f::PublicChainsKeepSourceRoundsAndTypedCastWitnesses<Backend>();
}

TEST(CpuCtPackedLinear, CapturedDynamicInputsRebindAcrossOwnedSlotsAndRetainResults) {
  f::CapturedDynamicInputsRebindAcrossOwnedSlotsAndRetainResults<Backend>();
}

TEST(CpuCtPackedLinear, PublishedViewAndControlEscapesExecuteTheOriginalDecomposition) {
  f::PublishedViewAndControlEscapesExecuteTheOriginalDecomposition<Backend>();
}

TEST(CpuCtPackedLinear, OriginalPackedConstantsDoNotRetainOrWorkspaceADenseWeightBank) {
  f::OriginalPackedConstantsDoNotRetainOrWorkspaceADenseWeightBank<Backend>();
}

TEST(CpuCtPackedLinear, DefaultMultirowKeepsOriginalRouteAndPreparedCanSelectPacked) {
  auto context = Backend::create_context();
  auto owner = Backend::create_owner(context);
  auto queue = context.create_queue().value();
  const auto graph = f::build({true, f::Type::kBFloat16, f::Type::kFloat16,
                              f::Type::kFloat32, 3});
  auto ordinary = Backend::compile_unrestricted(graph.graph, context, owner);
  ASSERT_TRUE(ordinary) << ordinary.error().message();
  f::selected<Backend>(graph, ordinary->report, false);
  auto prepared = Backend::compile(graph.graph, context, owner);
  ASSERT_TRUE(prepared) << prepared.error().message();
  f::selected<Backend>(graph, prepared->report, true);
  // The same small public graph, original quantized bytes and independent
  // source-round oracle exercise both routes without a large matrix payload.
  f::Bindings inputs(graph, owner, 3, true);
  for (const auto *program : {&ordinary->executable, &prepared->executable}) {
    auto slot = f::invocation(*program);
    f::bind(graph, inputs, slot, owner);
    f::submit<Backend>(slot, queue, false);
    f::expect_dot(graph, inputs, f::read(f::output(slot, graph.output)), true);
  }
  ::testing::Test::RecordProperty("default_packed_regions", "0");
  ::testing::Test::RecordProperty("prepared_packed_regions", "1");
}
} // namespace
