#include "../compute_models/gptq_reference_fixture.h"
#include "hyle/compute/apple_memory.h"
#include "hyle/graph/apple.h"
#include "hyle/graph/detail/region_adapter.h"
#include "../../src/graph/execution_source_internal.h"

namespace {
namespace f = gptq_reference_test;
namespace a = hyle::compute::apple;

template<class Run> void run_apple(Run run, bool packed) {
  auto context = f::take(a::Context::create(), "Apple context");
  ::testing::Test::RecordProperty("device", context.environment()->device_name);
  auto memory = f::take(a::MemoryResource::create(context), "Apple shared memory");
  const std::array domains{f::g::StorageDomain{memory.ref()}};
  const std::array targets{f::g::ExecutionTarget{
      f::take(f::g::apple::execution_source(context), "Apple execution source"),
      f::g::StorageDomainId{0}}};
  run(memory.ref(), std::span<const f::g::QueueBinding>{},
      [&](const f::g::Graph &graph, f::g::CompileOptions options) {
        if (!packed) return f::g::compile(graph, {domains,targets}, options);
        // Exercise the available bounded route without changing the measured
        // default preference. Its original compiler still proves closure.
        auto capability = f::take(
            f::g::detail::ExecutionSourceAccess::region(targets[0].source),
            "original Apple capability");
        std::vector<f::g::NodeId> nodes;
        for (std::size_t i = 0; i < graph.node_count(); ++i)
          nodes.push_back(f::take(graph.node_id(i), "GPTQ recipe node"));
        struct Choice {
          const f::g::RegionProviderVTable *native;
          void *context;
          std::span<const f::g::NodeId> nodes;
        } choice{static_cast<const f::g::RegionProviderVTable *>(
                     capability.native_interface()),
            f::take(f::g::detail::ExecutionSourceAccess::native_context(targets[0].source),
                "original Apple native context"), nodes};
        const f::g::RegionProviderVTable selected{
          +[](void *opaque, const f::g::RegionDiscoveryRequest &, f::m::AllocatorRef,
              f::g::RegionProposalWriter &writer) noexcept -> hyle::Status {
            const auto &source = *static_cast<const Choice *>(opaque);
            HYLE_ASSIGN_OR_RETURN(auto more, writer.try_add({
                .nodes = source.nodes, .route = 4,
                .replay_class = f::c::ProviderReplayClass::kManaged,
                .implementation = f::c::ProviderImplementationId{1},
                .policy = f::kPackedPolicy, .preference = 0}));
            (void)more;
            return hyle::Status::Ok();
          },
          +[](void *opaque, const f::g::RegionCompileRequest &request,
              f::m::AllocatorRef allocator,
              std::span<f::g::RegionInputRequirement> inputs,
              std::span<f::g::RegionOutputRequirement> outputs,
              f::g::RegionResourceProposal *proposal) noexcept -> hyle::Status {
            const auto &source = *static_cast<const Choice *>(opaque);
            return source.native->propose_resources(
                source.context, request, allocator, inputs, outputs, proposal);
          },
          +[](void *opaque, const f::g::RegionCompileRequest &request,
              f::m::AllocatorRef allocator, f::g::ProviderRegionPrepared *prepared)
              noexcept -> f::c::ProviderPrepareOutcome {
            const auto &source = *static_cast<const Choice *>(opaque);
            return source.native->compile(source.context, request, allocator, prepared);
          }};
        const std::array adapters{f::g::detail::RegionAdapter{
            .source = std::move(capability), .target = f::g::ExecutionTargetId{0},
            .interface = &selected, .context = &choice}};
        return f::g::detail::compile_with_adapters(
            graph, {domains,targets}, options, adapters);
      },
      [](f::m::BufferView destination, std::span<const std::byte> source) {
        return destination.write(source);
      },
      [](f::m::ConstBufferView source, std::span<std::byte> destination) {
        return source.read(destination);
      },
      [](f::g::Invocation &invocation,
         std::span<const f::g::QueueBinding> bindings) {
        return invocation.prepare_replay(bindings);
      });
}
} // namespace

TEST(AppleGptqReferenceGraph, OriginalPackedBanksKeepSourceRoundsSlotsAndIndexUpdates) {
  run_apple([](auto resource, auto queues, auto compile, auto write, auto read, auto prepare) {
    f::run_lifetimes(resource, queues, compile, write, read, prepare);
  }, true);
}
TEST(AppleGptqReferenceGraph, CompleteIndexValidationPrecedesSelectedScaleAndZeroAccess) {
  run_apple([](auto resource, auto queues, auto compile, auto write, auto read, auto prepare) {
    f::run_runtime_domain(resource, queues, compile, write, read, prepare);
  }, true);
}
TEST(AppleGptqReferenceGraph, PublishedDecodeRetainsOriginalF16ProductAndCheckedFailure) {
  run_apple([](auto resource, auto queues, auto compile, auto write, auto read, auto prepare) {
    f::run_escape(resource, queues, compile, write, read, prepare);
  }, false);
}
