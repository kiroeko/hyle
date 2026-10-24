#include "../compute_models/static_fp8_reference_fixture.h"
#include "hyle/compute/apple_memory.h"
#include "hyle/graph/apple.h"
#include "hyle/graph/detail/region_adapter.h"
#include "../../src/graph/execution_source_internal.h"

namespace {
namespace f = static_fp8_reference_test;
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
          nodes.push_back(f::take(graph.node_id(i), "static FP8 recipe node"));
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
                .nodes = source.nodes, .route = 5,
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

TEST(AppleStaticFloat8ReferenceGraph, OriginalCodesAndSeparateStoredScalesSurviveSlotsAndUpdates) {
  run_apple([](auto resource, auto queues, auto compile, auto write, auto read, auto prepare) {
    f::run_lifetimes(resource, queues, compile, write, read, prepare);
  }, true);
}
TEST(AppleStaticFloat8ReferenceGraph, IeeeScalarsAndCastBoundariesKeepTheCodeDomainTree) {
  run_apple([](auto resource, auto queues, auto compile, auto write, auto read, auto prepare) {
    f::run_runtime_domain(resource, queues, compile, write, read, prepare);
  }, true);
}
TEST(AppleStaticFloat8ReferenceGraph, PublishedCodeOrWidenedWeightKeepsTheOrdinaryGraph) {
  run_apple([](auto resource, auto queues, auto compile, auto write, auto read, auto prepare) {
    f::run_escape(resource, queues, compile, write, read, prepare);
  }, false);
}

TEST(AppleFloatingStorage, ScalarMultiplyAndCastPreserveSubnormalStorageBits) {
  run_apple([](auto resource, auto queues, auto compile, auto write, auto read, auto prepare) {
    const auto input = f::dense(f::Type::kFloat32,{2});
    const auto scale = f::dense(f::Type::kFloat32,{});
    auto builder = f::take(f::g::GraphBuilder::create(),"IEEE scalar graph");
    const auto x = f::take(builder.input(input),"IEEE input");
    const auto s = f::take(builder.input(scale),"IEEE scale");
    const auto mul = f::take(builder.add(f::c::Binary{f::c::BinaryKind::kMultiply},x,s),"scalar multiply");
    const auto cast = f::take(builder.add(f::c::Cast{f::Type::kBFloat16},mul),"product cast");
    const auto direct = f::take(builder.add(f::c::Cast{f::Type::kBFloat16},s),"direct scalar cast");
    for (auto port : {mul,cast,direct}) f::check(builder.publish(port),"publish exact boundary");
    auto graph = f::take(std::move(builder).finish(),"finish IEEE scalar graph");
    auto compiled = f::take(compile(graph,f::g::CompileOptions{}),"ordinary IEEE compile");
    auto run = f::take(compiled.executable.create_invocation({.resources=f::g::ResourceOptions{}}),"IEEE invocation");
    ASSERT_EQ(f::take(compiled.executable.output_ownership(),"IEEE ownership"),f::g::OutputOwnership::kInvocation);
    auto xs = f::take(f::m::Buffer::create(resource,8,256),"IEEE input storage");
    auto ss = f::take(f::m::Buffer::create(resource,4,256),"IEEE scale storage");
    const std::array<std::uint32_t,2> values{0x3f800000u,0x40000000u};
    const std::array<std::uint32_t,1> scales{0x00080000u};
    f::check(write(xs.view(),std::as_bytes(std::span(values))),"write IEEE values");
    f::check(write(ss.view(),std::as_bytes(std::span(scales))),"write IEEE scale");
    f::check(run.bind(x,f::take(f::c::ConstTensorView::create(input,xs.view()),"IEEE typed input")),"bind IEEE input");
    f::check(run.bind(s,f::take(f::c::ConstTensorView::create(scale,ss.view()),"IEEE typed scale")),"bind IEEE scale");
    f::check(prepare(run,queues),"prepare IEEE scalar graph");
    auto done = f::take(run.submit(queues),"IEEE scalar submit");
    f::check(done.wait(),"IEEE scalar completion");
    const auto load = [&](auto port,auto &destination) {
      auto result=f::take(run.result(port),"IEEE boundary result");
      auto tensor=f::take(result.tensor(),"IEEE boundary storage");
      f::check(read(tensor.storage(),std::as_writable_bytes(std::span(destination))),"read boundary bits");
    };
    std::array<std::uint32_t,2> multiplied{};
    std::array<std::uint16_t,2> narrowed{};
    std::array<std::uint16_t,1> direct_cast{};
    load(mul,multiplied);load(cast,narrowed);load(direct,direct_cast);
    ::testing::Test::RecordProperty("mul0_f32_bits",std::to_string(multiplied[0]));
    ::testing::Test::RecordProperty("mul1_f32_bits",std::to_string(multiplied[1]));
    ::testing::Test::RecordProperty("cast_product0_bf16_bits",std::to_string(narrowed[0]));
    ::testing::Test::RecordProperty("cast_direct_bf16_bits",std::to_string(direct_cast[0]));
    EXPECT_EQ(direct_cast[0],0x0008u);
    EXPECT_EQ(multiplied,(std::array<std::uint32_t,2>{0x00080000u,0x00100000u}));
    EXPECT_EQ(narrowed,(std::array<std::uint16_t,2>{0x0008u,0x0010u}));
  },false);
}
