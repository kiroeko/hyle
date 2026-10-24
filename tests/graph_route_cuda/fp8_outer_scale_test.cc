#include "../compute_models/fp8_outer_scale_fixture.h"
#include "../../src/graph/static_fp8_region.h"
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#include "hyle/compute/providers/torch_cuda.h"

#include <sstream>

namespace {
void record_default_report(const hyle::graph::CompileReport &report, std::size_t ordinal) {
  namespace f = static_fp8_reference_test;
  const auto prefix = "source_role_default_" + std::to_string(ordinal);
  std::ostringstream steps;
  for (std::size_t i = 0; i < report.step_count(); ++i) {
    const auto step = f::take(report.step(i), "default role step");
    steps << "step=" << i << ",begin=" << step.node_begin << ",count=" << step.node_count
          << ",contiguous=" << step.nodes_contiguous << ",nodes=";
    for (std::size_t n = 0; n < step.node_count; ++n)
      steps << f::take(report.step_node(i,n), "default role step node").ordinal() << ':';
    steps << ",provider=" << step.provider.name << ",provider_id="
          << step.provider.id.namespace_id << ':' << step.provider.id.provider_id
          << ",policy=" << step.policy.namespace_id << ':' << step.policy.policy_id
          << ':' << step.policy.version << ",workspace=" << step.workspace.size
          << ",persistent=";
    if (step.persistent_bytes) steps << *step.persistent_bytes;
    else steps << "unknown";
    steps << ",packed=" << step.packed_constant_bytes << ",copy=" << step.copy_bytes << ';';
  }
  ::testing::Test::RecordProperty(prefix+"_steps",steps.str());
  std::ostringstream physical;
  for (std::size_t i = 0; i < report.physical_value_count(); ++i) {
    const auto value = f::take(report.physical_value(i), "default role physical view");
    physical << "physical=" << i << ",value=" << value.value.ordinal()
             << ",span=" << f::take(value.description.required_bytes(), "default role view span")
             << ",alias=";
    if (value.alias_source) physical << *value.alias_source;
    else physical << "none";
    physical << ",constant=" << value.constant << ",workspace=" << value.workspace_backed << ';';
  }
  ::testing::Test::RecordProperty(prefix+"_physical_view_spans",physical.str());
  const auto storage=report.compiled_storage();
  std::ostringstream retained;
  retained << "materialized=" << storage.materialized_constant_bytes
           << ",persistent_known=" << storage.provider_persistent_bytes
           << ",packed=" << storage.packed_constant_bytes
           << ",unknown_providers=" << storage.unknown_provider_count
           << ",invocation_workspace=" << report.workspace_bytes();
  ::testing::Test::RecordProperty(prefix+"_storage",retained.str());
  ::testing::Test::RecordProperty(prefix+"_scope",
      "compile-only default; identical constants and budgets; released before constrained compile; view spans include aliases and are not allocation or traffic totals");
}
template<class Run> void run_cuda(Run run, bool contracted, bool pattern_only=false,bool default_probe=false) {
  namespace f = static_fp8_reference_test;
  auto context = f::take(f::g::cuda::create_context(), "CUDA context");
  auto memory = f::take(f::c::cuda::MemoryResource::create(context,
      f::c::cuda::MemoryKind::kDevice), "CUDA memory");
  const auto owner = f::take(f::c::StorageOwner::adopt(context, std::move(memory)),
      "CUDA storage owner");
  auto queue = f::take(context.create_queue(), "CUDA queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0}, &queue}};
  std::size_t default_ordinal=0;
  run(owner.ref(), queues,
      [&](const f::g::Graph &graph, f::g::CompileOptions options) {
        if(default_probe) {
          // Same constants/budgets, no provider constraints; no Invocation or submit.
          const auto observed=f::g::cuda::compile(graph,context,owner,options);
          if(observed) record_default_report(observed->report,default_ordinal);
          else ::testing::Test::RecordProperty("source_role_default_"+std::to_string(default_ordinal)+"_error",
              std::string(observed.error().status().message()));
          ++default_ordinal;
        } // Release the probe artifact before the actual selected compile.
        const std::array providers{f::c::torch_cuda::kProviderId};
        if (contracted) options.allowed_providers = providers;
        std::vector<f::g::NodeProviderConstraint> required;
        if (pattern_only) {
          using Rule = f::g::detail::StaticFloat8Pattern;
          std::size_t matches = 0;
          for (std::size_t i = 0; i < graph.node_count(); ++i) {
            Rule::Pattern pattern;
            if (!Rule::pattern(graph, f::take(graph.node_id(i), "static FP8 terminal"), pattern)) continue;
            ++matches;
            for (const auto node : pattern.span())
              required.push_back({node, f::c::torch_cuda::kProviderId, {}});
          }
          EXPECT_EQ(matches, std::size_t{1});
          EXPECT_EQ(required.size(), Rule::kMaxNodes);
          options.node_providers = required;
        }
        return f::g::cuda::compile(graph, context, owner, options);
      },
      [](f::m::BufferView destination, std::span<const std::byte> source) {
        return destination.write(source);
      },
      [](f::m::ConstBufferView source, std::span<std::byte> destination) {
        return source.read(destination);
      },
      [](f::g::Invocation &invocation, std::span<const f::g::QueueBinding> bindings) {
        const auto status = f::g::cuda::prepare_replay(invocation, *bindings[0].queue,
            {.mode = f::g::ReplayMode::kRequireCapture});
        if (status.ok()) EXPECT_EQ(invocation.replay_route(), f::g::ReplayRoute::kCaptured);
        return status;
      }, owner);
}
} // namespace

TEST(CudaFp8OuterScale, RowAndChannelScalesKeepWeightThenActivationOrder) {
  run_cuda([](auto r,auto q,auto compile,auto write,auto read,auto prepare,auto owner) {
    fp8_outer_scale_test::run_lifetimes(r,q,compile,write,read,prepare,owner);
  }, true);
}
TEST(CudaFp8OuterScale, DynamicRowPrologueKeepsActivationThenWeightOrder) {
  run_cuda([](auto r,auto q,auto compile,auto write,auto read,auto prepare,auto owner) {
    fp8_outer_scale_test::run_prologue(r,q,compile,write,read,prepare,owner);
  }, false, true);
}
TEST(CudaFp8OuterScale, PublishedCodesKeepTheOrdinaryOuterScaleGraph) {
  run_cuda([](auto r,auto q,auto compile,auto write,auto read,auto prepare,auto owner) {
    fp8_outer_scale_test::run_escape(r,q,compile,write,read,prepare,owner);
  }, false);
}
TEST(CudaFp8OuterScale, OriginalScalarLifetimeNeighbor) {
  run_cuda([](auto r,auto q,auto compile,auto write,auto read,auto prepare,auto owner) {
    static_fp8_reference_test::run_lifetimes(r,q,compile,write,read,prepare,owner);
  }, true);
}
TEST(CudaFp8OuterScale, OriginalScalarIeeeNeighbor) {
  run_cuda([](auto r,auto q,auto compile,auto write,auto read,auto prepare,auto) {
    static_fp8_reference_test::run_runtime_domain(r,q,compile,write,read,prepare);
  }, true);
}
TEST(CudaFp8OuterScale, OriginalScalarPublishedCodesNeighbor) {
  run_cuda([](auto r,auto q,auto compile,auto write,auto read,auto prepare,auto) {
    static_fp8_reference_test::run_escape(r,q,compile,write,read,prepare);
  }, false);
}

TEST(CudaFp8OuterScale, SourceBoundFaProjectionKeepsStoredCodesAndOuterScales) {
  run_cuda([](auto r,auto q,auto compile,auto write,auto read,auto prepare,auto owner) {
    fp8_outer_scale_test::run_source_role(r,q,compile,write,read,prepare,owner);
  }, false, true, true);
}
