#include "../compute_models/fp8_outer_scale_fixture.h"
#include "../../src/graph/static_fp8_region.h"
#include "../../src/compute_packed_shared/static_fp8_linear.h"
#include "hyle/graph/cpu.h"
#include "hyle/compute/providers/native.h"

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
template<class Run> void run_cpu(Run run,bool contracted,bool pattern_only=false,bool default_probe=false) {
  namespace f=static_fp8_reference_test;
  auto context=f::take(f::g::cpu::create_context(),"CPU static FP8 context");
  auto queue=f::take(context.create_queue(),"CPU queue");
  const std::array queues{f::g::QueueBinding{f::g::ExecutionLaneId{0},&queue}};
  std::size_t default_ordinal=0;
  run(f::m::system_resource(),queues,
      [&](const f::g::Graph &graph,f::g::CompileOptions options) {
        if(default_probe) {
          // Same constants/budgets, no provider constraints; no Invocation or submit.
          const auto observed=f::g::cpu::compile(graph,context,options);
          if(observed) record_default_report(observed->report,default_ordinal);
          else ::testing::Test::RecordProperty("source_role_default_"+std::to_string(default_ordinal)+"_error",
              std::string(observed.error().status().message()));
          ++default_ordinal;
        } // Release the probe artifact before the actual selected compile.
        std::vector<f::g::NodeProviderConstraint> required;
        if(contracted)for(std::size_t i=0;i<graph.node_count();++i)
          required.push_back({f::take(graph.node_id(i),"constrained original node"),f::c::native::kProviderId,{}});
        if(pattern_only) {
          using Rule=f::g::detail::StaticFloat8Pattern;
          std::size_t matches=0;
          for(std::size_t i=0;i<graph.node_count();++i) {
            Rule::Pattern pattern;
            if(!Rule::pattern(graph,f::take(graph.node_id(i),"static FP8 terminal"),pattern))continue;
            ++matches;
            for(const auto node:pattern.span())
              required.push_back({node,f::c::native::kProviderId,{}});
          }
          EXPECT_EQ(matches,std::size_t{1});
          EXPECT_EQ(required.size(),Rule::kMaxNodes);
        }
        options.node_providers=required;
        options.provider_replay=contracted?f::g::ProviderReplayPolicy::kRequirePrepared:f::g::ProviderReplayPolicy::kAllowAny;
        return f::g::cpu::compile(graph,context,options);
      },[](f::m::BufferView d,std::span<const std::byte> s) {return d.write(s);},
      [](f::m::ConstBufferView s,std::span<std::byte> d) {return s.read(d);},
      [](f::g::Invocation &i,std::span<const f::g::QueueBinding> q) {return i.prepare_replay(q);});
}
}
TEST(CpuFp8OuterScale, RowAndChannelScalesKeepWeightThenActivationOrder) {
  run_cpu([](auto r,auto q,auto compile,auto write,auto read,auto prepare) {
    fp8_outer_scale_test::run_lifetimes(r,q,compile,write,read,prepare);
  }, true);
}
TEST(CpuFp8OuterScale, DynamicRowPrologueKeepsActivationThenWeightOrder) {
  run_cpu([](auto r,auto q,auto compile,auto write,auto read,auto prepare) {
    fp8_outer_scale_test::run_prologue(r,q,compile,write,read,prepare);
  }, false, true);
}
TEST(CpuFp8OuterScale, PublishedCodesKeepTheOrdinaryOuterScaleGraph) {
  run_cpu([](auto r,auto q,auto compile,auto write,auto read,auto prepare) {
    fp8_outer_scale_test::run_escape(r,q,compile,write,read,prepare);
  }, false);
}
TEST(CpuFp8OuterScale, OriginalScalarLifetimeNeighbor) {
  run_cpu([](auto r,auto q,auto compile,auto write,auto read,auto prepare) {
    static_fp8_reference_test::run_lifetimes(r,q,compile,write,read,prepare);
  }, true);
}
TEST(CpuFp8OuterScale, OriginalScalarIeeeNeighbor) {
  run_cpu([](auto r,auto q,auto compile,auto write,auto read,auto prepare) {
    static_fp8_reference_test::run_runtime_domain(r,q,compile,write,read,prepare);
  }, true);
}
TEST(CpuFp8OuterScale, OriginalScalarPublishedCodesNeighbor) {
  run_cpu([](auto r,auto q,auto compile,auto write,auto read,auto prepare) {
    static_fp8_reference_test::run_escape(r,q,compile,write,read,prepare);
  }, false);
}

TEST(CpuFp8OuterScale, SourceBoundFaProjectionKeepsStoredCodesAndOuterScales) {
  run_cpu([](auto r,auto q,auto compile,auto write,auto read,auto prepare) {
    fp8_outer_scale_test::run_source_role(r,q,compile,write,read,prepare);
  }, false, true, true);
}
