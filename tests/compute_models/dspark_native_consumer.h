#pragma once

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "model_graph_run.h"
#include "model_resident_fragment.h"
#include "hyle/graph/weight_binding.h"
#if defined(HYLE_TEST_DSPARK_STATE_CPU)
#include "hyle/graph/cpu.h"
#endif
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
#include "hyle/compute/backends/cuda.h"
#include "hyle/graph/cuda.h"
#endif

// Ordinary native execution shared by the existing compressed sidecar and
// formal original consumers. Model construction, source acquisition, state
// policy and numerical reference remain in their caller.
namespace dspark_native_consumer {
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace m=hyle::mem;
namespace sem=d5_qwen38_24t_semantic_fragments;
using Bytes=std::vector<std::byte>;
template<class T> T take(hyle::Result<T> value,const char *where) {
  return model_graph_run::detail::take<T>(std::move(value),where);
}
inline void check(hyle::Status status,const char *where) {
  model_graph_run::detail::check(std::move(status),where);
}
template<class T> Bytes bytes(std::span<const T> value) {
  auto raw=std::as_bytes(value);return {raw.begin(),raw.end()};
}
inline Bytes read(const c::ReadLease &value) {
  Bytes result(value.size());check(take(value.view(),"readonly bytes").read(result),"read result bytes");return result;
}
template<class T> std::vector<T> values(const c::ReadLease &value) {
  const auto raw=read(value);if(raw.size()%sizeof(T)) throw std::runtime_error("typed result byte extent");
  std::vector<T> result(raw.size()/sizeof(T));std::memcpy(result.data(),raw.data(),raw.size());return result;
}
inline c::ReadLease upload(const c::StorageOwner &owner,std::span<const std::byte> source) {
  auto backing=take(m::Buffer::create(owner.ref(),source.size(),256),"explicit input backing");
  check(backing.view().write(source),"explicit one-time input upload");
  return take(owner.adopt_buffer(std::move(backing)),"ready input permission");
}
inline c::TensorDesc dense(c::DataType type,std::initializer_list<std::int64_t> dims) {
  return take(c::TensorDesc::dense(type,{dims.begin(),dims.size()}),"dense description");
}

// One ordinary public Graph lifecycle for the complete proposal and all model
// components. No additional executor, host tensor bridge or provider switch.
struct Run final {
  g::Graph graph;
  std::vector<c::TensorDesc> inputs,outputs;
  std::unique_ptr<model_graph_run::ColdProgram> program;
  g::Invocation invocation;
  c::LeasePool pool;
  hyle::inference::RetirementObservation last_failure_retirement{};
};
struct Backend final {
  c::Context &context;
  const c::StorageOwner &owner;
  c::Queue &queue;
  bool cuda;
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
  const c::cuda::MemoryResource *resident_device=nullptr;
#endif
  int submissions=0;
  std::optional<c::WorkloadContext> default_workload;
  hyle::Result<g::CompiledGraph> compile(const g::Graph &graph,g::CompileOptions options) {
#if defined(HYLE_TEST_DSPARK_STATE_CPU)
    if(!cuda) return g::cpu::compile(graph,context,options);
#endif
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
    if(cuda) return g::cuda::compile(graph,context,owner,options);
#endif
    return std::unexpected(hyle::Status::Unimplemented("DSpark state test backend"));
  }
  Run prepare(g::Graph graph,std::span<const g::ConstantBinding> constants={},std::uint32_t generations=4,
      std::optional<c::WorkloadContext> actual_workload=std::nullopt) {
    Run result{.graph=std::move(graph)};
    std::vector<model_graph_run::Port> input_ports,output_ports;
    for(std::size_t i=0;i<result.graph.input_count();++i) {
      result.inputs.push_back(take(result.graph.input_description(i),"input description"));
      input_ports.push_back({take(result.graph.input_id(i),"input ID"),result.inputs.back()});
    }
    std::size_t capacity=0;
    for(std::size_t i=0;i<result.graph.output_count();++i) {
      result.outputs.push_back(take(result.graph.output_description(i),"output description"));
      capacity+=(take(result.outputs.back().required_bytes(),"output bytes")+255u)/256u*256u;
    }
    const auto workload=actual_workload?actual_workload:default_workload;
    if(!workload)throw std::runtime_error("ordinary model workload was not selected by its caller");
    auto compiled=take(compile(result.graph,{.constants=constants,
        .workload=*workload}),"ordinary model compile");
    for(std::size_t i=0;i<result.outputs.size();++i)
      output_ports.push_back({take(result.graph.output_id(i),"output ID"),result.outputs[i]});
    result.program=model_graph_run::ColdProgram::create(std::move(compiled),input_ports,output_ports);
    result.invocation=take(result.program->compiled.executable.create_invocation({.resources=g::ResourceOptions{}}),"ordinary model Invocation");
    result.pool=take(c::LeasePool::create(owner.ref(),capacity*generations,
        {.max_blocks=std::uint32_t(result.outputs.size())*generations,.backing_alignment=256}),"independent result generations");
    return result;
  }
  std::vector<c::ReadLease> execute(Run &run,std::span<const c::ReadLease> inputs,
      const model_graph_run::InputPlan *input_plan=nullptr,
      const std::function<void(g::Invocation&)> &bind_external={}) {
    return model_graph_run::execute(*run.program,run.invocation,run.pool,queue,inputs,
        submissions,input_plan,bind_external,&run.last_failure_retirement);
  }
};

template<class Source,class CpuSupply,class Boundary,class Selection
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
    ,class CudaSupply
#endif
    > struct RoutedRun final {
  // Supplies and their native resources outlive every phase Invocation,
  // including exceptional unwind after a partial prepare/bind.
  std::shared_ptr<const Source> source;
  std::shared_ptr<CpuSupply> cpu_supply;
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
  std::shared_ptr<CudaSupply> cuda_supply;
#endif
  std::vector<Run> phases;
  std::optional<model_resident_fragment::ResidentDraft<Boundary,3>> plan;
  std::vector<sem::ValueId> full_inputs;
  std::vector<model_resident_fragment::Location> outputs;
  std::vector<sem::ValueId> observed_sources;
  std::size_t value_count=0;
  std::array<Selection,3> selections;
  std::function<Selection(std::span<const std::int32_t>)> select;
  std::array<std::vector<std::size_t>,3> union_sizes;
  std::size_t completed_phases=0,completed_episodes=0;
  std::size_t semantic_input_count=6,router_value_count=30,resident_capacity=30,
      resident_slot_bytes=0;
  std::array<std::uint64_t,4> bank_ids{10,20,30,40};
  struct ResidentInputs final {
    std::optional<model_graph_run::InputPlan> plan;
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
    std::array<g::WeightBinding,4> bindings{};
#endif
  };
  std::vector<ResidentInputs> resident_inputs;
  std::function<void(std::size_t,std::span<const std::int32_t>)> before_upload;
  void prepare_resident_inputs() {
    resident_inputs.resize(phases.size());
    if(!plan)return;
    if(phases.size()!=plan->phases.size())throw std::runtime_error("resident phase construction count");
    for(std::size_t index=0;index<phases.size();++index) {
      const auto &phase=plan->phases[index];
      if(!phase.resident_stage)continue;
      const auto &router=plan->routers[*phase.resident_stage];
      const std::array values{router.banks[0],router.banks[1],router.banks[2],router.map};
      std::array<std::size_t,4> dynamic{};
      for(std::size_t bank=0;bank<values.size();++bank) {
        const auto found=std::find_if(phase.original_inputs.begin(),phase.original_inputs.end(),
            [&](auto value){return value.ordinal()==values[bank].ordinal();});
        if(found==phase.original_inputs.end())throw std::runtime_error("resident phase missing one compressed/map input");
        dynamic[bank]=std::size_t(found-phase.original_inputs.begin());
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
        resident_inputs[index].bindings[bank]={phases[index].program->input(dynamic[bank]).value,bank_ids[bank]};
#endif
      }
      resident_inputs[index].plan=phases[index].program->input_plan(dynamic);
    }
  }
  const Run &run(std::size_t output) const{return phases.at(outputs.at(output).phase);}
  std::size_t port(std::size_t output) const{return outputs.at(output).port;}
};

template<class Routed> std::vector<c::ReadLease> execute_draft(Backend &backend,Routed &draft,std::span<const c::ReadLease> inputs) {
  if(!draft.plan)return backend.execute(draft.phases[0],inputs);
  if(inputs.size()!=draft.semantic_input_count||draft.full_inputs.size()!=draft.semantic_input_count+12)
    throw std::runtime_error("original resident semantic input contract");
  // Original six semantic inputs keep their identities. The twelve explicit
  // compressed/map inputs become available only after their actual router.
  std::vector<c::ReadLease> live(draft.value_count);
  for(std::size_t i=0;i<inputs.size();++i)live.at(draft.full_inputs[i].ordinal())=inputs[i];
  std::array<std::vector<c::ReadLease>,4> completed;
  for(std::size_t phase_index=0;phase_index<4;++phase_index) {
    const auto &phase=draft.plan->phases[phase_index];auto &run=draft.phases[phase_index];
    using CpuBatch=typename std::remove_reference_t<decltype(*draft.cpu_supply)>::Batch;
    std::optional<CpuBatch> cpu_batch;
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
    std::optional<c::cuda::WeightBatch> cuda_batch;
#endif
    const model_graph_run::InputPlan *input_plan=nullptr;
    std::function<void(g::Invocation&)> bind_external;
    if(phase.resident_stage) {
      const auto stage=*phase.resident_stage;const auto &router=draft.plan->routers[stage];
      // This bounded readback is scheduling metadata, not tensor arithmetic.
      const auto ids=values<std::int32_t>(live.at(router.ids.ordinal()));
      if(ids.size()!=draft.router_value_count)throw std::runtime_error("actual model router extent");
      if(!draft.select)throw std::runtime_error("resident model did not supply its typed global-ID selector");
      draft.selections[stage]=draft.select(ids);
      if(draft.before_upload)draft.before_upload(stage,draft.selections[stage].globals);
      const std::array bank_values{router.banks[0],router.banks[1],router.banks[2],router.map};
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
      if(backend.cuda) {
        cuda_batch.emplace(take(draft.cuda_supply->upload(stage%2,stage,draft.selections[stage],*draft.source,backend.queue),"one original encoded expert/map generation"));
        const auto &resident=draft.resident_inputs.at(phase_index);
        if(!resident.plan)throw std::runtime_error("resident phase input plan was not prepared");
        input_plan=&*resident.plan;
        bind_external=[&](g::Invocation &invocation){check(g::cuda::bind(invocation,draft.resident_inputs[phase_index].bindings,*cuda_batch),"atomic original expert/map weight bindings");};
      } else
#endif
      {
        cpu_batch.emplace(take(draft.cpu_supply->upload(stage,draft.selections[stage],*draft.source),"ready CPU original compressed experts"));
        for(std::size_t bank=0;bank<4;++bank)live.at(bank_values[bank].ordinal())=cpu_batch->banks[bank];
      }
    }
    std::vector<c::ReadLease> arguments;
    for(auto original:phase.original_inputs)arguments.push_back(live.at(original.ordinal()));
    completed[phase_index]=backend.execute(run,arguments,input_plan,bind_external);
    ++draft.completed_phases;
    if(phase.resident_stage) {
      const auto stage=*phase.resident_stage;
      draft.union_sizes[stage].push_back(draft.selections[stage].globals.size());
      std::printf("DSpark actual expert generation episode=%zu stage=%zu union=%zu physicalCapacity=%zu\n",
          draft.completed_episodes,stage,draft.selections[stage].globals.size(),draft.resident_capacity);
#if defined(HYLE_TEST_DSPARK_STATE_CUDA)
      if(cuda_batch) {
        const auto accounting=take(cuda_batch->report(),"actual retired CUDA expert upload accounting");
        EXPECT_EQ(accounting.logical_bytes,draft.resident_slot_bytes);EXPECT_EQ(accounting.staged_bytes,draft.resident_slot_bytes);
        EXPECT_EQ(accounting.completed_bytes,accounting.submitted_bytes);EXPECT_FALSE(accounting.copied_bytes_unknown);
        std::printf("DSpark CUDA expert upload episode=%zu stage=%zu slot=%zu logical=%zu staged=%zu submitted=%zu completed=%zu unknown=%d (API accounting,not measured VRAM/bus traffic)\n",
            draft.completed_episodes,stage,stage%2,accounting.logical_bytes,accounting.staged_bytes,
            accounting.submitted_bytes,accounting.completed_bytes,int(accounting.copied_bytes_unknown));
      }
#endif
    }
    for(std::size_t port=0;port<phase.original_outputs.size();++port)
      live.at(phase.original_outputs[port].ordinal())=completed[phase_index][port];
    if(phase.resident_stage) {
      const auto &router=draft.plan->routers[*phase.resident_stage];
      for(auto value:router.banks)live.at(value.ordinal())={};live.at(router.map.ordinal())={};
    }
    // Batch/argument permissions retire after the actual phase has completed
    // and its Invocation has reset; only independent activation/state leases
    // and bounded selection metadata survive into the next phase.
  }
  std::vector<c::ReadLease> outputs;
  for(const auto location:draft.outputs)outputs.push_back(completed.at(location.phase).at(location.port));
  ++draft.completed_episodes;
  return outputs;
}

} // namespace dspark_native_consumer
