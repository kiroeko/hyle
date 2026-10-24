#include <hyle/graph/apple.h>
#include <hyle/graph/executable.h>
#include <hyle/graph/result.h>
#include <hyle/compute/apple_memory.h>
#if !defined(HYLE_FOUNDATION_A_NATIVE_ONLY)
#include <hyle/graph/cpu.h>
#endif

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace a = c::apple;
using hyle::Status;
template<class T> using Result = hyle::Result<T>;
Status require(bool condition, const char *message) {
  return condition ? Status::Ok() : Status::Internal(message);
}
template<class T, std::size_t N>
Result<c::ReadLease> upload(const a::MemoryResource &memory,
                            const c::TensorDesc &desc, const std::array<T,N> &values) {
  HYLE_ASSIGN_OR_RETURN(auto native, memory.allocate(desc));
  HYLE_ASSIGN_OR_RETURN(auto host, hyle::mem::Buffer::create(
      hyle::mem::system_resource(), sizeof(values), alignof(T)));
  HYLE_RETURN_IF_ERROR(host.view().write(std::as_bytes(std::span(values))));
  HYLE_ASSIGN_OR_RETURN(auto source, c::ConstTensorView::create(desc, host.view()));
  HYLE_RETURN_IF_ERROR(memory.copy_from_host(native, source));
  return memory.import_readonly(native, desc);
}

Status run() {
  HYLE_ASSIGN_OR_RETURN(auto context, a::Context::create());
  HYLE_ASSIGN_OR_RETURN(auto memory, a::MemoryResource::create(context));
  HYLE_ASSIGN_OR_RETURN(auto source, g::apple::execution_source(context));
  const std::array domains{g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{source,g::StorageDomainId{0}}};
#if !defined(HYLE_FOUNDATION_A_NATIVE_ONLY)
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto queue, cpu.create_queue());
#endif
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto rows, builder.dimension({1,4}));
  HYLE_ASSIGN_OR_RETURN(auto spec, g::TensorSpec::dense(c::DataType::kFloat32,
      std::array{g::Dimension(rows), g::Dimension::fixed(8)}));
  HYLE_ASSIGN_OR_RETURN(auto x, builder.input(spec));
  HYLE_ASSIGN_OR_RETURN(auto first, builder.add_node(c::Operation(c::Unary{c::UnaryKind::kAbs}),
      std::array{x}, std::array{spec}));
  // Independent data branch; the first node is live only through control.
  HYLE_ASSIGN_OR_RETURN(auto last, builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}),
      std::array{x}, std::array{spec}));
  HYLE_RETURN_IF_ERROR(builder.depends_on(last, first));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.result(last));
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  const std::array one{g::ShapeBinding{rows,1}}, four{g::ShapeBinding{rows,4}};
  const std::array points{g::ExactShapeVariant{one}, g::ExactShapeVariant{four}};
  HYLE_ASSIGN_OR_RETURN(auto compiled, g::compile(graph, {domains,targets},
      {.shape_policy=g::CompileShapePolicy::kFiniteExact, .exact_variants=points}));
  const auto &family=compiled.executable;
  HYLE_RETURN_IF_ERROR(require(family.exact_variant_count()==2 && !family.has_execution_route(),
      "Apple finite family route contract differs"));
  const std::array missing{g::ShapeBinding{rows,2}};
  auto absent=family.create_invocation({.shape=missing});
  HYLE_RETURN_IF_ERROR(require(!absent && absent.error().IsUnimplemented(), "Apple covered undeclared shape"));
  HYLE_ASSIGN_OR_RETURN(auto first_slot, family.create_invocation({.shape=four}));
  HYLE_ASSIGN_OR_RETURN(auto second_slot, family.create_invocation({.shape=four}));
  HYLE_ASSIGN_OR_RETURN(auto point, g::specialize_exact(graph,four));
  HYLE_ASSIGN_OR_RETURN(auto input_desc, point.graph.input_description());
  std::array<float,32> values, changed;
  for (std::size_t i=0;i<values.size();++i) { values[i]=float(i)-16; changed[i]=2*values[i]; }
  HYLE_ASSIGN_OR_RETURN(auto input, upload(memory,input_desc,values));
  HYLE_ASSIGN_OR_RETURN(auto other, upload(memory,input_desc,changed));
  HYLE_RETURN_IF_ERROR(first_slot.prepare_resources());
  HYLE_RETURN_IF_ERROR(first_slot.bind(x,input_desc,input));
  HYLE_RETURN_IF_ERROR(first_slot.prepare_replay({}));
  {
    HYLE_ASSIGN_OR_RETURN(auto submitted, first_slot.submit());
    HYLE_RETURN_IF_ERROR(submitted.wait());
  }
  HYLE_ASSIGN_OR_RETURN(auto old, first_slot.result(output));
  HYLE_RETURN_IF_ERROR(second_slot.prepare_resources());
  HYLE_RETURN_IF_ERROR(second_slot.bind(x,input_desc,other));
  HYLE_RETURN_IF_ERROR(second_slot.prepare_replay({}));
  {
    HYLE_ASSIGN_OR_RETURN(auto next, second_slot.submit());
    HYLE_RETURN_IF_ERROR(next.wait());
  }
  HYLE_ASSIGN_OR_RETURN(auto fresh, second_slot.result(output));
  HYLE_ASSIGN_OR_RETURN(auto old_tensor, old.tensor());
  HYLE_ASSIGN_OR_RETURN(auto fresh_tensor, fresh.tensor());
  std::array<float,32> before,after;
  HYLE_RETURN_IF_ERROR(old_tensor.storage().read(std::as_writable_bytes(std::span(before))));
  HYLE_RETURN_IF_ERROR(fresh_tensor.storage().read(std::as_writable_bytes(std::span(after))));
  for (std::size_t i=0;i<values.size();++i)
    HYLE_RETURN_IF_ERROR(require(before[i]==-values[i] && after[i]==-changed[i], "Apple slots or controlled math differ"));
  HYLE_RETURN_IF_ERROR(require(!first_slot.bind(x,input_desc,other).ok(),
      "Apple finite result did not guard its original slot"));

#if !defined(HYLE_FOUNDATION_A_NATIVE_ONLY)
  const g::ExecutionTargetId apple_target{0}, cpu_target{1};
  const g::StorageDomainId host_storage{0}, metal_storage{1};
  const std::array mixed_domains{g::StorageDomain{hyle::mem::system_resource()}, g::StorageDomain{memory.ref()}};
  HYLE_ASSIGN_OR_RETURN(auto cpu_source, g::ExecutionSource::acquire(cpu));
  const std::array mixed_targets{g::ExecutionTarget{source,metal_storage},
      g::ExecutionTarget{cpu_source,host_storage}};
  const std::array placement{
      g::NodePlacement{first,cpu_target,g::PlacementStrength::kRequired},
      g::NodePlacement{last,apple_target,g::PlacementStrength::kRequired}};
  const std::array external{g::ExternalResourceContract{x,host_storage}};
  HYLE_ASSIGN_OR_RETURN(auto hybrids, g::compile(graph,{mixed_domains,mixed_targets},
      {.shape_policy=g::CompileShapePolicy::kFiniteExact,.exact_variants=points,
       .node_placements=placement,.external_resources=external}));
  HYLE_RETURN_IF_ERROR(require(hybrids.executable.exact_variant_count()==points.size() &&
      !hybrids.executable.has_execution_route(), "Apple hybrid finite parent invented a general route"));
  auto missing_hybrid=hybrids.executable.create_invocation({.shape=missing});
  HYLE_RETURN_IF_ERROR(require(!missing_hybrid && missing_hybrid.error().IsUnimplemented(),
      "Apple hybrid covered undeclared shape"));
  std::array<hyle::mem::Buffer,2> hybrid_inputs;
  std::array<g::GraphOutput,2> hybrid_retained;
  for (std::size_t point_index=0;point_index<points.size();++point_index) {
    const auto &shape=point_index==0?one:four;
    HYLE_ASSIGN_OR_RETURN(auto point,g::specialize_exact(graph,shape));
    HYLE_ASSIGN_OR_RETURN(auto desc,point.graph.input_description());
    HYLE_ASSIGN_OR_RETURN(hybrid_inputs[point_index],
        hyle::mem::Buffer::create(hyle::mem::system_resource(),*desc.required_bytes(),64));
    std::vector<float> data(*desc.element_count(),3.0f);
    HYLE_RETURN_IF_ERROR(hybrid_inputs[point_index].view().write(std::as_bytes(std::span(data))));
    HYLE_ASSIGN_OR_RETURN(auto tensor,c::ConstTensorView::create(desc,hybrid_inputs[point_index].view()));
    HYLE_ASSIGN_OR_RETURN(auto report,hybrids.report.exact_variant(point_index));
    std::vector<g::QueueBinding> queues;
    for(std::size_t lane_index=0;lane_index<report.lane_count();++lane_index) {
      HYLE_ASSIGN_OR_RETURN(auto lane,report.lane(lane_index));
      if(!lane.queue_required)continue;
      HYLE_RETURN_IF_ERROR(require(lane.target==cpu_target &&
          lane.runtime_mode==c::ProviderRuntimeMode::kSharedQueue,
          "Apple hybrid requested a synthetic native Queue"));
      queues.push_back({lane.lane,&queue});
    }
    HYLE_ASSIGN_OR_RETURN(auto slot,hybrids.executable.create_invocation({.shape=shape}));
    HYLE_RETURN_IF_ERROR(slot.bind(x,tensor));
    const auto forbidden=slot.prepare_replay(queues,{.host_wait=g::HostWaitPolicy::kReject});
    if(!forbidden.ok() && !forbidden.IsUnimplemented())return forbidden;
    HYLE_RETURN_IF_ERROR(slot.prepare_replay(queues));
    HYLE_ASSIGN_OR_RETURN(auto replay,slot.replay_report());
    HYLE_RETURN_IF_ERROR(require(forbidden.ok()==(replay.host_wait_dependency_count==0),
        "Apple hybrid host-wait policy differs from its actual dependency routes"));
    {
      HYLE_ASSIGN_OR_RETURN(auto submitted,slot.submit(queues));
      HYLE_RETURN_IF_ERROR(submitted.wait());
    }
    HYLE_ASSIGN_OR_RETURN(auto result,slot.result(output));
    HYLE_ASSIGN_OR_RETURN(auto actual,result.tensor());
    std::vector<float> observed(data.size());
    HYLE_RETURN_IF_ERROR(actual.storage().read(std::as_writable_bytes(std::span(observed))));
    for (std::size_t i=0;i<data.size();++i)
      HYLE_RETURN_IF_ERROR(require(observed[i]==-3.0f,"Apple hybrid exact/control math differs"));
    HYLE_RETURN_IF_ERROR(require(!slot.reset().ok(),"Apple hybrid old result failed to guard slot"));
    hybrid_retained[point_index]=std::move(result);
  }
  for(std::size_t point_index=0;point_index<hybrid_retained.size();++point_index) {
    HYLE_ASSIGN_OR_RETURN(auto actual,hybrid_retained[point_index].tensor());
    std::vector<float> observed(point_index==0?8:32);
    HYLE_RETURN_IF_ERROR(actual.storage().read(std::as_writable_bytes(std::span(observed))));
    for(const auto value:observed)
      HYLE_RETURN_IF_ERROR(require(value==-3.0f,"Apple hybrid retained exact slot lost its result"));
  }
#endif

  // The checked predecessor has no tensor consumer. Its accepted failure must
  // still prevent an otherwise valid, independent native successor result.
  HYLE_ASSIGN_OR_RETURN(auto checked, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto table_desc, c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t,2>{2,8}));
  HYLE_ASSIGN_OR_RETURN(auto index_desc, c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t,1>{1}));
  HYLE_ASSIGN_OR_RETURN(auto values_desc, c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t,1>{8}));
  HYLE_ASSIGN_OR_RETURN(auto table_input, checked.input(table_desc));
  HYLE_ASSIGN_OR_RETURN(auto index_input, checked.input(index_desc));
  HYLE_ASSIGN_OR_RETURN(auto values_input, checked.input(values_desc));
  HYLE_ASSIGN_OR_RETURN(auto predecessor, checked.add_node(
      c::Operation(c::Gather{0,c::IndexBoundsPolicy::kChecked}),
      std::array{table_input,index_input}));
  HYLE_ASSIGN_OR_RETURN(auto successor, checked.add_node(
      c::Operation(c::Unary{c::UnaryKind::kNeg}),std::array{values_input}));
  HYLE_RETURN_IF_ERROR(checked.depends_on(successor,predecessor));
  HYLE_ASSIGN_OR_RETURN(auto controlled_output, checked.result(successor));
  HYLE_RETURN_IF_ERROR(checked.publish(controlled_output));
  HYLE_ASSIGN_OR_RETURN(auto controlled_graph, std::move(checked).finish());
  HYLE_ASSIGN_OR_RETURN(auto controlled, g::compile(controlled_graph,{domains,targets}));
  HYLE_RETURN_IF_ERROR(require(controlled.report.node_count()==2,
      "Apple erased the control-only predecessor"));
  const std::array<float,16> table_values{};
  const std::array<float,8> independent{1,2,3,4,5,6,7,8};
  const std::array<std::int32_t,1> bad{2},good{1};
  HYLE_ASSIGN_OR_RETURN(auto native_table, upload(memory,table_desc,table_values));
  HYLE_ASSIGN_OR_RETURN(auto native_values, upload(memory,values_desc,independent));
  for(const auto &indices : {bad,good}) {
    HYLE_ASSIGN_OR_RETURN(auto native_ids, upload(memory,index_desc,indices));
    HYLE_ASSIGN_OR_RETURN(auto slot, controlled.executable.create_invocation());
    HYLE_RETURN_IF_ERROR(slot.prepare_resources());
    HYLE_RETURN_IF_ERROR(slot.bind(table_input,table_desc,native_table));
    HYLE_RETURN_IF_ERROR(slot.bind(index_input,index_desc,native_ids));
    HYLE_RETURN_IF_ERROR(slot.bind(values_input,values_desc,native_values));
    HYLE_RETURN_IF_ERROR(slot.prepare_replay({}));
    HYLE_ASSIGN_OR_RETURN(auto submitted, slot.submit());
    const auto terminal=submitted.wait();
    if(indices==bad) {
      HYLE_RETURN_IF_ERROR(require(terminal.IsOutOfRange() && !slot.result(controlled_output),
          "Apple erased a failing control-only predecessor"));
    } else {
      HYLE_RETURN_IF_ERROR(terminal);
      HYLE_ASSIGN_OR_RETURN(auto result, slot.result(controlled_output));
      HYLE_ASSIGN_OR_RETURN(auto tensor, result.tensor());
      std::array<float,8> actual;
      HYLE_RETURN_IF_ERROR(tensor.storage().read(std::as_writable_bytes(std::span(actual))));
      for(std::size_t i=0;i<independent.size();++i)
        HYLE_RETURN_IF_ERROR(require(actual[i]==-independent[i],"Apple checked control success differs"));
    }
  }
  return Status::Ok();
}
} // namespace

int main() {
  const auto status=run();
  if(!status.ok()){std::fprintf(stderr,"foundation A Apple: %.*s\n",int(status.message().size()),status.message().data());return 1;}
  std::puts("foundation A Apple exact/control/slot journey PASS");
}
