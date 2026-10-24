#include <hyle/graph/vulkan.h>
#include <hyle/graph/executable.h>
#include <hyle/compute/vulkan_memory.h>
#if defined(HYLE_FOUNDATION_A_HYBRID)
#include <hyle/graph/cpu.h>
#include <hyle/graph/progress.h>
#include <hyle/sched/scheduler.h>
#include <hyle/sched/blocking.h>
#endif

#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
namespace c = hyle::compute;
namespace g = hyle::graph;
namespace v = c::vulkan;
using hyle::Status;
template<class T> using Result = hyle::Result<T>;

Status require(bool condition, const char *message) {
  return condition ? Status::Ok() : Status::Internal(message);
}

struct Symbolic {
  g::Graph graph;
  g::DimensionId rows;
  g::ValueId input, output;
};
Result<Symbolic> symbolic_graph(bool return_to_cpu = false) {
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto rows, builder.dimension({1, 4}));
  HYLE_ASSIGN_OR_RETURN(auto spec, g::TensorSpec::dense(c::DataType::kFloat32,
      std::array{g::Dimension(rows), g::Dimension::fixed(8)}));
  HYLE_ASSIGN_OR_RETURN(auto input, builder.input(spec));
  HYLE_ASSIGN_OR_RETURN(auto first, builder.add_node(c::Operation(c::Unary{c::UnaryKind::kAbs}),
      std::array{input}, std::array{spec}));
  HYLE_ASSIGN_OR_RETURN(auto middle, builder.result(first));
  HYLE_ASSIGN_OR_RETURN(auto last, builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}),
      std::array{middle}, std::array{spec}));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.result(last));
  if (return_to_cpu) {
    HYLE_ASSIGN_OR_RETURN(auto back, builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}),
        std::array{output}, std::array{spec}));
    HYLE_ASSIGN_OR_RETURN(output, builder.result(back));
  }
  HYLE_RETURN_IF_ERROR(builder.publish(output));
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  return Symbolic{std::move(graph), rows, input, output};
}

Status variants(const v::Context &context) {
  HYLE_ASSIGN_OR_RETURN(auto f, symbolic_graph());
  HYLE_ASSIGN_OR_RETURN(auto memory, v::MemoryResource::create(context, v::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto source, g::vulkan::execution_source(context));
  const std::array domains{g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
  const std::array one{g::ShapeBinding{f.rows, 1}}, four{g::ShapeBinding{f.rows, 4}};
  const std::array table{g::ExactShapeVariant{one}, g::ExactShapeVariant{four}};
  HYLE_ASSIGN_OR_RETURN(auto compiled, g::compile(f.graph, {domains, targets},
      {.shape_policy = g::CompileShapePolicy::kFiniteExact, .exact_variants = table}));
  const auto &family = compiled.executable;
  HYLE_RETURN_IF_ERROR(require(family.exact_variant_count() == 2 && !family.has_execution_route(),
                               "native finite family route contract differs"));
  const std::array missing{g::ShapeBinding{f.rows, 2}};
  auto absent = family.create_invocation({.shape = missing});
  HYLE_RETURN_IF_ERROR(require(!absent && absent.error().IsUnimplemented(), "native family covered undeclared shape"));
  HYLE_ASSIGN_OR_RETURN(auto first, family.create_invocation({.shape = four}));
  HYLE_ASSIGN_OR_RETURN(auto second, family.create_invocation({.shape = four}));
  HYLE_ASSIGN_OR_RETURN(auto point, g::specialize_exact(f.graph, four));
  HYLE_ASSIGN_OR_RETURN(auto desc, point.graph.input_description());
  std::array<float, 32> values, changed;
  for (std::size_t i = 0; i < values.size(); ++i) { values[i] = float(i) - 16; changed[i] = 2 * values[i]; }
  HYLE_ASSIGN_OR_RETURN(auto input, context.upload(desc, std::as_bytes(std::span(values))));
  HYLE_ASSIGN_OR_RETURN(auto other, context.upload(desc, std::as_bytes(std::span(changed))));
  HYLE_ASSIGN_OR_RETURN(auto input_owner, memory.import_readonly(input));
  HYLE_ASSIGN_OR_RETURN(auto other_owner, memory.import_readonly(other));
  HYLE_RETURN_IF_ERROR(first.prepare_resources());
  HYLE_RETURN_IF_ERROR(first.bind(f.input, desc, input_owner));
  HYLE_RETURN_IF_ERROR(first.prepare_replay({}));
  {
    HYLE_ASSIGN_OR_RETURN(auto submitted, first.submit());
    HYLE_RETURN_IF_ERROR(submitted.wait());
  }
  HYLE_ASSIGN_OR_RETURN(auto old, first.result(f.output));
  HYLE_RETURN_IF_ERROR(second.prepare_resources());
  HYLE_RETURN_IF_ERROR(second.bind(f.input, desc, other_owner));
  HYLE_RETURN_IF_ERROR(second.prepare_replay({}));
  {
    HYLE_ASSIGN_OR_RETURN(auto next, second.submit());
    HYLE_RETURN_IF_ERROR(next.wait());
  }
  HYLE_ASSIGN_OR_RETURN(auto fresh, second.result(f.output));
  HYLE_ASSIGN_OR_RETURN(auto old_owner, old.storage_lease());
  HYLE_ASSIGN_OR_RETURN(auto fresh_owner, fresh.storage_lease());
  HYLE_ASSIGN_OR_RETURN(auto old_native, memory.binding(std::move(old_owner), desc));
  HYLE_ASSIGN_OR_RETURN(auto fresh_native, memory.binding(std::move(fresh_owner), desc));
  std::array<float, 32> actual, original;
  HYLE_RETURN_IF_ERROR(context.download(fresh_native, std::as_writable_bytes(std::span(actual))));
  HYLE_RETURN_IF_ERROR(context.download(old_native, std::as_writable_bytes(std::span(original))));
  for (std::size_t i = 0; i < values.size(); ++i)
    HYLE_RETURN_IF_ERROR(require(
        std::bit_cast<std::uint32_t>(actual[i]) == std::bit_cast<std::uint32_t>(-std::abs(changed[i])) &&
        std::bit_cast<std::uint32_t>(original[i]) == std::bit_cast<std::uint32_t>(-std::abs(values[i])),
                                 "native slots shared output or math differs"));
  HYLE_RETURN_IF_ERROR(require(!first.bind(f.input, desc, other_owner).ok(),
                               "native finite result did not guard its original slot"));
  HYLE_ASSIGN_OR_RETURN(auto small, family.create_invocation({.shape = one}));
  HYLE_ASSIGN_OR_RETURN(auto small_point, g::specialize_exact(f.graph, one));
  HYLE_ASSIGN_OR_RETURN(auto small_desc, small_point.graph.input_description());
  HYLE_ASSIGN_OR_RETURN(auto small_bytes, small_desc.required_bytes());
  HYLE_ASSIGN_OR_RETURN(auto small_owner, input_owner.subview(0, small_bytes));
  HYLE_RETURN_IF_ERROR(small.prepare_resources());
  HYLE_RETURN_IF_ERROR(small.bind(f.input, small_desc, small_owner));
  HYLE_RETURN_IF_ERROR(small.prepare_replay({}));
  {
    HYLE_ASSIGN_OR_RETURN(auto submitted, small.submit());
    HYLE_RETURN_IF_ERROR(submitted.wait());
  }
  HYLE_ASSIGN_OR_RETURN(auto small_result, small.result(f.output));
  HYLE_ASSIGN_OR_RETURN(auto small_output, small_result.storage_lease());
  HYLE_ASSIGN_OR_RETURN(auto small_native, memory.binding(std::move(small_output), small_desc));
  std::array<float, 8> small_actual{};
  HYLE_RETURN_IF_ERROR(context.download(small_native, std::as_writable_bytes(std::span(small_actual))));
  for (std::size_t i = 0; i < small_actual.size(); ++i)
    HYLE_RETURN_IF_ERROR(require(
        std::bit_cast<std::uint32_t>(small_actual[i]) == std::bit_cast<std::uint32_t>(-std::abs(values[i])),
        "native finite selection used the wrong concrete shape"));
  return Status::Ok();
}

struct Control {
  g::Graph graph;
  g::ValueId table, index, input, output;
};
Result<Control> control_graph() {
  HYLE_ASSIGN_OR_RETURN(auto builder, g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(auto table_desc, c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 2>{2, 8}));
  HYLE_ASSIGN_OR_RETURN(auto index_desc, c::TensorDesc::dense(c::DataType::kInt32,
      std::array<std::int64_t, 1>{1}));
  HYLE_ASSIGN_OR_RETURN(auto input_desc, c::TensorDesc::dense(c::DataType::kFloat32,
      std::array<std::int64_t, 1>{8}));
  HYLE_ASSIGN_OR_RETURN(auto table, builder.input(table_desc));
  HYLE_ASSIGN_OR_RETURN(auto index, builder.input(index_desc));
  HYLE_ASSIGN_OR_RETURN(auto input, builder.input(input_desc));
  HYLE_ASSIGN_OR_RETURN(auto prerequisite, builder.add_node(c::Operation(c::Gather{0, c::IndexBoundsPolicy::kChecked}),
      std::array{table, index}));
  HYLE_ASSIGN_OR_RETURN(auto successor, builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}), std::array{input}));
  HYLE_RETURN_IF_ERROR(builder.depends_on(successor, prerequisite));
  HYLE_ASSIGN_OR_RETURN(auto output, builder.result(successor));
  HYLE_RETURN_IF_ERROR(builder.publish(output)); // Gather has no data consumer.
  HYLE_ASSIGN_OR_RETURN(auto graph, std::move(builder).finish());
  return Control{std::move(graph), table, index, input, output};
}

Status control(const v::Context &context) {
  HYLE_ASSIGN_OR_RETURN(auto f, control_graph());
  HYLE_ASSIGN_OR_RETURN(auto memory, v::MemoryResource::create(context, v::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto source, g::vulkan::execution_source(context));
  const std::array domains{g::StorageDomain{memory.ref()}};
  const std::array targets{g::ExecutionTarget{source, g::StorageDomainId{0}}};
  HYLE_ASSIGN_OR_RETURN(auto compiled, g::compile(f.graph, {domains, targets}));
  HYLE_RETURN_IF_ERROR(require(compiled.report.node_count() == 2, "native compiler erased control-only predecessor"));
  std::array<float, 16> table{};
  const std::array<float, 8> values{1,2,3,4,5,6,7,8};
  const std::array<std::int32_t, 1> bad{2}, good{1};
  HYLE_ASSIGN_OR_RETURN(auto data, context.upload(*f.graph.input_description(0), std::as_bytes(std::span(table))));
  HYLE_ASSIGN_OR_RETURN(auto input, context.upload(*f.graph.input_description(2), std::as_bytes(std::span(values))));
  HYLE_ASSIGN_OR_RETURN(auto data_owner, memory.import_readonly(data));
  HYLE_ASSIGN_OR_RETURN(auto input_owner, memory.import_readonly(input));
  for (const auto &indices : {bad, good}) {
    HYLE_ASSIGN_OR_RETURN(auto index, context.upload(*f.graph.input_description(1), std::as_bytes(std::span(indices))));
    HYLE_ASSIGN_OR_RETURN(auto index_owner, memory.import_readonly(index));
    HYLE_ASSIGN_OR_RETURN(auto invocation, compiled.executable.create_invocation());
    HYLE_RETURN_IF_ERROR(invocation.prepare_resources());
    HYLE_RETURN_IF_ERROR(invocation.bind(f.table, *f.graph.input_description(0), data_owner));
    HYLE_RETURN_IF_ERROR(invocation.bind(f.index, *f.graph.input_description(1), index_owner));
    HYLE_RETURN_IF_ERROR(invocation.bind(f.input, *f.graph.input_description(2), input_owner));
    HYLE_RETURN_IF_ERROR(invocation.prepare_replay({}));
    HYLE_ASSIGN_OR_RETURN(auto submitted, invocation.submit());
    const auto status = submitted.wait();
    if (indices == bad) {
      HYLE_RETURN_IF_ERROR(require(status.IsOutOfRange() && !invocation.result(f.output),
                                   "control predecessor failure was erased"));
    } else {
      HYLE_RETURN_IF_ERROR(status);
      HYLE_ASSIGN_OR_RETURN(auto result, invocation.result(f.output));
      HYLE_ASSIGN_OR_RETURN(auto output_owner, result.storage_lease());
      HYLE_ASSIGN_OR_RETURN(auto native,
          memory.binding(std::move(output_owner), *f.graph.input_description(2)));
      std::array<float, 8> actual;
      HYLE_RETURN_IF_ERROR(context.download(native, std::as_writable_bytes(std::span(actual))));
      for (std::size_t i=0;i<actual.size();++i)
        HYLE_RETURN_IF_ERROR(require(actual[i] == -values[i], "controlled native math differs"));
    }
  }
  return Status::Ok();
}

#if defined(HYLE_FOUNDATION_A_HYBRID)
Result<std::vector<g::QueueBinding>> shared_queues(const g::CompileReport &report,
                                                 g::ExecutionTargetId cpu_target,
                                                 c::Queue &queue) {
  std::vector<g::QueueBinding> bindings;
  for (std::size_t index = 0; index < report.lane_count(); ++index) {
    HYLE_ASSIGN_OR_RETURN(auto lane, report.lane(index));
    if (!lane.queue_required) continue;
    HYLE_RETURN_IF_ERROR(require(lane.runtime_mode == c::ProviderRuntimeMode::kSharedQueue &&
                                 lane.target == cpu_target,
                                 "hybrid requested a synthetic native Compute Queue"));
    bindings.push_back({lane.lane, &queue});
  }
  return bindings;
}

Status hybrid(const v::Context &context) {
  namespace m = hyle::mem;
  hyle::sched::Scheduler scheduler(2, m::system_allocator(), 64);
  hyle::sched::BlockingPool workers(m::system_allocator(), 2);
  struct Drain final { hyle::sched::Scheduler &scheduler; ~Drain() { scheduler.drain(); } };
  HYLE_ASSIGN_OR_RETURN(auto cpu, g::cpu::create_context());
  HYLE_ASSIGN_OR_RETURN(auto queue, cpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto device, v::MemoryResource::create(context, v::MemoryKind::kDevice));
  HYLE_ASSIGN_OR_RETURN(auto staging, v::MemoryResource::create(context, v::MemoryKind::kStaging));
  const g::StorageDomainId host_storage{0}, device_storage{1};
  const g::ExecutionTargetId native_target{0}, cpu_target{1};
  const std::array domains{g::StorageDomain{m::system_resource()},
      g::StorageDomain{device.ref()}, g::StorageDomain{staging.ref()}};
  HYLE_ASSIGN_OR_RETURN(auto native_source,g::vulkan::execution_source(context));
  HYLE_ASSIGN_OR_RETURN(auto cpu_source,g::ExecutionSource::acquire(cpu));
  const std::array targets{
      g::ExecutionTarget{native_source, device_storage},
      g::ExecutionTarget{cpu_source, host_storage}};
  const g::CompileEnvironment environment{domains, targets};
  HYLE_ASSIGN_OR_RETURN(auto f, symbolic_graph());
  const std::array one{g::ShapeBinding{f.rows, 1}}, four{g::ShapeBinding{f.rows, 4}};
  const std::array table{g::ExactShapeVariant{one}, g::ExactShapeVariant{four}};
  const std::array placement{
      g::NodePlacement{*f.graph.node_id(0), cpu_target, g::PlacementStrength::kRequired},
      g::NodePlacement{*f.graph.node_id(1), native_target, g::PlacementStrength::kRequired}};
  const std::array external{g::ExternalResourceContract{f.input, host_storage}};
  HYLE_ASSIGN_OR_RETURN(auto family, g::compile(f.graph, environment,
      {.shape_policy=g::CompileShapePolicy::kFiniteExact, .exact_variants=table,
       .node_placements=placement, .external_resources=external}));
  HYLE_RETURN_IF_ERROR(require(!family.executable.has_execution_route() &&
      family.executable.exact_variant_count()==table.size(), "hybrid finite parent invented a general route"));
  const std::array missing{g::ShapeBinding{f.rows,2}};
  auto absent=family.executable.create_invocation({.shape=missing});
  HYLE_RETURN_IF_ERROR(require(!absent && absent.error().IsUnimplemented(), "hybrid covered undeclared shape"));
  std::array<m::Buffer,2> input_backing;
  std::array<g::GraphOutput,2> retained;
  for (std::size_t index=0; index<table.size(); ++index) {
    const auto &shape=index==0?one:four;
    HYLE_ASSIGN_OR_RETURN(auto point, g::specialize_exact(f.graph, shape));
    HYLE_ASSIGN_OR_RETURN(auto desc, point.graph.input_description());
    HYLE_ASSIGN_OR_RETURN(input_backing[index], m::Buffer::create(m::system_resource(), *desc.required_bytes(), 64));
    HYLE_ASSIGN_OR_RETURN(auto input, c::TensorView::create(desc, input_backing[index].view()));
    std::vector<float> values(*desc.element_count(), -3.0f), actual(values.size());
    HYLE_RETURN_IF_ERROR(input_backing[index].view().write(std::as_bytes(std::span(values))));
    HYLE_ASSIGN_OR_RETURN(auto report, family.report.exact_variant(index));
    HYLE_ASSIGN_OR_RETURN(auto bindings, shared_queues(report,cpu_target,queue));
    HYLE_ASSIGN_OR_RETURN(auto invocation, family.executable.create_invocation({.shape=shape}));
    HYLE_RETURN_IF_ERROR(g::prepare_progress(invocation, scheduler, workers));
    HYLE_RETURN_IF_ERROR(invocation.bind(f.input, input.as_const()));
    HYLE_RETURN_IF_ERROR(invocation.prepare_replay(bindings));
    Drain drain{scheduler};
    {
      HYLE_ASSIGN_OR_RETURN(auto accepted, invocation.submit(bindings));
      HYLE_RETURN_IF_ERROR(accepted.wait());
    }
    HYLE_ASSIGN_OR_RETURN(auto result, invocation.result(f.output));
    HYLE_ASSIGN_OR_RETURN(auto owner, result.storage_lease());
    HYLE_ASSIGN_OR_RETURN(auto native, device.binding(std::move(owner),desc));
    HYLE_RETURN_IF_ERROR(context.download(native, std::as_writable_bytes(std::span(actual))));
    for (auto value : actual) HYLE_RETURN_IF_ERROR(require(value == -3.0f, "hybrid exact math differs"));
    HYLE_RETURN_IF_ERROR(require(!invocation.reset().ok(), "old hybrid result failed to guard slot"));
    retained[index]=std::move(result);
  }
  for(std::size_t index=0;index<retained.size();++index){
    HYLE_ASSIGN_OR_RETURN(auto tensor,retained[index].tensor());
    HYLE_ASSIGN_OR_RETURN(auto owner,retained[index].storage_lease());
    HYLE_ASSIGN_OR_RETURN(auto native,device.binding(std::move(owner),tensor.description()));
    std::vector<float> actual(index==0?8:32);
    HYLE_RETURN_IF_ERROR(context.download(native,std::as_writable_bytes(std::span(actual))));
    for(auto value:actual)HYLE_RETURN_IF_ERROR(require(value==-3.0f,"hybrid independent slot lost its retained output"));
  }
  // Both real storage bridges execute in this same original Invocation. The
  // addresses and cold preparation stay fixed across changed payloads.
  HYLE_ASSIGN_OR_RETURN(auto roundtrip,symbolic_graph(true));
  const std::array small{g::ShapeBinding{roundtrip.rows,1}},large{g::ShapeBinding{roundtrip.rows,4}};
  const std::array roundtrip_points{g::ExactShapeVariant{small},g::ExactShapeVariant{large}};
  const std::array roundtrip_placement{
      g::NodePlacement{*roundtrip.graph.node_id(0),cpu_target,g::PlacementStrength::kRequired},
      g::NodePlacement{*roundtrip.graph.node_id(1),native_target,g::PlacementStrength::kRequired},
      g::NodePlacement{*roundtrip.graph.node_id(2),cpu_target,g::PlacementStrength::kRequired}};
  const std::array roundtrip_external{
      g::ExternalResourceContract{roundtrip.input,host_storage},
      g::ExternalResourceContract{roundtrip.output,host_storage}};
  HYLE_ASSIGN_OR_RETURN(auto replays,g::compile(roundtrip.graph,environment,
      {.shape_policy=g::CompileShapePolicy::kFiniteExact,.exact_variants=roundtrip_points,
       .node_placements=roundtrip_placement,.external_resources=roundtrip_external}));
  for(std::size_t index=0;index<roundtrip_points.size();++index){
    const auto&shape=index==0?small:large;
    HYLE_ASSIGN_OR_RETURN(auto point,g::specialize_exact(roundtrip.graph,shape));
    HYLE_ASSIGN_OR_RETURN(auto desc,point.graph.input_description());
    HYLE_ASSIGN_OR_RETURN(auto source,m::Buffer::create(m::system_resource(),*desc.required_bytes(),64));
    HYLE_ASSIGN_OR_RETURN(auto destination,m::Buffer::create(m::system_resource(),*desc.required_bytes(),64));
    HYLE_ASSIGN_OR_RETURN(auto input,c::ConstTensorView::create(desc,source.view()));
    HYLE_ASSIGN_OR_RETURN(auto output,c::TensorView::create(desc,destination.view()));
    std::vector<float> values(*desc.element_count()),actual(values.size());
    for(std::size_t i=0;i<values.size();++i)values[i]=float(i)-4.0f;
    values[0]=0.0f;
    HYLE_RETURN_IF_ERROR(source.view().write(std::as_bytes(std::span(values))));
    HYLE_ASSIGN_OR_RETURN(auto report,replays.report.exact_variant(index));
    HYLE_ASSIGN_OR_RETURN(auto replay_queues,shared_queues(report,cpu_target,queue));
    HYLE_ASSIGN_OR_RETURN(auto slot,replays.executable.create_invocation({.shape=shape}));
    HYLE_RETURN_IF_ERROR(g::prepare_progress(slot,scheduler,workers));
    HYLE_RETURN_IF_ERROR(slot.bind(roundtrip.input,input));
    HYLE_RETURN_IF_ERROR(slot.bind(roundtrip.output,output));
    HYLE_RETURN_IF_ERROR(slot.prepare_replay(replay_queues)); // Exactly once.
    Drain drain{scheduler};
    for(int phase=0;phase<2;++phase){
      for(std::size_t i=0;i<values.size();++i)
        values[i]=phase==0?float(i)-4.0f:0.5f*(float(i)-9.0f);
      values[0]=phase==0?0.0f:-0.0f;
      HYLE_RETURN_IF_ERROR(source.view().write(std::as_bytes(std::span(values))));
      {
        HYLE_ASSIGN_OR_RETURN(auto submitted,slot.submit(replay_queues));
        HYLE_RETURN_IF_ERROR(submitted.wait());
      }
      {
        HYLE_ASSIGN_OR_RETURN(auto observed,slot.result(roundtrip.output));
        HYLE_ASSIGN_OR_RETURN(auto tensor,observed.tensor());
        HYLE_RETURN_IF_ERROR(tensor.storage().read(std::as_writable_bytes(std::span(actual))));
        for(std::size_t i=0;i<actual.size();++i)
          HYLE_RETURN_IF_ERROR(require(std::bit_cast<std::uint32_t>(actual[i])==
              std::bit_cast<std::uint32_t>(std::abs(values[i])),"CPU/Vulkan/CPU strict replay raw bits differ"));
        HYLE_RETURN_IF_ERROR(require(!slot.reset().ok(),"hybrid replay output failed to guard its slot"));
      } // Drop every Completion/output observation before the next strict submit.
    }
    std::printf("CPU/Vulkan/CPU rows=%lld same-slot strict replay x2 PASS\n",static_cast<long long>(shape[0].value));
  }
  HYLE_ASSIGN_OR_RETURN(auto controlled, control_graph());
  const std::array last_cpu{
      g::NodePlacement{*controlled.graph.node_id(0),native_target,g::PlacementStrength::kRequired},
      g::NodePlacement{*controlled.graph.node_id(1),cpu_target,g::PlacementStrength::kRequired}};
  HYLE_ASSIGN_OR_RETURN(auto program, g::compile(controlled.graph, environment, {.node_placements=last_cpu}));
  HYLE_RETURN_IF_ERROR(require(program.report.node_count()==2 && program.executable.region_count()>=1,
      "hybrid erased control-only fragment"));
  HYLE_ASSIGN_OR_RETURN(auto bindings,shared_queues(program.report,cpu_target,queue));
  std::array<m::Buffer, 3> backing;
  std::array<c::ConstTensorView, 3> inputs;
  for (std::size_t i=0;i<inputs.size();++i) {
    HYLE_ASSIGN_OR_RETURN(auto desc, controlled.graph.input_description(i));
    HYLE_ASSIGN_OR_RETURN(backing[i], m::Buffer::create(m::system_resource(), *desc.required_bytes(), 64));
    HYLE_ASSIGN_OR_RETURN(inputs[i], c::ConstTensorView::create(desc, backing[i].view()));
    std::vector<std::byte> zero(*desc.required_bytes());
    HYLE_RETURN_IF_ERROR(backing[i].view().write(zero));
  }
  const std::array<std::int32_t, 1> bad{2};
  HYLE_RETURN_IF_ERROR(backing[1].view().write(std::as_bytes(std::span(bad))));
  HYLE_ASSIGN_OR_RETURN(auto invocation, program.executable.create_invocation());
  HYLE_RETURN_IF_ERROR(g::prepare_progress(invocation,scheduler,workers));
  for (std::size_t i=0;i<inputs.size();++i) HYLE_RETURN_IF_ERROR(invocation.bind(*controlled.graph.input_id(i), inputs[i]));
  HYLE_ASSIGN_OR_RETURN(auto desc,controlled.graph.value_description(controlled.output));
  HYLE_ASSIGN_OR_RETURN(auto destination,m::Buffer::create(m::system_resource(),*desc.required_bytes(),64));
  HYLE_ASSIGN_OR_RETURN(auto output,c::TensorView::create(desc,destination.view()));
  std::array<float,8> sentinel;
  sentinel.fill(99.0f);
  HYLE_RETURN_IF_ERROR(destination.view().write(std::as_bytes(std::span(sentinel))));
  HYLE_RETURN_IF_ERROR(invocation.bind(controlled.output,output));
  auto reject_wait = invocation.prepare_replay(bindings,{.host_wait=g::HostWaitPolicy::kReject});
  HYLE_RETURN_IF_ERROR(require(!reject_wait.ok(), "hybrid silently accepted forbidden host wait"));
  HYLE_RETURN_IF_ERROR(invocation.prepare_replay(bindings));
  Drain drain{scheduler};
  HYLE_ASSIGN_OR_RETURN(auto accepted,invocation.submit(bindings));
  const auto terminal=accepted.wait();
  HYLE_RETURN_IF_ERROR(require(terminal.IsOutOfRange() && !invocation.result(controlled.output),
      "hybrid erased failed control predecessor"));
  std::array<float,8> unchanged;
  HYLE_RETURN_IF_ERROR(destination.view().read(std::as_writable_bytes(std::span(unchanged))));
  for(const auto value:unchanged)
    HYLE_RETURN_IF_ERROR(require(std::bit_cast<std::uint32_t>(value)==std::bit_cast<std::uint32_t>(99.0f),
        "failed Vulkan control predecessor overwrote CPU caller output"));
  return Status::Ok();
}
#endif

Status run() {
  HYLE_ASSIGN_OR_RETURN(auto context, v::Context::create());
  HYLE_RETURN_IF_ERROR(variants(context));
  HYLE_RETURN_IF_ERROR(control(context));
#if defined(HYLE_FOUNDATION_A_HYBRID)
  HYLE_RETURN_IF_ERROR(hybrid(context));
#endif
  return Status::Ok();
}
} // namespace

int main() {
  const auto status = run();
  if (!status.ok()) { std::fprintf(stderr,"foundation A Vulkan: %.*s\n",int(status.message().size()),status.message().data()); return 1; }
  std::puts("foundation A Vulkan exact/control/slot journey PASS");
}
