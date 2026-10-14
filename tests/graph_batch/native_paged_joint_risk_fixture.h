#pragma once
#if defined(HYLE_JOINT_PAGE_APPLE)
#define HYLE_G01_PAGE_RISK_APPLE 1
#include <hyle/graph/cpu_apple.h>
#else
#include <hyle/graph/cpu_vulkan.h>
#endif
#include <hyle/graph/cpu.h>
#include <hyle/sched/scheduler.h>
#include "../compute_batch/native_paged_risk_fixture.h"
#include <cmath>

namespace hyle_joint_page_risk {
namespace r=hyle_g01_page_risk;
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace n=r::n;
#if defined(HYLE_JOINT_PAGE_APPLE)
namespace joint=g::cpu_apple;
#else
namespace joint=g::cpu_vulkan;
#endif
using hyle::Status;using hyle::Result;
using r::require;
struct Drain final {hyle::sched::Scheduler& scheduler;~Drain(){scheduler.drain();}};
struct GraphData final {
  g::Graph graph;
  g::ValueId query,key,value;
  c::TensorDesc query_desc;
  std::vector<g::NodePlacement> placements;
  std::vector<g::batch::NativePagedNodeBinding> bindings;
  std::vector<g::ValueId> outputs;
};
// A second page node deliberately shares the exact external page inputs.
// Its independent recipe/slot and different scale must still have one full
// layout set; neither node may widen the permission of the shared inputs.
inline Result<GraphData> graph(const r::Inputs& data,bool two=false,bool empty=false,unsigned misuse=0){
  GraphData result;
  HYLE_ASSIGN_OR_RETURN(result.query_desc,c::TensorDesc::dense(c::DataType::kFloat32,data.recipe.descriptions()[0].dimensions()));
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(result.query,builder.input(result.query_desc));
  HYLE_ASSIGN_OR_RETURN(result.key,builder.input(data.recipe.descriptions()[1]));
  HYLE_ASSIGN_OR_RETURN(result.value,builder.input(data.recipe.descriptions()[2]));
  std::vector<g::NodeId> cpu_nodes;
  const std::array query_port{result.query};
  HYLE_ASSIGN_OR_RETURN(auto absolute,builder.add_node(c::Operation(c::Unary{c::UnaryKind::kAbs}),query_port));
  cpu_nodes.push_back(absolute);HYLE_ASSIGN_OR_RETURN(auto qabs,builder.result(absolute));
  const std::array abs_port{qabs};
  HYLE_ASSIGN_OR_RETURN(auto cast,builder.add_node(c::Operation(c::Cast{c::DataType::kFloat16}),abs_port));
  cpu_nodes.push_back(cast);HYLE_ASSIGN_OR_RETURN(auto ready,builder.result(cast));
  auto layout=data.layout;
  if(empty){
    const std::array<std::int32_t,3> ptr{0,0,0};const std::array<std::int32_t,2> tails{0,0};
    HYLE_ASSIGN_OR_RETURN(layout,data.schema.version({ptr,ptr,{},tails}));
  }
  std::vector<g::ValueId> attended;
  for(unsigned index=0;index<(two?2U:1U);++index){
    auto recipe=data.recipe;
    if(index){HYLE_ASSIGN_OR_RETURN(recipe,c::NativePagedBatchProgram::prepare(data.schema,{.max_host_metadata_bytes=4096,.scale=.25F}));}
    HYLE_ASSIGN_OR_RETURN(auto descriptor,g::batch::native_paged_attention(recipe,layout));
    const std::array ports{ready,result.key,result.value};
    HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(c::Operation::custom(descriptor),ports));
    result.bindings.push_back({node,std::move(recipe),layout,{.max_metadata_backing_bytes=4096}});
    HYLE_ASSIGN_OR_RETURN(auto out,builder.result(node));attended.push_back(out);
  }
  for(auto value:attended){
    const std::array source{value};
    HYLE_ASSIGN_OR_RETURN(auto wide,builder.add_node(c::Operation(c::Cast{c::DataType::kFloat32}),source));
    cpu_nodes.push_back(wide);HYLE_ASSIGN_OR_RETURN(auto widened,builder.result(wide));
    const std::array expanded{widened};
    HYLE_ASSIGN_OR_RETURN(auto negative,builder.add_node(c::Operation(c::Unary{c::UnaryKind::kNeg}),expanded));
    cpu_nodes.push_back(negative);HYLE_ASSIGN_OR_RETURN(auto output,builder.result(negative));
    HYLE_RETURN_IF_ERROR(builder.publish(output));result.outputs.push_back(output);
  }
  if(misuse==1)HYLE_RETURN_IF_ERROR(builder.publish(result.key));
  if(misuse==2){
    HYLE_ASSIGN_OR_RETURN(auto dense,builder.add(c::Cast{c::DataType::kFloat32},result.key));
    HYLE_RETURN_IF_ERROR(builder.publish(dense));
  }
  HYLE_ASSIGN_OR_RETURN(result.graph,std::move(builder).finish());
  for(auto node:cpu_nodes)result.placements.push_back({node,joint::kCpuTarget,g::PlacementStrength::kRequired});
  return result;
}
inline joint::Environment environment(c::Context& cpu,n::Context& native,n::MemoryResource& device,n::MemoryResource& staging){
#if defined(HYLE_JOINT_PAGE_APPLE)
  (void)staging;return {.cpu=&cpu,.apple=&native,.native=&device};
#else
  return {.cpu=&cpu,.vulkan=&native,.device=&device,.staging=&staging};
#endif
}
inline Result<joint::Executable> compile(const GraphData& graph,joint::Environment env,
    hyle::mem::AllocatorRef allocator=hyle::mem::system_allocator(),c::ResourceBudget budget={}){
  auto bindings=graph.bindings;
  for(auto& binding:bindings){binding.options.allocator=allocator;binding.options.resource_budget=budget;}
  return joint::compile(graph.graph,env,{.node_placements=graph.placements,.resource_budget=budget,.allocator=allocator,.paged_nodes=bindings});
}
struct HostQuery final {c::LeasePool pool;c::ReadLease read;};
inline Result<HostQuery> query(const GraphData& graph){
  HYLE_ASSIGN_OR_RETURN(auto bytes,graph.query_desc.required_bytes());
  HostQuery result;HYLE_ASSIGN_OR_RETURN(result.pool,c::LeasePool::create(hyle::mem::system_resource(),bytes+512));
  HYLE_ASSIGN_OR_RETURN(auto writer,result.pool.allocate(bytes,64));
  std::vector<float> values(bytes/4);
  for(std::size_t i=0;i<values.size();++i)values[i]=float(int(i%11)-5)/32.F;
  HYLE_RETURN_IF_ERROR(writer.view()->write(std::as_bytes(std::span(values))));
  HYLE_ASSIGN_OR_RETURN(result.read,writer.publish());return result;
}
inline Status prepare(joint::Invocation& slot,const GraphData& graph,const r::Inputs& data,const HostQuery& source){
  std::vector<g::batch::NodeLayout> layouts;
  for(const auto& binding:graph.bindings)layouts.push_back({binding.node,binding.layout});
  HYLE_RETURN_IF_ERROR(slot.prepare_batch(layouts));
  HYLE_RETURN_IF_ERROR(slot.bind(graph.query,graph.query_desc,source.read));
  if(graph.bindings[0].layout.facts().query_rows){
    HYLE_RETURN_IF_ERROR(slot.bind(graph.key,data.key_frame));
    HYLE_RETURN_IF_ERROR(slot.bind(graph.value,data.value_frame));
  }
  return slot.prepare_replay();
}
inline Status result(joint::Invocation& slot,const GraphData& graph,bool empty=false){
  HYLE_ASSIGN_OR_RETURN(auto done,slot.run());
  for(auto id:graph.outputs){
    HYLE_ASSIGN_OR_RETURN(auto output,done->output(id));
    HYLE_ASSIGN_OR_RETURN(auto bytes,output.description().required_bytes());
    HYLE_RETURN_IF_ERROR(require(empty?bytes==0:bytes>0,"joint result active size differs"));
    std::vector<float> values(bytes/4);HYLE_RETURN_IF_ERROR(output.storage().read(std::as_writable_bytes(std::span(values))));
    for(float value:values)HYLE_RETURN_IF_ERROR(require(std::isfinite(value) && value<=0,"joint result nonfinite or missed real CPU negative suffix"));
  }
  return Status::Ok();
}
inline Status domain_checks(c::Context& cpu,n::Context& native,n::MemoryResource& device,n::MemoryResource& staging,r::Inputs& data){
  auto env=environment(cpu,native,device,staging);
  HYLE_ASSIGN_OR_RETURN(auto shape,graph(data));
  HYLE_RETURN_IF_ERROR(require(!joint::compile(shape.graph,env,{.node_placements=shape.placements}),"live page node without cold binding accepted"));
  auto duplicate=shape.bindings;duplicate.push_back(duplicate[0]);
  HYLE_RETURN_IF_ERROR(require(!joint::compile(shape.graph,env,{.node_placements=shape.placements,.paged_nodes=duplicate}),"duplicate cold page node accepted"));
  HYLE_ASSIGN_OR_RETURN(auto foreign,c::BatchSchema::create(data.schema.capacity()));
  const auto host=data.layout.host();HYLE_ASSIGN_OR_RETURN(auto foreign_layout,foreign.version(host));
  auto wrong=shape.bindings;wrong[0].layout=foreign_layout;
  HYLE_RETURN_IF_ERROR(require(!joint::compile(shape.graph,env,{.node_placements=shape.placements,.paged_nodes=wrong}),"foreign cold schema accepted"));
  HYLE_ASSIGN_OR_RETURN(auto foreign_graph,graph(data));wrong=shape.bindings;wrong[0].node=foreign_graph.bindings[0].node;
  HYLE_RETURN_IF_ERROR(require(!joint::compile(shape.graph,env,{.node_placements=shape.placements,.paged_nodes=wrong}),"foreign graph page node accepted"));
  for(unsigned misuse:{1U,2U}){
    HYLE_ASSIGN_OR_RETURN(auto bad,graph(data,false,false,misuse));
    HYLE_RETURN_IF_ERROR(require(!compile(bad,env),"page capacity publisher or ordinary dense reader accepted"));
  }
  HYLE_ASSIGN_OR_RETURN(auto two,graph(data,true));HYLE_ASSIGN_OR_RETURN(auto program,compile(two,env));
  auto mixed_cold=two.bindings;mixed_cold[1].layout=data.next;
  HYLE_RETURN_IF_ERROR(require(!joint::compile(two.graph,env,{.node_placements=two.placements,.paged_nodes=mixed_cold}),"shared page cold inputs accepted different owned layouts"));
  auto cpu_page=shape.placements;cpu_page.push_back({shape.bindings[0].node,joint::kCpuTarget,g::PlacementStrength::kRequired});
  HYLE_RETURN_IF_ERROR(require(!joint::compile(shape.graph,env,{.node_placements=cpu_page,.paged_nodes=shape.bindings}),"page node implicitly fell back to CPU"));
  hyle::sched::Scheduler scheduler(1,hyle::mem::system_allocator(),32);Drain drain{scheduler};
  HYLE_ASSIGN_OR_RETURN(auto queue,cpu.create_queue());HYLE_ASSIGN_OR_RETURN(auto source,query(two));
  HYLE_ASSIGN_OR_RETURN(auto slot,program.create_invocation(scheduler,queue));
  const std::array incomplete{g::batch::NodeLayout{two.bindings[0].node,data.layout}};
  const std::array mismatched{incomplete[0],g::batch::NodeLayout{two.bindings[1].node,data.next}};
  HYLE_RETURN_IF_ERROR(require(!slot.prepare_batch(incomplete).ok() && !slot.submit(),"incomplete page set admitted"));
  HYLE_RETURN_IF_ERROR(require(!slot.prepare_batch(mismatched).ok() && !slot.submit(),"shared page input mixed layouts admitted"));
  const std::array repeated{incomplete[0],incomplete[0]};
  HYLE_RETURN_IF_ERROR(require(!slot.prepare_batch(repeated).ok(),"duplicate page preparation node admitted"));
  const std::array foreign_versions{g::batch::NodeLayout{two.bindings[0].node,foreign_layout},g::batch::NodeLayout{two.bindings[1].node,foreign_layout}};
  HYLE_RETURN_IF_ERROR(require(!slot.prepare_batch(foreign_versions).ok(),"foreign runtime schema accepted"));
  const std::array<std::int32_t,3> one_query{0,1,1};
  HYLE_ASSIGN_OR_RETURN(auto changed_shape,data.schema.version({one_query,host.kv_indptr,host.page_indices,host.last_page_lengths}));
  const std::array changed_versions{g::batch::NodeLayout{two.bindings[0].node,changed_shape},g::batch::NodeLayout{two.bindings[1].node,changed_shape}};
  HYLE_RETURN_IF_ERROR(require(!slot.prepare_batch(changed_versions).ok(),"page preparation changed fixed active output shape"));
  HYLE_RETURN_IF_ERROR(prepare(slot,two,data,source));
  HYLE_RETURN_IF_ERROR(require(slot.bind(two.key,data.next_keys).IsInvalidArgument(),"wrong frame generation accepted"));
  HYLE_RETURN_IF_ERROR(require(slot.bind(two.key,c::PagedArenaRead{}).IsInvalidArgument(),"empty frame accepted for live page episode"));
  HYLE_RETURN_IF_ERROR(require(slot.bind(two.query,data.key_frame).IsFailedPrecondition(),"query port accepted a page frame"));
  HYLE_RETURN_IF_ERROR(require(!slot.bind(two.key,data.recipe.descriptions()[1],data.key_reads[0]).ok(),"raw lease widened arena permission"));
  HYLE_ASSIGN_OR_RETURN(auto other,c::PagedArena::create(data.schema,c::BatchCacheRole::kKey,hyle::mem::system_resource(),{.max_backing_bytes=1<<20}));
  std::array<c::ReadLease,2> other_reads;
  for(unsigned i=0;i<2;++i){HYLE_ASSIGN_OR_RETURN(auto writer,other.acquire(i?2:0));HYLE_RETURN_IF_ERROR(writer.lease().view()->fill(std::byte{0}));HYLE_ASSIGN_OR_RETURN(other_reads[i],writer.publish(data.schema.capacity().page_size));}
  HYLE_ASSIGN_OR_RETURN(auto wrong_resource,other.read(data.layout,other_reads));
  HYLE_RETURN_IF_ERROR(require(slot.bind(two.key,wrong_resource).IsInvalidArgument(),"foreign page resource accepted"));
  HYLE_RETURN_IF_ERROR(slot.prepare_replay());HYLE_RETURN_IF_ERROR(result(slot,two));
  HYLE_RETURN_IF_ERROR(require(source.pool.stats().active_uses==0,"completed joint query use retained"));
  HYLE_RETURN_IF_ERROR(slot.close());slot={};
  std::printf("%s joint two page nodes and exact refusal recovery PASS; starting empty output\n",r::route);std::fflush(stdout);
  HYLE_ASSIGN_OR_RETURN(auto zero,graph(data,false,true));HYLE_ASSIGN_OR_RETURN(auto empty_program,compile(zero,env));
  HYLE_ASSIGN_OR_RETURN(auto empty_query,query(zero));HYLE_ASSIGN_OR_RETURN(auto empty,empty_program.create_invocation(scheduler,queue));
  HYLE_RETURN_IF_ERROR(prepare(empty,zero,data,empty_query));
  HYLE_RETURN_IF_ERROR(require(empty.bind(zero.key,data.key_frame).IsInvalidArgument(),"empty episode accepted a live frame"));
  HYLE_RETURN_IF_ERROR(empty.prepare_replay());HYLE_RETURN_IF_ERROR(result(empty,zero,true));HYLE_RETURN_IF_ERROR(empty.close());
  std::printf("%s joint exact readers/node/layout/resource/refusal/2-page-nodes/empty PASS\n",r::route);return Status::Ok();
}
inline Status cold_failures(c::Context& cpu,n::Context& native,n::MemoryResource& device,n::MemoryResource& staging,r::Inputs& data){
  HYLE_ASSIGN_OR_RETURN(auto shape,graph(data));auto env=environment(cpu,native,device,staging);
  hyle_f5_fault_fixture::Allocator fault;
  {HYLE_ASSIGN_OR_RETURN(auto baseline,compile(shape,env,fault.ref()));}
  const auto compile_count=fault.attempts;HYLE_RETURN_IF_ERROR(require(fault.live==0,"joint baseline compile metadata leaked"));
  for(std::size_t nth=0;nth<compile_count;++nth){
    fault.attempts=0;fault.fail=nth;auto made=compile(shape,env,fault.ref());
    HYLE_RETURN_IF_ERROR(require(!made && made.error().IsResourceExhausted() && fault.live==0,"joint cold compile allocation rejection leaked or published"));
  }
  fault.fail=SIZE_MAX;fault.attempts=0;HYLE_ASSIGN_OR_RETURN(auto program,compile(shape,env,fault.ref()));
  hyle::sched::Scheduler scheduler(1,hyle::mem::system_allocator(),32);Drain drain{scheduler};HYLE_ASSIGN_OR_RETURN(auto queue,cpu.create_queue());
  const auto plan_live=fault.live;fault.attempts=0;
  {HYLE_ASSIGN_OR_RETURN(auto baseline,program.create_invocation(scheduler,queue));}
  const auto create_count=fault.attempts;HYLE_RETURN_IF_ERROR(require(fault.live==plan_live,"joint baseline invocation metadata leaked"));
  for(std::size_t nth=0;nth<create_count;++nth){
    fault.attempts=0;fault.fail=nth;auto made=program.create_invocation(scheduler,queue);
    HYLE_RETURN_IF_ERROR(require(!made && made.error().IsResourceExhausted() && fault.live==plan_live,"joint cold invocation allocation rejection leaked or published"));
  }
  fault.fail=SIZE_MAX;
  HYLE_ASSIGN_OR_RETURN(auto source,query(shape));
  const std::array versions{g::batch::NodeLayout{shape.bindings[0].node,data.layout}};
  fault.attempts=0;
  {HYLE_ASSIGN_OR_RETURN(auto baseline,program.create_invocation(scheduler,queue));
    fault.attempts=0;HYLE_RETURN_IF_ERROR(baseline.prepare_batch(versions));}
  const auto prepare_count=fault.attempts;
  HYLE_RETURN_IF_ERROR(require(fault.live==plan_live,"joint baseline prepare metadata leaked"));
  for(std::size_t nth=0;nth<prepare_count;++nth){
    fault.fail=SIZE_MAX;
    {HYLE_ASSIGN_OR_RETURN(auto slot,program.create_invocation(scheduler,queue));
      fault.attempts=0;fault.fail=nth;const auto denied=slot.prepare_batch(versions);
      HYLE_RETURN_IF_ERROR(require(denied.IsResourceExhausted() && !slot.submit(),"joint cold prepare OOM published an admitted episode"));
      fault.fail=SIZE_MAX;slot.close().IgnoreError();}
    HYLE_RETURN_IF_ERROR(require(fault.live==plan_live,"joint cold prepare OOM leaked metadata"));
  }
  fault.fail=SIZE_MAX;std::size_t bind_count=0;
  {HYLE_ASSIGN_OR_RETURN(auto baseline,program.create_invocation(scheduler,queue));
    HYLE_RETURN_IF_ERROR(prepare(baseline,shape,data,source));fault.attempts=0;
    HYLE_RETURN_IF_ERROR(baseline.bind(shape.key,data.key_frame));bind_count=fault.attempts;}
  HYLE_RETURN_IF_ERROR(require(fault.live==plan_live,"joint baseline exact-frame replacement leaked"));
  for(std::size_t nth=0;nth<bind_count;++nth){
    fault.fail=SIZE_MAX;
    {HYLE_ASSIGN_OR_RETURN(auto slot,program.create_invocation(scheduler,queue));
      HYLE_RETURN_IF_ERROR(prepare(slot,shape,data,source));fault.attempts=0;fault.fail=nth;
      const auto denied=slot.bind(shape.key,data.key_frame);fault.fail=SIZE_MAX;
      HYLE_RETURN_IF_ERROR(require(denied.IsResourceExhausted(),"joint exact-frame allocation failure was accepted"));
      HYLE_RETURN_IF_ERROR(slot.prepare_replay());HYLE_RETURN_IF_ERROR(result(slot,shape));HYLE_RETURN_IF_ERROR(slot.close());}
    HYLE_RETURN_IF_ERROR(require(fault.live==plan_live,"joint rejected frame replacement leaked or lost old binding"));
  }
  auto capped=shape;for(auto& binding:capped.bindings)binding.options.max_metadata_backing_bytes=236;
  HYLE_ASSIGN_OR_RETURN(auto narrow,joint::compile(capped.graph,env,{.node_placements=capped.placements,.paged_nodes=capped.bindings}));
  HYLE_RETURN_IF_ERROR(require(!narrow.create_invocation(scheduler,queue),"joint fixed page backing budget ignored"));
  auto no_host=env;no_host.host_workspace_budget_bytes=0;
  auto host_refused=compile(shape,no_host);
  if(host_refused)HYLE_RETURN_IF_ERROR(require(!host_refused->create_invocation(scheduler,queue),"joint host workspace limit ignored"));
  const std::array limits{c::ResourceBudgetLimit{hyle::mem::system_resource(),4096},c::ResourceBudgetLimit{device.ref(),SIZE_MAX},c::ResourceBudgetLimit{staging.ref(),SIZE_MAX}};
  HYLE_ASSIGN_OR_RETURN(auto budget,c::ResourceBudget::create({.limits=limits,.allow_unattributed=true}));
  HYLE_ASSIGN_OR_RETURN(auto usage,budget.usage(hyle::mem::system_resource()));
  HYLE_ASSIGN_OR_RETURN(auto exhausted,budget.reserve(hyle::mem::system_resource(),usage.limit_bytes-usage.live_bytes,c::ResourceBudgetCategory::kWorkspace));
  HYLE_RETURN_IF_ERROR(require(!compile(shape,env,hyle::mem::system_allocator(),budget),"joint shared metadata budget ignored"));
  std::printf("%s joint cold compile %zu / create %zu / prepare %zu / bind %zu injected OOM + backing/host/shared budget PASS\n",r::route,compile_count,create_count,prepare_count,bind_count);
  return Status::Ok();
}
} // namespace hyle_joint_page_risk
