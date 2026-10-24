#pragma once
#if defined(HYLE_JOINT_PAGE_APPLE)
#define HYLE_G01_PAGE_RISK_APPLE 1
#include <hyle/graph/apple.h>
#else
#include <hyle/graph/vulkan.h>
#endif
#include <hyle/graph/cpu.h>
#include <hyle/graph/batch.h>
#include <hyle/graph/progress.h>
#include <hyle/sched/blocking.h>
#include <hyle/sched/scheduler.h>
#include "../compute_batch/native_paged_risk_fixture.h"
#include <cmath>

namespace hyle_joint_page_risk {
namespace r=hyle_g01_page_risk;
namespace c=hyle::compute;
namespace g=hyle::graph;
namespace n=r::n;
#if defined(HYLE_JOINT_PAGE_APPLE)
namespace route=g::apple;
#else
namespace route=g::vulkan;
#endif
using hyle::Status;using hyle::Result;
using r::require;
inline constexpr g::ExecutionTargetId cpu_target{0},native_target{1};
inline constexpr g::StorageDomainId host_storage{0},native_storage{1};
struct Drain final {hyle::sched::Scheduler& scheduler;~Drain(){scheduler.drain();}};
struct GraphData final {
  g::Graph graph;
  g::ValueId query,key,value;
  c::TensorDesc query_desc;
  std::vector<g::NodePlacement> placements;
  std::vector<g::batch::NodeLayout> layouts;
  std::vector<g::ValueId> outputs;
};
// A second page node deliberately shares the exact external page inputs.
// Its independent recipe/slot and different scale must still have one full
// layout set; neither node may widen the permission of the shared inputs.
inline Result<GraphData> graph(const r::Inputs& data,bool two=false,bool empty=false,unsigned misuse=0){
  GraphData result;
  auto layout=data.layout;
  if(empty){
    const std::array<std::int32_t,3> ptr{0,0,0};const std::array<std::int32_t,2> tails{0,0};
    HYLE_ASSIGN_OR_RETURN(layout,data.schema.version({ptr,ptr,{},tails}));
  }
  HYLE_ASSIGN_OR_RETURN(auto query_description,data.recipe.input_description(layout,0));
  HYLE_ASSIGN_OR_RETURN(auto key_description,data.recipe.input_description(layout,1));
  HYLE_ASSIGN_OR_RETURN(auto value_description,data.recipe.input_description(layout,2));
  HYLE_ASSIGN_OR_RETURN(result.query_desc,c::TensorDesc::dense(c::DataType::kFloat32,query_description.dimensions()));
  HYLE_ASSIGN_OR_RETURN(auto builder,g::GraphBuilder::create());
  HYLE_ASSIGN_OR_RETURN(result.query,builder.input(result.query_desc));
  HYLE_ASSIGN_OR_RETURN(result.key,builder.input(key_description));
  HYLE_ASSIGN_OR_RETURN(result.value,builder.input(value_description));
  std::vector<g::NodeId> cpu_nodes;
  const std::array query_port{result.query};
  HYLE_ASSIGN_OR_RETURN(auto absolute,builder.add_node(c::Operation(c::Unary{c::UnaryKind::kAbs}),query_port));
  cpu_nodes.push_back(absolute);HYLE_ASSIGN_OR_RETURN(auto qabs,builder.result(absolute));
  const std::array abs_port{qabs};
  HYLE_ASSIGN_OR_RETURN(auto cast,builder.add_node(c::Operation(c::Cast{c::DataType::kFloat16}),abs_port));
  cpu_nodes.push_back(cast);HYLE_ASSIGN_OR_RETURN(auto ready,builder.result(cast));
  std::vector<g::ValueId> attended;
  for(unsigned index=0;index<(two?2U:1U);++index){
    auto recipe=data.recipe;
    if(index){HYLE_ASSIGN_OR_RETURN(recipe,c::NativePagedBatchProgram::prepare(data.schema,{.max_host_metadata_bytes=4096,.scale=.25F}));}
    HYLE_ASSIGN_OR_RETURN(auto descriptor,g::batch::paged_attention(layout,
        {.scale=recipe.scale(),.causal=recipe.causal(),.window_left=recipe.window_left()}));
    const std::array ports{ready,result.key,result.value};
    HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(c::Operation::custom(descriptor),ports));
    result.layouts.push_back({node,layout});
    result.placements.push_back({node,native_target,g::PlacementStrength::kRequired});
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
  for(auto node:cpu_nodes)result.placements.push_back({node,cpu_target,g::PlacementStrength::kRequired});
  return result;
}
// Test-owned cold environment only; all work uses the ordinary Invocation.
struct Environment final {
  std::array<g::StorageDomain,3> domains;
  std::array<g::ExecutionTarget,2> targets;
};
inline Result<Environment> environment(c::Context& cpu,n::Context& native,n::MemoryResource& device,n::MemoryResource& staging){
  HYLE_ASSIGN_OR_RETURN(auto host,g::ExecutionSource::acquire(cpu));
  HYLE_ASSIGN_OR_RETURN(auto gpu,route::execution_source(native));
  return Environment{{g::StorageDomain{hyle::mem::system_resource()},g::StorageDomain{device.ref()},g::StorageDomain{staging.ref()}},
      {g::ExecutionTarget{host,host_storage},g::ExecutionTarget{gpu,native_storage}}};
}
inline Result<g::CompiledGraph> compile(const GraphData& shape,const Environment& env,
    hyle::mem::AllocatorRef allocator=hyle::mem::system_allocator(),c::ResourceBudget budget={}){
  std::vector<g::ExternalResourceContract> external{{shape.query,host_storage},{shape.key,native_storage},{shape.value,native_storage}};
  for(auto output:shape.outputs)external.push_back({output,host_storage});
  // The common compiler takes its metadata allocator from the actual shared
  // scope. This is the public fault-injection boundary, not a second compiler.
  if(!budget){
    std::vector<c::ResourceBudgetLimit> limits;
    for(const auto&domain:env.domains)limits.push_back({domain.resource,SIZE_MAX});
    HYLE_ASSIGN_OR_RETURN(budget,c::ResourceBudget::create({.limits=limits,.allocator=allocator,.allow_unattributed=true}));
  }
  return g::batch::compile(shape.graph,{.storage_domains=env.domains,.targets=env.targets},
      {.resource_budget=budget,.node_placements=shape.placements,.external_resources=external});
}
inline Result<std::vector<g::QueueBinding>> queues(const g::CompileReport& report,c::Queue& queue){
  std::vector<g::QueueBinding> result;
  for(std::size_t i=0;i<report.lane_count();++i){HYLE_ASSIGN_OR_RETURN(auto lane,report.lane(i));
    if(lane.queue_required)result.push_back({lane.lane,&queue});}
  return result;
}
inline Result<g::Invocation> create(const g::CompiledGraph& program,hyle::sched::Scheduler& scheduler,
    hyle::sched::BlockingPool& workers,hyle::mem::AllocatorRef allocator=hyle::mem::system_allocator(),c::ResourceBudget budget={}){
  HYLE_ASSIGN_OR_RETURN(auto slot,program.executable.create_invocation({.resources=g::ResourceOptions{allocator,budget}}));
  HYLE_RETURN_IF_ERROR(g::prepare_progress(slot,scheduler,workers));return slot;
}
using OutputPools=std::vector<c::LeasePool>;
inline Result<OutputPools> output_pools(const GraphData& shape){
  OutputPools result;
  for(auto output:shape.outputs){HYLE_ASSIGN_OR_RETURN(auto desc,shape.graph.value_description(output));
    HYLE_ASSIGN_OR_RETURN(auto bytes,desc.required_bytes());
    HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(hyle::mem::system_resource(),bytes+128,{.max_blocks=1}));result.push_back(std::move(pool));}
  return result;
}
inline Status bind_outputs(g::Invocation& slot,const GraphData& shape,OutputPools& pools){
  for(std::size_t i=0;i<shape.outputs.size();++i){HYLE_ASSIGN_OR_RETURN(auto desc,shape.graph.value_description(shape.outputs[i]));
    HYLE_ASSIGN_OR_RETURN(auto bytes,desc.required_bytes());HYLE_ASSIGN_OR_RETURN(auto writer,pools[i].allocate(std::max<std::size_t>(bytes,1),64));
    HYLE_RETURN_IF_ERROR(slot.bind(shape.outputs[i],desc,std::move(writer)));}
  return Status::Ok();
}
struct HostQuery final {c::LeasePool pool;c::ReadLease read;};
inline Result<HostQuery> query(const GraphData& graph){
  HYLE_ASSIGN_OR_RETURN(auto bytes,graph.query_desc.required_bytes());
  HostQuery result;HYLE_ASSIGN_OR_RETURN(result.pool,c::LeasePool::create(hyle::mem::system_resource(),bytes+512));
  HYLE_ASSIGN_OR_RETURN(auto writer,result.pool.allocate(std::max<std::size_t>(bytes,1),64));
  std::vector<float> values(bytes/4);
  for(std::size_t i=0;i<values.size();++i)values[i]=float(int(i%11)-5)/32.F;
  if(bytes)HYLE_RETURN_IF_ERROR(writer.view()->write(std::as_bytes(std::span(values))));
  HYLE_ASSIGN_OR_RETURN(result.read,writer.publish());return result;
}
inline Status prepare(g::Invocation& slot,std::span<const g::QueueBinding> lanes,const GraphData& shape,
    const r::Inputs& data,const HostQuery& source,OutputPools& outputs,
    hyle::mem::AllocatorRef allocator=hyle::mem::system_allocator(),c::ResourceBudget budget={}){
  HYLE_RETURN_IF_ERROR(g::batch::prepare(slot,lanes,shape.layouts,{.allocator=allocator,.resource_budget=budget}));
  HYLE_RETURN_IF_ERROR(slot.bind(shape.query,shape.query_desc,source.read));
  if(shape.layouts[0].layout.facts().query_rows){
    HYLE_RETURN_IF_ERROR(g::batch::bind(slot,shape.key,data.key_frame));HYLE_RETURN_IF_ERROR(g::batch::bind(slot,shape.value,data.value_frame));
  }else{
    HYLE_ASSIGN_OR_RETURN(auto keys,data.keys.read(shape.layouts[0].layout,{}));
    HYLE_ASSIGN_OR_RETURN(auto values,data.values.read(shape.layouts[0].layout,{}));
    HYLE_RETURN_IF_ERROR(g::batch::bind(slot,shape.key,std::move(keys)));HYLE_RETURN_IF_ERROR(g::batch::bind(slot,shape.value,std::move(values)));
  }
  HYLE_RETURN_IF_ERROR(bind_outputs(slot,shape,outputs));return slot.prepare_replay(lanes);
}
inline Status result(g::Invocation& slot,std::span<const g::QueueBinding> lanes,const GraphData& shape,bool empty=false){
  HYLE_ASSIGN_OR_RETURN(auto done,slot.submit(lanes));HYLE_RETURN_IF_ERROR(done.wait());
  for(auto id:shape.outputs){HYLE_ASSIGN_OR_RETURN(auto output,slot.result(id));HYLE_ASSIGN_OR_RETURN(auto tensor,output.tensor());
    HYLE_ASSIGN_OR_RETURN(auto bytes,tensor.description().required_bytes());
    HYLE_RETURN_IF_ERROR(require(empty?bytes==0:bytes>0,"Graph result active size differs"));
    std::vector<float> values(bytes/4);HYLE_RETURN_IF_ERROR(tensor.storage().read(std::as_writable_bytes(std::span(values))));
    for(float value:values)HYLE_RETURN_IF_ERROR(require(std::isfinite(value) && value<=0,"Graph result nonfinite or missed real CPU negative suffix"));}
  return Status::Ok();
}
inline Status domain_checks(c::Context& cpu,n::Context& native,n::MemoryResource& device,n::MemoryResource& staging,r::Inputs& data){
  HYLE_ASSIGN_OR_RETURN(auto env,environment(cpu,native,device,staging));
  HYLE_ASSIGN_OR_RETURN(auto shape,graph(data));
  HYLE_ASSIGN_OR_RETURN(auto foreign,c::BatchSchema::create(data.schema.capacity()));
  const auto host=data.layout.host();HYLE_ASSIGN_OR_RETURN(auto foreign_layout,foreign.version(host));
  HYLE_ASSIGN_OR_RETURN(auto two,graph(data,true));HYLE_ASSIGN_OR_RETURN(auto program,compile(two,env));
  for(std::size_t i=0;i<program.report.step_count();++i){HYLE_ASSIGN_OR_RETURN(auto step,program.report.step(i));
    for(std::size_t j=0;j<step.node_count;++j){HYLE_ASSIGN_OR_RETURN(auto node,program.report.step_node(i,j));
      if(std::ranges::any_of(two.layouts,[&](const auto&entry){return entry.node==node;}))
        HYLE_RETURN_IF_ERROR(require(step.target==native_target && step.runtime.mode==c::ProviderRuntimeMode::kDelegated,
            "required native page silently selected another implementation"));}}
  hyle::sched::Scheduler scheduler(1,hyle::mem::system_allocator(),32);
  hyle::sched::BlockingPool workers(hyle::mem::system_allocator(),1);
  Drain drain{scheduler};
  HYLE_ASSIGN_OR_RETURN(auto queue,cpu.create_queue());HYLE_ASSIGN_OR_RETURN(auto source,query(two));
  HYLE_ASSIGN_OR_RETURN(auto slot,create(program,scheduler,workers));
  HYLE_ASSIGN_OR_RETURN(auto lanes,queues(program.report,queue));HYLE_ASSIGN_OR_RETURN(auto outputs,output_pools(two));
  HYLE_RETURN_IF_ERROR(require(!g::batch::prepare(slot,lanes,{}).ok() && !slot.submit(lanes),"missing owned layout set admitted"));
  for(unsigned misuse:{1U,2U}){
    HYLE_ASSIGN_OR_RETURN(auto bad,graph(data,false,false,misuse));auto made=compile(bad,env);
    if(made){HYLE_ASSIGN_OR_RETURN(auto denied,create(*made,scheduler,workers));HYLE_ASSIGN_OR_RETURN(auto bad_lanes,queues(made->report,queue));
      HYLE_RETURN_IF_ERROR(g::batch::prepare(denied,bad_lanes,bad.layouts));
      HYLE_RETURN_IF_ERROR(require(!g::batch::bind(denied,bad.key,data.key_frame).ok(),"capacity publisher or ordinary extra reader acquired page permission"));}
  }
  HYLE_ASSIGN_OR_RETURN(auto foreign_graph,graph(data,true));auto wrong_nodes=two.layouts;wrong_nodes[0].node=foreign_graph.layouts[0].node;
  HYLE_RETURN_IF_ERROR(require(!g::batch::prepare(slot,lanes,wrong_nodes).ok(),"foreign graph page node accepted"));
  const std::array incomplete{g::batch::NodeLayout{two.layouts[0].node,data.layout}};
  const std::array mismatched{incomplete[0],g::batch::NodeLayout{two.layouts[1].node,data.next}};
  HYLE_RETURN_IF_ERROR(require(!g::batch::prepare(slot,lanes,incomplete).ok() && !slot.submit(lanes),"incomplete page set admitted"));
  HYLE_RETURN_IF_ERROR(require(!g::batch::prepare(slot,lanes,mismatched).ok() && !slot.submit(lanes),"shared page input mixed layouts admitted"));
  const std::array repeated{incomplete[0],incomplete[0]};
  HYLE_RETURN_IF_ERROR(require(!g::batch::prepare(slot,lanes,repeated).ok(),"duplicate page preparation node admitted"));
  const std::array foreign_versions{g::batch::NodeLayout{two.layouts[0].node,foreign_layout},g::batch::NodeLayout{two.layouts[1].node,foreign_layout}};
  HYLE_RETURN_IF_ERROR(require(!g::batch::prepare(slot,lanes,foreign_versions).ok(),"foreign runtime schema accepted"));
  const std::array<std::int32_t,3> one_query{0,1,1};
  HYLE_ASSIGN_OR_RETURN(auto changed_shape,data.schema.version({one_query,host.kv_indptr,host.page_indices,host.last_page_lengths}));
  const std::array changed_versions{g::batch::NodeLayout{two.layouts[0].node,changed_shape},g::batch::NodeLayout{two.layouts[1].node,changed_shape}};
  HYLE_RETURN_IF_ERROR(require(!g::batch::prepare(slot,lanes,changed_versions).ok(),"page preparation changed fixed active output shape"));
  HYLE_RETURN_IF_ERROR(prepare(slot,lanes,two,data,source,outputs));
  HYLE_RETURN_IF_ERROR(require(g::batch::bind(slot,two.key,data.next_keys).IsInvalidArgument(),"wrong frame generation accepted"));
  HYLE_RETURN_IF_ERROR(require(g::batch::bind(slot,two.key,c::PagedArenaRead{}).IsInvalidArgument(),"empty frame accepted for live page episode"));
  HYLE_RETURN_IF_ERROR(require(g::batch::bind(slot,two.query,data.key_frame).IsInvalidArgument(),"query port accepted a page frame"));
  HYLE_ASSIGN_OR_RETURN(auto key_description,data.recipe.input_description(data.layout,1));
  HYLE_RETURN_IF_ERROR(require(!slot.bind(two.key,key_description,data.key_reads[0]).ok(),"raw lease widened arena permission"));
  HYLE_ASSIGN_OR_RETURN(auto other,c::PagedArena::create(data.schema,c::BatchCacheRole::kKey,hyle::mem::system_resource(),{.max_backing_bytes=1<<20}));
  std::array<c::ReadLease,2> other_reads;
  for(unsigned i=0;i<2;++i){HYLE_ASSIGN_OR_RETURN(auto writer,other.acquire(i?2:0));HYLE_RETURN_IF_ERROR(writer.lease().view()->fill(std::byte{0}));HYLE_ASSIGN_OR_RETURN(other_reads[i],writer.publish(data.schema.capacity().page_size));}
  HYLE_ASSIGN_OR_RETURN(auto wrong_resource,other.read(data.layout,other_reads));
  HYLE_RETURN_IF_ERROR(require(g::batch::bind(slot,two.key,wrong_resource).IsInvalidArgument(),"foreign page resource accepted"));
  HYLE_RETURN_IF_ERROR(slot.prepare_replay(lanes));HYLE_RETURN_IF_ERROR(result(slot,lanes,two));
  HYLE_RETURN_IF_ERROR(require(source.pool.stats().active_uses==0,"completed joint query use retained"));
  HYLE_RETURN_IF_ERROR(slot.close());slot={};
  std::printf("%s joint two page nodes and exact refusal recovery PASS; starting empty output\n",r::route);std::fflush(stdout);
  HYLE_ASSIGN_OR_RETURN(auto zero,graph(data,false,true));HYLE_ASSIGN_OR_RETURN(auto empty_program,compile(zero,env));
  HYLE_ASSIGN_OR_RETURN(auto empty_query,query(zero));
  HYLE_ASSIGN_OR_RETURN(auto empty,create(empty_program,scheduler,workers));
  HYLE_ASSIGN_OR_RETURN(auto empty_lanes,queues(empty_program.report,queue));HYLE_ASSIGN_OR_RETURN(auto empty_outputs,output_pools(zero));
  HYLE_RETURN_IF_ERROR(prepare(empty,empty_lanes,zero,data,empty_query,empty_outputs));
  HYLE_RETURN_IF_ERROR(require(g::batch::bind(empty,zero.key,data.key_frame).IsInvalidArgument(),"empty episode accepted a live frame"));
  HYLE_RETURN_IF_ERROR(empty.prepare_replay(empty_lanes));HYLE_RETURN_IF_ERROR(result(empty,empty_lanes,zero,true));HYLE_RETURN_IF_ERROR(empty.close());
  std::printf("%s joint exact readers/node/layout/resource/refusal/2-page-nodes/empty PASS\n",r::route);return Status::Ok();
}
inline Status cold_failures(c::Context& cpu,n::Context& native,n::MemoryResource& device,n::MemoryResource& staging,r::Inputs& data){
  HYLE_ASSIGN_OR_RETURN(auto shape,graph(data));HYLE_ASSIGN_OR_RETURN(auto env,environment(cpu,native,device,staging));
  hyle_f5_fault_fixture::Allocator fault;
  {HYLE_ASSIGN_OR_RETURN(auto baseline,compile(shape,env,fault.ref()));}
  const auto compile_count=fault.attempts;HYLE_RETURN_IF_ERROR(require(fault.live==0,"joint baseline compile metadata leaked"));
  for(std::size_t nth=0;nth<compile_count;++nth){
    fault.attempts=0;fault.fail=nth;auto made=compile(shape,env,fault.ref());
    if(made || !made.error().IsResourceExhausted() || fault.live!=0)
      std::fprintf(stderr,"compile OOM nth=%zu/%zu made=%d code=%u live=%zu attempts=%zu message=%.*s\n",
          nth,compile_count,made.has_value(),made?0u:unsigned(made.error().code()),fault.live,fault.attempts,
          made?0:int(made.error().message().size()),made?"":made.error().message().data());
    HYLE_RETURN_IF_ERROR(require(!made && made.error().IsResourceExhausted() && fault.live==0,"joint cold compile allocation rejection leaked or published"));
  }
  fault.fail=SIZE_MAX;fault.attempts=0;HYLE_ASSIGN_OR_RETURN(auto program,compile(shape,env,fault.ref()));
  hyle::sched::Scheduler scheduler(1,hyle::mem::system_allocator(),32);
  hyle::sched::BlockingPool workers(hyle::mem::system_allocator(),1);
  Drain drain{scheduler};HYLE_ASSIGN_OR_RETURN(auto queue,cpu.create_queue());
  HYLE_ASSIGN_OR_RETURN(auto lanes,queues(program.report,queue));HYLE_ASSIGN_OR_RETURN(auto outputs,output_pools(shape));
  const auto plan_live=fault.live;fault.attempts=0;
  {HYLE_ASSIGN_OR_RETURN(auto baseline,create(program,scheduler,workers,fault.ref()));}
  const auto create_count=fault.attempts;HYLE_RETURN_IF_ERROR(require(fault.live==plan_live,"joint baseline invocation metadata leaked"));
  for(std::size_t nth=0;nth<create_count;++nth){
    fault.attempts=0;fault.fail=nth;
    auto made=create(program,scheduler,workers,fault.ref());
    HYLE_RETURN_IF_ERROR(require(!made && made.error().IsResourceExhausted() && fault.live==plan_live,"joint cold invocation allocation rejection leaked or published"));
  }
  fault.fail=SIZE_MAX;
  HYLE_ASSIGN_OR_RETURN(auto source,query(shape));
  const std::array versions{g::batch::NodeLayout{shape.layouts[0].node,data.layout}};
  fault.attempts=0;
  {HYLE_ASSIGN_OR_RETURN(auto baseline,create(program,scheduler,workers,fault.ref()));
    fault.attempts=0;HYLE_RETURN_IF_ERROR(g::batch::prepare(baseline,lanes,versions,{.allocator=fault.ref()}));}
  const auto prepare_count=fault.attempts;
  HYLE_RETURN_IF_ERROR(require(fault.live==plan_live,"joint baseline prepare metadata leaked"));
  for(std::size_t nth=0;nth<prepare_count;++nth){
    fault.fail=SIZE_MAX;
    {HYLE_ASSIGN_OR_RETURN(auto slot,create(program,scheduler,workers,fault.ref()));
      fault.attempts=0;fault.fail=nth;const auto denied=g::batch::prepare(slot,lanes,versions,{.allocator=fault.ref()});
      HYLE_RETURN_IF_ERROR(require(denied.IsResourceExhausted() && !slot.submit(lanes),"joint cold prepare OOM published an admitted episode"));
      fault.fail=SIZE_MAX;slot.close().IgnoreError();}
    HYLE_RETURN_IF_ERROR(require(fault.live==plan_live,"joint cold prepare OOM leaked metadata"));
  }
  fault.fail=SIZE_MAX;std::size_t bind_count=0;
  {HYLE_ASSIGN_OR_RETURN(auto baseline,create(program,scheduler,workers,fault.ref()));
    HYLE_RETURN_IF_ERROR(prepare(baseline,lanes,shape,data,source,outputs,fault.ref()));fault.attempts=0;
    HYLE_RETURN_IF_ERROR(g::batch::bind(baseline,shape.key,data.key_frame));bind_count=fault.attempts;}
  HYLE_RETURN_IF_ERROR(require(fault.live==plan_live,"joint baseline exact-frame replacement leaked"));
  for(std::size_t nth=0;nth<bind_count;++nth){
    fault.fail=SIZE_MAX;
    {HYLE_ASSIGN_OR_RETURN(auto slot,create(program,scheduler,workers,fault.ref()));
      HYLE_RETURN_IF_ERROR(prepare(slot,lanes,shape,data,source,outputs,fault.ref()));fault.attempts=0;fault.fail=nth;
      const auto denied=g::batch::bind(slot,shape.key,data.key_frame);fault.fail=SIZE_MAX;
      HYLE_RETURN_IF_ERROR(require(denied.IsResourceExhausted(),"joint exact-frame allocation failure was accepted"));
      HYLE_RETURN_IF_ERROR(slot.prepare_replay(lanes));HYLE_RETURN_IF_ERROR(result(slot,lanes,shape));HYLE_RETURN_IF_ERROR(slot.close());}
    HYLE_RETURN_IF_ERROR(require(fault.live==plan_live,"joint rejected frame replacement leaked or lost old binding"));
  }
  // The removed per-node metadata cap is now the original shared resource
  // scope. A zero native allowance must reject its actual backing/mirrors.
  const std::array native_limits{c::ResourceBudgetLimit{hyle::mem::system_resource(),SIZE_MAX},c::ResourceBudgetLimit{device.ref(),0},c::ResourceBudgetLimit{staging.ref(),SIZE_MAX}};
  HYLE_ASSIGN_OR_RETURN(auto native_budget,c::ResourceBudget::create({.limits=native_limits,.allow_unattributed=true}));
  auto narrow=compile(shape,env,hyle::mem::system_allocator(),native_budget);
  if(narrow)HYLE_RETURN_IF_ERROR(require(!create(*narrow,scheduler,workers,hyle::mem::system_allocator(),native_budget),"native fixed backing scope budget ignored"));
  auto no_host=env;
  for(auto&domain:no_host.domains)if(domain.resource.properties().has(hyle::mem::MemoryProperty::kHostAccessible))
    domain.workspace_budget_bytes=0;
  auto host_refused=compile(shape,no_host);
  if(host_refused)HYLE_RETURN_IF_ERROR(require(!create(*host_refused,scheduler,workers),"host workspace limit ignored"));
  const std::array limits{c::ResourceBudgetLimit{hyle::mem::system_resource(),4096},c::ResourceBudgetLimit{device.ref(),SIZE_MAX},c::ResourceBudgetLimit{staging.ref(),SIZE_MAX}};
  HYLE_ASSIGN_OR_RETURN(auto budget,c::ResourceBudget::create({.limits=limits,.allow_unattributed=true}));
  HYLE_ASSIGN_OR_RETURN(auto usage,budget.usage(hyle::mem::system_resource()));
  HYLE_ASSIGN_OR_RETURN(auto exhausted,budget.reserve(hyle::mem::system_resource(),usage.limit_bytes-usage.live_bytes,c::ResourceBudgetCategory::kWorkspace));
  HYLE_RETURN_IF_ERROR(require(!compile(shape,env,hyle::mem::system_allocator(),budget),"joint shared metadata budget ignored"));
  std::printf("%s joint cold compile %zu / create %zu / prepare %zu / bind %zu injected OOM + backing/host/shared budget PASS\n",r::route,compile_count,create_count,prepare_count,bind_count);
  return Status::Ok();
}
} // namespace hyle_joint_page_risk
