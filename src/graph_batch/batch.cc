#include "hyle/graph/batch.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/graph/region_provider.h"
#include "hyle/compute/detail/batch_graph_access.h"
#include "hyle/compute/detail/paged_arena_access.h"
#include "hyle/compute/detail/context_ref.h"
#include "hyle/compute/detail/copy_outcome_source.h"
#include "hyle/mem/std_allocator.h"
#include "hyle/util/ref_counted.h"
#include "../compute/provider_internal.h"
#include <algorithm>
#include <array>
#include <mutex>
#include <vector>

namespace hyle::graph::batch {
namespace c=compute;namespace cd=c::detail;namespace gd=graph::detail;
namespace {
struct Config final:util::RefCounted<Config>{
  Config(c::Provider p,c::BatchSchema s,c::BatchProgramOptions o,mem::AllocatorRef a):base(std::move(p)),schema(std::move(s)),options(std::move(o)),allocator(a){}
  void destroy()noexcept{const auto a=allocator;this->~Config();a.deallocate(this,sizeof(*this),alignof(Config));}
  c::Provider base;c::BatchSchema schema;c::BatchProgramOptions options;mem::AllocatorRef allocator;
};
struct Context final {Config*config=nullptr;void*base=nullptr;const cd::ProviderVTable*base_table=nullptr;const cd::BatchProviderVTable*batch_table=nullptr;cd::BatchProviderVTable forwarding{};};
struct Region final {
  explicit Region(Config*p):config(p){p->incref();}
  ~Region(){config->decref();}
  Config*config;NodeId node;ValueId metadata;std::array<ValueId,4> values{};std::array<std::size_t,4> inputs{};
  c::BatchProgramOptions options;
  std::mutex cold_lock;c::BatchProgram program;
};
struct Slot final {Region*region=nullptr;c::BatchInvocation native;c::PreparedBatch token;
  c::BatchLayoutVersion layout;c::PagedArenaRead keys,values;};
bool same_storage(mem::ConstBufferView a,mem::ConstBufferView b)noexcept{
  return a.resource()==b.resource() && a.native_handle()==b.native_handle() && a.offset()==b.offset() && a.size()==b.size();
}
constexpr c::ProviderRuntimeProperties runtime{.mode=c::ProviderRuntimeMode::kSharedQueue,
  .workspace_owner=c::ProviderWorkspaceOwner::kProvider,.replay_class=c::ProviderReplayClass::kManaged,
  .blocks_host=false,.uses_private_threads=false,.supports_concurrent_invocations=true,.capture_compatible=false};
Status supports(const Graph&graph,NodeId node,const Config&config)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto op,graph.operation(node));
  if(const auto value=op.custom_as<c::PagedAttention>()){
    if(value->capacity!=config.schema.capacity() || value->scale!=config.options.scale || value->causal!=config.options.causal)
      return Status::Unimplemented("paged graph node differs from batch domain");
    return Status::Ok();
  }
  if(const auto value=op.custom_as<c::RaggedAttention>()){
    if(value->capacity!=config.schema.capacity() || value->scale!=config.options.scale || value->causal!=config.options.causal)
      return Status::Unimplemented("ragged graph node differs from batch domain");
    return Status::Ok();
  }return Status::Unimplemented("graph batch provider does not support this node");
}
Status propose(void*opaque,const Graph&graph,mem::AllocatorRef,RegionProposalWriter&writer)noexcept{
  const auto&context=*static_cast<Context*>(opaque);
  for(std::size_t i=0;i<graph.node_count();++i){HYLE_ASSIGN_OR_RETURN(auto node,graph.node_id(i));const auto accepted=supports(graph,node,*context.config);
    if(!accepted.ok()){if(accepted.IsUnimplemented())continue;return accepted;}
    const std::array nodes{node};HYLE_ASSIGN_OR_RETURN(auto added,writer.try_add({.nodes=nodes,.route=1,.replay_class=c::ProviderReplayClass::kManaged,.implementation=c::ProviderImplementationId{1}}));if(!added)break;
  }return Status::Ok();
}
Status resources(void*opaque,const RegionCompileRequest&r,mem::AllocatorRef,
    std::span<cd::ResourceBindingRequirement>inputs,std::span<cd::ResourceBindingRequirement>outputs,RegionResourceProposal*proposal)noexcept{
  const auto&context=*static_cast<Context*>(opaque);
  if(!r.graph || r.nodes.size()!=1 || inputs.size()!=4 || outputs.size()!=1 || r.has_unknown_effects || !r.effect_accesses.empty())
    return Status::Unimplemented("batch region requires one pure four-port node");
  HYLE_RETURN_IF_ERROR(supports(*r.graph,r.nodes[0],*context.config));
  // The fixed native domain can use coherent CPU storage as well as device
  // storage. Its provider qualifies that exact resource when preparing the
  // batch program; a GPU-only property here would reject valid CPU regions.
  const c::ResourceRequirement requirement{.minimum_alignment=16,.exact_resource=context.config->options.resource};
  std::fill(inputs.begin(),inputs.end(),requirement);inputs[3].minimum_alignment=8;outputs[0]=requirement;
  *proposal={.runtime=runtime,.workspace_resource=requirement};return Status::Ok();
}
void destroy_region(void*p,mem::AllocatorRef a)noexcept{a.destroy(static_cast<Region*>(p));}
c::AliasSupport alias(void*,std::size_t,std::size_t)noexcept{return {};}
Status create_slot(void*prepared,mem::AllocatorRef a,mem::BufferView,void**out)noexcept{
  *out=a.construct<Slot>();if(!*out)return Status::ResourceExhausted("Graph batch slot metadata");static_cast<Slot*>(*out)->region=static_cast<Region*>(prepared);return Status::Ok();
}
void destroy_slot(void*p,mem::AllocatorRef a)noexcept{a.destroy(static_cast<Slot*>(p));}
cd::EnqueueResult submit_region(void*prepared,void*invocation,c::Queue&queue,std::span<const c::ConstTensorView>inputs,std::span<const c::TensorView>outputs)noexcept{
  const auto&r=*static_cast<Region*>(prepared);auto&s=*static_cast<Slot*>(invocation);
  if(!s.token.valid() || inputs.size()!=4)return cd::EnqueueResult::rejected(Status::FailedPrecondition("Graph batch region has no explicit prepared token"));
  std::array<c::ConstTensorView,4> ordered;for(std::size_t i=0;i<4;++i)ordered[i]=inputs[r.inputs[i]];
  for(std::size_t port=1;port<3;++port){const auto&frame=port==1?s.keys:s.values;if(!frame.valid())continue;
    auto view=cd::PagedArenaAccess::tensor(frame);if(!view)return cd::EnqueueResult::rejected(view.error());
    if(frame.layout()!=s.layout || !same_storage(view->storage(),ordered[port].storage()))
      return cd::EnqueueResult::rejected(Status::InvalidArgument("Graph paged input differs from its prepared page permissions"));
  }
  return cd::BatchGraphAccess::enqueue(s.token,queue,ordered,outputs);
}
const ProviderRegionPreparedVTable region_table{.destroy_prepared=&destroy_region,.alias_support=&alias,.create_invocation=&create_slot,.destroy_invocation=&destroy_slot,.shared_submit=&submit_region};
c::ProviderPrepareOutcome compile_region(void*opaque,const RegionCompileRequest&r,mem::AllocatorRef a,ProviderRegionPrepared*out)noexcept{
  *out={};auto&context=*static_cast<Context*>(opaque);if(!r.graph || r.nodes.size()!=1 || r.inputs.size()!=4 || r.outputs.size()!=1)
    return c::ProviderPrepareOutcome::unsupported(Status::Unimplemented("Graph batch region boundary is not qualified"));
  const auto supported=supports(*r.graph,r.nodes[0],*context.config);if(!supported.ok())return c::ProviderPrepareOutcome::unsupported(supported);
  auto*p=a.construct<Region>(context.config);if(!p)return c::ProviderPrepareOutcome::error(Status::ResourceExhausted("Graph batch artifact metadata"));
  *out={.vtable=&region_table,.state=p,.properties={.workspace={0,16},.persistent_bytes=std::nullopt,.runtime=runtime,.invocation_private_bytes=std::nullopt}};
  p->node=r.nodes[0];
  p->options=context.config->options;p->options.resource=r.output_resources[0];
  for(std::size_t i=0;i<4;++i){auto value=r.graph->node_input(p->node,i);if(!value)return c::ProviderPrepareOutcome::error(value.error());auto found=std::find(r.inputs.begin(),r.inputs.end(),*value);if(found==r.inputs.end())return c::ProviderPrepareOutcome::error(Status::Internal("batch region port mapping"));p->inputs[i]=std::size_t(found-r.inputs.begin());p->values[i]=*value;if(i==3){if(value->kind()!=ValueKind::kInput)return c::ProviderPrepareOutcome::unsupported(Status::Unimplemented("batch metadata must be an explicit host-known Graph input"));p->metadata=*value;}}
  return c::ProviderPrepareOutcome::prepared();
}
const RegionProviderVTable region_provider{&propose,&resources,&compile_region};
Status qualify(void*opaque,c::BatchCapacity cap,const c::BatchProgramOptions&o)noexcept{auto&ctx=*static_cast<Context*>(opaque);return ctx.batch_table->qualify(ctx.base,cap,o);}
Status create_native(void*opaque,c::BatchCapacity cap,const c::BatchProgramOptions&o,mem::AllocatorRef a,void**out)noexcept{auto&ctx=*static_cast<Context*>(opaque);return ctx.batch_table->create(ctx.base,cap,o,a,out);}
Status create_context(const c::Provider&provider,c::Device device,void*backend,mem::AllocatorRef a,void**out)noexcept{
  *out=nullptr;auto*config=cd::ProviderAccess::payload<Config*>(provider);auto*p=a.construct<Context>();if(!p)return Status::ResourceExhausted("Graph batch context metadata");
  *out=p;p->config=config;config->incref();p->base_table=cd::ProviderAccess::vtable(config->base);
  HYLE_RETURN_IF_ERROR(p->base_table->create_context(config->base,device,backend,a,&p->base));
  const auto*interface_value=p->base_table->query_extension?p->base_table->query_extension(p->base,cd::kBatchPlanningExtensionId):nullptr;
  if(!interface_value)return Status::Unimplemented("base provider has no native batch planning capability");
  p->batch_table=static_cast<const cd::BatchProviderVTable*>(interface_value);p->forwarding=*p->batch_table;p->forwarding.qualify=&qualify;p->forwarding.create=&create_native;
  return p->batch_table->qualify(p->base,config->schema.capacity(),config->options);
}
void destroy_context(void*opaque,mem::AllocatorRef a)noexcept{auto*p=static_cast<Context*>(opaque);if(!p)return;if(p->base)p->base_table->destroy_context(p->base,a);p->config->decref();a.destroy(p);}
Status propose_single(void*,const cd::ProviderOperationRequest&,std::span<cd::ResourceBindingRequirement>,std::span<cd::ResourceBindingRequirement>,cd::ProviderOperationProposal*)noexcept{return Status::Unimplemented("batch Graph wrapper has no ordinary one-op route");}
c::ProviderPrepareOutcome prepare_single(void*,const cd::ProviderPrepareRequest&,mem::AllocatorRef,cd::ProviderPrepared*)noexcept{return c::ProviderPrepareOutcome::unsupported(Status::Unimplemented("batch Graph wrapper has no ordinary one-op route"));}
Status drain_context(void*opaque)noexcept{auto&ctx=*static_cast<Context*>(opaque);return ctx.base_table->drain_context(ctx.base);}
const void*query(void*opaque,c::ProviderExtensionId id)noexcept{auto&ctx=*static_cast<Context*>(opaque);if(id==kRegionProviderExtensionId)return &region_provider;if(id==cd::kBatchPlanningExtensionId)return &ctx.forwarding;return nullptr;}
void retain_config(const c::Provider&p)noexcept{cd::ProviderAccess::payload<Config*>(p)->incref();}
void release_config(const c::Provider&p)noexcept{cd::ProviderAccess::payload<Config*>(p)->decref();}
const cd::ProviderVTable provider_table{.identity={kProviderId,"graph-batch","1"},.create_context=&create_context,.destroy_context=&destroy_context,
  .propose=&propose_single,.prepare=&prepare_single,.drain_context=&drain_context,.query_extension=&query,.retain_configuration=&retain_config,.release_configuration=&release_config};
} // namespace
namespace detail {
struct InvocationData final {
  struct Binding{NodeId node;ValueId metadata;Region*region=nullptr;Slot*slot=nullptr;std::uint32_t step=0;};
  explicit InvocationData(mem::AllocatorRef a):allocator(a),queues(mem::StdAllocator<QueueBinding>(a)),bindings(mem::StdAllocator<Binding>(a)){}
  mem::AllocatorRef allocator;LeasedInvocation original;
  ReplayOptions replay{};
  std::vector<QueueBinding,mem::StdAllocator<QueueBinding>>queues;
  std::vector<Binding,mem::StdAllocator<Binding>>bindings;
};
struct PendingInvocationData final:util::RefCounted<PendingInvocationData>{
  struct Input final {
    ValueId value;
    c::PagedArenaPendingRead frame;
    std::vector<ExecutionLaneId,mem::StdAllocator<ExecutionLaneId>> lanes;
  };
  explicit PendingInvocationData(mem::AllocatorRef a):allocator(a),
      inputs(mem::StdAllocator<Input>(a)),admitted(mem::StdAllocator<unsigned char>(a)){}
  void destroy()noexcept{const auto a=allocator;a.destroy(this);}
  bool contains(ValueId value)const noexcept{
    return std::any_of(inputs.begin(),inputs.end(),[&](const auto&input){return input.value==value;});
  }
  Status reusable()noexcept{
    if(!source.available() || !dependent.available() || source.update().valid())
      return Status::Unavailable("pending Graph still owns source/outcome controls");
    return Status::Ok();
  }
  mem::AllocatorRef allocator;
  Invocation original;
  cd::CopyOutcomeSource<c::CopyBatchUpdate,cd::CopyStreamAccess> source;
  cd::DependentCompletion dependent;
  cd::OutcomeSources sources{};
  cd::CompletionRetention retention{};
  std::vector<Input,mem::StdAllocator<Input>> inputs;
  std::vector<unsigned char,mem::StdAllocator<unsigned char>> admitted;
  bool submitted=false;
};
void destroy(InvocationData*p)noexcept{if(p){const auto allocator=p->allocator;allocator.destroy(p);}}
} // namespace detail
Result<MaterializedGraph> materialized_graph(const c::MaterializedBatchProgram&program,GraphBuilderOptions options)noexcept{
  if(!program.valid())return std::unexpected(Status::InvalidArgument("materialized Graph program is empty"));
  HYLE_ASSIGN_OR_RETURN(auto builder,GraphBuilder::create(options));MaterializedGraph result;
  std::array<ValueId,24> values;const auto descriptions=program.descriptions();
  for(std::size_t i=0;i<5;++i){HYLE_ASSIGN_OR_RETURN(values[i],builder.input(descriptions[i]));result.inputs[i]=values[i];}
  for(const auto&step:program.steps()){
    std::array<ValueId,2> inputs;for(std::size_t i=0;i<step.input_count;++i)inputs[i]=values[step.inputs[i]];
    HYLE_ASSIGN_OR_RETURN(auto node,builder.add_node(step.operation,std::span(inputs).first(step.input_count)));
    HYLE_ASSIGN_OR_RETURN(values[step.output],builder.result(node));
  }
  result.output=values[program.output_value()];HYLE_RETURN_IF_ERROR(builder.publish(result.output));
  HYLE_ASSIGN_OR_RETURN(result.graph,std::move(builder).finish());return result;
}
Result<c::Provider> provider(c::Provider base,c::BatchSchema schema,c::BatchProgramOptions options)noexcept{
  if(!base.valid() || !schema.valid() || !options.allocator.valid())return std::unexpected(Status::InvalidArgument("Graph batch provider configuration"));
  const auto a=options.allocator;auto*p=a.construct<Config>(std::move(base),std::move(schema),std::move(options),a);if(!p)return std::unexpected(Status::ResourceExhausted("Graph batch provider configuration metadata"));return cd::ProviderAccess::make(&provider_table,p);
}
Invocation::Invocation(Invocation&&v)noexcept:data_(std::exchange(v.data_,nullptr)){}
Invocation&Invocation::operator=(Invocation&&v)noexcept{if(this!=&v){detail::destroy(data_);data_=std::exchange(v.data_,nullptr);}return *this;}
Invocation::~Invocation(){detail::destroy(data_);}
Result<Invocation> Invocation::create(const Executable&executable,std::span<const LaneBinding>lanes,LeasedInvocationOptions options,ReplayOptions replay)noexcept{
  if(!executable.valid() || !options.allocator.valid() || lanes.size()!=executable.required_queue_count())
    return std::unexpected(Status::InvalidArgument("Graph batch requires its complete active lanes"));
  auto*p=options.allocator.construct<detail::InvocationData>(options.allocator);if(!p)return std::unexpected(Status::ResourceExhausted("Graph batch coordinator metadata"));Invocation result(p);
  std::vector<const LaneBinding*,mem::StdAllocator<const LaneBinding*>>by_lane{mem::StdAllocator<const LaneBinding*>(options.allocator)};
  try{by_lane.resize(executable.lane_count(),nullptr);p->queues.reserve(lanes.size());
    for(const auto&binding:lanes){
      if(!binding.lane.valid() || binding.lane.ordinal()>=by_lane.size() || by_lane[binding.lane.ordinal()] ||
         !binding.context || !binding.context->valid() || !binding.context->is_open() || !binding.queue || !binding.queue->is_open())
        return std::unexpected(Status::InvalidArgument("Graph batch lane/context/queue is invalid, closed or duplicated"));
      HYLE_ASSIGN_OR_RETURN(auto required,executable.requires_queue(binding.lane));
      if(!required)return std::unexpected(Status::InvalidArgument("Graph batch binding names an inactive lane"));
      HYLE_RETURN_IF_ERROR(gd::RegionAccess::validate_queue(executable,binding.lane,*binding.queue));
      HYLE_ASSIGN_OR_RETURN(auto context,cd::ContextRef::acquire(*binding.context));
      if(!context.owns(*binding.queue))return std::unexpected(Status::InvalidArgument("Graph batch Queue does not belong to its Context"));
      by_lane[binding.lane.ordinal()]=&binding;p->queues.push_back({binding.lane,binding.queue});
    }
  }catch(...){return std::unexpected(Status::ResourceExhausted("Graph batch lane binding metadata"));}
  HYLE_ASSIGN_OR_RETURN(p->original,LeasedInvocation::create(executable,options));
  p->replay=replay;
  struct Call{detail::InvocationData*data;std::span<const LaneBinding*const>lanes;}call{p,by_lane};
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(p->original,[](void*opaque,const gd::RegionAccessView&view)noexcept->Status{
    if(view.owner->provider().id!=kProviderId)return Status::Ok();auto&call=*static_cast<Call*>(opaque);auto&r=*static_cast<Region*>(view.prepared);auto&s=*static_cast<Slot*>(view.invocation);
    if(!view.lane.valid() || view.lane.ordinal()>=call.lanes.size() || !call.lanes[view.lane.ordinal()])return Status::InvalidArgument("Graph batch region has no active lane binding");
    const auto&binding=*call.lanes[view.lane.ordinal()];
    if(!binding.host_workspace.valid())return Status::InvalidArgument("Graph native batch lane has no admitted host workspace");
    {std::lock_guard guard(r.cold_lock);if(!r.program.valid()){r.options.host_workspace=binding.host_workspace;HYLE_ASSIGN_OR_RETURN(r.program,cd::BatchGraphAccess::prepare(*binding.context,r.config->schema,r.options,view.owner->provider_instance_id()));}else if(r.options.host_workspace!=binding.host_workspace)return Status::InvalidArgument("Graph batch host resource differs from its cold program");}
    HYLE_ASSIGN_OR_RETURN(s.native,cd::BatchGraphAccess::create(r.program,*binding.queue));
    try{call.data->bindings.push_back({r.node,r.metadata,&r,&s,view.step});}catch(...){return Status::ResourceExhausted("Graph batch node binding metadata");}return Status::Ok();
  },&call));
  if(p->bindings.empty())return std::unexpected(Status::Unimplemented("executable has no selected native batch region"));
  std::uint64_t extra=0;for(const auto&binding:p->bindings)extra+=2ULL*binding.region->config->schema.capacity().max_page_refs;
  if(extra>UINT32_MAX-1)return std::unexpected(Status::ResourceExhausted("Graph page guard capacity overflow"));
  if(extra)HYLE_RETURN_IF_ERROR(gd::RegionAccess::reserve_reads(p->original,std::uint32_t(extra),options));
  return result;
}
Status Invocation::prepare_batch(std::span<const NodeLayout>layouts,bool update)noexcept{
  if(!data_ || layouts.size()!=data_->bindings.size())return Status::InvalidArgument("Graph batch needs a complete node/layout set");
  for(std::size_t i=0;i<layouts.size();++i){auto found=std::find_if(data_->bindings.begin(),data_->bindings.end(),[&](auto&b){return b.node==layouts[i].node;});if(found==data_->bindings.end() || !found->region->config->schema.owns(layouts[i].layout))return Status::InvalidArgument("Graph batch node/schema is foreign");for(std::size_t j=0;j<i;++j){if(layouts[j].node==layouts[i].node)return Status::InvalidArgument("Graph batch duplicate node layout");auto previous=std::find_if(data_->bindings.begin(),data_->bindings.end(),[&](auto&b){return b.node==layouts[j].node;});if(previous->metadata==found->metadata && layouts[j].layout!=layouts[i].layout)return Status::InvalidArgument("one Graph metadata input requires one layout version");}}
  HYLE_RETURN_IF_ERROR(data_->original.reset());
  for(auto&binding:data_->bindings){binding.slot->token={};binding.slot->layout={};binding.slot->keys={};binding.slot->values={};cd::BatchGraphAccess::clear_shared_metadata(binding.slot->native);}
  const auto prepare=[&]()noexcept->Status{
  for(const auto&layout:layouts){auto&binding=*std::find_if(data_->bindings.begin(),data_->bindings.end(),[&](auto&b){return b.node==layout.node;});binding.slot->token={};
    if(update){HYLE_ASSIGN_OR_RETURN(binding.slot->token,binding.slot->native.update_batch(layout.layout));}
    else{HYLE_ASSIGN_OR_RETURN(binding.slot->token,binding.slot->native.prepare_batch(layout.layout));}
    binding.slot->layout=layout.layout;
  }
  for(std::size_t i=0;i<data_->bindings.size();++i){auto&binding=data_->bindings[i];auto previous=std::find_if(data_->bindings.begin(),data_->bindings.begin()+i,[&](auto&b){return b.metadata==binding.metadata;});
    if(previous!=data_->bindings.begin()+i){HYLE_RETURN_IF_ERROR(cd::BatchGraphAccess::share_metadata(binding.slot->native,previous->slot->native));continue;}
    HYLE_ASSIGN_OR_RETURN(auto mirror,cd::BatchGraphAccess::metadata(binding.slot->native));HYLE_ASSIGN_OR_RETURN(auto description,binding.region->config->schema.metadata_description());HYLE_RETURN_IF_ERROR(data_->original.bind(binding.metadata,description,std::move(mirror)));
  }return Status::Ok();};
  const auto status=prepare();if(!status.ok()){for(auto&binding:data_->bindings){binding.slot->token={};cd::BatchGraphAccess::clear_shared_metadata(binding.slot->native);}data_->original.reset().IgnoreError();}return status;
}
Status Invocation::bind(ValueId value,c::TensorDesc description,c::ReadLease lease)noexcept{if(!data_)return Status::FailedPrecondition("Graph batch coordinator is empty");for(auto&b:data_->bindings){if(b.metadata==value)return Status::InvalidArgument("Graph batch metadata is library-owned");if((b.region->values[1]==value && b.slot->keys.valid()) || (b.region->values[2]==value && b.slot->values.valid()))return Status::FailedPrecondition("Graph projected input needs explicit prepare before rebinding");}return data_->original.bind(value,description,std::move(lease));}
Status Invocation::bind(ValueId value,c::TensorDesc description,c::WriteLease&&lease)noexcept{if(!data_)return Status::FailedPrecondition("Graph batch coordinator is empty");for(auto&b:data_->bindings){if(b.metadata==value)return Status::InvalidArgument("Graph batch metadata is library-owned");if((b.region->values[1]==value && b.slot->keys.valid()) || (b.region->values[2]==value && b.slot->values.valid()))return Status::FailedPrecondition("Graph projected input needs explicit prepare before rebinding");}return data_->original.bind(value,description,std::move(lease));}
Status Invocation::bind(ValueId value,c::PagedArenaRead frame)noexcept{
  if(!data_ || !frame.valid())return Status::InvalidArgument("Graph paged binding requires a coordinator and frame");
  std::vector<std::uint32_t,mem::StdAllocator<std::uint32_t>> readers{mem::StdAllocator<std::uint32_t>(data_->allocator)};
  try{readers.reserve(data_->bindings.size());for(const auto&b:data_->bindings){const auto&r=*b.region;
    if(r.values[0]==value || r.values[3]==value)return Status::InvalidArgument("paged arena cannot bind query or metadata");
    if(r.values[1]!=value && r.values[2]!=value)continue;
    if(r.config->schema.capacity().storage!=c::BatchStorage::kPaged || !r.config->schema.owns(frame.layout()) ||
       !b.slot->token.valid() || b.slot->layout!=frame.layout())return Status::InvalidArgument("Graph page layout/schema differs from its prepared slot");
    if((r.values[1]==value && b.slot->keys.valid()) || (r.values[2]==value && b.slot->values.valid()))
      return Status::FailedPrecondition("Graph projected input needs explicit prepare before rebinding");
    readers.push_back(b.step);
  }}catch(...){return Status::ResourceExhausted("Graph page reader qualification metadata");}
  HYLE_ASSIGN_OR_RETURN(auto tensor,cd::PagedArenaAccess::tensor(frame));
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::bind_projected(data_->original,value,tensor,cd::PagedArenaAccess::pages(frame),kProviderId,readers));
  for(auto&b:data_->bindings){if(b.region->values[1]==value)b.slot->keys=frame;if(b.region->values[2]==value)b.slot->values=frame;}
  return Status::Ok();
}
Result<c::Completion> Invocation::submit()noexcept{
  if(!data_)return std::unexpected(Status::FailedPrecondition("Graph batch coordinator is empty"));
  if(data_->original.replay_route()==ReplayRoute::kUnprepared)HYLE_RETURN_IF_ERROR(data_->original.prepare_replay(data_->queues,data_->replay));
  return data_->original.submit(data_->queues);
}
PendingInvocation::PendingInvocation(PendingInvocation&&other)noexcept:data_(std::exchange(other.data_,nullptr)){}
PendingInvocation&PendingInvocation::operator=(PendingInvocation&&other)noexcept{
  if(this!=&other){if(data_)data_->decref();data_=std::exchange(other.data_,nullptr);}return *this;
}
PendingInvocation::~PendingInvocation(){if(data_)data_->decref();}
Result<PendingInvocation> PendingInvocation::create(Invocation&&owner,std::size_t capacity,
    LeasedInvocationOptions options)noexcept{
  if(!owner.data_ || !capacity || !options.allocator.valid())
    return std::unexpected(Status::InvalidArgument("pending Graph requires an original coordinator/source capacity"));
  if(!gd::RegionAccess::outcome_available(owner.data_->original))
    return std::unexpected(Status::Unavailable("pending Graph original outcome is retained"));
  auto allocator=options.allocator;
  if(options.resource_budget.valid()){HYLE_ASSIGN_OR_RETURN(allocator,options.resource_budget.metadata_allocator());}
  auto*p=allocator.construct<detail::PendingInvocationData>(allocator);
  if(!p)return std::unexpected(Status::ResourceExhausted("pending Graph cold owner metadata"));
  PendingInvocation result(p);
  HYLE_ASSIGN_OR_RETURN(p->source,(cd::CopyOutcomeSource<c::CopyBatchUpdate,cd::CopyStreamAccess>::create(
      capacity,{.allocator=allocator,.resource_budget=options.resource_budget})));
  HYLE_ASSIGN_OR_RETURN(p->dependent,cd::DependentCompletion::create(allocator));
  if(owner.data_->bindings.size()>SIZE_MAX/2)
    return std::unexpected(Status::ResourceExhausted("pending Graph input capacity overflow"));
  try{p->inputs.reserve(2*owner.data_->bindings.size());p->admitted.resize(owner.data_->queues.size());}
  catch(...){return std::unexpected(Status::ResourceExhausted("pending Graph fixed input metadata"));}
  p->original=std::move(owner);return result;
}
Status PendingInvocation::prepare_batch(std::span<const NodeLayout>layouts,bool update)noexcept{
  if(!data_)return Status::FailedPrecondition("pending Graph is empty");
  HYLE_RETURN_IF_ERROR(data_->reusable());
  HYLE_RETURN_IF_ERROR(data_->original.prepare_batch(layouts,update));
  data_->inputs.clear();data_->submitted=false;return Status::Ok();
}
Status PendingInvocation::bind(ValueId value,c::TensorDesc description,c::ReadLease source)noexcept{
  if(!data_)return Status::FailedPrecondition("pending Graph is empty");
  HYLE_RETURN_IF_ERROR(data_->reusable());
  if(data_->contains(value))return Status::FailedPrecondition("pending Graph input needs prepare before rebinding");
  return data_->original.bind(value,description,std::move(source));
}
Status PendingInvocation::bind(ValueId value,c::TensorDesc description,c::WriteLease&&destination)noexcept{
  if(!data_)return Status::FailedPrecondition("pending Graph is empty");
  HYLE_RETURN_IF_ERROR(data_->reusable());
  if(data_->contains(value))return Status::FailedPrecondition("pending Graph input needs prepare before rebinding");
  return data_->original.bind(value,description,std::move(destination));
}
Status PendingInvocation::bind(ValueId value,c::PagedArenaRead frame)noexcept{
  if(!data_)return Status::FailedPrecondition("pending Graph is empty");
  HYLE_RETURN_IF_ERROR(data_->reusable());
  if(data_->contains(value))return Status::FailedPrecondition("pending Graph input needs prepare before rebinding");
  return data_->original.bind(value,std::move(frame));
}
Status PendingInvocation::bind(ValueId value,c::PagedArenaPendingRead frame)noexcept{
  if(!data_ || !frame.valid())return Status::InvalidArgument("pending Graph binding requires its typed frame");
  auto&p=*data_;HYLE_RETURN_IF_ERROR(p.reusable());
  if(p.contains(value))return Status::FailedPrecondition("pending Graph input needs prepare before rebinding");
  auto&original=*p.original.data_;
  std::vector<std::uint32_t,mem::StdAllocator<std::uint32_t>> readers{mem::StdAllocator<std::uint32_t>(p.allocator)};
  detail::PendingInvocationData::Input input{value,std::move(frame),
      std::vector<ExecutionLaneId,mem::StdAllocator<ExecutionLaneId>>(mem::StdAllocator<ExecutionLaneId>(p.allocator))};
  try{
    readers.reserve(original.bindings.size());input.lanes.reserve(original.queues.size());
    for(const auto&b:original.bindings){const auto&r=*b.region;
      if(r.values[0]==value || r.values[3]==value)return Status::InvalidArgument("pending pages cannot bind query/metadata");
      if(r.values[1]!=value && r.values[2]!=value)continue;
      if(r.config->schema.capacity().storage!=c::BatchStorage::kPaged || !r.config->schema.owns(input.frame.layout()) ||
          !b.slot->token.valid() || b.slot->layout!=input.frame.layout())
        return Status::InvalidArgument("pending Graph frame differs from its prepared reader/layout");
      if((r.values[1]==value && b.slot->keys.valid()) || (r.values[2]==value && b.slot->values.valid()))
        return Status::FailedPrecondition("ready Graph input needs prepare before pending binding");
      readers.push_back(b.step);
    }
  }catch(...){return Status::ResourceExhausted("pending Graph exact reader metadata");}
  struct Readers final {std::span<const std::uint32_t> steps;decltype(input.lanes)*lanes;};
  Readers selected{readers,&input.lanes};
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(original.original,[](void*opaque,const gd::RegionAccessView&view)noexcept->Status{
    auto&readers=*static_cast<Readers*>(opaque);
    if(std::find(readers.steps.begin(),readers.steps.end(),view.step)!=readers.steps.end() &&
        std::find(readers.lanes->begin(),readers.lanes->end(),view.lane)==readers.lanes->end()){
      if(readers.lanes->size()==readers.lanes->capacity())return Status::ResourceExhausted("pending Graph lane capacity");
      readers.lanes->push_back(view.lane);
    }return Status::Ok();
  },&selected));
  if(input.lanes.empty())return Status::InvalidArgument("pending Graph input has no exact paged reader");
  HYLE_ASSIGN_OR_RETURN(auto tensor,cd::PagedArenaAccess::tensor(input.frame));
  if(p.inputs.size()==p.inputs.capacity())return Status::ResourceExhausted("pending Graph input capacity");
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::bind_pending_projected(original.original,value,tensor,kProviderId,readers));
  p.inputs.push_back(std::move(input));return Status::Ok();
}
Result<c::Completion> PendingInvocation::submit(c::CopyBatchUpdate&&update)noexcept{
  if(!data_)return std::unexpected(Status::FailedPrecondition("pending Graph is empty"));
  auto&p=*data_;HYLE_RETURN_IF_ERROR(p.reusable());auto&original=*p.original.data_;
  if(p.inputs.empty() || p.submitted)return std::unexpected(Status::FailedPrecondition("pending Graph has no unconsumed page inputs"));
  for(const auto&input:p.inputs)
    HYLE_RETURN_IF_ERROR((cd::PagedArenaAccess::validate<c::CopyBatchUpdate,cd::CopyStreamAccess>(input.frame,update)));
  if(original.original.replay_route()==ReplayRoute::kUnprepared)
    HYLE_RETURN_IF_ERROR(original.original.prepare_replay(original.queues,original.replay));
  HYLE_ASSIGN_OR_RETURN(p.sources,p.source.take(std::move(update)));
  std::fill(p.admitted.begin(),p.admitted.end(),static_cast<unsigned char>(0));
  struct Call final {detail::PendingInvocationData*data;detail::InvocationData*original;};
  Call call{&p,&original};
  // The callbacks borrow only this synchronous submission scope; aggregate
  // retention owns the cold state after acceptance, including abandonment.
  const gd::OrderedGraphSubmission ordered{
    [](void*opaque,ExecutionLaneId lane,cd::QueueSequence&sequence,c::LeaseUseBatch&uses)noexcept->Status{
      auto&call=*static_cast<Call*>(opaque);auto&p=*call.data;
      const auto queued=std::find_if(call.original->queues.begin(),call.original->queues.end(),
          [&](const auto&binding){return binding.lane==lane;});
      if(queued==call.original->queues.end())return Status::Internal("pending Graph lane is missing");
      const auto index=std::size_t(queued-call.original->queues.begin());
      if(p.admitted[index])return Status::Ok();
      const auto reads=[&](const auto&input){return std::find(input.lanes.begin(),input.lanes.end(),lane)!=input.lanes.end();};
      if(std::none_of(p.inputs.begin(),p.inputs.end(),reads))return Status::Ok();
      HYLE_ASSIGN_OR_RETURN(auto prefix,cd::CopyStreamAccess::dependency(p.source.update()));
      if(prefix.pending()){
        auto waited=sequence.enqueue_resolved_wait(prefix);const auto status=waited.status;
        sequence.record(std::move(waited));HYLE_RETURN_IF_ERROR(status);
      }
      for(const auto&input:p.inputs)if(reads(input))
        HYLE_RETURN_IF_ERROR((cd::PagedArenaAccess::admit_read<c::CopyBatchUpdate,cd::CopyStreamAccess>(uses,input.frame,p.source.update())));
      p.admitted[index]=1;return Status::Ok();
    },
    [](void*opaque,c::Completion&&suffix)noexcept->c::Completion{
      auto&p=*static_cast<Call*>(opaque)->data;p.submitted=true;p.incref();
      p.retention.own={&p,[](void*,const Status&)noexcept{},
          [](void*owner)noexcept{static_cast<detail::PendingInvocationData*>(owner)->decref();}};
      auto aggregate=p.dependent.publish(std::move(suffix),std::exchange(p.sources,{}));
      HYLE_FATAL_ASSERT(cd::CompletionAccess::attach_retention(aggregate,p.retention).ok());
      return aggregate;
    },&call,[](void*opaque,ValueId value)noexcept->bool{
      return static_cast<Call*>(opaque)->data->contains(value);
    }};
  auto result=gd::RegionAccess::submit(original.original,original.queues,ordered);
  if(!result){update=std::move(p.source.update());auto sources=std::exchange(p.sources,{});sources.release(sources.state);}
  return result;
}
Result<LeasedGraphOutput> PendingInvocation::output(ValueId value)noexcept{
  if(!data_ || !data_->submitted)return std::unexpected(Status::FailedPrecondition("pending Graph has no submitted output"));
  return data_->original.output(value);
}
Result<c::CopyBatchUpdate> PendingInvocation::take_update()noexcept{
  if(!data_ || !data_->dependent.available() || !data_->source.available() ||
      !data_->source.update().valid() || data_->source.update().in_flight() ||
      !gd::RegionAccess::outcome_available(data_->original.data_->original))
    return std::unexpected(Status::Unavailable("pending Graph source/outcome remains retained"));
  return std::move(data_->source.update());
}
Result<c::BatchPlanReport> PendingInvocation::report(NodeId node)const noexcept{
  if(!data_)return std::unexpected(Status::FailedPrecondition("pending Graph is empty"));return data_->original.report(node);
}
Status PendingInvocation::close()noexcept{return data_?data_->original.close():Status::Ok();}
Status PendingInvocation::drain()noexcept{return data_?data_->original.drain():Status::Ok();}
Result<LeasedGraphOutput> Invocation::output(ValueId value)noexcept{if(!data_)return std::unexpected(Status::FailedPrecondition("Graph batch coordinator is empty"));return data_->original.output(value);}
Result<c::BatchPlanReport> Invocation::report(NodeId node)const noexcept{if(!data_)return std::unexpected(Status::FailedPrecondition("Graph batch coordinator is empty"));for(const auto&b:data_->bindings)if(b.node==node)return b.slot->native.report();return std::unexpected(Status::InvalidArgument("Graph batch report node is foreign"));}
Status Invocation::close()noexcept{return data_?data_->original.close():Status::Ok();}
Status Invocation::drain()noexcept{return data_?data_->original.drain():Status::Ok();}
} // namespace hyle::graph::batch
