#include "hyle/graph/batch.h"
#include "hyle/compute/detail/batch_provider.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/graph/region_provider.h"
#include "hyle/compute/detail/batch_graph_access.h"
#include "hyle/compute/detail/paged_arena_access.h"
#include "hyle/compute/detail/context_ref.h"
#include "hyle/compute/detail/queue_ref.h"
#include "hyle/graph/detail/region_adapter.h"
#include "hyle/compute/detail/copy_outcome_source.h"
#include "hyle/mem/std_allocator.h"
#include "../graph/compile_internal.h"
#include "../graph/execution_source_internal.h"
#include "native_batch_regions.h"
#include "row_cache_node.h"
#include "paged_append.h"
#include "../graph/slots_region.h"
#include "hyle/graph/slots.h"
#include <algorithm>
#include <array>
#include <vector>

namespace hyle::graph::batch {
namespace c=compute;namespace cd=c::detail;namespace gd=graph::detail;
namespace gs=::hyle::graph::slots;
namespace {
struct Region final {
  NodeId node;ValueId metadata;
  std::array<ValueId,4> values{};std::array<std::size_t,4> inputs{};
  c::BatchProgramOptions options;
  cd::ContextRef context;cd::ProviderExtensionRef source;
  c::BatchCapacity capacity{};c::MlaCacheLayout mla=c::MlaCacheLayout::kNone;
  std::uint32_t query_rows=0; // Exact concrete Q/output extent, not schema maximum.
  c::BatchLayoutVersion initial_layout;
  mem::MemoryResourceRef resource{};std::size_t boundary_count=0;
  bool graph_capture=false;
};
struct Slot final {
  struct WrittenBank final {c::BatchLayoutVersion layout;c::ConstTensorView capacity;NodeId producer;};
  Region*region=nullptr;c::BatchInvocation native;c::PreparedBatch token;
  std::uint32_t selected_step=UINT32_MAX;
  c::BatchLayoutVersion layout;c::PagedArenaRead keys,values;
  WrittenBank written_keys,written_values;
  c::BatchProgram program;PrepareOptions prepared_options{};cd::QueueRef queue;
  // Physical metadata ownership only, not a retained read permission. The
  // selected top-level slots survive reset and flatten shared owners here.
  Slot*metadata_owner=nullptr;
};
std::size_t payloads(const Region&r)noexcept{return r.mla==c::MlaCacheLayout::kNone?3u:2u;}
std::size_t ports(const Region&r)noexcept{return payloads(r)+(r.initial_layout.valid()?0u:1u);}
cd::BatchGraphContract contract(const Region&r)noexcept{return r.initial_layout.valid()?
    cd::BatchGraphContract::kOwnedLayoutActive:cd::BatchGraphContract::kPreparedMetadataCapacity;}
bool has_value_port(const Region&r)noexcept{return r.mla==c::MlaCacheLayout::kNone;}
bool same_storage(mem::ConstBufferView a,mem::ConstBufferView b)noexcept{
  return a.resource()==b.resource() && a.native_handle()==b.native_handle() && a.offset()==b.offset() && a.size()==b.size();
}
constexpr c::ProviderRuntimeProperties default_runtime{.mode=c::ProviderRuntimeMode::kSharedQueue,
  .workspace_owner=c::ProviderWorkspaceOwner::kProvider,.replay_class=c::ProviderReplayClass::kManaged,
  .blocks_host=true,.uses_private_threads=false,.supports_concurrent_invocations=true,.capture_compatible=false};
void destroy_region(void*p,mem::AllocatorRef a)noexcept{a.destroy(static_cast<Region*>(p));}
c::AliasSupport alias(void*,std::size_t,std::size_t)noexcept{return {};}
Status create_slot(void*prepared,mem::AllocatorRef a,mem::BufferView,void**out)noexcept{
  *out=a.construct<Slot>();if(!*out)return Status::ResourceExhausted("Graph batch slot metadata");static_cast<Slot*>(*out)->region=static_cast<Region*>(prepared);return Status::Ok();
}
void destroy_slot(void*p,mem::AllocatorRef a)noexcept{a.destroy(static_cast<Slot*>(p));}
cd::EnqueueResult submit_region(void*prepared,void*invocation,cd::QueueSequence&sequence,std::span<const c::ConstTensorView>inputs,std::span<const c::TensorView>outputs)noexcept{
  const auto&r=*static_cast<Region*>(prepared);auto&s=*static_cast<Slot*>(invocation);
  if(!s.token.valid() || inputs.size()!=r.boundary_count)return cd::EnqueueResult::rejected(Status::FailedPrecondition("Graph batch region has no explicit prepared token"));
  std::array<c::ConstTensorView,4> ordered;for(std::size_t i=0;i<ports(r);++i)ordered[i]=inputs[r.inputs[i]];
  for(std::size_t port=1;port<payloads(r);++port){const auto&frame=port==1?s.keys:s.values;if(!frame.valid())continue;
    auto view=cd::PagedArenaAccess::tensor(frame);if(!view)return cd::EnqueueResult::rejected(view.error());
    if(frame.layout()!=s.layout || !same_storage(view->storage(),ordered[port].storage()))
      return cd::EnqueueResult::rejected(Status::InvalidArgument("Graph paged input differs from its prepared page permissions"));
  }
  for(std::size_t port=1;port<payloads(r);++port){const auto&write=port==1?s.written_keys:s.written_values;if(!write.producer.valid())continue;
    if(write.layout!=s.layout || write.capacity.description()!=ordered[port].description() ||
        !same_storage(write.capacity.storage(),ordered[port].storage()))
      return cd::EnqueueResult::rejected(Status::InvalidArgument("Graph cache reader differs from its exact same-Graph append binding"));
  }
  return cd::BatchGraphAccess::enqueue(s.token,sequence,std::span(ordered).first(ports(r)),outputs,contract(r));
}
const gd::RegionProjectedReadVTable projected_table{
  [](const ProviderRegionPrepared&prepared,void*invocation,std::uint32_t port,
      cd::OperationPreparedAccess*out)noexcept->Status{
    *out={};
    if(!prepared.state || !invocation)return Status::InvalidArgument("shared page projection needs its actual slot");
    const auto&r=*static_cast<const Region*>(prepared.state);const auto&s=*static_cast<const Slot*>(invocation);
    bool cache=false;
    for(std::size_t input=1;input<payloads(r);++input)cache|=r.inputs[input]==port;
    if(!cache)return Status::Unimplemented("shared batch port is not a projected cache reader");
    HYLE_ASSIGN_OR_RETURN(const bool empty,cd::BatchGraphAccess::empty_cache_reads(s.token,contract(r)));
    if(!empty)return Status::Unimplemented("shared batch slot has no actual empty-cache proof");
    *out={.projected_input=port,.allows_empty_footprint=true};return Status::Ok();
  }};
const gd::RegionCaptureBindingsVTable capture_bindings{
  [](void*prepared,void*invocation)noexcept->bool{
    if(!prepared || !invocation)return false;
    const auto&r=*static_cast<const Region*>(prepared);
    const auto&s=*static_cast<const Slot*>(invocation);
    // The original BatchInvocation retains its plan, workspaces and metadata.
    // before_command_change invalidates this slot and metadata dependents
    // before fresh planning or any private carrier/program replacement.
    return r.graph_capture && s.region==&r && s.program.valid() &&
        s.native.valid() && s.token.valid() && s.layout.valid() &&
        s.program.schema().owns(s.layout) && s.program.schema().capacity()==r.capacity &&
        s.program.schema().mla_layout()==r.mla && s.queue.is_open() &&
        s.token.generation()==s.layout.generation() && !s.native.in_flight();
  }};
const ProviderRegionPreparedVTable region_table{.destroy_prepared=&destroy_region,.alias_support=&alias,.create_invocation=&create_slot,.destroy_invocation=&destroy_slot,.shared_submit=&submit_region,
  .seal_bindings=[](void*prepared,void*invocation)noexcept->Status{
    const auto&r=*static_cast<const Region*>(prepared);
    if(!r.graph_capture)return Status::Ok();
    const auto&s=*static_cast<const Slot*>(invocation);
    // Ordinary Graph binding preparation already compares every exact port
    // description/resource/offset/span. This checks the fresh native episode
    // even when that unchanged physical recording remains dormant.
    return cd::BatchGraphAccess::validate_prepared(s.token,contract(r));
  },
  .query_extension=[](void*,c::ProviderExtensionId id)noexcept->const void*{
    if(id==gd::kRegionCaptureBindingsExtensionId)return &capture_bindings;
    return id==gd::kRegionProjectedReadExtensionId?&projected_table:nullptr;
  }};
struct BatchGeometry final {
  c::BatchCapacity capacity{};
  c::MlaCacheLayout mla=c::MlaCacheLayout::kNone;
  c::BatchProgramOptions math{};
  c::BatchLayoutVersion initial_layout;
};
Result<BatchGeometry> geometry(const c::Operation&operation)noexcept{
  BatchGeometry result;
  const auto set=[&](const auto&op){
    result.math.scale=op.scale;result.math.causal=op.causal;
    result.math.window_left=op.window_left;
  };
  if(const auto paged=operation.custom_as<PagedAttention>()){
    result.initial_layout=paged->layout;result.capacity=paged->layout.origin_schema().capacity();
    set(paged->options);result.math.allow_bf16_partials=paged->options.allow_bf16_partials;
    if(!result.initial_layout.valid() || result.capacity.storage!=c::BatchStorage::kPaged ||
        paged->layout.origin_schema().mla_layout()!=c::MlaCacheLayout::kNone)
      return std::unexpected(Status::InvalidArgument("owned page Graph needs its GQA layout"));
  }else if(const auto paged_mla=operation.custom_as<PagedMlaAttention>()){
    result.initial_layout=paged_mla->layout;result.capacity=paged_mla->layout.origin_schema().capacity();
    result.mla=paged_mla->layout.origin_schema().mla_layout();set(paged_mla->options);
    result.math.allow_bf16_partials=paged_mla->options.allow_bf16_partials;
    if(!result.initial_layout.valid() || result.capacity.storage!=c::BatchStorage::kPaged ||
        result.mla==c::MlaCacheLayout::kNone)
      return std::unexpected(Status::InvalidArgument("owned MLA Graph needs its compact-cache layout"));
  }else if(const auto explicit_paged=operation.custom_as<c::PagedAttention>()){
    result.capacity=explicit_paged->capacity;set(*explicit_paged);result.math.allow_bf16_partials=explicit_paged->allow_bf16_partials;
  }else if(const auto ragged=operation.custom_as<c::RaggedAttention>()){
    result.capacity=ragged->capacity;set(*ragged);result.math.allow_bf16_partials=ragged->allow_bf16_partials;
  }else if(const auto explicit_paged_mla=operation.custom_as<c::PagedMlaAttention>()){
    HYLE_ASSIGN_OR_RETURN(result.capacity,c::mla_batch_capacity(explicit_paged_mla->capacity));
    result.mla=explicit_paged_mla->capacity.layout;set(*explicit_paged_mla);
  }else if(const auto ragged_mla=operation.custom_as<c::RaggedMlaAttention>()){
    HYLE_ASSIGN_OR_RETURN(result.capacity,c::mla_batch_capacity(ragged_mla->capacity));
    result.mla=ragged_mla->capacity.layout;set(*ragged_mla);
  }else return std::unexpected(Status::Unimplemented("not a typed native batch operation"));
  return result;
}
struct AdapterContext final {
  cd::ContextRef context;cd::ProviderExtensionRef source;mem::MemoryResourceRef resource;
  c::ProviderRuntimeProperties runtime=default_runtime;
};
Result<c::ProviderRuntimeProperties> batch_runtime(const cd::ProviderExtensionRef&source)noexcept{
  struct Call{c::ProviderRuntimeProperties properties=default_runtime;}call;
  HYLE_RETURN_IF_ERROR(cd::ProviderExtensionAccess::call(source,
      [](void*context,const void*table,void*opaque)noexcept->Status{
        const auto*provider=static_cast<const cd::BatchProviderVTable*>(table);
        auto&properties=static_cast<Call*>(opaque)->properties;
        if(provider->runtime)properties=provider->runtime(context);
        properties.capture_compatible=properties.capture_compatible &&
            provider->supports_command_capture && !properties.blocks_host &&
            !properties.uses_private_threads;
        return Status::Ok();
      },&call));
  const auto&properties=call.properties;
  if(!properties.valid() || properties.mode!=c::ProviderRuntimeMode::kSharedQueue ||
      properties.workspace_owner!=c::ProviderWorkspaceOwner::kProvider ||
      properties.replay_class!=c::ProviderReplayClass::kManaged)
    return std::unexpected(Status::InvalidArgument("batch capability needs its shared managed slot runtime"));
  return properties;
}
Status qualify_geometry(const AdapterContext&adapter,const BatchGeometry&shape,
    mem::MemoryResourceRef resource={})noexcept{
  const auto capability=shape.mla==c::MlaCacheLayout::kNone?
      cd::kBatchPlanningExtensionId:cd::kMlaBatchPlanningExtensionId;
  if(adapter.source.id()!=capability)return Status::Unimplemented("native batch capability has different payload semantics");
  auto bound=shape;bound.math.resource=resource.valid()?resource:adapter.resource;
  struct Call{const BatchGeometry*shape;};Call call{&bound};
  return cd::ProviderExtensionAccess::call(adapter.source,[](void*context,const void*table,void*opaque)noexcept{
    const auto&shape=*static_cast<Call*>(opaque)->shape;
    const auto*table_value=static_cast<const cd::BatchProviderVTable*>(table);
    if(!table_value->qualify)return Status::FailedPrecondition("native batch capability has no qualifier");
    return table_value->qualify(context,shape.capacity,shape.math);
  },&call);
}
Status propose_native(void*opaque,const RegionDiscoveryRequest&request,mem::AllocatorRef,RegionProposalWriter&writer)noexcept{
  const auto&graph=request.graph;
  if(!request.allowed_nodes.empty() && request.allowed_nodes.size()!=graph.node_count())
    return Status::InvalidArgument("native batch discovery mask size differs");
  const auto&adapter=*static_cast<AdapterContext*>(opaque);
  for(std::size_t i=0;i<graph.node_count();++i){
    if(!request.allowed_nodes.empty() && !request.allowed_nodes[i])continue;
    HYLE_ASSIGN_OR_RETURN(auto node,graph.node_id(i));
    HYLE_ASSIGN_OR_RETURN(auto kind,graph.operation_kind(node));
    if(kind!=c::OperationKind::kCustom)continue;
    HYLE_ASSIGN_OR_RETURN(auto operation,graph.operation(node));
    auto shape=geometry(operation);if(!shape){if(shape.error().IsUnimplemented())continue;return shape.error();}
    const auto admitted=qualify_geometry(adapter,*shape);
    if(!admitted.ok()){if(admitted.IsUnimplemented())continue;return admitted;}
    const std::array nodes{node};
    HYLE_ASSIGN_OR_RETURN(auto added,writer.try_add({.nodes=nodes,.route=1,
        .replay_class=adapter.runtime.replay_class,
        .implementation=c::ProviderImplementationId{0x42415443484e4154ULL}}));
    if(!added)break;
  }return Status::Ok();
}
Status native_resources(void*opaque,const RegionCompileRequest&r,mem::AllocatorRef,
    std::span<RegionInputRequirement>inputs,std::span<RegionOutputRequirement>outputs,
    RegionResourceProposal*proposal)noexcept{
  if(!r.graph || r.nodes.size()!=1 || outputs.size()!=1 || r.has_unknown_effects || !r.effect_accesses.empty())
    return Status::Unimplemented("native batch requires its pure typed node");
  HYLE_ASSIGN_OR_RETURN(auto operation,r.graph->operation(r.nodes[0]));HYLE_ASSIGN_OR_RETURN(auto shape,geometry(operation));
  const auto count=(shape.mla==c::MlaCacheLayout::kNone?3u:2u)+(shape.initial_layout.valid()?0u:1u);
  if(inputs.size()!=r.inputs.size() || inputs.empty() || inputs.size()>count)return Status::InvalidArgument("native batch boundary port count differs");
  HYLE_RETURN_IF_ERROR(qualify_geometry(*static_cast<AdapterContext*>(opaque),shape));
  const auto&adapter=*static_cast<AdapterContext*>(opaque);
  const c::ResourceRequirement requirement{
      .required_properties=adapter.context.device().device_class()==c::DeviceClass::kCpu?
          mem::MemoryProperty::kHostAccessible:mem::MemoryProperty::kDeviceAccessible,
      .minimum_alignment=16};
  std::fill(inputs.begin(),inputs.end(),RegionInputRequirement{.resource=requirement});
  if(!shape.initial_layout.valid()){
    HYLE_ASSIGN_OR_RETURN(auto metadata,r.graph->node_input(r.nodes[0],count-1));
    const auto found=std::find(r.inputs.begin(),r.inputs.end(),metadata);
    if(found==r.inputs.end())return Status::InvalidArgument("native batch metadata boundary is missing");
    inputs[std::size_t(found-r.inputs.begin())].resource.minimum_alignment=8;
  }
  outputs[0]={.resource=requirement};
  *proposal={.runtime=adapter.runtime,.workspace_resource=requirement};return Status::Ok();
}
c::ProviderPrepareOutcome compile_native_region(void*opaque,const RegionCompileRequest&r,
    mem::AllocatorRef allocator,ProviderRegionPrepared*out)noexcept{
  *out={};
  if(!r.graph || r.nodes.size()!=1 || r.outputs.size()!=1 || r.output_resources.size()!=1)
    return c::ProviderPrepareOutcome::unsupported(Status::Unimplemented("native batch requires one typed region"));
  auto operation=r.graph->operation(r.nodes[0]);if(!operation)return c::ProviderPrepareOutcome::error(operation.error());
  auto shape=geometry(*operation);if(!shape)return c::ProviderPrepareOutcome::unsupported(shape.error());
  const auto&adapter=*static_cast<AdapterContext*>(opaque);
  const auto admitted=qualify_geometry(adapter,*shape,r.output_resources[0]);
  if(!admitted.ok())return admitted.IsUnimplemented()?c::ProviderPrepareOutcome::unsupported(admitted):c::ProviderPrepareOutcome::error(admitted);
  const auto count=(shape->mla==c::MlaCacheLayout::kNone?3u:2u)+(shape->initial_layout.valid()?0u:1u);
  if(r.inputs.empty() || r.inputs.size()>count)return c::ProviderPrepareOutcome::unsupported(Status::Unimplemented("native batch region port count differs"));
  auto query_value=r.graph->node_input(r.nodes[0],0);
  if(!query_value)return c::ProviderPrepareOutcome::error(query_value.error());
  auto query=r.graph->value_description(*query_value),output=r.graph->value_description(r.outputs[0]);
  if(!query || !output)return c::ProviderPrepareOutcome::error(!query?query.error():output.error());
  if(query->rank()!=3 || query->dimensions()[0]<0 ||
      std::uint64_t(query->dimensions()[0])>shape->capacity.max_query_rows)
    return c::ProviderPrepareOutcome::error(Status::InvalidArgument("native batch concrete query extent exceeds schema"));
  const auto query_rows=static_cast<std::uint32_t>(query->dimensions()[0]);
  auto expected_query=c::batch_query_description(shape->capacity,query_rows);
  auto expected_output=c::batch_output_description(shape->capacity,query_rows);
  if(!expected_query || !expected_output)
    return c::ProviderPrepareOutcome::error(!expected_query?expected_query.error():expected_output.error());
  if(*query!=*expected_query || *output!=*expected_output ||
      (shape->initial_layout.valid() && query_rows!=shape->initial_layout.facts().query_rows) ||
      (!shape->initial_layout.valid() && shape->mla!=c::MlaCacheLayout::kNone &&
       query_rows!=shape->capacity.max_query_rows))
    return c::ProviderPrepareOutcome::error(Status::InvalidArgument("native batch concrete Q/output geometry differs"));
  auto*p=allocator.construct<Region>();if(!p)return c::ProviderPrepareOutcome::error(Status::ResourceExhausted("native batch region metadata"));
  *out={.vtable=&region_table,.state=p,.properties={.workspace={0,16},
      .persistent_bytes=sizeof(Region),.runtime=adapter.runtime,.invocation_private_bytes=std::nullopt}};
  p->node=r.nodes[0];p->context=adapter.context;p->source=adapter.source;
  p->capacity=shape->capacity;p->mla=shape->mla;p->options=shape->math;p->resource=r.output_resources[0];p->boundary_count=r.inputs.size();
  p->initial_layout=shape->initial_layout;p->query_rows=query_rows;p->graph_capture=adapter.runtime.capture_compatible;
  for(std::size_t i=0;i<count;++i){
    auto value=r.graph->node_input(p->node,i);if(!value)return c::ProviderPrepareOutcome::error(value.error());
    const auto found=std::find(r.inputs.begin(),r.inputs.end(),*value);
    if(found==r.inputs.end())return c::ProviderPrepareOutcome::error(Status::Internal("native batch port mapping differs"));
    p->inputs[i]=std::size_t(found-r.inputs.begin());p->values[i]=*value;
    if(!p->initial_layout.valid() && i+1==count){
      if(value->kind()!=ValueKind::kInput)return c::ProviderPrepareOutcome::unsupported(Status::Unimplemented("native batch layout needs its owned external binding"));
      p->metadata=*value;
    }
  }
  return c::ProviderPrepareOutcome::prepared();
}
const RegionProviderVTable native_region_provider{&propose_native,&native_resources,&compile_native_region};
} // namespace
Result<CompiledGraph> compile(const Graph&graph,CompileEnvironment environment,CompileOptions options)noexcept{
  auto allocator=mem::system_allocator();
  if(!environment.targets.empty() && environment.targets[0].source)
    allocator=environment.targets[0].source.metadata_allocator();
  if(options.resource_budget.valid()){HYLE_ASSIGN_OR_RETURN(allocator,options.resource_budget.metadata_allocator());}
  if(!allocator.valid())return std::unexpected(Status::InvalidArgument("native capability assembly allocator is invalid"));
  if(environment.targets.size()>UINT32_MAX)return std::unexpected(Status::OutOfRange("native target identity count overflows"));
  bool has_append=false,has_store=false;
  for(std::size_t i=0;i<graph.node_count();++i){HYLE_ASSIGN_OR_RETURN(auto node,graph.node_id(i));
    // Logical views have no concrete Operation until specialization. Batch
    // discovery only inspects its typed custom nodes in this mixed graph.
    HYLE_ASSIGN_OR_RETURN(auto kind,graph.operation_kind(node));
    if(kind!=c::OperationKind::kCustom)continue;
    HYLE_ASSIGN_OR_RETURN(auto operation,graph.operation(node));has_append|=operation.custom_as<PagedAppend>().has_value();
    has_store|=operation.custom_as<gs::Store>().has_value();}
  try{
    std::size_t maximum=has_append?environment.targets.size():0;
    if(has_store){if(maximum>SIZE_MAX-environment.targets.size())return std::unexpected(Status::OutOfRange("slot capability count overflows"));maximum+=environment.targets.size();}
    for(const auto&target:environment.targets){
      if(!target.source)return std::unexpected(Status::InvalidArgument("native capability target has no execution source"));
      auto context=gd::ExecutionSourceAccess::compute(target.source);
      if(!context){if(context.error().IsUnimplemented())continue;return std::unexpected(context.error());}
      if((*context)->provider_count()>UINT32_MAX)return std::unexpected(Status::OutOfRange("native provider identity count overflows"));
      if((*context)->provider_count()>(SIZE_MAX-maximum)/2)return std::unexpected(Status::OutOfRange("native capability count overflows"));
      maximum+=2*(*context)->provider_count();
    }
    std::vector<AdapterContext,mem::StdAllocator<AdapterContext>>contexts{mem::StdAllocator<AdapterContext>(allocator)};
    std::vector<gd::RegionAdapter,mem::StdAllocator<gd::RegionAdapter>>adapters{mem::StdAllocator<gd::RegionAdapter>(allocator)};
    std::vector<detail::AppendAdapter,mem::StdAllocator<detail::AppendAdapter>>append_contexts{mem::StdAllocator<detail::AppendAdapter>(allocator)};
    std::vector<gs::detail::StoreAdapter,mem::StdAllocator<gs::detail::StoreAdapter>>store_contexts{mem::StdAllocator<gs::detail::StoreAdapter>(allocator)};
    contexts.reserve(maximum);adapters.reserve(maximum);
    append_contexts.reserve(environment.targets.size());
    if(has_store)store_contexts.reserve(environment.targets.size());
    for(std::size_t target=0;target<environment.targets.size();++target){
      const auto&endpoint=environment.targets[target].source;
      auto retained=gd::ExecutionSourceAccess::compute(endpoint);
      if(!retained){if(retained.error().IsUnimplemented())continue;return std::unexpected(retained.error());}
      const auto&context=**retained;
      const auto storage=environment.targets[target].default_storage;
      if(!storage.valid() || storage.ordinal()>=environment.storage_domains.size())return std::unexpected(Status::InvalidArgument("native capability target storage is invalid"));
      if(has_store){
        const auto*provider=gs::detail::store_provider();
        HYLE_ASSIGN_OR_RETURN(auto source,gd::GraphRegionSource::shared_queue(endpoint,provider));
        store_contexts.push_back({environment.storage_domains[storage.ordinal()].resource,context.device().device_class()});
        adapters.push_back({std::move(source),UINT32_MAX,ExecutionTargetId{static_cast<std::uint32_t>(target)},provider,&store_contexts.back()});
      }
      if(has_append){
        HYLE_ASSIGN_OR_RETURN(auto append_source,gd::GraphRegionSource::shared_queue(endpoint,detail::PagedAppendAccess::provider()));
        detail::AppendAdapter append_adapter{environment.storage_domains[storage.ordinal()].resource,context.device().device_class(),{}};
        auto native=cd::BackendExtensionAccess::query(context,cd::kPagedAppendExtensionId);
        if(native){
          HYLE_RETURN_IF_ERROR(cd::BackendExtensionAccess::call(*native,[](void*,const void*table,void*user)noexcept->Status{
            const auto*t=static_cast<const cd::PagedAppendBackendVTable*>(table);
            *static_cast<bool*>(user)=t && t->prepare && t->destroy && t->enqueue && t->supports_command_capture;
            return Status::Ok();
          },&append_adapter.native_capture));
          if(append_adapter.native_capture)append_adapter.native=std::move(*native);
        }else if(!native.error().IsUnimplemented())return std::unexpected(native.error());
        append_contexts.push_back(std::move(append_adapter));
        adapters.push_back({std::move(append_source),UINT32_MAX,ExecutionTargetId{static_cast<std::uint32_t>(target)},
            detail::PagedAppendAccess::provider(),&append_contexts.back()});
      }
      for(std::size_t provider=0;provider<context.provider_count();++provider){
        for(const auto capability:{cd::kBatchPlanningExtensionId,cd::kMlaBatchPlanningExtensionId}){
          auto source=cd::ProviderExtensionAccess::query(context,provider,capability);
          if(!source){if(source.error().IsUnimplemented())continue;return std::unexpected(source.error());}
          HYLE_ASSIGN_OR_RETURN(auto runtime,batch_runtime(*source));
          HYLE_ASSIGN_OR_RETURN(auto region_source,gd::GraphRegionSource::shared(endpoint,*source));
          contexts.push_back({context,std::move(*source),environment.storage_domains[storage.ordinal()].resource,runtime});
          adapters.push_back({std::move(region_source),static_cast<std::uint32_t>(provider),
              ExecutionTargetId{static_cast<std::uint32_t>(target)},&native_region_provider,&contexts.back()});
        }
      }
    }
    return gd::compile_with_adapters(graph,environment,std::move(options),adapters);
  }catch(const std::bad_alloc&){return std::unexpected(Status::ResourceExhausted("native capability assembly metadata"));}
   catch(...){return std::unexpected(Status::Internal("native capability assembly failed"));}
}

namespace {
struct PreparedEntry final {
  Region*region=nullptr;Slot*slot=nullptr;ExecutionLaneId lane;
  const NodeLayout*binding=nullptr;compute::Queue*queue=nullptr;
  cd::QueueRef queue_owner{};
  PreparedEntry*metadata_source=nullptr;
  const cd::ProviderExtensionRef*owner=nullptr;
  std::uint32_t step=UINT32_MAX;
};
using PreparedEntries=std::vector<PreparedEntry,mem::StdAllocator<PreparedEntry>>;
struct CommandChange final {
  Invocation*invocation;Slot*slot;std::span<const PreparedEntry>entries;
};
Status before_command_change(void*opaque)noexcept{
  const auto&change=*static_cast<const CommandChange*>(opaque);
  // A dependent recording may be in a different outer range. Invalidate it
  // before its original metadata owner can release that carrier, including
  // after reset has correctly dropped all old read permissions.
  for(const auto&entry:change.entries)
    if(entry.slot==change.slot || entry.slot->metadata_owner==change.slot)
      HYLE_RETURN_IF_ERROR(gd::RegionAccess::invalidate_commands(*change.invocation,entry.step));
  return Status::Ok();
}
// Only these top-level selected identities survive the call. Current layouts,
// Queue references and the links into this workspace are always call-scoped.
struct EntryBindings final {
  PreparedEntries&entries;
  ~EntryBindings(){for(auto&entry:entries){entry.binding=nullptr;entry.queue=nullptr;
    entry.queue_owner={};entry.metadata_source=nullptr;}}
};
bool same_allocator(mem::AllocatorRef a,mem::AllocatorRef b)noexcept{
  return a.state()==b.state() && a.alloc_fn()==b.alloc_fn() &&
      a.free_fn()==b.free_fn() && a.realloc_fn()==b.realloc_fn();
}
Status bind_metadata(graph::Invocation&invocation,ValueId value,c::TensorDesc description,
    c::ReadLease lease)noexcept{
  return gd::RegionAccess::bind_configuration(invocation,value,description,std::move(lease));
}
Status prepare_native(graph::Invocation&invocation,std::span<const QueueBinding>queues,
    std::span<const NodeLayout>layouts,PrepareOptions options,
    std::span<const detail::NativeBatchRegionRef>native_regions,
    std::span<const NodeLayout>all_layouts,PreparedEntries*workspace=nullptr,
    bool selected=false)noexcept{
  if(!options.allocator.valid() || !options.host_workspace.valid() ||
      !options.max_float_workspace_bytes || !options.max_int_workspace_bytes)
    return Status::InvalidArgument("native Graph preparation resources are invalid");
  if(options.resource_budget.valid()){HYLE_ASSIGN_OR_RETURN(options.allocator,options.resource_budget.metadata_allocator());}
  try{
    PreparedEntries local{mem::StdAllocator<PreparedEntry>(options.allocator)};
    auto&entries=workspace?*workspace:local;
    EntryBindings bindings{entries};
    // Every cached entry still passes the original owner and idle gate before
    // any token, metadata binding or source permission is changed.
    if(selected){
      for(const auto&entry:entries){const auto&r=*entry.region;
        if(!entry.owner || entry.owner->id()!=r.source.id() ||
            entry.owner->provider_instance_id()!=r.source.provider_instance_id())
          return Status::InvalidArgument("native Graph capability owner differs");
      }
    }else{
      entries.clear();
      struct Visit{PreparedEntries*entries;};Visit visit{&entries};
      HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(invocation,[](void*opaque,const gd::RegionAccessView&view)noexcept->Status{
        if(view.prepared_vtable!=&region_table)return Status::Ok();
        auto&r=*static_cast<Region*>(view.prepared);
        if(!view.owner || view.owner->id()!=r.source.id() ||
            view.owner->provider_instance_id()!=r.source.provider_instance_id())
          return Status::InvalidArgument("native Graph capability owner differs");
        try{PreparedEntry entry{&r,static_cast<Slot*>(view.invocation),view.lane};
          entry.owner=view.owner;entry.step=view.step;
          static_cast<Visit*>(opaque)->entries->push_back(std::move(entry));}
        catch(...){return Status::ResourceExhausted("native Graph cold binding metadata");}
        return Status::Ok();
      },&visit));
    }
    if(entries.empty())return layouts.empty()?Status::Ok():Status::Unimplemented("Graph has no selected native batch region");
    std::sort(entries.begin(),entries.end(),[](const auto&a,const auto&b){return a.region->node.ordinal()<b.region->node.ordinal();});
    // Preflight the frozen concrete extent before changing any native token,
    // metadata owner or plan. Schema capacity alone cannot authorize a layout
    // with more active rows than this exact selected Graph port can hold.
    for(const auto&binding:layouts){
      if(!binding.node.valid() || !binding.layout.valid())continue;
      const auto found=std::lower_bound(entries.begin(),entries.end(),binding.node.ordinal(),
          [](const auto&entry,std::size_t ordinal){return entry.region->node.ordinal()<ordinal;});
      if(found!=entries.end() && found->region->node==binding.node &&
          binding.layout.facts().query_rows>found->region->query_rows)
        return Status::InvalidArgument("native Graph active queries exceed its concrete Q/output extent");
    }
    for(const auto&entry:entries)if(!entry.region->graph_capture)
      HYLE_RETURN_IF_ERROR(gd::RegionAccess::invalidate_commands(invocation,entry.step));
    for(auto&entry:entries){entry.slot->token={};entry.slot->layout={};entry.slot->keys={};entry.slot->values={};
      cd::BatchGraphAccess::clear_shared_metadata(entry.slot->native);}
    const auto apply=[&]()noexcept->Status{
      if(layouts.size()!=entries.size())return Status::InvalidArgument("native Graph requires every selected node layout");
      for(const auto&binding:layouts){
        auto found=std::lower_bound(entries.begin(),entries.end(),binding.node.ordinal(),
            [](const auto&entry,std::size_t ordinal){return entry.region->node.ordinal()<ordinal;});
        if(found==entries.end() || found->region->node!=binding.node || found->binding || !binding.layout.valid())
          return Status::InvalidArgument("native Graph node/layout is foreign or duplicated");
        const auto&r=*found->region;const auto schema=binding.layout.origin_schema();
        if(!cd::BatchLayoutAccess::same_topology(schema.capacity(),r.capacity) || schema.mla_layout()!=r.mla)
          return Status::InvalidArgument("native Graph layout capacity differs from its typed node");
        if(r.initial_layout.valid() &&
            (!r.initial_layout.origin_schema().owns(binding.layout) ||
             binding.layout.facts().query_rows!=r.initial_layout.facts().query_rows))
          return Status::InvalidArgument("owned page Graph layout identity/active shape differs");
        HYLE_RETURN_IF_ERROR(c::validate_batch_attention_layout(binding.layout,r.options.causal,r.options.window_left));
        found->binding=&binding;
      }
      for(std::size_t i=0;i<entries.size();++i){
        for(std::size_t j=0;j<i;++j){
          if(!entries[i].region->metadata.valid() ||
              entries[j].region->metadata!=entries[i].region->metadata)continue;
          if(entries[j].binding->layout!=entries[i].binding->layout)
            return Status::InvalidArgument("one Graph metadata input requires one owned layout version");
          if(entries[i].region->resource==entries[j].region->resource){
            entries[i].metadata_source=&entries[j];break;
          }
        }
      }
      for(const auto&entry:entries){const auto&r=*entry.region;
        for(const auto&native:native_regions){
          const auto layout=std::find_if(all_layouts.begin(),all_layouts.end(),
              [&](const auto&binding){return binding.node==native.node;});
          if(layout==all_layouts.end())return Status::InvalidArgument("native Graph page layout is missing");
          if(r.metadata.valid() && r.metadata==native.metadata_input && entry.binding->layout!=layout->layout)
            return Status::InvalidArgument("shared metadata input needs identical native/shared layout");
          if(r.capacity.storage==c::BatchStorage::kPaged && native.capacity.storage==c::BatchStorage::kPaged)
            for(std::size_t p=1;p<payloads(r);++p)for(std::size_t q=1;q<native.inputs.size();++q)
              if(r.values[p]==native.inputs[q] && entry.binding->layout!=layout->layout)
                return Status::InvalidArgument("shared page input needs identical native/shared layout");
        }
      }
      for(std::size_t i=0;i<queues.size();++i){
        if(!queues[i].lane.valid() || !queues[i].queue || !queues[i].queue->is_open())return Status::InvalidArgument("native Graph Queue binding is invalid");
        for(std::size_t j=0;j<i;++j)if(queues[j].lane==queues[i].lane)return Status::InvalidArgument("duplicate native Graph Queue binding");
      }
      for(auto&entry:entries){
        const auto queued=std::find_if(queues.begin(),queues.end(),[&](const auto&value){return value.lane==entry.lane;});
        if(queued==queues.end() || !entry.region->context.owns(*queued->queue))return Status::InvalidArgument("native Graph Queue does not belong to its original Context");
        entry.queue=queued->queue;
        HYLE_ASSIGN_OR_RETURN(entry.queue_owner,cd::QueueRef::acquire(*queued->queue));
      }
      for(std::size_t i=0;i<entries.size();++i){
        const auto&a=entries[i];
        for(std::size_t j=0;j<i;++j){
          const auto&b=entries[j];
          if(a.region->capacity.storage==c::BatchStorage::kPaged && b.region->capacity.storage==c::BatchStorage::kPaged){
            for(std::size_t p=1;p<payloads(*a.region);++p)for(std::size_t q=1;q<payloads(*b.region);++q)
              if(a.region->values[p]==b.region->values[q] && a.binding->layout!=b.binding->layout)
                return Status::InvalidArgument("shared page input needs one owned layout version");
          }
          if(a.region->initial_layout.valid() && b.region->initial_layout.valid() &&
              a.binding->layout==b.binding->layout && a.region->resource==b.region->resource &&
              a.queue_owner.same_owner(b.queue_owner)){
            if(!entries[i].metadata_source)entries[i].metadata_source=&entries[j];
          }
        }
      }
      for(auto&entry:entries){
        auto&r=*entry.region;auto&s=*entry.slot;const auto&layout=entry.binding->layout;
        const auto&prior=s.prepared_options;
        const bool reuse=s.program.valid() && s.program.schema().owns(layout) &&
            s.program.schema().capacity()==r.capacity && s.program.schema().mla_layout()==r.mla &&
            s.queue.same_owner(entry.queue_owner) &&
            prior.host_workspace==options.host_workspace &&
            prior.max_float_workspace_bytes==options.max_float_workspace_bytes &&
            prior.max_int_workspace_bytes==options.max_int_workspace_bytes && same_allocator(prior.allocator,options.allocator);
        CommandChange change{&invocation,&s,entries};
        const cd::BeforeCommandChange before_change{&change,&before_command_change};
        if(!reuse){
          HYLE_ASSIGN_OR_RETURN(auto schema,layout.origin_schema().with_cache_storage(
              r.capacity.key_storage,r.capacity.value_storage));
          HYLE_RETURN_IF_ERROR(before_command_change(&change));
          // The actual operation's immutable storage interpretation is cold-bound
          // to this explicit topology domain, never inferred from metadata bytes.
          s.native={};s.program={};
          auto native_options=r.options;native_options.resource=r.resource;
          native_options.host_workspace=options.host_workspace;
          native_options.max_float_workspace_bytes=options.max_float_workspace_bytes;
          native_options.max_int_workspace_bytes=options.max_int_workspace_bytes;
          native_options.max_invocations=1;native_options.allocator=options.allocator;
          native_options.resource_budget=options.resource_budget;
          HYLE_ASSIGN_OR_RETURN(s.program,cd::BatchGraphAccess::prepare(r.context,r.source,std::move(schema),native_options));
          HYLE_ASSIGN_OR_RETURN(s.native,cd::BatchGraphAccess::create(s.program,*entry.queue));
          s.prepared_options=options;s.queue=entry.queue_owner;
        }
        if(entry.metadata_source){
          HYLE_ASSIGN_OR_RETURN(s.token,cd::BatchGraphAccess::prepare_shared(
              s.native,layout,options.update,entry.metadata_source->slot->native,before_change));
        }else{HYLE_ASSIGN_OR_RETURN(s.token,cd::BatchGraphAccess::prepare(
            s.native,layout,options.update,before_change));}
        const auto*source=entry.metadata_source?entry.metadata_source->slot:nullptr;
        s.metadata_owner=source?(source->metadata_owner?source->metadata_owner:entry.metadata_source->slot):&s;
        s.layout=layout;
      }
      return Status::Ok();
    };
    // The enclosing configuration transaction owns cleanup, including native
    // partial failure. It must not discard unrelated caller bindings here.
    return apply();
  }catch(const std::bad_alloc&){return Status::ResourceExhausted("native Graph preparation metadata");}
   catch(...){return Status::Internal("native Graph preparation failed");}
}
Result<c::BatchPlanReport> shared_native_report(graph::Invocation&invocation,NodeId node)noexcept{
  struct Call{NodeId node;bool found=false;c::BatchPlanReport report{};};Call call{node};
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(invocation,[](void*opaque,const gd::RegionAccessView&view)noexcept->Status{
    if(view.prepared_vtable!=&region_table)return Status::Ok();
    auto&r=*static_cast<Region*>(view.prepared);auto&call=*static_cast<Call*>(opaque);
    if(r.node!=call.node)return Status::Ok();
    auto&s=*static_cast<Slot*>(view.invocation);
    if(!s.native.valid())return Status::FailedPrecondition("native Graph batch has not been prepared");
    call.found=true;call.report=s.native.report();return Status::Ok();
  },&call));
  if(!call.found)return std::unexpected(Status::InvalidArgument("node is not a selected native batch region"));
  return call.report;
}
Status capture_native(graph::Invocation&invocation,std::span<const QueueBinding>queues)noexcept{
  // Legacy providers may replace their inner artifact here. Actual outer-
  // capture providers only validate the current bound ports; no nested
  // recording or native command replacement takes place.
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(invocation,[](void*opaque,const gd::RegionAccessView&view)noexcept->Status{
    if(view.prepared_vtable!=&region_table)return Status::Ok();
    if(static_cast<const Region*>(view.prepared)->graph_capture)return Status::Ok();
    return gd::RegionAccess::invalidate_commands(*static_cast<Invocation*>(opaque),view.step);
  },&invocation));
  struct Call{std::span<const QueueBinding>queues;};Call call{queues};
  return gd::RegionAccess::visit_bound(invocation,[](void*opaque,const gd::RegionAccessView&view)noexcept->Status{
    if(view.prepared_vtable!=&region_table)return Status::Ok();
    auto&r=*static_cast<Region*>(view.prepared);
    auto&s=*static_cast<Slot*>(view.invocation);const auto&call=*static_cast<Call*>(opaque);
    const auto found=std::find_if(call.queues.begin(),call.queues.end(),[&](const auto&queue){return queue.lane==view.lane;});
    if(found==call.queues.end() || !found->queue || !s.queue.same_owner(*found->queue))
      return Status::InvalidArgument("native Graph replay Queue differs from preparation");
    if(view.inputs.size()!=r.boundary_count || view.outputs.size()!=1)
      return Status::InvalidArgument("native Graph replay needs its complete original ports");
    std::array<c::ConstTensorView,4>ordered;
    for(std::size_t i=0;i<ports(r);++i)ordered[i]=view.inputs[r.inputs[i]];
    if(r.graph_capture)
      return cd::BatchGraphAccess::validate_replay(s.token,*found->queue,std::span(ordered).first(ports(r)),view.outputs,contract(r));
    return cd::BatchGraphAccess::prepare_replay(s.token,*found->queue,std::span(ordered).first(ports(r)),view.outputs,contract(r));
  },&call);
}
} // namespace
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
namespace {
struct BatchResources final{
  struct Metadata final {
    ValueId input;c::TensorDesc description;
    c::LeasePool pool;c::WriteLease mirror;c::ReadLease ready;
    c::BatchLayoutVersion layout;
    bool native_owner=false,bound=false;
  };
  struct Row final {
    PagedRowCacheGather reader;detail::RowCacheFacts facts;
    c::LeasePool pool;c::WriteLease mirror;c::ReadLease ready;
    c::RowCacheLayoutVersion layout;c::PagedRowCacheRead frame;
    ExecutionLaneId lane{};bool prepared=false;
  };
  using Pending=detail::PendingPageInput;
  explicit BatchResources(mem::AllocatorRef a):allocator(a),metadata(mem::StdAllocator<Metadata>(a)),rows(mem::StdAllocator<Row>(a)),
    frame_slots(mem::StdAllocator<Slot*>(a)),selected_entries(mem::StdAllocator<PreparedEntry>(a)),
    append_queues(mem::StdAllocator<detail::AppendQueueBinding>(a)),
    shared_layouts(mem::StdAllocator<NodeLayout>(a)),native_regions(mem::StdAllocator<detail::NativeBatchRegionRef>(a)),
    pending_regions(mem::StdAllocator<detail::NativePendingRegionRef>(a)),inputs(mem::StdAllocator<Pending>(a)),
    admitted(mem::StdAllocator<ExecutionLaneId>(a)),page_reads(a){}
  void destroy()noexcept{const auto a=allocator;a.destroy(this);}
  Status reusable()noexcept{
    if((dependent_ready && !dependent.available()) ||
       (source_capacity && (!source.available() || source.update().valid())))
      return Status::Unavailable("Graph still retains its typed source controls");
    if(native_source.state)HYLE_RETURN_IF_ERROR(native_source.table->reusable(native_source.state));
    return Status::Ok();
  }
  bool contains(ValueId value)const noexcept{return std::any_of(inputs.begin(),inputs.end(),
    [&](const auto&input){return input.value==value;});}
  mem::AllocatorRef allocator;compute::ResourceBudget budget;
  std::vector<Metadata,mem::StdAllocator<Metadata>>metadata;
  std::vector<Row,mem::StdAllocator<Row>>rows;
  // Borrowed only for reset of actual guards owned by the original slots.
  // Destruction never dereferences them; State owns their lifetime/order.
  std::vector<Slot*,mem::StdAllocator<Slot*>>frame_slots;
  // Exact selections belong to this original InvocationState/Plan. No reset
  // replaces their top-level slots; destruction never dereferences them.
  PreparedEntries selected_entries;
  std::vector<detail::AppendQueueBinding,mem::StdAllocator<detail::AppendQueueBinding>>append_queues;
  mem::MemoryResourceRef append_host;
  bool has_appends=false,append_ready=false;
  detail::NodeLayouts shared_layouts;
  bool selection_ready=false;
  detail::NativeBatchRegions native_regions;
  detail::NativePendingRegions pending_regions;
  std::vector<Pending,mem::StdAllocator<Pending>>inputs;
  std::vector<ExecutionLaneId,mem::StdAllocator<ExecutionLaneId>>admitted;
  cd::CopyOutcomeSource<c::CopyBatchUpdate,cd::CopyStreamAccess>source;
  detail::NativePendingSourceOwner native_source;
  cd::DependentCompletion dependent;cd::OutcomeSources sources{};
  cd::BatchPageReads page_reads;
  std::size_t source_capacity=0,native_source_capacity=0,pending_capacity=0;
  bool dependent_ready=false;
};
const gd::ResourcePayloadVTable resource_table{
  [](void*opaque,Invocation&invocation)noexcept->Status{
    auto&p=*static_cast<BatchResources*>(opaque);HYLE_RETURN_IF_ERROR(p.reusable());
    HYLE_RETURN_IF_ERROR(detail::PagedAppendAccess::reset(invocation));
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::revoke_preparation(invocation));
    p.append_ready=false;for(auto&q:p.append_queues)q.queue={};p.append_host={};
    // Readiness and page permissions expire on every reset. Unknown/native
    // reset callbacks still discard their recordings before releasing any
    // command resources. Qualified shared slots retain their native owners;
    // their before-change gate covers actual later resource replacement.
    for(const auto&region:p.native_regions)
      HYLE_RETURN_IF_ERROR(gd::RegionAccess::invalidate_commands(invocation,region.step));
    for(const auto&region:p.pending_regions)
      HYLE_RETURN_IF_ERROR(gd::RegionAccess::invalidate_commands(invocation,region.step));
    for(const auto*slot:p.frame_slots)if(!slot->region->graph_capture)
      HYLE_RETURN_IF_ERROR(gd::RegionAccess::invalidate_commands(invocation,slot->selected_step));
    HYLE_RETURN_IF_ERROR(detail::reset_native_batch_regions(p.native_regions));
    detail::reset_native_pending_regions(p.pending_regions);
    // All rejecting qualifications precede this infallible resource release.
    for(auto*slot:p.frame_slots){slot->keys={};slot->values={};slot->written_keys={};slot->written_values={};slot->token={};slot->layout={};
      cd::BatchGraphAccess::clear_shared_metadata(slot->native);}
    for(auto&row:p.rows)row.frame={};
    for(auto&metadata:p.metadata){
      metadata.bound=false;
      // A provider's original mirror must become uniquely writable before its
      // next cold prepare. Only our own transfer mirror persists across reset.
      if(metadata.native_owner){metadata.ready={};metadata.layout={};metadata.native_owner=false;}
    }
    p.inputs.clear();p.admitted.clear();p.page_reads.reset();return Status::Ok();
  },
  [](void*opaque)noexcept{static_cast<BatchResources*>(opaque)->destroy();}};
Result<BatchResources*> resources(graph::Invocation&invocation)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto value,gd::RegionAccess::resource_payload(invocation,&resource_table));
  return static_cast<BatchResources*>(value);
}
Result<BatchResources*> ensure_resources(graph::Invocation&invocation,PrepareOptions options)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto current,resources(invocation));if(current)return current;
  if(!options.allocator.valid())return std::unexpected(Status::InvalidArgument("typed Graph allocator is invalid"));
  if(options.resource_budget.valid()){HYLE_ASSIGN_OR_RETURN(options.allocator,options.resource_budget.metadata_allocator());}
  HYLE_RETURN_IF_ERROR(invocation.prepare_resources({.allocator=options.allocator,.resource_budget=options.resource_budget}));
  auto*p=options.allocator.construct<BatchResources>(options.allocator);
  if(!p)return std::unexpected(Status::ResourceExhausted("typed Graph resource owner metadata"));
  p->budget=options.resource_budget;
  auto status=gd::RegionAccess::attach_resource_payload(invocation,&resource_table,p);
  if(!status.ok()){p->destroy();return std::unexpected(status);}return p;
}
Status register_slots(graph::Invocation&invocation,BatchResources&p)noexcept{
  HYLE_RETURN_IF_ERROR(p.reusable());p.frame_slots.clear();p.append_queues.clear();p.has_appends=false;
  struct Call{BatchResources*p;};Call call{&p};
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(invocation,[](void*opaque,const gd::RegionAccessView&view)noexcept->Status{
    auto&p=*static_cast<Call*>(opaque)->p;
    if(detail::PagedAppendAccess::selected(view)){
      p.has_appends=true;
      if(detail::PagedAppendAccess::needs_queue(view) &&
          std::none_of(p.append_queues.begin(),p.append_queues.end(),[&](const auto&q){return q.lane==view.lane;})){
        try{p.append_queues.push_back({view.lane,{}});}
        catch(...){return Status::ResourceExhausted("writer-only Graph Queue lane metadata");}
      }
    }
    if(view.prepared_vtable!=&region_table)return Status::Ok();
    auto*slot=static_cast<Slot*>(view.invocation);slot->selected_step=view.step;
    try{p.frame_slots.push_back(slot);}
    catch(...){return Status::ResourceExhausted("typed Graph guard slot metadata");}return Status::Ok();
  },&call));return Status::Ok();
}
Status prepare_pending(BatchResources&p,PrepareOptions options)noexcept{
  HYLE_RETURN_IF_ERROR(p.reusable());p.pending_capacity=0;
  const auto capacity=options.max_source_completions;
  if(!capacity)return Status::Ok();
  const auto native_row=[&](const auto&row){return std::any_of(p.pending_regions.begin(),p.pending_regions.end(),
      [&](const auto&region){return std::find(region.inputs.begin(),region.inputs.end(),row.reader.bank_input())!=region.inputs.end();});};
  const bool shared_source=std::any_of(p.rows.begin(),p.rows.end(),[&](const auto&row){return !native_row(row);}) || std::any_of(p.frame_slots.begin(),p.frame_slots.end(),
      [](const auto*slot){return slot->region->capacity.storage==c::BatchStorage::kPaged;});
  detail::NativePendingSourceFactory factory;void*prepared=nullptr;
  for(const auto&region:p.pending_regions){
    const auto selected=region.source;
    if(!selected.table || !selected.table->destroy || !selected.table->reusable ||
        !selected.identity || !selected.create)
      return Status::FailedPrecondition("selected native pending source factory is incomplete");
    if(factory.table && (selected.table!=factory.table || selected.identity!=factory.identity))
      return Status::Unimplemented("one pending source episode requires one actual source owner");
    factory=selected;prepared=region.prepared;
  }
  if(p.frame_slots.size()>SIZE_MAX/2 || p.native_regions.size()>SIZE_MAX/2 ||
      2*p.native_regions.size()>SIZE_MAX-2*p.frame_slots.size() ||
      p.rows.size()>SIZE_MAX-2*p.frame_slots.size()-2*p.native_regions.size())
    return Status::ResourceExhausted("typed Graph pending input capacity overflow");
  std::size_t pages=0;
  for(const auto*slot:p.frame_slots){if(slot->region->capacity.storage!=c::BatchStorage::kPaged)continue;const auto refs=slot->region->capacity.max_page_refs;
    const auto payloads=has_value_port(*slot->region)?2u:1u;
    if(refs>(SIZE_MAX-pages)/payloads)return Status::ResourceExhausted("typed Graph pending page capacity overflow");
    pages+=refs*payloads;}
  for(const auto&row:p.rows){if(native_row(row))continue;const auto refs=row.reader.schema().capacity().max_page_refs;
    if(refs>SIZE_MAX-pages)return Status::ResourceExhausted("typed row Graph page capacity overflow");
    pages+=refs;}
  // A failed cold reservation must leave the prior controls and their charges
  // intact. Keep new owners/capacities local until the last fallible reserve.
  cd::CopyOutcomeSource<c::CopyBatchUpdate,cd::CopyStreamAccess> source;
  const bool replace_source=shared_source && p.source_capacity!=capacity;
  if(replace_source){
    HYLE_ASSIGN_OR_RETURN(source,(cd::CopyOutcomeSource<c::CopyBatchUpdate,cd::CopyStreamAccess>::create(
      capacity,{.allocator=p.allocator,.resource_budget=p.budget})));
  }
  detail::NativePendingSourceOwner native_source;
  const bool replace_native=factory.table && (p.native_source_capacity!=capacity ||
      p.native_source.table!=factory.table || p.native_source.identity!=factory.identity);
  if(replace_native){
    HYLE_ASSIGN_OR_RETURN(auto state,factory.create(prepared,capacity,p.allocator,p.budget));
    if(!state)return Status::Internal("native pending source factory returned no owner");
    native_source.table=factory.table;native_source.state=state;native_source.identity=factory.identity;
  }
  cd::DependentCompletion dependent;
  if(!p.dependent_ready){HYLE_ASSIGN_OR_RETURN(dependent,cd::DependentCompletion::create(p.allocator));}
  const auto input_capacity=2*p.frame_slots.size()+2*p.native_regions.size()+p.rows.size();
  const auto admitted_capacity=p.frame_slots.size()+p.rows.size();
  const bool grow_inputs=p.inputs.capacity()<input_capacity;
  const bool grow_admitted=p.admitted.capacity()<admitted_capacity;
  decltype(p.inputs) inputs(p.inputs.get_allocator());
  decltype(p.admitted) admitted(p.admitted.get_allocator());
  try{if(grow_inputs)inputs.reserve(input_capacity);if(grow_admitted)admitted.reserve(admitted_capacity);}
  catch(...){return Status::ResourceExhausted("typed Graph fixed pending metadata");}
  HYLE_RETURN_IF_ERROR(p.page_reads.reserve(pages));
  // prepare_all revoked the previous configuration before this transaction.
  HYLE_FATAL_ASSERT(p.inputs.empty() && p.admitted.empty());
  if(grow_inputs)p.inputs.swap(inputs);
  if(grow_admitted)p.admitted.swap(admitted);
  if(replace_source){p.source=std::move(source);p.source_capacity=capacity;}
  if(!shared_source){p.source={};p.source_capacity=0;}
  if(replace_native){p.native_source=std::move(native_source);p.native_source_capacity=capacity;}
  if(!factory.table){p.native_source.reset();p.native_source_capacity=0;}
  if(!p.dependent_ready){p.dependent=std::move(dependent);p.dependent_ready=true;}
  p.pending_capacity=capacity;return Status::Ok();
}
Status prepare_rows(graph::Invocation&invocation,const Graph&graph,BatchResources&p,
    std::span<const RowCacheBinding>rows)noexcept{
  // Fully validate private SSA roles against this same immutable source before
  // changing mirrors or uploading any row metadata. The payload keeps facts only.
  for(std::size_t i=0;i<rows.size();++i){
    HYLE_RETURN_IF_ERROR(detail::RowCacheAccess::validate(graph,rows[i].reader));
    for(std::size_t j=0;j<i;++j)if(rows[i].reader.node()==rows[j].reader.node())
      return Status::InvalidArgument("duplicate typed row Graph binding");
  }
  bool same=p.rows.size()==rows.size();
  for(std::size_t i=0;same && i<rows.size();++i)same=p.rows[i].reader.node()==rows[i].reader.node() &&
    p.rows[i].reader.bank_input()==rows[i].reader.bank_input() && p.rows[i].reader.schema().owns(rows[i].layout);
  if(!same){
    std::vector<BatchResources::Row,mem::StdAllocator<BatchResources::Row>>next{mem::StdAllocator<BatchResources::Row>(p.allocator)};
    try{next.reserve(rows.size());}catch(...){return Status::ResourceExhausted("typed row Graph binding metadata");}
    for(const auto&binding:rows){BatchResources::Row row;row.reader=binding.reader;
      HYLE_ASSIGN_OR_RETURN(row.facts,detail::RowCacheAccess::facts(graph,row.reader));
      HYLE_ASSIGN_OR_RETURN(auto resource,detail::RowCacheAccess::resource(invocation.executable(),row.reader,&row.lane));
      const auto bytes=row.reader.schema().metadata_offsets().bytes;
      HYLE_ASSIGN_OR_RETURN(row.pool,c::LeasePool::create(resource,bytes,{.max_blocks=1,.allocator=p.allocator,.resource_budget=p.budget}));
      HYLE_ASSIGN_OR_RETURN(row.mirror,row.pool.allocate(bytes,8));next.push_back(std::move(row));
    }p.rows=std::move(next);
  }
  for(std::size_t i=0;i<rows.size();++i){auto&row=p.rows[i];
    HYLE_RETURN_IF_ERROR(detail::RowCacheAccess::validate_layout(row.reader,rows[i].layout,row.facts));
  }
  return Status::Ok();
}
// A cold owned-metadata transfer uses an existing Source that can copy into the
// actual planned input carrier. No provider selection or transfer exists on submit.
Status copy_metadata(const Executable&executable,BatchResources&p,c::TensorDesc description,
    mem::ConstBufferView source,mem::BufferView target)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto destination,c::TensorView::create(description,target));
  HYLE_ASSIGN_OR_RETURN(auto input,c::ConstTensorView::create(description,source));
  const std::array copies{gd::ConstantCopy{destination,input}};
  for(std::uint32_t lane=0;lane<executable.lane_count();++lane){
    HYLE_ASSIGN_OR_RETURN(auto owner,gd::RegionAccess::execution_source(executable,ExecutionLaneId{lane}));
    HYLE_ASSIGN_OR_RETURN(const bool native,gd::ExecutionSourceAccess::can_copy_constant(owner,copies.front()));
    if(!native)continue;
    c::ResourceBudget::Reservation scratch;
    if(p.budget){
      HYLE_ASSIGN_OR_RETURN(auto bytes,gd::ExecutionSourceAccess::constant_copy_scratch_bytes(owner,copies));
      HYLE_ASSIGN_OR_RETURN(scratch,p.budget.reserve_unattributed(bytes));
    }
    return gd::ExecutionSourceAccess::copy_constants(owner,copies,p.allocator);
  }
  HYLE_ASSIGN_OR_RETURN(auto mapped,source.map_host());
  auto status=target.write(mapped.bytes());status.Update(mapped.close());return status;
}
Status bind_layout_metadata(graph::Invocation&invocation,BatchResources&p,ValueId value,
    const c::BatchLayoutVersion&layout,c::ReadLease native={})noexcept{
    HYLE_ASSIGN_OR_RETURN(auto description,layout.origin_schema().metadata_description());
    HYLE_ASSIGN_OR_RETURN(auto resource,gd::RegionAccess::input_resource(invocation.executable(),value));
    auto found=std::find_if(p.metadata.begin(),p.metadata.end(),[&](const auto&entry){return entry.input==value;});
    if(found==p.metadata.end()){
      BatchResources::Metadata next;next.input=value;next.description=description;
      try{p.metadata.push_back(std::move(next));}catch(...){return Status::ResourceExhausted("native metadata mirror ownership");}
      found=p.metadata.end()-1;
    }
    auto&entry=*found;
    if(entry.description!=description)return Status::InvalidArgument("native metadata input capacity changed");
    if(entry.bound){
      return entry.layout==layout?Status::Ok():Status::InvalidArgument("metadata input has different owned versions");
    }
    if(native.valid()){
      HYLE_ASSIGN_OR_RETURN(auto view,native.view());
      if(view.resource()==resource){
        entry.ready=std::move(native);entry.layout=layout;entry.native_owner=true;
      }
    }
    if(!entry.ready.valid() || entry.layout!=layout){
      if(!entry.pool.valid()){
        const auto bytes=layout.origin_schema().metadata_offsets().bytes;
        HYLE_ASSIGN_OR_RETURN(entry.pool,c::LeasePool::create(resource,bytes,{.max_blocks=1,.allocator=p.allocator,.resource_budget=p.budget}));
      }
      if(entry.ready.valid()){HYLE_ASSIGN_OR_RETURN(entry.mirror,entry.ready.try_write());}
      if(!entry.mirror.valid()){
        HYLE_ASSIGN_OR_RETURN(entry.mirror,entry.pool.allocate(layout.origin_schema().metadata_offsets().bytes,8));
      }
      HYLE_ASSIGN_OR_RETURN(auto metadata,layout.metadata());HYLE_ASSIGN_OR_RETURN(auto source,metadata.view());
      HYLE_ASSIGN_OR_RETURN(auto target,entry.mirror.view());
      HYLE_RETURN_IF_ERROR(copy_metadata(invocation.executable(),p,description,source,target));
      HYLE_ASSIGN_OR_RETURN(entry.ready,entry.mirror.publish());entry.layout=layout;entry.native_owner=false;
    }
    HYLE_RETURN_IF_ERROR(bind_metadata(invocation,entry.input,description,entry.ready));
    entry.bound=true;
    return Status::Ok();
}
Status upload_batch_metadata(graph::Invocation&invocation,BatchResources&p,
    std::span<const NodeLayout>layouts,bool selected=false)noexcept{
  if(selected){
    // frame_slots retain the original plan order (entries are node-sorted).
    for(const auto*slot:p.frame_slots){const auto&r=*slot->region;
      if(!r.metadata.valid())continue;
      const auto&s=*slot;
      HYLE_ASSIGN_OR_RETURN(auto metadata,cd::BatchGraphAccess::metadata(s.native));
      HYLE_RETURN_IF_ERROR(bind_layout_metadata(invocation,p,r.metadata,s.layout,std::move(metadata)));
    }
  }else{
    struct Call{Invocation*invocation;BatchResources*resources;}call{&invocation,&p};
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(invocation,[](void*opaque,const gd::RegionAccessView&view)noexcept->Status{
      if(view.prepared_vtable!=&region_table)return Status::Ok();
      const auto&r=*static_cast<Region*>(view.prepared);if(!r.metadata.valid())return Status::Ok();
      const auto&s=*static_cast<Slot*>(view.invocation);const auto&call=*static_cast<Call*>(opaque);
      HYLE_ASSIGN_OR_RETURN(auto metadata,cd::BatchGraphAccess::metadata(s.native));
      return bind_layout_metadata(*call.invocation,*call.resources,r.metadata,s.layout,std::move(metadata));
    },&call));
  }
  for(const auto&region:p.native_regions){
    if(!region.metadata_input.valid())continue;
    const auto binding=std::find_if(layouts.begin(),layouts.end(),[&](const auto&entry){return entry.node==region.node;});
    if(binding==layouts.end())return Status::InvalidArgument("native metadata layout is missing");
    HYLE_RETURN_IF_ERROR(bind_layout_metadata(invocation,p,region.metadata_input,binding->layout));
  }
  return Status::Ok();
}
Status upload_rows(graph::Invocation&invocation,BatchResources&p,std::span<const RowCacheBinding>rows)noexcept{
  for(std::size_t i=0;i<rows.size();++i){auto&row=p.rows[i];const auto&layout=rows[i].layout;
    if(!row.prepared || row.layout!=layout){row.prepared=false;
      HYLE_RETURN_IF_ERROR(detail::RowCacheAccess::configure(invocation,row.reader,layout));
      if(row.ready.valid()){HYLE_ASSIGN_OR_RETURN(row.mirror,row.ready.try_write());}
      HYLE_ASSIGN_OR_RETURN(auto metadata,layout.metadata());HYLE_ASSIGN_OR_RETURN(auto source,metadata.view());
      HYLE_ASSIGN_OR_RETURN(auto target,row.mirror.view());
      HYLE_RETURN_IF_ERROR(copy_metadata(invocation.executable(),p,row.facts.metadata,source,target));
      HYLE_ASSIGN_OR_RETURN(row.ready,row.mirror.publish());row.layout=layout;row.prepared=true;
    }
    HYLE_RETURN_IF_ERROR(bind_metadata(invocation,detail::RowCacheAccess::metadata(row.reader),row.facts.metadata,row.ready));
  }return Status::Ok();
}
Status clear_native_frames(graph::Invocation&invocation)noexcept{
  return gd::RegionAccess::visit(invocation,[](void*opaque,const gd::RegionAccessView&view)noexcept->Status{
    if(view.prepared_vtable!=&region_table)return Status::Ok();
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::invalidate_commands(*static_cast<Invocation*>(opaque),view.step));
    auto&slot=*static_cast<Slot*>(view.invocation);slot.keys={};slot.values={};slot.token={};slot.layout={};
    cd::BatchGraphAccess::clear_shared_metadata(slot.native);return Status::Ok();
  },&invocation);
}
Status prepare_all(graph::Invocation&invocation,const Graph*graph,std::span<const QueueBinding>queues,
    std::span<const NodeLayout>layouts,std::span<const RowCacheBinding>rows,PrepareOptions options)noexcept{
  if(!rows.empty() && !graph)return Status::InvalidArgument("typed row preparation needs its immutable source Graph");
  // Revoke this layout episode through the original outcome/result gate.
  // Stable caller bindings and their owners survive; page permissions do not.
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::reset_configuration(invocation));
  HYLE_ASSIGN_OR_RETURN(auto*p,resources(invocation));
  const bool cached=p && p->selection_ready && !options.max_source_completions;
  if(p && options.max_source_completions)p->selection_ready=false;
  // A successful cached selection was already cleared by the original payload
  // configuration reset above. Fresh/partial discovery keeps the complete sweep.
  if(!cached)HYLE_RETURN_IF_ERROR(clear_native_frames(invocation));
  const auto apply=[&]()noexcept->Status{
    if(!options.allocator.valid())return Status::InvalidArgument("typed Graph allocator is invalid");
    if(options.resource_budget.valid()){HYLE_ASSIGN_OR_RETURN(options.allocator,options.resource_budget.metadata_allocator());}
    detail::NativeBatchRegions local_native{mem::StdAllocator<detail::NativeBatchRegionRef>(options.allocator)};
    auto&native=cached?p->native_regions:local_native;
    if(!cached)HYLE_RETURN_IF_ERROR(detail::collect_native_batch_regions(invocation,native));
    detail::NodeLayouts local_layouts{mem::StdAllocator<NodeLayout>(options.allocator)};
    auto&shared_layouts=cached?p->shared_layouts:local_layouts;
    HYLE_RETURN_IF_ERROR(detail::split_native_batch_layouts(native,layouts,shared_layouts));
    if(!cached){
      bool typed=!native.empty();
      HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(invocation,[](void*opaque,const gd::RegionAccessView&view)noexcept->Status{
        if(view.prepared_vtable==&region_table || detail::PagedAppendAccess::selected(view))
          *static_cast<bool*>(opaque)=true;
        return Status::Ok();
      },&typed));
      if(typed || !rows.empty() || options.max_source_completions){HYLE_ASSIGN_OR_RETURN(p,ensure_resources(invocation,options));}
    }
    if(p){p->pending_capacity=0;
      if(!cached)HYLE_RETURN_IF_ERROR(register_slots(invocation,*p));
      if(p->has_appends){
        // Retain the supplied real Queue identities, never caller Queue pointers.
        // Publish append_ready only after this whole cold preparation succeeds.
        p->append_ready=false;
        for(auto&prepared:p->append_queues){
          const auto q=std::find_if(queues.begin(),queues.end(),[&](const auto&binding){return binding.lane==prepared.lane;});
          if(q==queues.end() || !q->lane.valid() || !q->queue || !q->queue->is_open())
            return Status::InvalidArgument("writer-only Graph Queue binding is missing or invalid");
          if(std::count_if(queues.begin(),queues.end(),[&](const auto&binding){return binding.lane==prepared.lane;})!=1)
            return Status::InvalidArgument("duplicate writer-only Graph Queue lane");
          HYLE_RETURN_IF_ERROR(gd::RegionAccess::validate_queue(invocation.executable(),q->lane,*q->queue));
          HYLE_ASSIGN_OR_RETURN(prepared.queue,cd::QueueRef::acquire(*q->queue));
        }
        p->append_host=options.host_workspace;
      }
      p->pending_regions.clear();
      if(options.max_source_completions)HYLE_RETURN_IF_ERROR(detail::collect_native_pending_regions(invocation,p->pending_regions));
      if(!cached){
        try{p->native_regions.assign(native.begin(),native.end());}
        catch(...){return Status::ResourceExhausted("native page original slot references");}
      }
      if(graph){HYLE_RETURN_IF_ERROR(prepare_rows(invocation,*graph,*p,rows));}
      else if(!p->rows.empty())return Status::InvalidArgument("row Graph requires its complete cold binding set");
      HYLE_ASSIGN_OR_RETURN(auto extra,detail::native_paged_read_capacity(p->native_regions));
      for(const auto*slot:p->frame_slots)if(slot->region->capacity.storage==c::BatchStorage::kPaged)
        extra+=(has_value_port(*slot->region)?2ULL:1ULL)*slot->region->capacity.max_page_refs;
      for(const auto&row:p->rows)extra+=row.reader.schema().capacity().max_page_refs;
      if(extra>UINT32_MAX-1)return Status::ResourceExhausted("typed Graph page guard capacity overflow");
      // Reserve before native metadata becomes an owned binding. Ready paged
      // inputs need the same exact page-guard capacity as pending inputs.
      HYLE_RETURN_IF_ERROR(gd::RegionAccess::reserve_reads(invocation,std::uint32_t(extra),{.allocator=p->allocator,.resource_budget=p->budget}));
    }
    const auto native_regions=p?std::span<const detail::NativeBatchRegionRef>(p->native_regions):
        std::span<const detail::NativeBatchRegionRef>{};
    const bool cache_selection=p && !options.max_source_completions;
    if(cache_selection){
      try{p->shared_layouts.reserve(shared_layouts.size());}
      catch(...){return Status::ResourceExhausted("native Graph cached layout partition");}
    }
    HYLE_RETURN_IF_ERROR(prepare_native(invocation,queues,shared_layouts,options,native_regions,layouts,
        cache_selection?&p->selected_entries:nullptr,cached));
    if(p){
      for(const auto&region:p->native_regions)
        HYLE_RETURN_IF_ERROR(gd::RegionAccess::invalidate_commands(invocation,region.step));
      HYLE_RETURN_IF_ERROR(detail::prepare_native_batch_regions(p->native_regions,layouts));
    }
    if(p){
      HYLE_RETURN_IF_ERROR(upload_batch_metadata(invocation,*p,layouts,cache_selection));
      HYLE_RETURN_IF_ERROR(upload_rows(invocation,*p,rows));
      HYLE_RETURN_IF_ERROR(prepare_pending(*p,options));
    }
    return Status::Ok();
  };
  auto status=apply();
  if(!status.ok()){
    if(p){p->selection_ready=false;p->pending_capacity=0;
      for(auto&row:p->rows){row.prepared=false;row.frame={};}}
    status.Update(clear_native_frames(invocation));
    status.Update(gd::RegionAccess::reset_configuration(invocation));
  }else if(p){p->append_ready=p->has_appends;if(!options.max_source_completions)p->selection_ready=true;}
  // Cached layout capacity is reusable, but this call's owned versions are not
  // another persistent configuration. Dynamic validation ran above as before.
  if(p)p->shared_layouts.clear();
  return status;
}
Result<BatchResources*> frame_resources(graph::Invocation&invocation)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto p,resources(invocation));
  if(!p)return std::unexpected(Status::FailedPrecondition("prepare typed paged Graph resources before binding frames"));
  return p;
}
Status invalidate_projected_commands(graph::Invocation&invocation,
    std::span<const gd::ProjectedReader>readers)noexcept{
  for(const auto&reader:readers){
    // This shared implementation reads the same ordinary bank carrier and
    // current runtime metadata. Original Graph binding comparison invalidates
    // changed physical ports; fresh page permissions alone change no command.
    if(reader.implementation==&region_table &&
        static_cast<const Region*>(reader.prepared)->graph_capture)continue;
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::invalidate_commands(invocation,reader.step));
  }
  return Status::Ok();
}
template<class Frame>
Status native_frame(graph::Invocation&invocation,ValueId value,const Frame&frame,
    std::vector<gd::ProjectedReader,mem::StdAllocator<gd::ProjectedReader>>&proofs,
    std::vector<ExecutionLaneId,mem::StdAllocator<ExecutionLaneId>>*lanes=nullptr,
    bool pending=false)noexcept{
  using Proofs=std::vector<gd::ProjectedReader,mem::StdAllocator<gd::ProjectedReader>>;
  using Lanes=std::vector<ExecutionLaneId,mem::StdAllocator<ExecutionLaneId>>;
  struct Call{ValueId value;const Frame*frame;Proofs*proofs;Lanes*lanes;bool pending;};
  Call call{value,&frame,&proofs,lanes,pending};
  return gd::RegionAccess::visit(invocation,[](void*opaque,const gd::RegionAccessView&view)noexcept->Status{
    if(view.prepared_vtable!=&region_table)return Status::Ok();
    auto&call=*static_cast<Call*>(opaque);
    const auto&r=*static_cast<Region*>(view.prepared);const auto&s=*static_cast<Slot*>(view.invocation);
    if(r.values[0]==call.value || r.metadata==call.value)return Status::InvalidArgument("typed pages cannot bind query or metadata");
    if(r.values[1]!=call.value && (!has_value_port(r) || r.values[2]!=call.value))return Status::Ok();
    if(r.capacity.storage!=c::BatchStorage::kPaged || !s.layout.valid() || s.layout!=call.frame->layout() || !s.token.valid())
      return Status::InvalidArgument("typed page frame differs from its original prepared reader/layout");
    if(r.values[1]==call.value)HYLE_RETURN_IF_ERROR(cd::PagedArenaAccess::validate_storage(*call.frame,r.capacity.key_storage));
    if(has_value_port(r) && r.values[2]==call.value)HYLE_RETURN_IF_ERROR(cd::PagedArenaAccess::validate_storage(*call.frame,r.capacity.value_storage));
    if(call.pending && ((r.values[1]==call.value && s.keys.valid()) ||
       (has_value_port(r) && r.values[2]==call.value && s.values.valid())))
      return Status::FailedPrecondition("typed Graph page input needs reset before rebinding");
    try{
      if(r.values[1]==call.value)call.proofs->push_back({view.step,&region_table,&r,std::uint32_t(r.inputs[1])});
      if(has_value_port(r) && r.values[2]==call.value && (r.values[1]!=call.value || r.inputs[1]!=r.inputs[2]))
        call.proofs->push_back({view.step,&region_table,&r,std::uint32_t(r.inputs[2])});
      if(call.lanes && std::find(call.lanes->begin(),call.lanes->end(),view.lane)==call.lanes->end())call.lanes->push_back(view.lane);
    }catch(...){return Status::ResourceExhausted("typed Graph exact reader metadata");}return Status::Ok();
  },&call);
}
} // namespace

Status prepare(graph::Invocation&invocation,std::span<const QueueBinding>queues,
    std::span<const NodeLayout>layouts,PrepareOptions options)noexcept{
  return prepare_all(invocation,nullptr,queues,layouts,{},std::move(options));
}
Status prepare(graph::Invocation&invocation,const Graph&graph,std::span<const QueueBinding>queues,
    std::span<const NodeLayout>layouts,std::span<const RowCacheBinding>rows,PrepareOptions options)noexcept{
  return prepare_all(invocation,&graph,queues,layouts,rows,std::move(options));
}
Result<c::BatchPlanReport> report(graph::Invocation&invocation,NodeId node)noexcept{return shared_native_report(invocation,node);}
Result<c::NativePagedBatchReport> native_report(graph::Invocation&invocation,NodeId node)noexcept{return detail::native_paged_report(invocation,node);}
Status prepare_replay(graph::Invocation&invocation,std::span<const QueueBinding>queues)noexcept{return capture_native(invocation,queues);}
Result<gd::OutputBinding>detail::reader_binding(Invocation&invocation,ValueId bank,
    PagedAppendSource&source,c::BatchLayoutVersion layout)noexcept{
  HYLE_RETURN_IF_ERROR(PagedAppendSourceAccess::covers(source,layout));
  HYLE_ASSIGN_OR_RETURN(auto facts,PagedAppendSourceAccess::facts(source));
  HYLE_ASSIGN_OR_RETURN(auto p,frame_resources(invocation));HYLE_RETURN_IF_ERROR(p->reusable());
  if(p->contains(bank))return std::unexpected(Status::FailedPrecondition("another pending source owns this cache input"));
  detail::ProjectedReaders proofs{mem::StdAllocator<gd::ProjectedReader>(p->allocator)};
  std::vector<gd::ResourceReadyInput,mem::StdAllocator<gd::ResourceReadyInput>>metadata{
      mem::StdAllocator<gd::ResourceReadyInput>(p->allocator)};
  // A delegated reader's ready-frame/writer-import hooks do not establish
  // this shared-Queue submission contract. Never forge a ready frame for it.
  for(const auto&native:p->native_regions)
    if(std::find(native.inputs.begin()+1,native.inputs.end(),bank)!=native.inputs.end())
      return std::unexpected(Status::Unimplemented("Graph submission page sources require an actual shared Queue reader"));
  try{for(const auto*slot:p->frame_slots){const auto&r=*slot->region;const auto&s=*slot;
    const bool key=r.values[1]==bank,value=has_value_port(r) && r.values[2]==bank;
    if(!key && !value)continue;
    // This first submission-source bridge proves the actual readonly layout
    // through its original configuration lease on every prepare/submit. An
    // owned-layout node has private mutable metadata, not such a ready input.
    if(!r.metadata.valid())return std::unexpected(Status::Unimplemented("pending append reader needs its explicit owned metadata input"));
    const auto role=key?(r.mla==c::MlaCacheLayout::kNone?c::BatchCacheRole::kKey:c::BatchCacheRole::kMla):c::BatchCacheRole::kValue;
    if(facts.role!=role || facts.storage!=(key?r.capacity.key_storage:r.capacity.value_storage) ||
        r.capacity.storage!=c::BatchStorage::kPaged || s.layout!=layout || !s.token.valid() || !s.queue.is_open() ||
        (key && (s.keys.valid() || s.written_keys.producer.valid())) ||
        (value && (s.values.valid() || s.written_values.producer.valid())))
      return std::unexpected(Status::InvalidArgument("pending append pages differ from the prepared reader role/storage/layout"));
    proofs.push_back({s.selected_step,&region_table,&r,std::uint32_t(r.inputs[key?1:2]),r.graph_capture});
    if(r.metadata.valid()){
      const auto existing=std::ranges::find(metadata,r.metadata,&gd::ResourceReadyInput::input);
      if(existing!=metadata.end())continue;
      const auto ready=std::ranges::find(p->metadata,r.metadata,&BatchResources::Metadata::input);
      if(ready==p->metadata.end() || !ready->bound || ready->layout!=layout || !ready->ready.valid())
        return std::unexpected(Status::FailedPrecondition("pending page reader has no exact owned ready metadata"));
      HYLE_ASSIGN_OR_RETURN(auto tensor,ready->ready.tensor(ready->description));
      metadata.push_back({r.metadata,tensor,ready->ready});
    }
  }}catch(...){return std::unexpected(Status::ResourceExhausted("pending append reader cold proof metadata"));}
  if(proofs.empty())return std::unexpected(Status::Unimplemented("pending append bank has no selected shared page reader"));
  return PagedAppendSourceAccess::binding(source,bank,std::move(layout),proofs,metadata);
}
Status bind_append(Invocation&invocation,const Graph&graph,NodeId append,ValueId value,PagedAppendPages&&pages)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto p,frame_resources(invocation));HYLE_RETURN_IF_ERROR(p->reusable());
  if(p->contains(value))return Status::FailedPrecondition("external pending source already owns the cache input");
  HYLE_ASSIGN_OR_RETURN(auto facts,detail::PagedAppendAccess::facts(pages));
  // Preserve the original idle/source gate before borrowing cached selected
  // slots. The actual bind below still checks the append artifact and port.
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(invocation,append,
      [](void*,const gd::RegionAccessView&)noexcept->Status{return Status::Ok();},nullptr));
  detail::ProjectedReaders proofs{mem::StdAllocator<gd::ProjectedReader>(p->allocator)};
  auto native_writer=detail::PagedAppendAccess::native(invocation,append);
  if(!native_writer && !native_writer.error().IsUnimplemented())
    return native_writer.error();
  cd::QueueRef append_owner;mem::MemoryResourceRef append_host;
  for(const auto*slot:p->frame_slots){
    const auto&r=*slot->region;const auto&s=*slot;
    const bool key=r.values[1]==value,is_value=has_value_port(r) && r.values[2]==value;
    if(!key && !is_value)continue;
    const auto expected=key?(r.mla==c::MlaCacheLayout::kNone?c::BatchCacheRole::kKey:c::BatchCacheRole::kMla):c::BatchCacheRole::kValue;
    if(facts.role!=expected || facts.storage!=(key?r.capacity.key_storage:r.capacity.value_storage) ||
        s.layout!=facts.layout || !s.token.valid() ||
        r.capacity.storage!=c::BatchStorage::kPaged || (key && s.keys.valid()) || (is_value && s.values.valid()))
      return Status::InvalidArgument("append pages differ from the prepared original cache reader/role/layout");
    if(!s.queue.valid() || (append_owner.valid() && (!append_owner.same_owner(s.queue) || append_host!=s.prepared_options.host_workspace)))
      return Status::InvalidArgument("append readers need one actual prepared Queue and host workspace");
    append_owner=s.queue;append_host=s.prepared_options.host_workspace;
    // This exact shared reader records ordinary bank/metadata addresses. Its
    // page guards live in the current Graph episode; native topology/carrier
    // replacement remains covered by before_command_change above.
    try{proofs.push_back({s.selected_step,&region_table,&r,std::uint32_t(r.inputs[key?1:2]),
        r.graph_capture});}
    catch(...){return Status::ResourceExhausted("append exact selected reader metadata");}
  }
  // Native page readers keep their original Runtime and slot. They borrow the
  // selected in-Graph writer, never an invented ready frame or external update.
  for(const auto&reader:p->native_regions){
    if(reader.capacity.storage!=c::BatchStorage::kPaged)continue;
    const bool used=std::find(reader.inputs.begin()+1,reader.inputs.end(),value)!=reader.inputs.end();
    if(!used)continue;
    if(!native_writer || !reader.table->validate_written || !reader.table->commit_written)
      return Status::Unimplemented("native page reader needs its selected native append capability");
    HYLE_ASSIGN_OR_RETURN(auto selected,detail::inspect_native_batch_region(reader));
    if(selected.layout!=facts.layout)
      return Status::InvalidArgument("native append differs from the reader's exact prepared layout");
    for(std::size_t port=1;port<reader.inputs.size();++port){
      if(reader.inputs[port]!=value)continue;
      const auto expected=port==1?(selected.mla==c::MlaCacheLayout::kNone?
          c::BatchCacheRole::kKey:c::BatchCacheRole::kMla):c::BatchCacheRole::kValue;
      if(facts.role!=expected || facts.storage!=(port==1?selected.capacity.key_storage:selected.capacity.value_storage))
        return Status::InvalidArgument("native append cache role/storage differs from its selected reader port");
    }
    HYLE_RETURN_IF_ERROR(gd::RegionAccess::invalidate_commands(invocation,reader.step));
    HYLE_RETURN_IF_ERROR(reader.table->validate_written(reader.prepared,reader.invocation,
        value,facts.layout,facts.capacity,*native_writer));
    try{
      for(std::size_t port=1;port<reader.inputs.size();++port){
        if(reader.inputs[port]!=value)continue;
        const auto physical=reader.input_ports[port];
        if(std::none_of(proofs.begin(),proofs.end(),[&](const auto&proof){
            return proof.step==reader.step && proof.input_port==physical;}))
          proofs.push_back({reader.step,reader.implementation,reader.prepared,physical});
      }
    }catch(...){return Status::ResourceExhausted("native append exact reader metadata");}
  }
  if(proofs.empty())return Status::Unimplemented("paged append requires an actual selected cache reader");
  HYLE_RETURN_IF_ERROR(detail::PagedAppendAccess::bind(invocation,graph,append,value,std::move(pages),proofs,
      append_owner,append_host,{.allocator=p->allocator,.resource_budget=p->budget}));
  // The selected shared native slot acknowledges this exact ordered producer;
  // it never stores a fictitious ready frame for pages this Graph will write.
  for(auto*slot:p->frame_slots){if(slot->region->values[1]==value)slot->written_keys={facts.layout,facts.capacity,append};
    if(has_value_port(*slot->region) && slot->region->values[2]==value)slot->written_values={facts.layout,facts.capacity,append};}
  if(native_writer)for(const auto&reader:p->native_regions)
    if(reader.capacity.storage==c::BatchStorage::kPaged && std::find(reader.inputs.begin()+1,reader.inputs.end(),value)!=reader.inputs.end())
      reader.table->commit_written(reader.prepared,reader.invocation,value,
          facts.layout,facts.capacity,*native_writer);
  return Status::Ok();
}
Status bind_append(Invocation&invocation,const Graph&graph,NodeId append,PagedAppendPages&&pages)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto p,frame_resources(invocation));HYLE_RETURN_IF_ERROR(p->reusable());
  if(!p->append_ready)return Status::FailedPrecondition("prepare the original writer-only Graph Queue before binding append pages");
  return detail::PagedAppendAccess::bind(invocation,graph,append,std::move(pages),p->append_queues,
      p->append_host,{.allocator=p->allocator,.resource_budget=p->budget});
}
Result<PagedAppendPages> take_append(Invocation&invocation,NodeId node)noexcept{return detail::PagedAppendAccess::take(invocation,node);}
Status bind(graph::Invocation&invocation,ValueId value,c::PagedArenaRead frame)noexcept{
  if(!frame.valid())return Status::InvalidArgument("typed Graph page frame is empty");
  HYLE_ASSIGN_OR_RETURN(auto p,frame_resources(invocation));HYLE_RETURN_IF_ERROR(p->reusable());
  if(p->contains(value))return Status::FailedPrecondition("pending Graph input needs reset before ready binding");
  std::vector<gd::ProjectedReader,mem::StdAllocator<gd::ProjectedReader>>proofs{mem::StdAllocator<gd::ProjectedReader>(p->allocator)};
  HYLE_RETURN_IF_ERROR(native_frame(invocation,value,frame,proofs));
  HYLE_RETURN_IF_ERROR(detail::native_paged_frame(p->native_regions,value,frame,proofs));
  HYLE_ASSIGN_OR_RETURN(auto tensor,cd::PagedArenaAccess::tensor(frame));
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::bind_projected(invocation,value,tensor,cd::PagedArenaAccess::pages(frame),proofs));
  HYLE_RETURN_IF_ERROR(invalidate_projected_commands(invocation,proofs));
  for(auto*slot:p->frame_slots){if(slot->region->values[1]==value)slot->keys=frame;
    if(has_value_port(*slot->region) && slot->region->values[2]==value)slot->values=frame;}
  detail::commit_native_paged_frame(p->native_regions,value,std::move(frame));
  return Status::Ok();
}
Status bind(graph::Invocation&invocation,ValueId value,c::PagedRowCacheRead frame)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto p,resources(invocation));if(!p || !frame.valid())return Status::InvalidArgument("typed row Graph has no prepared resource/frame");
  HYLE_RETURN_IF_ERROR(p->reusable());if(p->contains(value))return Status::FailedPrecondition("pending row input needs reset before ready binding");
  auto found=std::find_if(p->rows.begin(),p->rows.end(),[&](const auto&row){return row.reader.bank_input()==value;});
  if(found==p->rows.end() || !found->prepared || found->layout!=frame.layout() || !found->reader.schema().owns(frame.layout()))
    return Status::InvalidArgument("typed row Graph frame differs from its prepared reader/layout");
  if(found->frame.valid())return Status::FailedPrecondition("typed row frame needs reset before rebinding");
  const auto cap=found->reader.schema().capacity();HYLE_RETURN_IF_ERROR(cd::PagedArenaAccess::validate_storage(frame,cap.encoding));
  HYLE_ASSIGN_OR_RETURN(auto storage,cd::PagedArenaAccess::tensor(frame));
  HYLE_ASSIGN_OR_RETURN(auto description,c::row_cache_carrier_description(cap.encoding,std::int64_t(std::uint64_t(cap.max_pages)*cap.page_size)));
  HYLE_ASSIGN_OR_RETURN(auto tensor,c::ConstTensorView::create(description,storage.storage()));
  HYLE_RETURN_IF_ERROR(detail::RowCacheAccess::bind(invocation,found->reader,tensor,cd::PagedArenaAccess::pages(frame)));found->frame=std::move(frame);return Status::Ok();
}
Status bind(graph::Invocation&invocation,ValueId value,c::PagedArenaPendingRead frame)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto p,resources(invocation));if(!p || !p->pending_capacity || !frame.valid())return Status::InvalidArgument("pending Graph requires cold source capacity and typed frame");
  HYLE_RETURN_IF_ERROR(p->reusable());if(p->contains(value))return Status::FailedPrecondition("pending Graph input needs reset before rebinding");
  BatchResources::Pending input{value,std::move(frame),std::vector<ExecutionLaneId,mem::StdAllocator<ExecutionLaneId>>(mem::StdAllocator<ExecutionLaneId>(p->allocator))};
  std::vector<gd::ProjectedReader,mem::StdAllocator<gd::ProjectedReader>>proofs{mem::StdAllocator<gd::ProjectedReader>(p->allocator)};
  HYLE_RETURN_IF_ERROR(native_frame(invocation,value,input.frame,proofs,&input.lanes,true));
  const bool shared=!input.lanes.empty();
  HYLE_ASSIGN_OR_RETURN(input.native,detail::native_pending_frame(
      p->pending_regions,value,input.frame,p->native_source,proofs,input.lanes));
  if(shared && input.native)return Status::Unimplemented("pending physical readers require different source protocols");
  if(input.lanes.empty())return Status::InvalidArgument("pending Graph has no qualified native reader");
  if(p->inputs.size()==p->inputs.capacity())return Status::ResourceExhausted("pending Graph fixed input capacity");
  HYLE_ASSIGN_OR_RETURN(auto tensor,cd::PagedArenaAccess::tensor(input.frame));
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::bind_pending_projected(invocation,value,tensor,proofs));
  HYLE_RETURN_IF_ERROR(invalidate_projected_commands(invocation,proofs));
  detail::commit_native_pending_frame(p->pending_regions,value,input.frame,p->native_source);
  p->inputs.push_back(std::move(input));return Status::Ok();
}
Status bind(graph::Invocation&invocation,ValueId value,c::PagedRowCachePendingRead frame)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto p,resources(invocation));if(!p || !p->pending_capacity || !frame.valid())return Status::InvalidArgument("pending row Graph has no cold source/frame");
  HYLE_RETURN_IF_ERROR(p->reusable());if(p->contains(value))return Status::FailedPrecondition("pending row input needs reset before rebinding");
  const auto row=std::find_if(p->rows.begin(),p->rows.end(),[&](const auto&r){return r.reader.bank_input()==value;});
  if(row==p->rows.end() || !row->prepared || row->layout!=frame.layout() || !row->reader.schema().owns(frame.layout()))
    return Status::InvalidArgument("pending row frame differs from its prepared reader/layout");
  if(row->frame.valid())return Status::FailedPrecondition("ready row input needs reset before pending binding");
  HYLE_RETURN_IF_ERROR(cd::PagedArenaAccess::validate_storage(frame,row->reader.schema().capacity().encoding));
  BatchResources::Pending input{value,cd::PagedArenaAccess::owner(frame),std::vector<ExecutionLaneId,mem::StdAllocator<ExecutionLaneId>>(mem::StdAllocator<ExecutionLaneId>(p->allocator))};
  std::vector<gd::ProjectedReader,mem::StdAllocator<gd::ProjectedReader>>proofs{mem::StdAllocator<gd::ProjectedReader>(p->allocator)};
  HYLE_ASSIGN_OR_RETURN(input.native,detail::native_pending_frame(
      p->pending_regions,value,input.frame,p->native_source,proofs,input.lanes));
  if(!input.native){try{input.lanes.push_back(row->lane);}catch(...){return Status::ResourceExhausted("pending row lane metadata");}}
  if(p->inputs.size()==p->inputs.capacity())return Status::ResourceExhausted("pending row fixed input capacity");
  HYLE_ASSIGN_OR_RETURN(auto storage,cd::PagedArenaAccess::tensor(input.frame));const auto cap=row->reader.schema().capacity();
  HYLE_ASSIGN_OR_RETURN(auto description,c::row_cache_carrier_description(cap.encoding,std::int64_t(std::uint64_t(cap.max_pages)*cap.page_size)));
  HYLE_ASSIGN_OR_RETURN(auto tensor,c::ConstTensorView::create(description,storage.storage()));
  HYLE_RETURN_IF_ERROR(detail::RowCacheAccess::bind(invocation,row->reader,tensor,{},true));
  HYLE_RETURN_IF_ERROR(invalidate_projected_commands(invocation,proofs));
  detail::commit_native_pending_frame(p->pending_regions,value,input.frame,p->native_source);
  p->inputs.push_back(std::move(input));return Status::Ok();
}
Result<c::Completion> submit(graph::Invocation&invocation,std::span<const QueueBinding>queues,c::CopyBatchUpdate&&update)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto p,resources(invocation));if(!p)return std::unexpected(Status::FailedPrecondition("Graph has no typed pending resources"));
  HYLE_RETURN_IF_ERROR(p->reusable());if(p->inputs.empty())return std::unexpected(Status::FailedPrecondition("Graph has no pending typed inputs"));
  if(!p->source_capacity || std::any_of(p->inputs.begin(),p->inputs.end(),[](const auto&input){return input.native;}))
    return std::unexpected(Status::Unimplemented("pending Graph inputs need their actual native source submit"));
  for(const auto&row:p->rows)if(!row.prepared || (!row.frame.valid() && !p->contains(row.reader.bank_input())))
    return std::unexpected(Status::FailedPrecondition("typed row Graph has no prepared frame set"));
  for(const auto&input:p->inputs)HYLE_RETURN_IF_ERROR((cd::PagedArenaAccess::validate<c::CopyBatchUpdate,cd::CopyStreamAccess>(input.frame,update)));
  if(invocation.replay_route()==ReplayRoute::kUnprepared)return std::unexpected(Status::FailedPrecondition("prepare the original Graph replay before typed pending submit"));
  HYLE_ASSIGN_OR_RETURN(p->sources,p->source.take(std::move(update)));p->admitted.clear();
  const gd::OrderedGraphSubmission ordered{
    [](void*opaque,ExecutionLaneId lane,cd::QueueSequence&sequence,c::LeaseUseBatch&uses)noexcept->Status{
      auto&p=*static_cast<BatchResources*>(opaque);
      if(std::find(p.admitted.begin(),p.admitted.end(),lane)!=p.admitted.end())return Status::Ok();
      const auto reads=[&](const auto&input){return std::find(input.lanes.begin(),input.lanes.end(),lane)!=input.lanes.end();};
      if(std::none_of(p.inputs.begin(),p.inputs.end(),reads))return Status::Ok();
      HYLE_ASSIGN_OR_RETURN(auto prefix,cd::CopyStreamAccess::dependency(p.source.update()));
      if(prefix.pending()){auto waited=sequence.enqueue_resolved_wait(prefix);const auto status=waited.status;sequence.record(std::move(waited));HYLE_RETURN_IF_ERROR(status);}
      p.page_reads.reset();for(const auto&input:p.inputs)if(reads(input))HYLE_RETURN_IF_ERROR((cd::PagedArenaAccess::collect_reads<c::CopyBatchUpdate,cd::CopyStreamAccess>(p.page_reads,input.frame,p.source.update())));
      HYLE_RETURN_IF_ERROR(p.page_reads.admit(uses));
      if(p.admitted.size()==p.admitted.capacity())return Status::ResourceExhausted("pending Graph admitted lane capacity");
      p.admitted.push_back(lane);return Status::Ok();
    },
    [](void*opaque,c::Completion&&suffix)noexcept->c::Completion{
      auto&p=*static_cast<BatchResources*>(opaque);
      // Core retains its one original InvocationState on this aggregate;
      // that same State owns this resource payload through actual retirement.
      return p.dependent.publish(std::move(suffix),std::exchange(p.sources,{}));
    },p,[](void*opaque,ValueId value)noexcept->bool{return static_cast<BatchResources*>(opaque)->contains(value);}};
  auto result=gd::RegionAccess::submit(invocation,queues,ordered);
  if(!result){update=std::move(p->source.update());auto sources=std::exchange(p->sources,{});sources.release(sources.state);}
  return result;
}
Result<c::CopyBatchUpdate> take_update(graph::Invocation&invocation)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto p,resources(invocation));
  if(!p || !p->source_capacity || !p->dependent.available() || !p->source.available() || !p->source.update().valid() ||
      p->source.update().in_flight() || !gd::RegionAccess::outcome_available(invocation))
    return std::unexpected(Status::Unavailable("Graph still retains source/outcome controls"));
  auto result=std::move(p->source.update());p->inputs.clear();return result;
}

namespace detail {
Result<NativePendingAccess> native_pending_access(Invocation&invocation,
    const NativePendingSourceVTable*table)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto p,resources(invocation));
  if(!p || !table || p->native_source.table!=table || !p->native_source.state ||
      !p->pending_capacity || p->inputs.empty())
    return std::unexpected(Status::FailedPrecondition("Graph has no prepared native pending source/frame"));
  HYLE_RETURN_IF_ERROR(p->reusable());
  HYLE_RETURN_IF_ERROR(gd::RegionAccess::visit(invocation,
      [](void*,const gd::RegionAccessView&)noexcept{return Status::Ok();},nullptr));
  // Each binding committed only after every physical reader accepted its
  // exact frame/source protocol. Lane identity alone never grants admission.
  if(std::any_of(p->inputs.begin(),p->inputs.end(),[](const auto&input){return !input.native;}))
    return std::unexpected(Status::Unimplemented("native pending source has a reader with a different import protocol"));
  return NativePendingAccess{p->native_source.state,p->inputs};
}
Result<c::Completion> submit_native_pending(Invocation&invocation,
    std::span<const QueueBinding>queues,const NativePendingSourceVTable*table,
    cd::OutcomeSources sources)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto p,resources(invocation));
  if(!p || !table || p->native_source.table!=table || !p->native_source.state ||
      !p->pending_capacity || p->inputs.empty() || !p->dependent_ready ||
      !p->dependent.available() || p->sources.valid() || !sources.valid())
    return std::unexpected(Status::FailedPrecondition("native pending Graph source/outcome is unavailable"));
  if(invocation.replay_route()==ReplayRoute::kUnprepared)
    return std::unexpected(Status::FailedPrecondition("prepare the original Graph replay before native pending submit"));
  p->sources=sources;
  const gd::OrderedGraphSubmission ordered{
    .map=[](void*opaque,c::Completion&&suffix)noexcept->c::Completion{
      auto&p=*static_cast<BatchResources*>(opaque);
      return p.dependent.publish(std::move(suffix),std::exchange(p.sources,{}));
    },.owner=p,
    .pending_input=[](void*opaque,ValueId value)noexcept->bool{
      return static_cast<BatchResources*>(opaque)->contains(value);
    }};
  auto result=gd::RegionAccess::submit(invocation,queues,ordered);
  if(!result)p->sources={}; // The typed caller still owns this source reference.
  return result;
}
Result<void*> native_pending_recovery(Invocation&invocation,
    const NativePendingSourceVTable*table)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto p,resources(invocation));
  if(!p || !table || p->native_source.table!=table || !p->native_source.state ||
      !p->dependent_ready || !p->dependent.available() ||
      !gd::RegionAccess::outcome_available(invocation))
    return std::unexpected(Status::Unavailable("Graph still retains native source/outcome controls"));
  return p->native_source.state;
}
void finish_native_pending_recovery(Invocation&invocation)noexcept{
  auto p=resources(invocation);
  if(p && *p)(*p)->inputs.clear();
}
} // namespace detail

} // namespace hyle::graph::batch
