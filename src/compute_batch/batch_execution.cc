#include "hyle/compute/batch_execution.h"
#include "hyle/compute/detail/batch_provider.h"
#include "hyle/compute/detail/batch_graph_access.h"
#include "hyle/compute/detail/batch_device_choice.h"
#include "hyle/compute/transfer.h"
#include "hyle/compute/detail/command_capture.h"
#include "hyle/compute/detail/paged_arena_access.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/compute/detail/retained_completion.h"
#include "hyle/compute/detail/copy_outcome_source.h"
#include "hyle/compute/detail/queue_ref.h"
#include "hyle/core/assert.h"
#include "hyle/util/ref_counted.h"
#include "hyle/util/scope_guard.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <new>

namespace hyle::compute {
namespace d=detail;
namespace detail {
struct BatchProgramData final:util::RefCounted<BatchProgramData>{
  BatchProgramData(BatchSchema s,BatchProgramOptions o,mem::AllocatorRef a):schema(std::move(s)),options(std::move(o)),allocator(a){}
  void destroy()noexcept{const auto a=allocator;this->~BatchProgramData();a.deallocate(this,sizeof(*this),alignof(BatchProgramData));}
  BatchSchema schema;BatchProgramOptions options;mem::AllocatorRef allocator;
  ContextRef context;
  ProviderExtensionRef extension;const BatchProviderVTable*table=nullptr;
  ProviderRuntimeProperties runtime{};
  CopyBatchProgram upload;std::atomic<std::uint32_t> invocations{0};
};
struct BatchBuffers final:util::RefCounted<BatchBuffers>{
  explicit BatchBuffers(BatchProgramData*p):program(p),allocator(p->allocator){p->incref();}
  void destroy()noexcept{auto*p=program;const auto a=allocator;this->~BatchBuffers();p->invocations.fetch_sub(1,std::memory_order_relaxed);p->decref();a.deallocate(this,sizeof(*this),alignof(BatchBuffers));}
  BatchProgramData*program;mem::AllocatorRef allocator;
  // F2 borrows its Queue wrapper at a fixed address. Original copy bindings
  // retain these buffer domains through actual retirement, including failed
  // observation, so the optional upload Queue outlives those borrowed uses.
  std::shared_ptr<Queue> metadata_queue;
  ProviderExtensionRef provider_owner;LeasePool device,host;
};
struct BatchReplay final {
  CapturedCommands commands;
  std::array<mem::ConstBufferView,5> bindings{};
};
struct BatchInvocationData final:util::RefCounted<BatchInvocationData>{
  explicit BatchInvocationData(BatchProgramData*p):program(p),read_set(p->allocator){p->incref();}
  void destroy()noexcept{
    auto*p=program;const auto a=p->allocator;
    if(replay)a.destroy(replay);
    if(native)p->table->destroy(native,a);
    if(buffers)buffers->decref();else p->invocations.fetch_sub(1,std::memory_order_relaxed);
    this->~BatchInvocationData();p->decref();
    a.deallocate(this,sizeof(*this),alignof(BatchInvocationData));
  }
  void release_user()noexcept{
    HYLE_FATAL_ASSERT(users);if(--users==0){closed=true;completion.reset();upload.close().IgnoreError();}
    decref();
  }
  BatchProgramData*program;BatchBuffers*buffers=nullptr;
  QueueRef queue_owner;Queue*queue=nullptr;void*native=nullptr;
  BatchReplay*replay=nullptr;
  WriteLease mirror,output,float_workspace,int_workspace,host_workspace,staging;
  // Coordinates borrowed only by the trusted outer Graph region. Its original
  // root/native-use lifetime owns both payload and permission, not this view.
  mem::BufferView graph_float_workspace;
  ReadLease mirror_ready,output_ready,staging_ready;
  ReadLease graph_shared_metadata;
  ProviderExtensionRef choice_extension;
  const BatchDeviceChoiceVTable *choice_table=nullptr;
  std::array<BatchLayoutVersion,2> choice_layouts;
  LeasePool choice_device_pool,choice_host_pool;
  WriteLease choice_pair,choice_selected,choice_staging;
  ReadLease choice_ready,choice_staging_ready;
  bool device_choices=false,preparing_choices=false;
  // Non-owning recording facts only. Actual buffers remain in this original
  // slot; Graph owns readiness/permissions and the outer command artifact.
  mem::ConstBufferView graph_metadata_binding;
  std::uint64_t graph_metadata_epoch=0;
  CopyBatchInvocation upload;LeaseUsePool uses;RetainedCompletion completion;
  BatchLayoutVersion layout,mirror_layout;BatchPlanReport report;
  BatchPageReads read_set;
  CompletionRetention native_retention{};std::atomic<std::uint32_t> pending{0};
  Status terminal;std::uint64_t epoch=0;std::uint32_t users=1,tickets=0;
  bool closed=false,failed=false,prepared=false,ran=false,external_output=false;
};
struct PagedReceiverData final:util::RefCounted<PagedReceiverData>{
  explicit PagedReceiverData(mem::AllocatorRef a):allocator(a){}
  void destroy()noexcept{const auto a=allocator;this->~PagedReceiverData();a.deallocate(this,sizeof(*this),alignof(PagedReceiverData));}
  mem::AllocatorRef allocator;BatchInvocation original;PreparedBatch token;
  CopyOutcomeSource<CopyBatchUpdate,CopyStreamAccess> source;DependentCompletion dependent;
  CompletionRetention retention;std::atomic<bool> pending{false};
  Status terminal;bool submitted=false;
  Status reusable()noexcept{
    return !pending.load(std::memory_order_acquire) && dependent.available() && source.available() && !source.update().valid()?
        Status::Ok():Status::Unavailable("common pending receiver outcome/source retained");
  }
};
} // namespace detail
namespace {
Result<std::size_t> output_bytes(BatchCapacity c)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto desc,batch_output_description(c,c.max_query_rows));return desc.required_bytes();
}
std::size_t input_count(const BatchSchema&schema)noexcept{return schema.mla_layout()==MlaCacheLayout::kNone?3:2;}
Result<std::array<TensorDesc,3>> input_descriptions(const BatchSchema&schema)noexcept{
  const auto c=schema.capacity();
  std::array<TensorDesc,3> result;
  HYLE_ASSIGN_OR_RETURN(result[0],batch_query_description(c));
  HYLE_ASSIGN_OR_RETURN(result[1],batch_cache_description(schema,schema.mla_layout()==MlaCacheLayout::kNone?BatchCacheRole::kKey:BatchCacheRole::kMla));
  if(input_count(schema)==3){HYLE_ASSIGN_OR_RETURN(result[2],batch_cache_description(schema,BatchCacheRole::kValue));}return result;
}
void native_terminal(void*opaque,const Status&status)noexcept{
  auto&s=*static_cast<d::BatchInvocationData*>(opaque);s.terminal=status;
}
void native_release(void*opaque)noexcept{
  auto*s=static_cast<d::BatchInvocationData*>(opaque);s->pending.fetch_sub(1,std::memory_order_release);s->pending.notify_all();s->decref();
}
Result<d::BatchWorkspaceViews> workspace_views(d::BatchInvocationData&s)noexcept{
  d::BatchWorkspaceViews views;
  HYLE_ASSIGN_OR_RETURN(views.metadata,(s.graph_shared_metadata.valid()?s.graph_shared_metadata:s.mirror_ready).view());
  if(s.graph_float_workspace.valid())views.float_workspace=s.graph_float_workspace;
  else if(s.float_workspace.valid()){HYLE_ASSIGN_OR_RETURN(views.float_workspace,s.float_workspace.view());}
  if(s.int_workspace.valid()){HYLE_ASSIGN_OR_RETURN(views.int_workspace,s.int_workspace.view());}
  if(s.host_workspace.valid()){HYLE_ASSIGN_OR_RETURN(views.host_workspace,s.host_workspace.view());}
  if(s.output.valid()){HYLE_ASSIGN_OR_RETURN(views.output,s.output.view());}return views;
}
void retain_domain(d::BatchBuffers&owner,LeasePool&pool)noexcept{
  owner.incref();d::LeaseAccess::retain_resource_owner(pool,&owner,[](void*p)noexcept{static_cast<d::BatchBuffers*>(p)->decref();});
  d::LeaseAccess::detach_pool_root(pool);
}
Queue& planning_queue(d::BatchInvocationData&s)noexcept{
  return s.buffers && s.buffers->metadata_queue?*s.buffers->metadata_queue:*s.queue;
}
Result<BatchPlanReport> query_workspace(d::BatchInvocationData&s,const BatchLayoutVersion&layout,bool allow_update)noexcept{
  auto&p=*s.program;BatchPlanReport report;
  if(!p.table->uses_caller_workspace)return report;
  struct Call{d::BatchInvocationData*state;const BatchLayoutVersion*layout;bool allow_update;BatchPlanReport*report;}call{&s,&layout,allow_update,&report};
  HYLE_RETURN_IF_ERROR(d::ProviderExtensionAccess::call(p.extension,[](void*,const void*table,void*opaque)noexcept{
    auto&call=*static_cast<Call*>(opaque);
    return static_cast<const d::BatchProviderVTable*>(table)->workspace(call.state->native,planning_queue(*call.state),*call.layout,call.allow_update,call.report);
  },&call));
  const auto floating=report.float_workspace_bytes,integer=report.int_workspace_bytes;
  if(floating>p.options.max_float_workspace_bytes || integer>p.options.max_int_workspace_bytes)
    return std::unexpected(Status::ResourceExhausted("batch layout workspace exceeds program hard caps"));
  if(s.graph_float_workspace.valid() && floating>s.graph_float_workspace.size())
    return std::unexpected(Status::ResourceExhausted("batch layout exceeds its fixed Graph float workspace"));
  if(floating>SIZE_MAX-integer || floating+integer>SIZE_MAX-512u)
    return std::unexpected(Status::OutOfRange("batch scratch extent overflow"));
  return report;
}
struct PreparationChangeGate final {
  const d::BeforeCommandChange*callback=nullptr;bool notified=false;
  Status before_change()noexcept{
    if(notified)return Status::Ok();
    if(callback && callback->call)HYLE_RETURN_IF_ERROR(callback->call(callback->state));
    notified=true;return Status::Ok();
  }
};
Status invalidate_recorded_commands(d::BatchInvocationData&s,PreparationChangeGate&gate)noexcept{
  HYLE_RETURN_IF_ERROR(gate.before_change());
  if(s.replay)s.replay->commands.reset();
  s.graph_metadata_epoch=0;
  return Status::Ok();
}
Result<bool> prepare_workspace(d::BatchInvocationData&s,const BatchPlanReport&report,
    PreparationChangeGate&gate)noexcept{
  auto&p=*s.program;if(!p.table->uses_caller_workspace)return false;
  const auto floating=report.float_workspace_bytes,integer=report.int_workspace_bytes;
  const bool borrowed=s.graph_float_workspace.valid();
  if(borrowed && floating>s.graph_float_workspace.size())
    return std::unexpected(Status::ResourceExhausted("batch layout exceeds its fixed Graph float workspace"));
  if((borrowed || s.float_workspace.size()>=floating) && s.int_workspace.size()>=integer && s.host_workspace.size()>=integer)return false;
  // All ticket/completion/output guards ran before this cold mutation. Release
  // an old capture before its addresses, and old charges before the new exact
  // request so a hard budget never needs two generations of idle scratch.
  // Retain an actual high-water capacity across smaller layouts; there is no
  // fixed eager allocation floor and independent workspaces grow separately.
  HYLE_RETURN_IF_ERROR(invalidate_recorded_commands(s,gate));
  const auto grow=[&](WriteLease&lease,mem::MemoryResourceRef resource,std::size_t bytes)noexcept->Status{
    if(lease.size()>=bytes)return Status::Ok();
    lease.reset();
    HYLE_ASSIGN_OR_RETURN(auto pool,LeasePool::create(resource,bytes,
        {.max_blocks=1,.allocator=p.allocator,.resource_budget=p.options.resource_budget}));
    HYLE_ASSIGN_OR_RETURN(auto next,pool.allocate(bytes));retain_domain(*s.buffers,pool);lease=std::move(next);return Status::Ok();
  };
  if(!borrowed)HYLE_RETURN_IF_ERROR(grow(s.float_workspace,p.options.resource,floating));
  HYLE_RETURN_IF_ERROR(grow(s.int_workspace,p.options.resource,integer));
  HYLE_RETURN_IF_ERROR(grow(s.host_workspace,p.options.host_workspace,integer));
  return true;
}
std::array<mem::ConstBufferView,5> replay_bindings(std::span<const ConstTensorView>inputs,
    d::BatchWorkspaceViews views)noexcept{
  return {inputs[0].storage(),inputs[1].storage(),inputs.size()==3?inputs[2].storage():mem::ConstBufferView{},views.metadata,mem::ConstBufferView(views.output)};
}
bool same_storage(mem::ConstBufferView a,mem::ConstBufferView b)noexcept{
  return a.resource()==b.resource() && a.native_handle()==b.native_handle() && a.offset()==b.offset() && a.size()==b.size();
}
bool same_replay(const d::BatchReplay&replay,std::span<const ConstTensorView>inputs,
    d::BatchWorkspaceViews views)noexcept{
  const auto bindings=replay_bindings(inputs,views);
  for(std::size_t i=0;i<bindings.size();++i)if(!same_storage(bindings[i],replay.bindings[i]))return false;
  return true;
}
Status validate_replay_storage(const d::BatchInvocationData&s,std::span<const ConstTensorView>inputs,
    d::BatchWorkspaceViews views)noexcept{
  if(s.replay && s.replay->commands.valid() && !same_replay(*s.replay,inputs,views))
    return Status::FailedPrecondition("batch replay storage changed; explicitly prepare_replay again");
  return Status::Ok();
}
Status qualify_graph_recording(const d::BatchInvocationData&s)noexcept{
  const auto&runtime=s.program->runtime;
  if(!s.program->table->supports_command_capture || !runtime.valid() ||
      !runtime.capture_compatible || runtime.mode!=ProviderRuntimeMode::kSharedQueue ||
      runtime.workspace_owner!=ProviderWorkspaceOwner::kProvider ||
      runtime.replay_class!=ProviderReplayClass::kManaged || runtime.blocks_host ||
      runtime.uses_private_threads)
    return Status::Unimplemented("native batch slot is not qualified for outer Graph recording");
  return Status::Ok();
}
Result<Completion> retain_native(d::BatchInvocationData&s,LeaseUseBatch&&use,Completion&&completion)noexcept{
  HYLE_FATAL_ASSERT(use.attach(completion).ok());
  s.incref();s.pending.fetch_add(1,std::memory_order_relaxed);
  s.native_retention.own={&s,&native_terminal,&native_release};
  HYLE_FATAL_ASSERT(d::CompletionAccess::attach_retention(completion,s.native_retention).ok());
  return std::move(completion);
}
Result<Completion> finish_native(d::BatchInvocationData&s,bool planning,
    std::span<const ConstTensorView> inputs,d::BatchWorkspaceViews views,
    LeaseUseBatch&&use,d::QueueSequence&sequence)noexcept{
  struct Call{d::BatchInvocationData*state;bool planning;std::span<const ConstTensorView> inputs;d::BatchWorkspaceViews views;}call{&s,planning,inputs,views};
  if(sequence.terminal().ok()){
    if(!planning && s.replay && s.replay->commands.valid()){
      const auto valid=validate_replay_storage(s,inputs,views);
      sequence.record(valid.ok()?s.replay->commands.enqueue(sequence):d::EnqueueResult::rejected(valid));
    }else sequence.record(d::ProviderExtensionAccess::enqueue_admitted(s.program->extension,sequence,inputs,{},
      [](void*,const void*interface_value,void*opaque,d::QueueSequence&sequence)noexcept{
        const auto&call=*static_cast<Call*>(opaque);auto&state=*call.state;const auto*table=static_cast<const d::BatchProviderVTable*>(interface_value);
        return call.planning?table->plan(state.native,sequence,state.layout,call.views,&state.report):table->run(state.native,sequence,call.inputs,call.views);
      },&call));
  }
  HYLE_ASSIGN_OR_RETURN(auto completion,sequence.finish());
  return retain_native(s,std::move(use),std::move(completion));
}
Result<Completion> submit_native(d::BatchInvocationData&s,bool planning,
    std::span<const ReadLease> leases,std::span<const ConstTensorView> inputs,d::BatchWorkspaceViews views,
    std::span<const ReadLease> key_pages={},std::span<const ReadLease> value_pages={})noexcept{
  if(!s.completion.available() || s.pending.load(std::memory_order_acquire) || s.uses.active_batches())
    return std::unexpected(Status::Unavailable("batch native slot or old completion retained"));
  HYLE_ASSIGN_OR_RETURN(auto use,s.uses.begin());
  auto&reads=s.read_set;reads.reset();
  for(const auto&lease:leases)HYLE_RETURN_IF_ERROR(reads.add(lease));
  for(const auto&lease:key_pages)HYLE_RETURN_IF_ERROR(reads.add(lease));
  for(const auto&lease:value_pages)HYLE_RETURN_IF_ERROR(reads.add(lease));
  HYLE_RETURN_IF_ERROR(reads.add(s.graph_shared_metadata.valid()?s.graph_shared_metadata:s.mirror_ready));
  HYLE_RETURN_IF_ERROR(reads.admit(use));
  if(s.float_workspace.valid())HYLE_RETURN_IF_ERROR(use.write(s.float_workspace));
  if(s.int_workspace.valid())HYLE_RETURN_IF_ERROR(use.write(s.int_workspace));
  if(planning){if(s.host_workspace.valid())HYLE_RETURN_IF_ERROR(use.write(s.host_workspace));}
  else HYLE_RETURN_IF_ERROR(use.write(s.output));
  HYLE_ASSIGN_OR_RETURN(auto sequence,d::QueueSequence::begin(planning?planning_queue(s):*s.queue));
  HYLE_ASSIGN_OR_RETURN(auto completion,finish_native(s,planning,inputs,views,std::move(use),sequence));
  return s.completion.publish(std::move(completion));
}
Status capture_batch(d::BatchInvocationData&s,std::span<const ConstTensorView>inputs,
    d::BatchWorkspaceViews views)noexcept{
  if(!s.program->table->supports_command_capture || !s.layout.facts().query_rows)
    return Status::Unimplemented("native batch command capture is not qualified for this provider/empty batch");
  if(inputs.size()!=input_count(s.program->schema) || s.closed || s.failed || !s.prepared || s.ran)
    return Status::FailedPrecondition("batch replay needs its live prepared token");
  if(s.pending.load(std::memory_order_acquire) || s.uses.active_batches() || !s.completion.available())
    return Status::Unavailable("batch replay has outstanding native work/completion");
  HYLE_ASSIGN_OR_RETURN(auto extension,d::CommandCaptureAccess::query(*s.queue));
  if(s.replay && s.replay->commands.valid() && same_replay(*s.replay,inputs,views)){
    s.report.capture_compatible=true;return Status::Ok();
  }
  if(!s.replay){s.replay=s.program->allocator.construct<d::BatchReplay>();
    if(!s.replay)return Status::ResourceExhausted("batch replay metadata allocation");}
  // Recording captures argument identities, never tensor execution. Only the
  // original private workspace/mirror uses are needed through its checked
  // preparation event; actual Q/K/V/page permissions are admitted at replay.
  HYLE_ASSIGN_OR_RETURN(auto use,s.uses.begin());
  HYLE_RETURN_IF_ERROR(use.read(s.graph_shared_metadata.valid()?s.graph_shared_metadata:s.mirror_ready));
  if(s.float_workspace.valid())HYLE_RETURN_IF_ERROR(use.write(s.float_workspace));
  if(s.int_workspace.valid())HYLE_RETURN_IF_ERROR(use.write(s.int_workspace));
  HYLE_ASSIGN_OR_RETURN(auto sequence,d::QueueSequence::begin(*s.queue));
  const auto begun=d::CommandCaptureAccess::begin(extension,sequence);
  if(!begun.ok()){
    sequence.record(d::EnqueueResult::rejected(begun));
    auto finished=sequence.finish();if(!finished)return finished.error();return begun;
  }
  struct Call{d::BatchInvocationData*state;std::span<const ConstTensorView>inputs;d::BatchWorkspaceViews views;}call{&s,inputs,views};
  sequence.record(d::ProviderExtensionAccess::enqueue_admitted(s.program->extension,sequence,inputs,{},
    [](void*,const void*interface_value,void*opaque,d::QueueSequence&sequence)noexcept{
      auto&call=*static_cast<Call*>(opaque);
      return static_cast<const d::BatchProviderVTable*>(interface_value)->run(call.state->native,sequence,call.inputs,call.views);
    },&call));
  d::CapturedCommands captured;
  if(sequence.terminal().ok()){
    auto ended=d::CommandCaptureAccess::end(extension,sequence,s.program->allocator);
    if(ended)captured=std::move(*ended);else sequence.record(d::EnqueueResult::rejected(ended.error()));
  }else{
    const auto aborted=d::CommandCaptureAccess::abort(extension,sequence);
    if(!aborted.ok())sequence.record(d::EnqueueResult::rejected(aborted));
  }
  const bool accepted=sequence.accepted();
  auto finished=sequence.finish();
  if(!finished){if(accepted)s.failed=true;return finished.error();}
  HYLE_ASSIGN_OR_RETURN(auto retained,retain_native(s,std::move(use),std::move(*finished)));
  auto completion=s.completion.publish(std::move(retained));
  const auto terminal=completion.wait();completion={};++s.report.host_waits;
  if(!terminal.ok() || s.pending.load(std::memory_order_acquire) || s.uses.active_batches()){
    s.failed=true;return terminal.ok()?Status::Unavailable("batch capture preparation retirement unproven"):terminal;
  }
  if(!captured.valid())return Status::Internal("batch capture returned no artifact");
  s.replay->commands=std::move(captured);s.replay->bindings=replay_bindings(inputs,views);
  s.report.capture_compatible=true;return Status::Ok();
}
template<class QueueIdentity>
Result<d::BatchWorkspaceViews> graph_views(d::BatchInvocationData*s,const QueueIdentity&queue,
    std::span<const ConstTensorView>inputs,std::span<const TensorView>outputs,
    d::BatchGraphContract contract)noexcept{
  if(contract!=d::BatchGraphContract::kPreparedMetadataCapacity &&
      contract!=d::BatchGraphContract::kOwnedLayoutActive && contract!=d::BatchGraphContract::kDeviceLayoutChoice)
    return std::unexpected(Status::InvalidArgument("Graph batch port contract is invalid"));
  const bool choice=contract==d::BatchGraphContract::kDeviceLayoutChoice;
  const bool owned=contract==d::BatchGraphContract::kOwnedLayoutActive || choice;
  if(!s || !s->external_output || !s->queue_owner.same_owner(queue) || s->closed || s->failed ||
      !s->prepared || (!owned && s->ran))
    return std::unexpected(Status::FailedPrecondition("Graph batch token/slot/queue is stale"));
  const auto payloads=input_count(s->program->schema);
  if(inputs.size()!=payloads+(owned && !choice?0u:1u) || outputs.size()!=1)return std::unexpected(Status::InvalidArgument("Graph batch region port count"));
  if(choice && (!s->device_choices || !s->choice_table || !s->choice_ready.valid() || !s->choice_selected.valid()))
    return std::unexpected(Status::FailedPrecondition("device batch choice was not explicitly prepared"));
  HYLE_ASSIGN_OR_RETURN(auto descriptions,input_descriptions(s->program->schema));
  const auto capacity=s->program->schema.capacity();
  auto query_rows=choice?2u:owned?s->layout.facts().query_rows:capacity.max_query_rows;
  if(!owned && s->program->schema.mla_layout()==MlaCacheLayout::kNone){
    // Graph has already admitted this immutable concrete port description.
    // Its physical extent may be smaller than the schema upper bound; the
    // actual native plan still uses only the owned layout's active rows.
    const auto&query=inputs[0].description();
    if(!query.valid() || query.rank()!=3 || query.dimensions()[0]<0 ||
        std::uint64_t(query.dimensions()[0])>capacity.max_query_rows)
      return std::unexpected(Status::InvalidArgument("Graph batch query extent exceeds its schema"));
    query_rows=static_cast<std::uint32_t>(query.dimensions()[0]);
  }
  if(s->layout.facts().query_rows>query_rows)
    return std::unexpected(Status::InvalidArgument("Graph batch active queries exceed its concrete port"));
  HYLE_ASSIGN_OR_RETURN(descriptions[0],batch_query_description(capacity,query_rows));
  for(std::size_t i=0;i<payloads;++i)if(inputs[i].description()!=descriptions[i])
    return std::unexpected(Status::InvalidArgument("Graph batch input differs from its prepared port contract"));
  HYLE_ASSIGN_OR_RETURN(auto mirror,(s->graph_shared_metadata.valid()?s->graph_shared_metadata:s->mirror_ready).view());
  if(choice){
    HYLE_ASSIGN_OR_RETURN(auto selector,TensorDesc::dense(DataType::kInt32,std::array<std::int64_t,1>{1}));
    if(inputs[payloads].description()!=selector || inputs[payloads].storage().resource()!=s->program->options.resource)
      return std::unexpected(Status::InvalidArgument("device batch choice needs its original I32[1] selector/resource"));
    HYLE_ASSIGN_OR_RETURN(auto selected,s->choice_selected.view());mirror=selected;
  }
  if(!owned){
    HYLE_ASSIGN_OR_RETURN(auto metadata_description,s->program->schema.metadata_description());
    if(inputs[payloads].description()!=metadata_description)
      return std::unexpected(Status::InvalidArgument("Graph batch metadata description differs from its owned configuration"));
  }
  HYLE_ASSIGN_OR_RETURN(auto output,batch_output_description(capacity,query_rows));
  if(outputs[0].description()!=output)return std::unexpected(Status::InvalidArgument("Graph batch output differs from its prepared port contract"));
  HYLE_ASSIGN_OR_RETURN(auto views,workspace_views(*s));views.metadata=mirror;views.output=outputs[0].storage();return views;
}
} // namespace
BatchProgram::BatchProgram(const BatchProgram&p)noexcept:data_(p.data_){if(data_)data_->incref();}
BatchProgram&BatchProgram::operator=(const BatchProgram&p)noexcept{if(this!=&p){BatchProgram next(p);std::swap(data_,next.data_);}return *this;}
BatchProgram::BatchProgram(BatchProgram&&p)noexcept:data_(std::exchange(p.data_,nullptr)){}
BatchProgram&BatchProgram::operator=(BatchProgram&&p)noexcept{if(this!=&p){BatchProgram next(std::move(p));std::swap(data_,next.data_);}return *this;}
BatchProgram::~BatchProgram(){if(data_)data_->decref();}
BatchSchema BatchProgram::schema()const noexcept{return data_?data_->schema:BatchSchema{};}
Result<BatchProgram> BatchProgram::prepare(const Context&context,BatchSchema schema,BatchProgramOptions options)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto retained,d::ContextRef::acquire(context));
  d::ProviderExtensionRef selected;
  for(std::size_t index=0;index<context.provider_count();++index){
    auto extension=d::ProviderExtensionAccess::query(context,index,schema.mla_layout()==MlaCacheLayout::kNone?d::kBatchPlanningExtensionId:d::kMlaBatchPlanningExtensionId);
    if(!extension){if(extension.error().IsUnimplemented())continue;return std::unexpected(extension.error());}
    selected=std::move(*extension);break;
  }
  return prepare(retained,selected,std::move(schema),std::move(options));
}
Result<BatchProgram> BatchProgram::prepare(const d::ContextRef&context,const d::ProviderExtensionRef&extension,
    BatchSchema schema,BatchProgramOptions options)noexcept{
  if(d::BatchLayoutAccess::is_row_cache(schema))return std::unexpected(Status::InvalidArgument("row cache is not an attention program"));
  if(!context.valid() || !schema.valid() || !options.resource.valid() || !options.host_workspace.valid() || !options.allocator.valid() ||
     !options.max_invocations || options.max_invocations==UINT32_MAX || !options.max_float_workspace_bytes || !options.max_int_workspace_bytes || !std::isfinite(options.scale))
    return std::unexpected(Status::InvalidArgument("batch program options are invalid"));
  HYLE_RETURN_IF_ERROR(validate_batch_attention_options(options.causal,options.window_left));
  if(!context.is_open())return std::unexpected(Status::FailedPrecondition("batch context is closed"));
  if(!extension.valid())return std::unexpected(Status::Unimplemented("selected route has no native BatchPlanning provider"));
  if(!context.owns(extension) || extension.id()!=(schema.mla_layout()==MlaCacheLayout::kNone?d::kBatchPlanningExtensionId:d::kMlaBatchPlanningExtensionId))
    return std::unexpected(Status::InvalidArgument("batch capability does not belong to the exact context/schema"));
  HYLE_RETURN_IF_ERROR(context.validate_resource(options.resource));
  HYLE_RETURN_IF_ERROR(context.validate_resource(options.host_workspace));
  if(options.resource_budget.valid() && !options.resource_budget.allows_unattributed())
    return std::unexpected(Status::Unimplemented("native batch planning private bytes are unknown"));
  auto a=options.allocator;if(options.resource_budget.valid()){HYLE_ASSIGN_OR_RETURN(a,options.resource_budget.metadata_allocator());}
  auto*p=a.construct<d::BatchProgramData>(std::move(schema),std::move(options),a);if(!p)return std::unexpected(Status::ResourceExhausted("batch program metadata"));
  BatchProgram result(p);
  p->context=context;
  p->extension=extension;
  HYLE_RETURN_IF_ERROR(d::ProviderExtensionAccess::call(p->extension,[](void*context,const void*table,void*opaque)noexcept{
    auto&p=*static_cast<d::BatchProgramData*>(opaque);p.table=static_cast<const d::BatchProviderVTable*>(table);
    if(!p.table->qualify || !p.table->create || !p.table->destroy || !p.table->update || !p.table->plan || !p.table->run ||
        (p.table->uses_caller_workspace && !p.table->workspace))return Status::FailedPrecondition("batch provider interface is incomplete");
    if(p.table->runtime)p.runtime=p.table->runtime(context);
    if(!p.runtime.valid() || (p.runtime.capture_compatible &&
        (!p.table->supports_command_capture || p.runtime.mode!=ProviderRuntimeMode::kSharedQueue ||
         p.runtime.workspace_owner!=ProviderWorkspaceOwner::kProvider ||
         p.runtime.replay_class!=ProviderReplayClass::kManaged || p.runtime.blocks_host ||
         p.runtime.uses_private_threads)))
      return Status::FailedPrecondition("batch provider recording runtime is invalid");
    return p.table->qualify(context,p.schema.capacity(),p.options);
  },p));
  const auto bytes=p->schema.metadata_offsets().bytes;
  const std::array specs{CopyResourceSpec{p->options.host_workspace,bytes},CopyResourceSpec{p->options.resource,bytes}};
  HYLE_ASSIGN_OR_RETURN(p->upload,d::CopyStreamAccess::prepare(context,specs,{.max_resources=2,.max_copies=5,.max_phases=1,.max_bytes=bytes,.allocator=a,.resource_budget=p->options.resource_budget}));
  return result;
}
BatchInvocation::BatchInvocation(BatchInvocation&&v)noexcept:data_(std::exchange(v.data_,nullptr)){}
BatchInvocation&BatchInvocation::operator=(BatchInvocation&&v)noexcept{if(this!=&v){BatchInvocation old(std::move(*this));data_=std::exchange(v.data_,nullptr);}return *this;}
BatchInvocation::~BatchInvocation(){if(data_)data_->release_user();}
Result<BatchInvocation> BatchInvocation::create(const BatchProgram&program,Queue&queue)noexcept{
  return create(program,queue,false);
}
namespace {
Result<d::BatchInvocationData*> create_batch_invocation(d::BatchProgramData*p,Queue&queue,
    bool external_output,std::shared_ptr<Queue>metadata_queue={},
    const d::BatchGraphFloatWorkspace*graph_float=nullptr)noexcept{
  if(!p || !queue.valid())return std::unexpected(Status::InvalidArgument("batch invocation program/queue is invalid"));
  if(graph_float){
    const auto storage=graph_float->storage;
    if(!external_output || !p->table->uses_caller_workspace)
      return std::unexpected(Status::Unimplemented("borrowed float workspace requires a caller-workspace Graph batch"));
    if(!queue.is_open() || !p->context.owns(queue) || !storage.valid() ||
        storage.resource()!=p->options.resource || !storage.native_handle())
      return std::unexpected(Status::InvalidArgument("Graph float workspace differs from the original resource/Context"));
    HYLE_RETURN_IF_ERROR(p->context.validate_resource(storage.resource()));
    const auto properties=storage.resource().properties();
    if(properties.domain!=p->options.resource.domain() ||
        properties.minimum_alignment<16 || properties.minimum_alignment%16 || storage.offset()%16)
      return std::unexpected(Status::InvalidArgument("Graph float workspace requires its original domain and 16-byte alignment"));
    if(!storage.size())
      return std::unexpected(Status::ResourceExhausted("Graph float workspace is empty"));
    if(storage.offset()>SIZE_MAX-storage.size())
      return std::unexpected(Status::OutOfRange("Graph float workspace range overflows"));
  }
  if(metadata_queue && (!queue.is_open() || !metadata_queue->is_open() ||
      !p->context.owns(queue) || !p->context.owns(*metadata_queue)))
    return std::unexpected(Status::InvalidArgument("batch metadata and execution Queues must belong to the original open Context"));
  auto count=p->invocations.load(std::memory_order_relaxed);
  do{if(count==p->options.max_invocations)return std::unexpected(Status::Unavailable("batch invocation capacity retained"));}
  while(!p->invocations.compare_exchange_weak(count,count+1,std::memory_order_relaxed));
  auto*s=p->allocator.construct<d::BatchInvocationData>(p);if(!s){p->invocations.fetch_sub(1,std::memory_order_relaxed);return std::unexpected(Status::ResourceExhausted("batch invocation metadata"));}
  util::ScopeGuard rollback{[&]()noexcept{s->release_user();}};
  HYLE_ASSIGN_OR_RETURN(s->queue_owner,d::QueueRef::acquire(queue));
  s->queue=&queue;s->external_output=external_output;
  if(graph_float)s->graph_float_workspace=graph_float->storage;
  s->buffers=p->allocator.construct<d::BatchBuffers>(p);if(!s->buffers)return std::unexpected(Status::ResourceExhausted("batch buffer owner"));
  s->buffers->metadata_queue=std::move(metadata_queue);
  s->buffers->provider_owner=p->extension;
  HYLE_ASSIGN_OR_RETURN(auto output_capacity,output_bytes(p->schema.capacity()));const auto output=external_output?0u:output_capacity;const auto mirror=p->schema.metadata_offsets().bytes;
  std::size_t capacity=2u*256u;
  for(const auto bytes:{output,mirror}){if(bytes>SIZE_MAX-capacity)return std::unexpected(Status::OutOfRange("batch pool extent overflow"));capacity+=bytes;}
  HYLE_ASSIGN_OR_RETURN(s->buffers->device,LeasePool::create(p->options.resource,capacity,{.max_blocks=2,.allocator=p->allocator,.resource_budget=p->options.resource_budget}));
  HYLE_ASSIGN_OR_RETURN(s->buffers->host,LeasePool::create(p->options.host_workspace,mirror,{.max_blocks=1,.allocator=p->allocator,.resource_budget=p->options.resource_budget}));
  HYLE_ASSIGN_OR_RETURN(s->mirror,s->buffers->device.allocate(mirror));if(!external_output){HYLE_ASSIGN_OR_RETURN(s->output,s->buffers->device.allocate(output));}
  HYLE_ASSIGN_OR_RETURN(s->staging,s->buffers->host.allocate(mirror));
  HYLE_RETURN_IF_ERROR(s->mirror.view()->fill(std::byte{0}));
  HYLE_RETURN_IF_ERROR(s->staging.view()->fill(std::byte{0}));
  retain_domain(*s->buffers,s->buffers->device);retain_domain(*s->buffers,s->buffers->host);
  auto&upload_queue=s->buffers->metadata_queue?*s->buffers->metadata_queue:queue;
  HYLE_ASSIGN_OR_RETURN(s->upload,CopyBatchInvocation::create(p->upload,upload_queue));
  const auto refs=p->schema.capacity().max_page_refs;
  if(refs>(UINT32_MAX-8u)/2u)return std::unexpected(Status::OutOfRange("batch page-use capacity overflow"));
  HYLE_RETURN_IF_ERROR(s->read_set.reserve(8u+2u*refs));
  HYLE_ASSIGN_OR_RETURN(s->uses,LeaseUsePool::create({.max_batches=1,.max_resources_per_batch=8u+2u*refs,.allocator=p->allocator,.resource_budget=p->options.resource_budget}));
  HYLE_ASSIGN_OR_RETURN(s->completion,d::RetainedCompletion::create(p->allocator));
  HYLE_RETURN_IF_ERROR(d::ProviderExtensionAccess::call(p->extension,[](void*context,const void*table,void*opaque)noexcept{
    auto&s=*static_cast<d::BatchInvocationData*>(opaque);return static_cast<const d::BatchProviderVTable*>(table)->create(context,s.program->schema.capacity(),s.program->options,s.program->allocator,&s.native);
  },s));rollback.dismiss();return s;
}
} // namespace
Result<BatchInvocation> BatchInvocation::create(const BatchProgram&program,Queue&queue,bool external_output)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto*data,create_batch_invocation(program.data_,queue,external_output));
  return BatchInvocation(data);
}
Result<PreparedBatch> BatchInvocation::prepare_batch(const BatchLayoutVersion&layout)noexcept{
  return prepare(layout,false);
}
Result<PreparedBatch> BatchInvocation::update_batch(const BatchLayoutVersion&layout)noexcept{
  return prepare(layout,true);
}
Result<PreparedBatch> BatchInvocation::prepare(const BatchLayoutVersion&layout,bool update,
    const BatchInvocation*metadata_source,const d::BeforeCommandChange*before_change)noexcept{
  if(!data_ || data_->closed || data_->failed)return std::unexpected(Status::FailedPrecondition("batch invocation is closed/failed"));
  auto&s=*data_;
  if(!s.program->schema.owns(layout))return std::unexpected(Status::InvalidArgument("batch layout belongs to another schema"));
  ReadLease shared;
  if(metadata_source){
    const auto*source=metadata_source->data_;
    if(!source || source==&s || !s.external_output || !source->external_output || source->closed || source->failed || !source->prepared || source->ran || source->layout!=layout ||
        source->program->options.resource!=s.program->options.resource)
      return std::unexpected(Status::InvalidArgument("shared Graph metadata needs a prepared same-layout resource"));
    shared=source->graph_shared_metadata.valid()?source->graph_shared_metadata:source->mirror_ready;
    if(!shared.valid())return std::unexpected(Status::FailedPrecondition("shared Graph metadata is not ready"));
  }
  HYLE_RETURN_IF_ERROR(validate_batch_attention_layout(layout,s.program->options.causal,s.program->options.window_left));
  if(s.tickets || in_flight() || !s.completion.available())return std::unexpected(Status::Unavailable("batch slot has a live ticket/native work/old completion"));
  s.device_choices=false;
  // Native callbacks publish terminal before releasing pending. Check only
  // after actual retirement; an observation error must never reset the slot.
  if(!s.terminal.ok()){s.failed=true;return std::unexpected(Status::FailedPrecondition("batch invocation native terminal failed"));}
  if(s.output_ready.valid() && !d::LeaseAccess::unique_read_owner(s.output_ready))return std::unexpected(Status::Unavailable("old batch output prevents slot overwrite"));
  if(s.epoch==UINT64_MAX)return std::unexpected(Status::OutOfRange("batch slot epoch exhausted"));
  // Sizing is read-only: a failed query/cap check preserves the old episode.
  // Never use the native update callback as a sizing probe.
  // A completed preparation or retired run retains the original plan and
  // scratch. A partial preparation cleared both flags before any resize; its
  // retry must size and create a fresh plan, even if native metadata survived.
  const bool may_update=update && (s.prepared || s.ran);
  HYLE_ASSIGN_OR_RETURN(auto workspace_report,query_workspace(s,layout,may_update));
  PreparationChangeGate gate{before_change};
  if(s.graph_metadata_binding.valid()){
    mem::ConstBufferView next_metadata;
    if(s.preparing_choices && s.choice_selected.valid()){HYLE_ASSIGN_OR_RETURN(auto next,s.choice_selected.view());next_metadata=next;}
    else if(shared.valid()){HYLE_ASSIGN_OR_RETURN(next_metadata,shared.view());}
    else if(s.mirror_ready.valid()){HYLE_ASSIGN_OR_RETURN(next_metadata,s.mirror_ready.view());}
    else {HYLE_ASSIGN_OR_RETURN(auto mirror,s.mirror.view());next_metadata=mirror;}
    if(!same_storage(s.graph_metadata_binding,next_metadata))
      HYLE_RETURN_IF_ERROR(invalidate_recorded_commands(s,gate));
  }
  if(s.output_ready.valid()){HYLE_ASSIGN_OR_RETURN(s.output,s.output_ready.try_write());}
  ++s.epoch;s.prepared=false;s.ran=false;s.layout=layout;s.report=workspace_report;s.terminal=Status::Ok();
  HYLE_ASSIGN_OR_RETURN(const bool workspace_changed,prepare_workspace(s,workspace_report,gate));
  s.graph_shared_metadata=std::move(shared);
  CopyBatchReport upload_report{};std::size_t staging_bytes=0;std::uint32_t upload_waits=0;
  if(!s.graph_shared_metadata.valid()){
    const auto next=d::BatchLayoutAccess::segments(layout),old=d::BatchLayoutAccess::segments(s.mirror_layout);
    std::array<CopyRange,5> ranges{};std::size_t count=0;
    // Only active words are compared. A shrinking segment includes its old
    // active suffix; initialized mirrors contain zero everywhere else.
    for(std::size_t part=0;part<next.size();++part){
      const auto a=next[part].bytes,b=old[part].bytes;const auto extent=std::max(a.size(),b.size());
      std::size_t first=extent,last=0;
      for(std::size_t at=0;at<extent;at+=4){
        std::uint32_t x=0,y=0;if(at<a.size())std::memcpy(&x,a.data()+at,4);if(at<b.size())std::memcpy(&y,b.data()+at,4);
        if(x!=y){first=std::min(first,at);last=at+4;}
      }
      if(first<last)ranges[count++]={{0},{1},{0},next[part].offset+first,next[part].offset+first,last-first};
    }
    if(count){
      HYLE_RETURN_IF_ERROR(s.upload.reset());
      if(s.mirror_ready.valid()){HYLE_ASSIGN_OR_RETURN(s.mirror,s.mirror_ready.try_write());}
      if(s.staging_ready.valid()){HYLE_ASSIGN_OR_RETURN(s.staging,s.staging_ready.try_write());}
      {
        HYLE_ASSIGN_OR_RETURN(auto view,s.staging.view());HYLE_ASSIGN_OR_RETURN(auto mapping,view.map_host());
        HYLE_ASSIGN_OR_RETURN(auto bytes,mapping.writable_bytes());
        for(const auto&range:std::span(ranges).first(count)){
          auto target=bytes.subspan(range.source_offset,range.bytes);std::fill(target.begin(),target.end(),std::byte{0});
          for(const auto&segment:next){
            if(range.source_offset<segment.offset || range.source_offset>=segment.offset+segment.bytes.size())continue;
            const auto from=range.source_offset-segment.offset;const auto length=std::min(target.size(),segment.bytes.size()-from);
            std::memcpy(target.data(),segment.bytes.data()+from,length);break;
          }
          staging_bytes+=range.bytes;
        }
        HYLE_RETURN_IF_ERROR(mapping.close());
      }
      HYLE_ASSIGN_OR_RETURN(s.staging_ready,s.staging.publish());
      HYLE_RETURN_IF_ERROR(s.upload.bind_source({0},s.staging_ready));HYLE_RETURN_IF_ERROR(s.upload.bind_mutable({1},std::move(s.mirror)));
      const std::array<CopyPhase,1> phases{{{}}};
      HYLE_RETURN_IF_ERROR(s.upload.prepare_replay(phases,std::span(ranges).first(count)));HYLE_ASSIGN_OR_RETURN(auto uploaded,s.upload.submit());
      const auto terminal=uploaded.wait();if(!terminal.ok()){s.failed=true;return std::unexpected(terminal);}
      upload_report=s.upload.report();upload_waits=1;uploaded={};
      // F2 certifies only this episode's changed ranges. Retake the original
      // writer after actual retirement: our cold zero initialization plus all
      // successful deltas, not a fabricated whole-range F2 output, proves the
      // complete mirror initialized before publication.
      HYLE_ASSIGN_OR_RETURN(s.mirror,d::CopyStreamAccess::take_mutable(s.upload,{1}));
      HYLE_ASSIGN_OR_RETURN(s.mirror_ready,s.mirror.publish());
      HYLE_RETURN_IF_ERROR(s.upload.reset());
    }
    if(!s.mirror_ready.valid()){HYLE_ASSIGN_OR_RETURN(s.mirror_ready,s.mirror.publish());}
    s.mirror_layout=layout;
  }
  bool reused=false,planned_native=false;
  if(may_update && !workspace_changed){struct Call{d::BatchInvocationData*state;bool reused=false;}call{&s};
    const auto status=d::ProviderExtensionAccess::call(s.program->extension,[](void*,const void*table,void*opaque)noexcept->Status{
      auto&call=*static_cast<Call*>(opaque);HYLE_ASSIGN_OR_RETURN(call.reused,static_cast<const d::BatchProviderVTable*>(table)->update(call.state->native,call.state->layout,&call.state->report));return Status::Ok();
    },&call);if(!status.ok()){s.failed=true;return std::unexpected(status);}reused=call.reused;
  }
  if(!reused){
    // A read-only, exact-layout workspace query may prove that refreshing the
    // scheduler's contents preserves every recorded launch argument/offset.
    // Retain only unchanged carriers on the original qualified runtime. The
    // native plan still performs its real upload and checked retirement below.
    const bool preserves_commands=may_update && !workspace_changed && !gate.notified &&
        workspace_report.reused_native_plan && qualify_graph_recording(s).ok();
    if(!preserves_commands)HYLE_RETURN_IF_ERROR(invalidate_recorded_commands(s,gate));
    s.report.reused_native_plan=preserves_commands;
    HYLE_ASSIGN_OR_RETURN(auto views,workspace_views(s));
    auto planned=submit_native(s,true,{}, {},views);
    if(!planned){s.failed=true;return std::unexpected(planned.error());}
    planned_native=true;
    const auto terminal=planned->wait();*planned={};
    if(!terminal.ok() || s.pending.load(std::memory_order_acquire) || s.uses.active_batches()){s.failed=true;return std::unexpected(!terminal.ok()?terminal:Status::Unavailable("batch planning retirement unproven"));}
    if(preserves_commands && !s.report.reused_native_plan){s.failed=true;return std::unexpected(Status::Internal("batch planning did not preserve its recorded-command proof"));}
    reused=preserves_commands;
  }
  s.report.metadata_upload=upload_report;s.report.metadata_staging_bytes=staging_bytes;s.report.host_waits=upload_waits+(planned_native?1u:0u);s.report.reused_native_plan=reused;s.report.capture_compatible=s.replay && s.replay->commands.valid();s.prepared=true;
  ++s.tickets;++s.users;s.incref();return PreparedBatch(&s,s.epoch);
}
bool BatchInvocation::in_flight()const noexcept{return data_ && (data_->pending.load(std::memory_order_acquire) || data_->uses.active_batches() || data_->upload.in_flight());}
BatchPlanReport BatchInvocation::report()const noexcept{return data_?data_->report:BatchPlanReport{};}
Result<ReadLease> BatchInvocation::output()noexcept{
  if(!data_ || data_->closed || !data_->ran)return std::unexpected(Status::FailedPrecondition("batch has no submitted output"));
  auto&s=*data_;if(in_flight())return std::unexpected(Status::Unavailable("batch output is pending"));
  if(!s.terminal.ok())return std::unexpected(s.terminal);
  if(!s.output_ready.valid()){
    HYLE_ASSIGN_OR_RETURN(s.output_ready,s.output.publish());
  }return s.output_ready;
}
Status BatchInvocation::request_cancel()noexcept{if(!data_)return Status::Ok();data_->closed=true;data_->prepared=false;return Status::Ok();}
Status BatchInvocation::close()noexcept{request_cancel().IgnoreError();return in_flight()?Status::Unavailable("batch native work still retires"):Status::Ok();}
Status BatchInvocation::drain()noexcept{
  if(!data_)return Status::Ok();
  close().IgnoreError();const auto copy=data_->upload.drain();const auto native=data_->completion.wait();
  // An unknown observation can finish the public Completion while the original
  // backend still owns its accepted prefix. Drain waits for actual retention.
  const auto uses=data_->uses.drain();
  for(auto pending=data_->pending.load(std::memory_order_acquire);pending;pending=data_->pending.load(std::memory_order_acquire))
    data_->pending.wait(pending,std::memory_order_acquire);
  return !copy.ok()?copy:!native.ok()?native:uses;
}
PreparedBatch::PreparedBatch(PreparedBatch&&b)noexcept:data_(std::exchange(b.data_,nullptr)),epoch_(b.epoch_){}
PreparedBatch&PreparedBatch::operator=(PreparedBatch&&b)noexcept{if(this!=&b){PreparedBatch old(std::move(*this));data_=std::exchange(b.data_,nullptr);epoch_=b.epoch_;}return *this;}
PreparedBatch::~PreparedBatch(){if(data_){--data_->tickets;data_->release_user();}}
std::uint64_t PreparedBatch::generation()const noexcept{return data_?data_->layout.generation():0;}
Status PreparedBatch::prepare_replay(std::span<const ReadLease>leases)noexcept{
  auto*s=data_;
  if(!s || s->external_output || s->closed || s->failed || !s->prepared || s->ran || epoch_!=s->epoch)
    return Status::FailedPrecondition("batch replay token is closed/stale/already consumed");
  if(leases.size()!=input_count(s->program->schema))return Status::InvalidArgument("batch replay payload count differs from schema");
  HYLE_ASSIGN_OR_RETURN(auto descriptions,input_descriptions(s->program->schema));
  std::array<ConstTensorView,3>inputs;
  for(std::size_t i=0;i<leases.size();++i){HYLE_ASSIGN_OR_RETURN(inputs[i],leases[i].tensor(descriptions[i]));}
  HYLE_ASSIGN_OR_RETURN(auto views,workspace_views(*s));return capture_batch(*s,std::span(inputs).first(input_count(s->program->schema)),views);
}
Status PreparedBatch::prepare_replay(const ReadLease&query,const PagedArenaRead&keys,const PagedArenaRead&values)noexcept{
  auto*s=data_;
  if(!s || s->external_output || s->closed || s->failed || !s->prepared || s->ran || epoch_!=s->epoch ||
      s->program->schema.capacity().storage!=BatchStorage::kPaged || s->program->schema.mla_layout()!=MlaCacheLayout::kNone)
    return Status::FailedPrecondition("paged batch replay token is closed/stale/already consumed");
  if(keys.layout()!=s->layout || values.layout()!=s->layout)return Status::InvalidArgument("paged replay layout differs from prepared metadata");
  const auto capacity=s->program->schema.capacity();
  HYLE_RETURN_IF_ERROR(d::PagedArenaAccess::validate_storage(keys,capacity.key_storage));
  HYLE_RETURN_IF_ERROR(d::PagedArenaAccess::validate_storage(values,capacity.value_storage));
  HYLE_ASSIGN_OR_RETURN(auto descriptions,input_descriptions(s->program->schema));std::array<ConstTensorView,3>inputs;
  HYLE_ASSIGN_OR_RETURN(inputs[0],query.tensor(descriptions[0]));
  HYLE_ASSIGN_OR_RETURN(inputs[1],d::PagedArenaAccess::tensor(keys));HYLE_ASSIGN_OR_RETURN(inputs[2],d::PagedArenaAccess::tensor(values));
  for(std::size_t i=1;i<3;++i)if(inputs[i].description()!=descriptions[i] || inputs[i].storage().resource()!=s->program->options.resource)
    return Status::InvalidArgument("paged replay capacity/resource differs from native slot");
  HYLE_ASSIGN_OR_RETURN(auto views,workspace_views(*s));return capture_batch(*s,std::span(inputs).first(input_count(s->program->schema)),views);
}
Result<Completion> PreparedBatch::run(std::span<const ReadLease>leases)noexcept{
  if(!data_ || data_->closed || data_->failed || !data_->prepared || data_->ran || epoch_!=data_->epoch)
    return std::unexpected(Status::FailedPrecondition("batch token is closed/stale/already consumed"));
  if(leases.size()!=input_count(data_->program->schema))return std::unexpected(Status::InvalidArgument("batch payload count differs from schema"));
  auto&s=*data_;HYLE_ASSIGN_OR_RETURN(auto descriptions,input_descriptions(s.program->schema));
  if(s.external_output)return std::unexpected(Status::FailedPrecondition("Graph batch slot requires its admitted original region"));
  std::array<ConstTensorView,3> inputs;
  for(std::size_t i=0;i<leases.size();++i){HYLE_ASSIGN_OR_RETURN(inputs[i],leases[i].tensor(descriptions[i]));}
  HYLE_ASSIGN_OR_RETURN(auto views,workspace_views(s));
  HYLE_RETURN_IF_ERROR(validate_replay_storage(s,std::span(inputs).first(leases.size()),views));
  auto submitted=submit_native(s,false,leases,std::span(inputs).first(leases.size()),views);
  if(!submitted){s.failed=true;s.prepared=false;return std::unexpected(submitted.error());}
  s.ran=true;s.prepared=false;return std::move(*submitted);
}
Result<BatchProgram> d::BatchGraphAccess::prepare(const ContextRef&context,const ProviderExtensionRef&extension,
    BatchSchema schema,BatchProgramOptions options)noexcept{
  return BatchProgram::prepare(context,extension,std::move(schema),std::move(options));
}
Result<Completion> PreparedBatch::run(const ReadLease&query,const PagedArenaRead&keys,const PagedArenaRead&values)noexcept{
  if(!data_ || data_->closed || data_->failed || !data_->prepared || data_->ran || epoch_!=data_->epoch)
    return std::unexpected(Status::FailedPrecondition("batch token is closed/stale/already consumed"));
  auto&s=*data_;if(s.external_output || s.program->schema.capacity().storage!=BatchStorage::kPaged || s.program->schema.mla_layout()!=MlaCacheLayout::kNone)
    return std::unexpected(Status::FailedPrecondition("paged arena Direct needs its prepared paged native slot"));
  if(keys.layout()!=s.layout || values.layout()!=s.layout)
    return std::unexpected(Status::InvalidArgument("paged arena read layout differs from prepared metadata"));
  const auto capacity=s.program->schema.capacity();
  HYLE_RETURN_IF_ERROR(d::PagedArenaAccess::validate_storage(keys,capacity.key_storage));
  HYLE_RETURN_IF_ERROR(d::PagedArenaAccess::validate_storage(values,capacity.value_storage));
  HYLE_ASSIGN_OR_RETURN(auto descriptions,input_descriptions(s.program->schema));std::array<ConstTensorView,3> inputs;
  HYLE_ASSIGN_OR_RETURN(inputs[0],query.tensor(descriptions[0]));HYLE_ASSIGN_OR_RETURN(inputs[1],d::PagedArenaAccess::tensor(keys));HYLE_ASSIGN_OR_RETURN(inputs[2],d::PagedArenaAccess::tensor(values));
  for(std::size_t i=1;i<3;++i)if(inputs[i].description()!=descriptions[i] || inputs[i].storage().resource()!=s.program->options.resource)
    return std::unexpected(Status::InvalidArgument("paged arena capacity/resource differs from native slot"));
  HYLE_ASSIGN_OR_RETURN(auto views,workspace_views(s));const std::array ordinary{query};
  HYLE_RETURN_IF_ERROR(validate_replay_storage(s,inputs,views));
  auto submitted=submit_native(s,false,ordinary,inputs,views,d::PagedArenaAccess::pages(keys),d::PagedArenaAccess::pages(values));
  if(!submitted){s.failed=true;s.prepared=false;return std::unexpected(submitted.error());}s.ran=true;s.prepared=false;return std::move(*submitted);
}
Status PreparedBatch::prepare_replay(MlaBatchInputs inputs)noexcept{
  if(!data_ || data_->program->schema.mla_layout()==MlaCacheLayout::kNone)
    return Status::InvalidArgument("MLA inputs require an MLA schema");
  const std::array leases{std::move(inputs.query),std::move(inputs.cache)};
  return prepare_replay(leases);
}
Result<Completion> PreparedBatch::run(MlaBatchInputs inputs)noexcept{
  if(!data_ || data_->program->schema.mla_layout()==MlaCacheLayout::kNone)
    return std::unexpected(Status::InvalidArgument("MLA inputs require an MLA schema"));
  const std::array leases{std::move(inputs.query),std::move(inputs.cache)};
  return run(leases);
}
Status PreparedBatch::prepare_replay(const ReadLease&query,const PagedArenaRead&cache)noexcept{
  auto*s=data_;
  if(!s || s->external_output || s->closed || s->failed || !s->prepared || s->ran || epoch_!=s->epoch ||
      s->program->schema.mla_layout()==MlaCacheLayout::kNone || s->program->schema.capacity().storage!=BatchStorage::kPaged)
    return Status::FailedPrecondition("MLA replay needs a prepared paged MLA token");
  if(cache.layout()!=s->layout)return Status::InvalidArgument("MLA cache layout differs from prepared metadata");
  HYLE_RETURN_IF_ERROR(d::PagedArenaAccess::validate_storage(cache,s->program->schema.capacity().key_storage));
  HYLE_ASSIGN_OR_RETURN(auto descriptions,input_descriptions(s->program->schema));
  std::array<ConstTensorView,2>inputs;
  HYLE_ASSIGN_OR_RETURN(inputs[0],query.tensor(descriptions[0]));
  HYLE_ASSIGN_OR_RETURN(inputs[1],d::PagedArenaAccess::tensor(cache));
  if(inputs[1].description()!=descriptions[1] || inputs[1].storage().resource()!=s->program->options.resource)
    return Status::InvalidArgument("MLA cache capacity/resource differs from native slot");
  HYLE_ASSIGN_OR_RETURN(auto views,workspace_views(*s));return capture_batch(*s,inputs,views);
}
Result<Completion> PreparedBatch::run(const ReadLease&query,const PagedArenaRead&cache)noexcept{
  auto*s=data_;
  if(!s || s->external_output || s->closed || s->failed || !s->prepared || s->ran || epoch_!=s->epoch ||
      s->program->schema.mla_layout()==MlaCacheLayout::kNone || s->program->schema.capacity().storage!=BatchStorage::kPaged)
    return std::unexpected(Status::FailedPrecondition("MLA run needs a prepared paged MLA token"));
  if(cache.layout()!=s->layout)return std::unexpected(Status::InvalidArgument("MLA cache layout differs from prepared metadata"));
  HYLE_RETURN_IF_ERROR(d::PagedArenaAccess::validate_storage(cache,s->program->schema.capacity().key_storage));
  HYLE_ASSIGN_OR_RETURN(auto descriptions,input_descriptions(s->program->schema));
  std::array<ConstTensorView,2>inputs;
  HYLE_ASSIGN_OR_RETURN(inputs[0],query.tensor(descriptions[0]));
  HYLE_ASSIGN_OR_RETURN(inputs[1],d::PagedArenaAccess::tensor(cache));
  if(inputs[1].description()!=descriptions[1] || inputs[1].storage().resource()!=s->program->options.resource)
    return std::unexpected(Status::InvalidArgument("MLA cache capacity/resource differs from native slot"));
  HYLE_ASSIGN_OR_RETURN(auto views,workspace_views(*s));
  HYLE_RETURN_IF_ERROR(validate_replay_storage(*s,inputs,views));
  const std::array ordinary{query};
  auto submitted=submit_native(*s,false,ordinary,inputs,views,d::PagedArenaAccess::pages(cache));
  if(!submitted){s->failed=true;s->prepared=false;return std::unexpected(submitted.error());}
  s->ran=true;s->prepared=false;return std::move(*submitted);
}
PagedBatchReceiver::PagedBatchReceiver(PagedBatchReceiver&&v)noexcept:data_(std::exchange(v.data_,nullptr)){}
PagedBatchReceiver&PagedBatchReceiver::operator=(PagedBatchReceiver&&v)noexcept{
  if(this!=&v){if(data_)data_->decref();data_=std::exchange(v.data_,nullptr);}return *this;
}
PagedBatchReceiver::~PagedBatchReceiver(){if(data_)data_->decref();}
Result<PagedBatchReceiver> PagedBatchReceiver::create(BatchInvocation&&owner,std::size_t capacity)noexcept{
  auto*s=owner.data_;
  if(!s || !capacity || s->closed || s->failed || s->external_output ||
      s->program->schema.capacity().storage!=BatchStorage::kPaged)
    return std::unexpected(Status::InvalidArgument("pending receiver needs an open paged Direct controller/capacity"));
  if(s->users!=1 || s->tickets || owner.in_flight() || !s->completion.available())
    return std::unexpected(Status::Unavailable("pending receiver controller has an external ticket/native outcome"));
  const auto allocator=s->program->allocator;
  auto*p=allocator.construct<d::PagedReceiverData>(allocator);
  if(!p)return std::unexpected(Status::ResourceExhausted("common pending receiver metadata"));
  PagedBatchReceiver result(p);
  HYLE_ASSIGN_OR_RETURN(p->source,(d::CopyOutcomeSource<CopyBatchUpdate,d::CopyStreamAccess>::create(
      capacity,{.allocator=allocator,.resource_budget=s->program->options.resource_budget})));
  HYLE_ASSIGN_OR_RETURN(p->dependent,d::DependentCompletion::create(allocator));
  p->original=std::move(owner);return result;
}
Status PagedBatchReceiver::prepare_batch(const BatchLayoutVersion&layout,bool update)noexcept{
  if(!data_)return Status::FailedPrecondition("empty common pending receiver");
  auto&p=*data_;HYLE_RETURN_IF_ERROR(p.reusable());
  if(p.submitted && !p.terminal.ok())return Status::FailedPrecondition("common pending receiver episode failed");
  p.token={};
  auto prepared=update?p.original.update_batch(layout):p.original.prepare_batch(layout);
  if(!prepared)return prepared.error();
  p.token=std::move(*prepared);p.submitted=false;p.terminal=Status::Ok();return Status::Ok();
}
Result<Completion> PagedBatchReceiver::submit(const ReadLease&query,
    const PagedArenaPendingRead&keys,const PagedArenaPendingRead&values,CopyBatchUpdate&&update)noexcept{
  const std::array frames{keys,values};
  return submit_pages(query,frames,std::move(update));
}
Result<Completion> PagedBatchReceiver::submit(const ReadLease&query,
    const PagedArenaPendingRead&cache,CopyBatchUpdate&&update)noexcept{
  return submit_pages(query,std::span(&cache,1),std::move(update));
}
Result<Completion> PagedBatchReceiver::submit_pages(const ReadLease&query,
    std::span<const PagedArenaPendingRead>frames,CopyBatchUpdate&&update)noexcept{
  if(!data_)return std::unexpected(Status::FailedPrecondition("empty common pending receiver"));
  auto&p=*data_;HYLE_RETURN_IF_ERROR(p.reusable());auto*s=p.token.data_;
  if(!s || s!=p.original.data_ || s->closed || s->failed || !s->prepared || s->ran || p.token.epoch_!=s->epoch)
    return std::unexpected(Status::FailedPrecondition("common pending receiver prepared token is stale/consumed"));
  if(frames.size()+1!=input_count(s->program->schema))
    return std::unexpected(Status::InvalidArgument("pending cache count differs from the prepared schema"));
  const auto capacity=s->program->schema.capacity();
  for(std::size_t i=0;i<frames.size();++i){
    if(frames[i].layout()!=s->layout)
      return std::unexpected(Status::InvalidArgument("common pending frame differs from the prepared layout"));
    HYLE_RETURN_IF_ERROR(d::PagedArenaAccess::validate_storage(frames[i],i?capacity.value_storage:capacity.key_storage));
    HYLE_RETURN_IF_ERROR((d::PagedArenaAccess::validate<CopyBatchUpdate,d::CopyStreamAccess>(frames[i],update)));
  }
  HYLE_ASSIGN_OR_RETURN(auto descriptions,input_descriptions(s->program->schema));
  std::array<ConstTensorView,3> inputs;
  HYLE_ASSIGN_OR_RETURN(inputs[0],query.tensor(descriptions[0]));
  for(std::size_t i=0;i<frames.size();++i){
    HYLE_ASSIGN_OR_RETURN(inputs[i+1],d::PagedArenaAccess::tensor(frames[i]));
    if(inputs[i+1].description()!=descriptions[i+1] || inputs[i+1].storage().resource()!=s->program->options.resource)
      return std::unexpected(Status::InvalidArgument("common pending arena capacity/resource differs"));
  }
  if(!s->completion.available() || s->pending.load(std::memory_order_acquire) || s->uses.active_batches())
    return std::unexpected(Status::Unavailable("common pending native slot is still retained"));
  HYLE_ASSIGN_OR_RETURN(auto views,workspace_views(*s));HYLE_ASSIGN_OR_RETURN(auto use,s->uses.begin());
  HYLE_RETURN_IF_ERROR(use.read(query));HYLE_RETURN_IF_ERROR(use.read(s->mirror_ready));
  if(s->float_workspace.valid())HYLE_RETURN_IF_ERROR(use.write(s->float_workspace));
  if(s->int_workspace.valid())HYLE_RETURN_IF_ERROR(use.write(s->int_workspace));
  HYLE_RETURN_IF_ERROR(use.write(s->output));
  auto sources=p.source.take(std::move(update));if(!sources)return std::unexpected(sources.error());
  const auto restore=[&]()noexcept{update=std::move(p.source.update());sources->release(sources->state);};
  auto sequence=d::QueueSequence::begin(*s->queue);
  if(!sequence){restore();return std::unexpected(sequence.error());}
  auto prefix=d::CopyStreamAccess::dependency(p.source.update());
  if(!prefix)sequence->record(d::EnqueueResult::rejected(prefix.error()));
  else if(prefix->pending())sequence->record(sequence->enqueue_resolved_wait(*prefix));
  if(sequence->terminal().ok()){
    auto&reads=s->read_set;reads.reset();
    Status admitted;
    for(const auto&frame:frames){
      admitted=d::PagedArenaAccess::collect_reads<CopyBatchUpdate,d::CopyStreamAccess>(reads,frame,p.source.update());
      if(!admitted.ok())break;
    }
    if(admitted.ok())admitted=reads.admit(use);
    if(!admitted.ok())sequence->record(d::EnqueueResult::rejected(admitted));
  }
  auto suffix=finish_native(*s,false,std::span(inputs).first(input_count(s->program->schema)),views,std::move(use),*sequence);
  if(!suffix){restore();return std::unexpected(suffix.error());}
  s->ran=true;s->prepared=false;p.submitted=true;p.pending.store(true,std::memory_order_release);p.incref();
  p.retention.own={&p,[](void*opaque,const Status&terminal)noexcept{
      static_cast<d::PagedReceiverData*>(opaque)->terminal=terminal;
    },[](void*opaque)noexcept{
      auto*p=static_cast<d::PagedReceiverData*>(opaque);p->pending.store(false,std::memory_order_release);p->decref();
    }};
  auto aggregate=p.dependent.publish(std::move(*suffix),*sources);
  HYLE_FATAL_ASSERT(d::CompletionAccess::attach_retention(aggregate,p.retention).ok());
  return s->completion.publish(std::move(aggregate));
}
Result<ReadLease> PagedBatchReceiver::output()noexcept{
  if(!data_ || !data_->submitted)return std::unexpected(Status::FailedPrecondition("common pending receiver has no submitted output"));
  if(data_->pending.load(std::memory_order_acquire))return std::unexpected(Status::Unavailable("common pending aggregate is not observed/retired"));
  if(!data_->terminal.ok())return std::unexpected(data_->terminal);
  return data_->original.output();
}
Result<CopyBatchUpdate> PagedBatchReceiver::take_update()noexcept{
  if(!data_ || data_->pending.load(std::memory_order_acquire) || !data_->dependent.available() ||
      !data_->source.available() || !data_->source.update().valid() || data_->source.update().in_flight() ||
      !data_->original.data_->completion.available())
    return std::unexpected(Status::Unavailable("common pending receiver outcome/source is retained"));
  return std::move(data_->source.update());
}
BatchPlanReport PagedBatchReceiver::report()const noexcept{return data_?data_->original.report():BatchPlanReport{};}
Status PagedBatchReceiver::close()noexcept{return data_?data_->original.close():Status::Ok();}
Status PagedBatchReceiver::drain()noexcept{return data_?data_->original.drain():Status::Ok();}
Result<BatchInvocation> d::BatchGraphAccess::create(const BatchProgram&program,Queue&queue)noexcept{return BatchInvocation::create(program,queue,true);}
Result<BatchInvocation> d::BatchGraphAccess::create(const BatchProgram&program,Queue&execution,
    std::shared_ptr<Queue>metadata_queue)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto*data,create_batch_invocation(program.data_,execution,true,std::move(metadata_queue)));
  return BatchInvocation(data);
}
Result<BatchInvocation> d::BatchGraphAccess::create(const BatchProgram&program,Queue&execution,
    std::shared_ptr<Queue>metadata_queue,BatchGraphFloatWorkspace workspace)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto*data,create_batch_invocation(program.data_,execution,true,
      std::move(metadata_queue),&workspace));
  return BatchInvocation(data);
}
Result<ReadLease> d::BatchGraphAccess::metadata(const BatchInvocation&invocation)noexcept{
  if(!invocation.data_ || !invocation.data_->prepared)return std::unexpected(Status::FailedPrecondition("Graph batch slot is not prepared"));
  return invocation.data_->graph_shared_metadata.valid()?invocation.data_->graph_shared_metadata:invocation.data_->mirror_ready;
}
Result<PreparedBatch> d::BatchGraphAccess::prepare(BatchInvocation&target,
    const BatchLayoutVersion&layout,bool update,BeforeCommandChange before_change)noexcept{
  return target.prepare(layout,update,nullptr,&before_change);
}
Result<PreparedBatch> d::BatchGraphAccess::prepare_choices(BatchInvocation&target,
    const std::array<BatchLayoutVersion,2>&layouts,bool update,BeforeCommandChange before_change)noexcept{
  auto*s=target.data_;
  if(!s || !s->external_output || !layouts[0].valid() || !layouts[1].valid() ||
      !s->program->schema.owns(layouts[0]) || !s->program->schema.owns(layouts[1]))
    return std::unexpected(Status::InvalidArgument("device batch choices need two versions of the original schema"));
  for(const auto&layout:layouts){
    const auto facts=layout.facts();
    if(facts.requests!=1 || !facts.query_rows || facts.query_rows>2)
      return std::unexpected(Status::Unimplemented("device batch choices require one request and one/two active rows"));
    HYLE_RETURN_IF_ERROR(validate_batch_attention_layout(layout,s->program->options.causal,s->program->options.window_left));
  }
  if(!s->buffers || !s->buffers->metadata_queue)
    return std::unexpected(Status::Unimplemented("device batch choices require the original independent metadata Queue"));
  if(!s->choice_extension.valid()){
    HYLE_ASSIGN_OR_RETURN(s->choice_extension,ProviderExtensionAccess::query(s->program->extension,kBatchDeviceChoiceExtensionId));
    HYLE_RETURN_IF_ERROR(ProviderExtensionAccess::call(s->choice_extension,[](void*,const void*table,void*opaque)noexcept{
      auto&s=*static_cast<BatchInvocationData*>(opaque);s.choice_table=static_cast<const BatchDeviceChoiceVTable*>(table);
      return s.choice_table && s.choice_table->qualify && s.choice_table->select?Status::Ok():
          Status::FailedPrecondition("device batch choice capability is incomplete");
    },s));
  }
  const auto bigger=[](const BatchLayoutVersion&a,const BatchLayoutVersion&b){
    return a.facts().query_rows>b.facts().query_rows ||
        (a.facts().query_rows==b.facts().query_rows && a.facts().kv_rows>b.facts().kv_rows);
  };
  const auto&anchor=bigger(layouts[0],layouts[1])?layouts[0]:layouts[1];
  if(anchor.facts().query_rows!=2)
    return std::unexpected(Status::Unimplemented("device batch choice physical plan requires an active Q2 alternative"));
  s->preparing_choices=true;
  util::ScopeGuard preparing{[&]()noexcept{s->preparing_choices=false;}};
  HYLE_ASSIGN_OR_RETURN(auto token,target.prepare(anchor,update,nullptr,&before_change));
  struct Qualify{BatchInvocationData*s;const std::array<BatchLayoutVersion,2>*layouts;}qualify{s,&layouts};
  HYLE_RETURN_IF_ERROR(ProviderExtensionAccess::call(s->choice_extension,[](void*,const void*,void*opaque)noexcept{
    auto&call=*static_cast<Qualify*>(opaque);
    return call.s->choice_table->qualify(call.s->native,planning_queue(*call.s),(*call.layouts)[0],(*call.layouts)[1],
        BatchDeviceChoiceStage::kPrepared);
  },&qualify));
  const auto bytes=s->program->schema.metadata_offsets().bytes;
  if(bytes>SIZE_MAX/3 || !bytes)return std::unexpected(Status::OutOfRange("device batch choice metadata extent"));
  if(!s->choice_selected.valid()){
    if(before_change.call)HYLE_RETURN_IF_ERROR(before_change.call(before_change.state));
    const LeasePoolOptions options{.max_blocks=2,.backing_alignment=16,.allocator=s->program->allocator,
        .resource_budget=s->program->options.resource_budget};
    HYLE_ASSIGN_OR_RETURN(s->choice_device_pool,LeasePool::create(s->program->options.resource,3*bytes+32,options));
    HYLE_ASSIGN_OR_RETURN(s->choice_host_pool,LeasePool::create(s->program->options.host_workspace,2*bytes,
        {.max_blocks=1,.backing_alignment=16,.allocator=s->program->allocator,.resource_budget=s->program->options.resource_budget}));
    HYLE_ASSIGN_OR_RETURN(s->choice_pair,s->choice_device_pool.allocate(2*bytes,16));
    HYLE_ASSIGN_OR_RETURN(s->choice_selected,s->choice_device_pool.allocate(bytes,16));
    HYLE_ASSIGN_OR_RETURN(s->choice_staging,s->choice_host_pool.allocate(2*bytes,16));
    retain_domain(*s->buffers,s->choice_device_pool);retain_domain(*s->buffers,s->choice_host_pool);
  }
  if(s->choice_ready.valid()){HYLE_ASSIGN_OR_RETURN(s->choice_pair,s->choice_ready.try_write());}
  if(s->choice_staging_ready.valid()){HYLE_ASSIGN_OR_RETURN(s->choice_staging,s->choice_staging_ready.try_write());}
  HYLE_ASSIGN_OR_RETURN(auto staging,s->choice_staging.view());
  HYLE_ASSIGN_OR_RETURN(auto map,staging.map_host());HYLE_ASSIGN_OR_RETURN(auto writable,map.writable_bytes());
  for(std::size_t i=0;i<2;++i){
    HYLE_ASSIGN_OR_RETURN(auto ready,layouts[i].metadata());HYLE_ASSIGN_OR_RETURN(auto view,ready.view());
    HYLE_ASSIGN_OR_RETURN(auto source,view.map_host());
    if(source.bytes().size()!=bytes)return std::unexpected(Status::InvalidArgument("device choice metadata encoding extent changed"));
    std::memcpy(writable.data()+i*bytes,source.bytes().data(),bytes);
  }
  HYLE_RETURN_IF_ERROR(map.close());
  HYLE_ASSIGN_OR_RETURN(s->choice_staging_ready,s->choice_staging.publish());
  HYLE_ASSIGN_OR_RETURN(auto source,s->choice_staging_ready.view());HYLE_ASSIGN_OR_RETURN(auto destination,s->choice_pair.view());
  HYLE_ASSIGN_OR_RETURN(auto use,s->uses.begin());HYLE_RETURN_IF_ERROR(use.read(s->choice_staging_ready));
  HYLE_RETURN_IF_ERROR(use.write(s->choice_pair));
  HYLE_ASSIGN_OR_RETURN(auto sequence,QueueSequence::begin(planning_queue(*s)));
  sequence.record(TransferAccess::enqueue_copy(sequence,destination,source));
  HYLE_ASSIGN_OR_RETURN(auto copied,sequence.finish());HYLE_FATAL_ASSERT(use.attach(copied).ok());
  const auto terminal=copied.wait();copied={};
  if(!terminal.ok() || s->uses.active_batches()){
    s->failed=true;return std::unexpected(terminal.ok()?Status::Unavailable("device choice metadata upload retirement unproven"):terminal);
  }
  HYLE_ASSIGN_OR_RETURN(s->choice_ready,s->choice_pair.publish());
  s->choice_layouts=layouts;s->device_choices=true;return token;
}
Status d::BatchGraphAccess::validate_choices(const BatchInvocation&target,
    const std::array<BatchLayoutVersion,2>&layouts)noexcept{
  auto*s=target.data_;
  if(!s || !s->external_output || !s->prepared || s->closed || s->failed || target.in_flight())
    return Status::Unavailable("device choice preflight needs its retired original prepared native slot");
  for(const auto&layout:layouts){
    if(!layout.valid() || !s->program->schema.owns(layout))return Status::InvalidArgument("device choice preflight layout is foreign");
    HYLE_RETURN_IF_ERROR(validate_batch_attention_layout(layout,s->program->options.causal,s->program->options.window_left));
  }
  HYLE_ASSIGN_OR_RETURN(auto extension,ProviderExtensionAccess::query(s->program->extension,kBatchDeviceChoiceExtensionId));
  struct Call{BatchInvocationData*s;const std::array<BatchLayoutVersion,2>*layouts;}call{s,&layouts};
  return ProviderExtensionAccess::call(extension,[](void*,const void*table,void*opaque)noexcept{
    const auto*choice=static_cast<const BatchDeviceChoiceVTable*>(table);auto&c=*static_cast<Call*>(opaque);
    if(!choice || !choice->qualify)return Status::Unimplemented("device choice preflight capability is absent");
    return choice->qualify(c.s->native,planning_queue(*c.s),(*c.layouts)[0],(*c.layouts)[1],
        BatchDeviceChoiceStage::kPreflight);
  },&call);
}
Result<PreparedBatch> d::BatchGraphAccess::prepare_shared(BatchInvocation&target,
    const BatchLayoutVersion&layout,bool update,const BatchInvocation&source,
    BeforeCommandChange before_change)noexcept{
  return target.prepare(layout,update,&source,&before_change);
}
void d::BatchGraphAccess::clear_shared_metadata(BatchInvocation&invocation)noexcept{
  if(invocation.data_)invocation.data_->graph_shared_metadata.reset();
}
Result<std::uint64_t> d::BatchGraphAccess::preparation_epoch(const PreparedBatch&token)noexcept{
  const auto*s=token.data_;
  if(!s || token.epoch_!=s->epoch || !s->external_output || s->closed || s->failed ||
      !s->prepared || s->ran || !s->program->schema.owns(s->layout) || !s->queue_owner.is_open())
    return std::unexpected(Status::FailedPrecondition("Graph batch preparation epoch is stale"));
  if(s->pending.load(std::memory_order_acquire) || s->uses.active_batches() || !s->completion.available())
    return std::unexpected(Status::Unavailable("Graph batch preparation still has native work/completion"));
  return token.epoch_;
}
Status d::BatchGraphAccess::validate_prepared(const PreparedBatch&token,BatchGraphContract contract)noexcept{
  auto*s=token.data_;
  if(contract!=BatchGraphContract::kPreparedMetadataCapacity && contract!=BatchGraphContract::kOwnedLayoutActive &&
      contract!=BatchGraphContract::kDeviceLayoutChoice)
    return Status::InvalidArgument("Graph batch port contract is invalid");
  const bool choice=contract==BatchGraphContract::kDeviceLayoutChoice;
  const bool owned=contract==BatchGraphContract::kOwnedLayoutActive || choice;
  if(!s || token.epoch_!=s->epoch || !s->external_output || s->closed || s->failed ||
      !s->prepared || (!owned && s->ran) || !s->program->schema.owns(s->layout) ||
      !s->queue_owner.is_open())
    return Status::FailedPrecondition("Graph batch seal needs its current prepared slot/Queue");
  if(s->pending.load(std::memory_order_acquire) || s->uses.active_batches() || !s->completion.available())
    return Status::Unavailable("Graph batch seal has outstanding native work/completion");
  HYLE_RETURN_IF_ERROR(qualify_graph_recording(*s));
  HYLE_RETURN_IF_ERROR(ProviderExtensionAccess::call(s->program->extension,
      [](void*,const void*,void*)noexcept{return Status::Ok();},nullptr));
  HYLE_ASSIGN_OR_RETURN(auto metadata,(s->graph_shared_metadata.valid()?s->graph_shared_metadata:s->mirror_ready).view());
  if(choice){
    if(!s->device_choices || !s->choice_ready.valid() || !s->choice_table)
      return Status::FailedPrecondition("device batch choice seal has no finite prepared alternatives");
    HYLE_ASSIGN_OR_RETURN(auto selected,s->choice_selected.view());metadata=selected;
  }
  if(metadata.resource()!=s->program->options.resource ||
      metadata.size()!=s->program->schema.metadata_offsets().bytes)
    return Status::InvalidArgument("Graph batch metadata carrier differs from its prepared owner");
  s->graph_metadata_binding=metadata;s->graph_metadata_epoch=s->epoch;
  s->report.capture_compatible=true;
  return Status::Ok();
}
Status d::BatchGraphAccess::prepare_replay(PreparedBatch&token,Queue&queue,
    std::span<const ConstTensorView>inputs,std::span<const TensorView>outputs,BatchGraphContract contract)noexcept{
  if(!token.data_ || token.epoch_!=token.data_->epoch)return Status::FailedPrecondition("Graph batch replay token is stale");
  HYLE_ASSIGN_OR_RETURN(auto views,graph_views(token.data_,queue,inputs,outputs,contract));
  if(contract==BatchGraphContract::kDeviceLayoutChoice)
    return Status::Unimplemented("device layout choices require the original outer Graph capture");
  return capture_batch(*token.data_,inputs.first(input_count(token.data_->program->schema)),views);
}
Status d::BatchGraphAccess::validate_replay(PreparedBatch&token,Queue&queue,
    std::span<const ConstTensorView>inputs,std::span<const TensorView>outputs,BatchGraphContract contract)noexcept{
  if(!token.data_ || token.epoch_!=token.data_->epoch)return Status::FailedPrecondition("Graph batch replay token is stale");
  HYLE_ASSIGN_OR_RETURN(auto views,graph_views(token.data_,queue,inputs,outputs,contract));
  if(token.data_->program->runtime.capture_compatible){
    HYLE_RETURN_IF_ERROR(validate_prepared(token,contract));
    return validate_replay_storage(*token.data_,inputs.first(input_count(token.data_->program->schema)),views);
  }
  if(!token.data_->replay || !token.data_->replay->commands.valid())
    return Status::FailedPrecondition("Graph batch replay needs explicit preparation");
  return validate_replay_storage(*token.data_,inputs.first(input_count(token.data_->program->schema)),views);
}
d::EnqueueResult d::BatchGraphAccess::enqueue(PreparedBatch&token,QueueSequence&sequence,std::span<const ConstTensorView>inputs,std::span<const TensorView>outputs,BatchGraphContract contract)noexcept{
  auto*s=token.data_;
  if(!s || token.epoch_!=s->epoch)return EnqueueResult::rejected(Status::FailedPrecondition("Graph batch token/slot/queue is stale"));
  auto views=graph_views(s,sequence,inputs,outputs,contract);if(!views)return EnqueueResult::rejected(views.error());
  const auto payloads=inputs.first(input_count(s->program->schema));
  const auto valid=validate_replay_storage(*s,payloads,*views);
  if(!valid.ok())return EnqueueResult::rejected(valid);
  const auto choose=[&]()noexcept->EnqueueResult{
    if(contract!=BatchGraphContract::kDeviceLayoutChoice)return EnqueueResult::no_op();
    auto paired=s->choice_ready.view();if(!paired)return EnqueueResult::rejected(paired.error());
    const auto bytes=s->program->schema.metadata_offsets().bytes;
    auto zero=paired->subview(0,bytes),nonzero=paired->subview(bytes,bytes);
    auto selected=s->choice_selected.view();
    if(!zero || !nonzero || !selected)return EnqueueResult::rejected(Status::FailedPrecondition("device batch choice carriers changed"));
    return s->choice_table->select(s->native,sequence,inputs[input_count(s->program->schema)],*zero,*nonzero,*selected,*views);
  };
  if(sequence.recording()){
    const auto qualified=qualify_graph_recording(*s);
    if(!qualified.ok())return EnqueueResult::rejected(qualified);
    if(s->graph_metadata_epoch!=s->epoch || !same_storage(s->graph_metadata_binding,views->metadata))
      return EnqueueResult::rejected(Status::FailedPrecondition("Graph batch recording needs a current cold metadata seal"));
    const auto chosen=choose();if(!chosen.status.ok())return chosen;
    // Capture records arguments, not an execution episode. Preserve the token
    // and run directly in the admitted outer sequence, never a child artifact.
    auto recorded=s->program->table->run(s->native,sequence,payloads,*views);
    recorded.accepted|=chosen.accepted;return recorded;
  }
  const auto chosen=choose();if(!chosen.status.ok())return chosen;
  auto enqueued=s->replay && s->replay->commands.valid()?s->replay->commands.enqueue(sequence):
      s->program->table->run(s->native,sequence,payloads,*views);
  enqueued.accepted|=chosen.accepted;
  if(!enqueued.status.ok()){s->failed=true;s->prepared=false;s->ran=true;}
  else if(contract==BatchGraphContract::kPreparedMetadataCapacity){s->prepared=false;s->ran=true;}
  // Owned Graph keeps its exact cold token/bindings. The original Graph
  // admission, result and retirement gates still precede every replay.
  return enqueued;
}
Result<bool> d::BatchGraphAccess::empty_cache_reads(
    const PreparedBatch&token,BatchGraphContract contract)noexcept{
  const auto*s=token.data_;
  if(contract!=BatchGraphContract::kOwnedLayoutActive || !s || token.epoch_!=s->epoch ||
      !s->external_output || s->closed || s->failed || !s->prepared ||
      !s->program->schema.owns(s->layout) || s->layout.facts().query_rows ||
      s->layout.facts().page_refs || s->program->schema.capacity().storage!=BatchStorage::kPaged ||
      !s->program->table->empty_cache_reads)
    return false;
  struct Call{const d::BatchInvocationData*state;bool empty=false;}call{s};
  HYLE_RETURN_IF_ERROR(ProviderExtensionAccess::call(s->program->extension,
      [](void*,const void*table,void*opaque)noexcept->Status{
        auto&call=*static_cast<Call*>(opaque);
        call.empty=static_cast<const BatchProviderVTable*>(table)->empty_cache_reads(call.state->native);
        return Status::Ok();
      },&call));
  return call.empty;
}
} // namespace hyle::compute
