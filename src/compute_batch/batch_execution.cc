#include "hyle/compute/batch_execution.h"
#include "hyle/compute/detail/batch_provider.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/compute/detail/retained_completion.h"
#include "hyle/core/assert.h"
#include "hyle/util/ref_counted.h"
#include <array>
#include <atomic>
#include <cmath>
#include <new>

namespace hyle::compute {
namespace d=detail;
namespace detail {
struct BatchProgramData final:util::RefCounted<BatchProgramData>{
  BatchProgramData(BatchSchema s,BatchProgramOptions o,mem::AllocatorRef a):schema(std::move(s)),options(std::move(o)),allocator(a){}
  void destroy()noexcept{const auto a=allocator;this->~BatchProgramData();a.deallocate(this,sizeof(*this),alignof(BatchProgramData));}
  BatchSchema schema;BatchProgramOptions options;mem::AllocatorRef allocator;
  ProviderExtensionRef extension;const BatchProviderVTable*table=nullptr;
  CopyBatchProgram upload;std::atomic<std::uint32_t> invocations{0};
};
struct BatchBuffers final:util::RefCounted<BatchBuffers>{
  explicit BatchBuffers(BatchProgramData*p):program(p),allocator(p->allocator){p->incref();}
  void destroy()noexcept{auto*p=program;const auto a=allocator;this->~BatchBuffers();p->invocations.fetch_sub(1,std::memory_order_relaxed);p->decref();a.deallocate(this,sizeof(*this),alignof(BatchBuffers));}
  BatchProgramData*program;mem::AllocatorRef allocator;ProviderExtensionRef provider_owner;LeasePool device,host;
};
struct BatchInvocationData final:util::RefCounted<BatchInvocationData>{
  explicit BatchInvocationData(BatchProgramData*p):program(p){p->incref();}
  void destroy()noexcept{
    auto*p=program;const auto a=p->allocator;
    if(native)p->table->destroy(native,a);
    if(buffers)buffers->decref();else p->invocations.fetch_sub(1,std::memory_order_relaxed);
    this->~BatchInvocationData();p->decref();
    a.deallocate(this,sizeof(*this),alignof(BatchInvocationData));
  }
  void release_user()noexcept{
    HYLE_FATAL_ASSERT(users);if(--users==0){closed=true;completion.reset();upload.close().IgnoreError();}
    decref();
  }
  BatchProgramData*program;BatchBuffers*buffers=nullptr;Queue*queue=nullptr;void*native=nullptr;
  WriteLease mirror,output,float_workspace,int_workspace,host_workspace,staging;
  ReadLease mirror_ready,output_ready,staging_ready;
  CopyBatchInvocation upload;LeaseUsePool uses;RetainedCompletion completion;
  BatchLayoutVersion layout;BatchPlanReport report;
  CompletionRetention native_retention{};std::atomic<std::uint32_t> pending{0};
  Status terminal;std::uint64_t epoch=0;std::uint32_t users=1,tickets=0;
  bool closed=false,failed=false,prepared=false,ran=false;
};
} // namespace detail
namespace {
Result<std::size_t> output_bytes(BatchCapacity c)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto desc,TensorDesc::dense(c.data_type,std::array<std::int64_t,3>{c.max_query_rows,c.query_heads,c.head_dim}));return desc.required_bytes();
}
Result<std::array<TensorDesc,3>> input_descriptions(BatchCapacity c)noexcept{
  std::array<TensorDesc,3> result;
  HYLE_ASSIGN_OR_RETURN(result[0],TensorDesc::dense(c.data_type,std::array<std::int64_t,3>{c.max_query_rows,c.query_heads,c.head_dim}));
  if(c.storage==BatchStorage::kPaged){HYLE_ASSIGN_OR_RETURN(result[1],TensorDesc::dense(c.data_type,std::array<std::int64_t,4>{c.max_pages,c.page_size,c.kv_heads,c.head_dim}));}
  else{HYLE_ASSIGN_OR_RETURN(result[1],TensorDesc::dense(c.data_type,std::array<std::int64_t,3>{c.max_kv_rows,c.kv_heads,c.head_dim}));}
  result[2]=result[1];return result;
}
void native_terminal(void*opaque,const Status&status)noexcept{
  auto&s=*static_cast<d::BatchInvocationData*>(opaque);s.terminal=status;
}
void native_release(void*opaque)noexcept{
  auto*s=static_cast<d::BatchInvocationData*>(opaque);s->pending.fetch_sub(1,std::memory_order_release);s->decref();
}
Result<d::BatchWorkspaceViews> workspace_views(d::BatchInvocationData&s)noexcept{
  d::BatchWorkspaceViews views;
  HYLE_ASSIGN_OR_RETURN(views.metadata,s.mirror_ready.view());
  HYLE_ASSIGN_OR_RETURN(views.float_workspace,s.float_workspace.view());
  HYLE_ASSIGN_OR_RETURN(views.int_workspace,s.int_workspace.view());
  HYLE_ASSIGN_OR_RETURN(views.host_workspace,s.host_workspace.view());
  HYLE_ASSIGN_OR_RETURN(views.output,s.output.view());return views;
}
Result<Completion> submit_native(d::BatchInvocationData&s,bool planning,
    std::span<const ReadLease> leases,std::span<const ConstTensorView> inputs,d::BatchWorkspaceViews views)noexcept{
  if(!s.completion.available() || s.pending.load(std::memory_order_acquire) || s.uses.active_batches())
    return std::unexpected(Status::Unavailable("batch native slot or old completion retained"));
  HYLE_ASSIGN_OR_RETURN(auto use,s.uses.begin());
  for(const auto&lease:leases)HYLE_RETURN_IF_ERROR(use.read(lease));
  HYLE_RETURN_IF_ERROR(use.read(s.mirror_ready));
  HYLE_RETURN_IF_ERROR(use.write(s.float_workspace));HYLE_RETURN_IF_ERROR(use.write(s.int_workspace));
  if(planning)HYLE_RETURN_IF_ERROR(use.write(s.host_workspace));
  else HYLE_RETURN_IF_ERROR(use.write(s.output));
  struct Call{d::BatchInvocationData*state;bool planning;std::span<const ConstTensorView> inputs;d::BatchWorkspaceViews views;}call{&s,planning,inputs,views};
  auto completion=d::ProviderExtensionAccess::submit(s.program->extension,*s.queue,inputs,{},
    [](void*,const void*interface_value,void*opaque,Queue&queue)noexcept{
      const auto&call=*static_cast<Call*>(opaque);auto&state=*call.state;const auto*table=static_cast<const d::BatchProviderVTable*>(interface_value);
      return call.planning?table->plan(state.native,queue,state.layout,call.views,&state.report):table->run(state.native,queue,call.inputs,call.views);
    },&call);
  if(!completion)return std::unexpected(completion.error());
  HYLE_FATAL_ASSERT(use.attach(*completion).ok());
  s.incref();s.pending.fetch_add(1,std::memory_order_relaxed);
  s.native_retention.own={&s,&native_terminal,&native_release};
  HYLE_FATAL_ASSERT(d::CompletionAccess::attach_retention(*completion,s.native_retention).ok());
  return s.completion.publish(std::move(*completion));
}
} // namespace
BatchProgram::BatchProgram(const BatchProgram&p)noexcept:data_(p.data_){if(data_)data_->incref();}
BatchProgram&BatchProgram::operator=(const BatchProgram&p)noexcept{if(this!=&p){BatchProgram next(p);std::swap(data_,next.data_);}return *this;}
BatchProgram::BatchProgram(BatchProgram&&p)noexcept:data_(std::exchange(p.data_,nullptr)){}
BatchProgram&BatchProgram::operator=(BatchProgram&&p)noexcept{if(this!=&p){BatchProgram next(std::move(p));std::swap(data_,next.data_);}return *this;}
BatchProgram::~BatchProgram(){if(data_)data_->decref();}
BatchSchema BatchProgram::schema()const noexcept{return data_?data_->schema:BatchSchema{};}
Result<BatchProgram> BatchProgram::prepare(const Context&context,BatchSchema schema,BatchProgramOptions options)noexcept{
  if(!context.valid() || !schema.valid() || !options.resource.valid() || !options.host_workspace.valid() || !options.allocator.valid() ||
     !options.max_invocations || options.max_invocations==UINT32_MAX || !options.max_float_workspace_bytes || !options.max_int_workspace_bytes || !std::isfinite(options.scale))
    return std::unexpected(Status::InvalidArgument("batch program options are invalid"));
  if(options.resource_budget.valid() && !options.resource_budget.allows_unattributed())
    return std::unexpected(Status::Unimplemented("native batch planning private bytes are unknown"));
  auto a=options.allocator;if(options.resource_budget.valid()){HYLE_ASSIGN_OR_RETURN(a,options.resource_budget.metadata_allocator());}
  auto*p=a.construct<d::BatchProgramData>(std::move(schema),std::move(options),a);if(!p)return std::unexpected(Status::ResourceExhausted("batch program metadata"));
  BatchProgram result(p);
  for(std::size_t index=0;index<context.provider_count();++index){
    auto extension=d::ProviderExtensionAccess::query(context,index,d::kBatchPlanningExtensionId);
    if(!extension){if(extension.error().IsUnimplemented())continue;return std::unexpected(extension.error());}
    p->extension=std::move(*extension);break;
  }
  if(!p->extension.valid())return std::unexpected(Status::Unimplemented("selected route has no native BatchPlanning provider"));
  HYLE_RETURN_IF_ERROR(d::ProviderExtensionAccess::call(p->extension,[](void*context,const void*table,void*opaque)noexcept{
    auto&p=*static_cast<d::BatchProgramData*>(opaque);p.table=static_cast<const d::BatchProviderVTable*>(table);
    if(!p.table->qualify || !p.table->create || !p.table->destroy || !p.table->update || !p.table->plan || !p.table->run)return Status::FailedPrecondition("batch provider interface is incomplete");return p.table->qualify(context,p.schema.capacity(),p.options);
  },p));
  const auto bytes=p->schema.metadata_offsets().bytes;
  const std::array specs{CopyResourceSpec{p->options.host_workspace,bytes},CopyResourceSpec{p->options.resource,bytes}};
  HYLE_ASSIGN_OR_RETURN(p->upload,CopyBatchProgram::prepare(context,specs,{.max_resources=2,.max_copies=1,.max_phases=1,.max_bytes=bytes,.allocator=a,.resource_budget=p->options.resource_budget}));
  return result;
}
BatchInvocation::BatchInvocation(BatchInvocation&&v)noexcept:data_(std::exchange(v.data_,nullptr)){}
BatchInvocation&BatchInvocation::operator=(BatchInvocation&&v)noexcept{if(this!=&v){BatchInvocation old(std::move(*this));data_=std::exchange(v.data_,nullptr);}return *this;}
BatchInvocation::~BatchInvocation(){if(data_)data_->release_user();}
Result<BatchInvocation> BatchInvocation::create(const BatchProgram&program,Queue&queue)noexcept{
  auto*p=program.data_;if(!p || !queue.valid())return std::unexpected(Status::InvalidArgument("batch invocation program/queue is invalid"));
  auto count=p->invocations.load(std::memory_order_relaxed);
  do{if(count==p->options.max_invocations)return std::unexpected(Status::Unavailable("batch invocation capacity retained"));}
  while(!p->invocations.compare_exchange_weak(count,count+1,std::memory_order_relaxed));
  auto*s=p->allocator.construct<d::BatchInvocationData>(p);if(!s){p->invocations.fetch_sub(1,std::memory_order_relaxed);return std::unexpected(Status::ResourceExhausted("batch invocation metadata"));}
  BatchInvocation result(s);s->queue=&queue;
  s->buffers=p->allocator.construct<d::BatchBuffers>(p);if(!s->buffers)return std::unexpected(Status::ResourceExhausted("batch buffer owner"));
  s->buffers->provider_owner=p->extension;
  HYLE_ASSIGN_OR_RETURN(auto output,output_bytes(p->schema.capacity()));const auto mirror=p->schema.metadata_offsets().bytes;
  std::size_t capacity=4u*256u;
  for(const auto bytes:{output,mirror,p->options.max_float_workspace_bytes,p->options.max_int_workspace_bytes}){if(bytes>SIZE_MAX-capacity)return std::unexpected(Status::OutOfRange("batch pool extent overflow"));capacity+=bytes;}
  HYLE_ASSIGN_OR_RETURN(s->buffers->device,LeasePool::create(p->options.resource,capacity,{.max_blocks=4,.allocator=p->allocator,.resource_budget=p->options.resource_budget}));
  if(mirror>SIZE_MAX-p->options.max_int_workspace_bytes-512u)return std::unexpected(Status::OutOfRange("batch host workspace extent overflow"));
  HYLE_ASSIGN_OR_RETURN(s->buffers->host,LeasePool::create(p->options.host_workspace,p->options.max_int_workspace_bytes+mirror+512u,{.max_blocks=2,.allocator=p->allocator,.resource_budget=p->options.resource_budget}));
  HYLE_ASSIGN_OR_RETURN(s->mirror,s->buffers->device.allocate(mirror));HYLE_ASSIGN_OR_RETURN(s->output,s->buffers->device.allocate(output));
  HYLE_ASSIGN_OR_RETURN(s->float_workspace,s->buffers->device.allocate(p->options.max_float_workspace_bytes));
  HYLE_ASSIGN_OR_RETURN(s->int_workspace,s->buffers->device.allocate(p->options.max_int_workspace_bytes));HYLE_ASSIGN_OR_RETURN(s->host_workspace,s->buffers->host.allocate(p->options.max_int_workspace_bytes));
  HYLE_ASSIGN_OR_RETURN(s->staging,s->buffers->host.allocate(mirror));
  const auto retain_domain=[&](LeasePool&pool){s->buffers->incref();d::LeaseAccess::retain_resource_owner(pool,s->buffers,[](void*p)noexcept{static_cast<d::BatchBuffers*>(p)->decref();});d::LeaseAccess::detach_pool_root(pool);};
  retain_domain(s->buffers->device);retain_domain(s->buffers->host);
  HYLE_ASSIGN_OR_RETURN(s->upload,CopyBatchInvocation::create(p->upload,queue));
  HYLE_ASSIGN_OR_RETURN(s->uses,LeaseUsePool::create({.max_batches=1,.max_resources_per_batch=8,.allocator=p->allocator,.resource_budget=p->options.resource_budget}));
  HYLE_ASSIGN_OR_RETURN(s->completion,d::RetainedCompletion::create(p->allocator));
  HYLE_RETURN_IF_ERROR(d::ProviderExtensionAccess::call(p->extension,[](void*context,const void*table,void*opaque)noexcept{
    auto&s=*static_cast<d::BatchInvocationData*>(opaque);return static_cast<const d::BatchProviderVTable*>(table)->create(context,s.program->schema.capacity(),s.program->options,s.program->allocator,&s.native);
  },s));return result;
}
Result<PreparedBatch> BatchInvocation::prepare_batch(const BatchLayoutVersion&layout)noexcept{
  return prepare(layout,false);
}
Result<PreparedBatch> BatchInvocation::update_batch(const BatchLayoutVersion&layout)noexcept{
  return prepare(layout,true);
}
Result<PreparedBatch> BatchInvocation::prepare(const BatchLayoutVersion&layout,bool update)noexcept{
  if(!data_ || data_->closed || data_->failed)return std::unexpected(Status::FailedPrecondition("batch invocation is closed/failed"));
  auto&s=*data_;
  if(!s.program->schema.owns(layout))return std::unexpected(Status::InvalidArgument("batch layout belongs to another schema"));
  if(s.tickets || in_flight() || !s.completion.available())return std::unexpected(Status::Unavailable("batch slot has a live ticket/native work/old completion"));
  if(s.output_ready.valid() && !d::LeaseAccess::unique_read_owner(s.output_ready))return std::unexpected(Status::Unavailable("old batch output prevents slot overwrite"));
  if(s.epoch==UINT64_MAX)return std::unexpected(Status::OutOfRange("batch slot epoch exhausted"));
  if(s.output_ready.valid()){HYLE_ASSIGN_OR_RETURN(s.output,s.output_ready.try_write());}
  if(s.mirror_ready.valid()){HYLE_ASSIGN_OR_RETURN(s.mirror,s.mirror_ready.try_write());}
  ++s.epoch;s.prepared=false;s.ran=false;s.layout=layout;s.report={};s.terminal=Status::Ok();
  HYLE_ASSIGN_OR_RETURN(auto source,layout.metadata());
  HYLE_RETURN_IF_ERROR(s.upload.reset());
  if(s.staging_ready.valid()){HYLE_ASSIGN_OR_RETURN(s.staging,s.staging_ready.try_write());}
  HYLE_ASSIGN_OR_RETURN(auto source_view,source.view());HYLE_ASSIGN_OR_RETURN(auto source_mapping,source_view.map_host());
  HYLE_ASSIGN_OR_RETURN(auto staging_view,s.staging.view());HYLE_RETURN_IF_ERROR(staging_view.write(source_mapping.bytes()));
  HYLE_ASSIGN_OR_RETURN(s.staging_ready,s.staging.publish());
  HYLE_RETURN_IF_ERROR(s.upload.bind_source({0},s.staging_ready));HYLE_RETURN_IF_ERROR(s.upload.bind_mutable({1},std::move(s.mirror)));
  const std::array<CopyPhase,1> phases{{{}}};const std::array<CopyRange,1> ranges{{{{0},{1},{0},0,0,source.size()}}};
  HYLE_RETURN_IF_ERROR(s.upload.prepare_replay(phases,ranges));HYLE_ASSIGN_OR_RETURN(auto uploaded,s.upload.submit());
  auto terminal=uploaded.wait();if(!terminal.ok()){s.failed=true;return std::unexpected(terminal);}
  HYLE_ASSIGN_OR_RETURN(s.mirror_ready,uploaded.output({0},{1},0,source.size()));uploaded={};
  const auto upload_report=s.upload.report();HYLE_RETURN_IF_ERROR(s.upload.reset());
  HYLE_ASSIGN_OR_RETURN(auto views,workspace_views(s));
  bool reused=false;
  if(update){struct Call{d::BatchInvocationData*state;bool reused=false;}call{&s};
    const auto status=d::ProviderExtensionAccess::call(s.program->extension,[](void*,const void*table,void*opaque)noexcept->Status{
      auto&call=*static_cast<Call*>(opaque);HYLE_ASSIGN_OR_RETURN(call.reused,static_cast<const d::BatchProviderVTable*>(table)->update(call.state->native,call.state->layout,&call.state->report));return Status::Ok();
    },&call);if(!status.ok()){s.failed=true;return std::unexpected(status);}reused=call.reused;
  }
  if(!reused){
    auto planned=submit_native(s,true,{}, {},views);
    if(!planned){s.failed=true;return std::unexpected(planned.error());}
    terminal=planned->wait();*planned={};
    if(!terminal.ok() || s.pending.load(std::memory_order_acquire) || s.uses.active_batches()){s.failed=true;return std::unexpected(!terminal.ok()?terminal:Status::Unavailable("batch planning retirement unproven"));}
  }
  s.report.metadata_upload=upload_report;s.report.metadata_staging_bytes=source.size();s.report.host_waits=reused?1u:2u;s.report.reused_native_plan=reused;s.prepared=true;
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
Status BatchInvocation::drain()noexcept{if(!data_)return Status::Ok();close().IgnoreError();const auto copy=data_->upload.drain();const auto native=data_->completion.wait();return !copy.ok()?copy:native;}
PreparedBatch::PreparedBatch(PreparedBatch&&b)noexcept:data_(std::exchange(b.data_,nullptr)),epoch_(b.epoch_){}
PreparedBatch&PreparedBatch::operator=(PreparedBatch&&b)noexcept{if(this!=&b){PreparedBatch old(std::move(*this));data_=std::exchange(b.data_,nullptr);epoch_=b.epoch_;}return *this;}
PreparedBatch::~PreparedBatch(){if(data_){--data_->tickets;data_->release_user();}}
std::uint64_t PreparedBatch::generation()const noexcept{return data_?data_->layout.generation():0;}
Result<Completion> PreparedBatch::run(std::span<const ReadLease>leases)noexcept{
  if(!data_ || data_->closed || data_->failed || !data_->prepared || data_->ran || epoch_!=data_->epoch)
    return std::unexpected(Status::FailedPrecondition("batch token is closed/stale/already consumed"));
  if(leases.size()!=3)return std::unexpected(Status::InvalidArgument("batch run needs Q/K/V readonly leases"));
  auto&s=*data_;HYLE_ASSIGN_OR_RETURN(auto descriptions,input_descriptions(s.program->schema.capacity()));
  std::array<ConstTensorView,3> inputs;
  for(std::size_t i=0;i<3;++i){HYLE_ASSIGN_OR_RETURN(inputs[i],leases[i].tensor(descriptions[i]));}
  HYLE_ASSIGN_OR_RETURN(auto views,workspace_views(s));
  auto submitted=submit_native(s,false,leases,inputs,views);
  if(!submitted){s.failed=true;s.prepared=false;return std::unexpected(submitted.error());}
  s.ran=true;s.prepared=false;return std::move(*submitted);
}
} // namespace hyle::compute
