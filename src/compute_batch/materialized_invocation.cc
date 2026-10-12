#include "hyle/compute/batch_materialize.h"
#include "hyle/compute/detail/paged_arena_access.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/compute/detail/retained_completion.h"
#include "hyle/core/assert.h"

namespace hyle::compute {
namespace d=detail;
namespace detail {
struct MaterializedDirectBacking final {
  MaterializedDirectBacking(MaterializedBatchProgram p,mem::AllocatorRef a,ResourceBudget b):program(std::move(p)),allocator(a),budget(std::move(b)){}
  MaterializedBatchProgram program;mem::AllocatorRef allocator;ResourceBudget budget;
};
struct MaterializedBatchInvocationData final {
  explicit MaterializedBatchInvocationData(mem::AllocatorRef a):allocator(a){}
  mem::AllocatorRef allocator;Queue*queue=nullptr;MaterializedBatchProgram program;
  std::array<WriteLease,24> buffers;std::array<WriteLease,19> workspaces;
  std::array<OperationInvocation,19> operations;
  std::array<ConstTensorView,24> views;
  std::array<ReadLease,5> inputs;PagedArenaRead keys,values;
  MaterializedBatchVersion version;ReadLease output_ready;
  LeaseUsePool uses;RetainedCompletion completion;MaterializedBatchRunReport report;
  Status terminal;bool prepared=false,bound=false,closed=false,failed=false;
};
void destroy_direct(MaterializedBatchInvocationData*p)noexcept{if(p){const auto a=p->allocator;a.destroy(p);}}
} // namespace detail
namespace {
Status ready(d::MaterializedBatchInvocationData*p)noexcept{
  if(!p || p->closed || p->failed)return Status::FailedPrecondition("host Direct batch is empty/closed/failed");
  if(p->uses.active_batches() || !p->completion.available())return Status::Unavailable("host Direct native use or old completion retained");
  return Status::Ok();
}
Status add_bytes(std::size_t&total,std::size_t bytes,std::size_t alignment)noexcept{
  if(!alignment || bytes>SIZE_MAX-(alignment-1) || bytes+alignment-1>SIZE_MAX-total)
    return Status::OutOfRange("host Direct buffer extent overflow");
  total+=bytes+alignment-1;return Status::Ok();
}
}
MaterializedBatchInvocation::MaterializedBatchInvocation(MaterializedBatchInvocation&&v)noexcept:data_(std::exchange(v.data_,nullptr)){}
MaterializedBatchInvocation&MaterializedBatchInvocation::operator=(MaterializedBatchInvocation&&v)noexcept{if(this!=&v){d::destroy_direct(data_);data_=std::exchange(v.data_,nullptr);}return *this;}
MaterializedBatchInvocation::~MaterializedBatchInvocation(){d::destroy_direct(data_);}
Result<MaterializedBatchInvocation> MaterializedBatchInvocation::create(const Context&context,
    MaterializedBatchProgram program,Queue&queue,MaterializedBatchInvocationOptions options)noexcept{
  if(!program.valid() || !context.valid() || !queue.valid() || !options.allocator.valid())
    return std::unexpected(Status::InvalidArgument("host Direct batch needs recipe/context/queue/allocator"));
  const auto cap=program.schema().capacity();if(cap.max_page_refs>(UINT32_MAX-64)/2)
    return std::unexpected(Status::ResourceExhausted("host Direct page permission capacity"));
  auto allocator=options.allocator;if(options.resource_budget.valid()){
    HYLE_ASSIGN_OR_RETURN(allocator,options.resource_budget.metadata_allocator());
  }
  auto*p=allocator.construct<d::MaterializedBatchInvocationData>(allocator);
  if(!p)return std::unexpected(Status::ResourceExhausted("host Direct batch metadata"));
  MaterializedBatchInvocation result(p);p->program=std::move(program);p->queue=&queue;
  const auto steps=p->program.steps();const auto descriptions=p->program.descriptions();
  std::array<PreparedOperation,19> prepared;std::size_t payload=0,workspace=0;std::uint32_t blocks=0;
  for(std::size_t i=0;i<steps.size();++i){const auto&step=steps[i];if(step.operation.kind()==OperationKind::kView)continue;
    HYLE_ASSIGN_OR_RETURN(auto bytes,descriptions[step.output].required_bytes());HYLE_RETURN_IF_ERROR(add_bytes(payload,bytes,64));++blocks;
    if(payload>options.max_payload_bytes)return std::unexpected(Status::ResourceExhausted("host Direct payload byte cap"));
    std::array<TensorDesc,2> inputs;for(std::size_t port=0;port<step.input_count;++port)inputs[port]=descriptions[step.inputs[port]];
    HYLE_ASSIGN_OR_RETURN(prepared[i],compute::prepare(context,step.operation,std::span(inputs).first(step.input_count),
        std::span(&descriptions[step.output],1),mem::system_resource()));
    if(!prepared[i].runtime_properties().supports_concurrent_invocations)
      return std::unexpected(Status::Unimplemented("host Direct reference requires independently usable native invocations"));
    const auto required=prepared[i].workspace_requirement();if(required.size){HYLE_RETURN_IF_ERROR(add_bytes(workspace,required.size,required.alignment));++blocks;}
    if(workspace>options.max_workspace_bytes)return std::unexpected(Status::ResourceExhausted("host Direct workspace byte cap"));
  }
  if(payload>SIZE_MAX-workspace)return std::unexpected(Status::OutOfRange("host Direct combined backing overflow"));
  HYLE_ASSIGN_OR_RETURN(auto pool,LeasePool::create(mem::system_resource(),payload+workspace,
      {.max_blocks=blocks,.allocator=allocator,.resource_budget=options.resource_budget}));
  for(std::size_t i=0;i<steps.size();++i){const auto&step=steps[i];if(!prepared[i].valid())continue;
    HYLE_ASSIGN_OR_RETURN(auto bytes,descriptions[step.output].required_bytes());HYLE_ASSIGN_OR_RETURN(p->buffers[step.output],pool.allocate(bytes,64));
    const auto required=prepared[i].workspace_requirement();mem::BufferView view;
    if(required.size){HYLE_ASSIGN_OR_RETURN(p->workspaces[i],pool.allocate(required.size,required.alignment));HYLE_ASSIGN_OR_RETURN(view,p->workspaces[i].view());}
    HYLE_ASSIGN_OR_RETURN(p->operations[i],prepared[i].create_invocation(view));
  }
  auto*owner=allocator.construct<d::MaterializedDirectBacking>(p->program,allocator,options.resource_budget);
  if(!owner)return std::unexpected(Status::ResourceExhausted("host Direct backing owner metadata"));
  d::LeaseAccess::retain_resource_owner(pool,owner,[](void*opaque)noexcept{auto*p=static_cast<d::MaterializedDirectBacking*>(opaque);const auto a=p->allocator;a.destroy(p);});d::LeaseAccess::detach_pool_root(pool);
  HYLE_ASSIGN_OR_RETURN(p->uses,LeaseUsePool::create({.max_batches=1,.max_resources_per_batch=64+2*cap.max_page_refs,.allocator=allocator,.resource_budget=options.resource_budget}));
  HYLE_ASSIGN_OR_RETURN(p->completion,d::RetainedCompletion::create(allocator));return result;
}
Status MaterializedBatchInvocation::prepare_batch(BatchLayoutVersion layout)noexcept{
  HYLE_RETURN_IF_ERROR(ready(data_));HYLE_ASSIGN_OR_RETURN(auto version,data_->program.version(std::move(layout)));
  if(data_->output_ready.valid()){
    HYLE_ASSIGN_OR_RETURN(data_->buffers[data_->program.output_value()],data_->output_ready.try_write());
  }
  data_->inputs={};data_->keys={};data_->values={};data_->views={};data_->bound=false;data_->prepared=false;
  HYLE_ASSIGN_OR_RETURN(data_->inputs[3],version.indices());HYLE_ASSIGN_OR_RETURN(data_->inputs[4],version.mask());
  data_->version=std::move(version);data_->report={.layout_generation=data_->version.layout().generation()};data_->prepared=true;return Status::Ok();
}
Status MaterializedBatchInvocation::bind(ReadLease query,PagedArenaRead keys,PagedArenaRead values)noexcept{
  HYLE_RETURN_IF_ERROR(ready(data_));if(!data_->prepared || data_->bound)return Status::FailedPrecondition("host Direct needs an unbound prepared episode");
  if(data_->program.schema().capacity().storage!=BatchStorage::kPaged || keys.layout()!=data_->version.layout() || values.layout()!=data_->version.layout())
    return Status::InvalidArgument("host Direct page frames differ from prepared layout");
  HYLE_ASSIGN_OR_RETURN(auto q,query.tensor(data_->program.descriptions()[0]));HYLE_ASSIGN_OR_RETURN(auto k,d::PagedArenaAccess::tensor(keys));HYLE_ASSIGN_OR_RETURN(auto v,d::PagedArenaAccess::tensor(values));
  for(const auto&view:std::array{q,k,v})if(view.storage().resource()!=mem::system_resource())return Status::InvalidArgument("host Direct requires system resource");
  data_->inputs[0]=std::move(query);data_->keys=std::move(keys);data_->values=std::move(values);data_->views[0]=q;data_->views[1]=k;data_->views[2]=v;data_->bound=true;return Status::Ok();
}
Status MaterializedBatchInvocation::bind(ReadLease query,ReadLease keys,ReadLease values)noexcept{
  HYLE_RETURN_IF_ERROR(ready(data_));if(!data_->prepared || data_->bound)return Status::FailedPrecondition("host Direct needs an unbound prepared episode");
  if(data_->program.schema().capacity().storage!=BatchStorage::kRagged)return Status::InvalidArgument("host Direct paged mode requires exact page frames");
  const std::array inputs{query,keys,values};std::array<ConstTensorView,3> views;
  for(std::size_t i=0;i<3;++i){HYLE_ASSIGN_OR_RETURN(views[i],inputs[i].tensor(data_->program.descriptions()[i]));if(views[i].storage().resource()!=mem::system_resource())return Status::InvalidArgument("host Direct requires system resource");}
  for(std::size_t i=0;i<3;++i){data_->inputs[i]=inputs[i];data_->views[i]=views[i];}data_->bound=true;return Status::Ok();
}
Result<Completion> MaterializedBatchInvocation::submit()noexcept{
  HYLE_RETURN_IF_ERROR(ready(data_));if(!data_->prepared || !data_->bound)return std::unexpected(Status::FailedPrecondition("host Direct episode is not prepared/bound"));
  auto&p=*data_;p.prepared=false;p.bound=false;const auto descriptions=p.program.descriptions();
  for(std::size_t i=3;i<5;++i){HYLE_ASSIGN_OR_RETURN(p.views[i],p.inputs[i].tensor(descriptions[i]));}
  const auto execute=[&]()noexcept->Status{
    for(std::size_t i=0;i<p.program.steps().size();++i){const auto&step=p.program.steps()[i];
      if(step.operation.kind()==OperationKind::kView){HYLE_ASSIGN_OR_RETURN(p.views[step.output],ConstTensorView::create(descriptions[step.output],p.views[step.inputs[0]].storage()));continue;}
      std::array<ConstTensorView,2> inputs;for(std::size_t port=0;port<step.input_count;++port)inputs[port]=p.views[step.inputs[port]];
      HYLE_ASSIGN_OR_RETURN(auto output,p.buffers[step.output].tensor(descriptions[step.output]));
      HYLE_ASSIGN_OR_RETURN(auto use,p.uses.begin());
      for(const auto&lease:p.inputs)if(lease.valid())HYLE_RETURN_IF_ERROR(use.read(lease));
      for(const auto&page:d::PagedArenaAccess::pages(p.keys))HYLE_RETURN_IF_ERROR(use.read(page));
      for(const auto&page:d::PagedArenaAccess::pages(p.values))HYLE_RETURN_IF_ERROR(use.read(page));
      for(const auto&writer:p.buffers)if(writer.valid())HYLE_RETURN_IF_ERROR(use.write(writer));
      for(const auto&writer:p.workspaces)if(writer.valid())HYLE_RETURN_IF_ERROR(use.write(writer));
      HYLE_ASSIGN_OR_RETURN(auto done,p.operations[i].submit(*p.queue,std::span(inputs).first(step.input_count),output));
      HYLE_FATAL_ASSERT(done.valid() && use.attach(done).ok());++p.report.submitted_stages;
      auto observed=p.completion.publish(std::move(done));++p.report.host_waits;const auto status=observed.wait();observed={};
      if(!status.ok())return status;
      if(p.uses.active_batches())return Status::Unavailable("host Direct observation preceded actual retirement");
      p.views[step.output]=output.as_const();
    }
    HYLE_ASSIGN_OR_RETURN(p.output_ready,p.buffers[p.program.output_value()].publish());return Status::Ok();
  };
  const auto status=execute();if(!status.ok()){p.failed=true;p.terminal=status;return std::unexpected(status);}
  return p.completion.publish(Completion::completed());
}
Result<ReadLease> MaterializedBatchInvocation::output()const noexcept{
  if(!data_ || data_->failed || data_->uses.active_batches() || !data_->output_ready.valid())
    return std::unexpected(Status::FailedPrecondition("host Direct output is not successfully ready"));
  return data_->output_ready;
}
MaterializedBatchRunReport MaterializedBatchInvocation::report()const noexcept{return data_?data_->report:MaterializedBatchRunReport{};}
Status MaterializedBatchInvocation::close()noexcept{if(!data_)return Status::Ok();data_->closed=true;data_->uses.close().IgnoreError();return data_->uses.active_batches()?Status::Unavailable("host Direct native work remains"):Status::Ok();}
Status MaterializedBatchInvocation::drain()noexcept{
  if(!data_)return Status::Ok();
  close().IgnoreError();data_->terminal.Update(data_->completion.wait());
  HYLE_RETURN_IF_ERROR(data_->uses.drain());return data_->terminal;
}
} // namespace hyle::compute
