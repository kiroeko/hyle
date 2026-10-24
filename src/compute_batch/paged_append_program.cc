#include "hyle/compute/detail/paged_append_program.h"
#include "hyle/compute/transfer.h"
#include "hyle/core/assert.h"
#include <cstring>
#include <limits>
#include <utility>

namespace hyle::compute::detail {
PagedAppendProgram::PagedAppendProgram(PagedAppendProgram&&p)noexcept
    :extension_(std::move(p.extension_)),table_(p.table_),program_(std::exchange(p.program_,nullptr)),allocator_(p.allocator_){}
PagedAppendProgram&PagedAppendProgram::operator=(PagedAppendProgram&&p)noexcept{
  if(this!=&p){reset();extension_=std::move(p.extension_);table_=p.table_;program_=std::exchange(p.program_,nullptr);allocator_=p.allocator_;}return *this;
}
void PagedAppendProgram::reset()noexcept{if(program_){table_->destroy(program_,allocator_);program_=nullptr;}extension_={};table_=nullptr;}
PagedAppendProgram::~PagedAppendProgram(){reset();}
Result<PagedAppendProgram>PagedAppendProgram::prepare(BackendExtensionRef extension,
    PagedAppendNativeSpec spec,mem::AllocatorRef allocator)noexcept{
  if(!extension.valid() || !allocator.valid() || !spec.row_bytes || !spec.page_rows ||
      !spec.physical_pages || !spec.input_rows || !spec.instruction_capacity)
    return std::unexpected(Status::InvalidArgument("paged append native geometry/capability is empty"));
  PagedAppendProgram result;result.extension_=std::move(extension);result.allocator_=allocator;
  struct Call{PagedAppendProgram*p;PagedAppendNativeSpec spec;}call{&result,spec};
  HYLE_RETURN_IF_ERROR(BackendExtensionAccess::call(result.extension_,[](void*context,const void*opaque,void*user)noexcept->Status{
    auto&call=*static_cast<Call*>(user);const auto*t=static_cast<const PagedAppendBackendVTable*>(opaque);
    if(!t || !t->prepare || !t->destroy || !t->enqueue || !t->supports_command_capture)
      return Status::Unimplemented("backend has no capturable paged append program");
    call.p->table_=t;return t->prepare(context,call.spec,call.p->allocator_,&call.p->program_);
  },&call));
  if(!result.program_)return std::unexpected(Status::Internal("paged append backend returned no program"));
  return result;
}
EnqueueResult PagedAppendProgram::enqueue(QueueSequence&sequence,PagedAppendNativeBindings bindings)const noexcept{
  if(!valid())return EnqueueResult::rejected(Status::FailedPrecondition("paged append native program is empty"));
  struct Call{const PagedAppendProgram*p;PagedAppendNativeBindings bindings;}call{this,bindings};
  return BackendExtensionAccess::enqueue_admitted(extension_,sequence,[](void*context,const void*,void*queue,void*user)noexcept{
    auto&call=*static_cast<Call*>(user);return call.p->table_->enqueue(context,queue,call.p->program_,call.bindings);
  },&call);
}
Result<PagedAppendMirror>PagedAppendMirror::create(mem::MemoryResourceRef resource,
    mem::MemoryResourceRef host,std::uint32_t capacity,mem::AllocatorRef allocator,ResourceBudget budget)noexcept{
  if(!capacity || capacity>(SIZE_MAX-sizeof(PagedAppendMetadata))/sizeof(PagedAppendInstruction))
    return std::unexpected(Status::OutOfRange("paged append descriptor capacity overflows"));
  const auto bytes=sizeof(PagedAppendMetadata)+std::size_t(capacity)*sizeof(PagedAppendInstruction);
  PagedAppendMirror result;result.capacity_=capacity;
  const LeasePoolOptions options{.max_blocks=1,.backing_alignment=16,.allocator=allocator,.resource_budget=budget};
  HYLE_ASSIGN_OR_RETURN(result.device_pool_,LeasePool::create(resource,bytes,options));
  HYLE_ASSIGN_OR_RETURN(result.host_pool_,LeasePool::create(host,bytes,options));
  HYLE_ASSIGN_OR_RETURN(result.writer_,result.device_pool_.allocate(bytes,16));
  HYLE_ASSIGN_OR_RETURN(result.staging_,result.host_pool_.allocate(bytes,16));
  HYLE_ASSIGN_OR_RETURN(result.uses_,LeaseUsePool::create({.max_batches=1,.max_resources_per_batch=2,.allocator=allocator,.resource_budget=budget}));
  return result;
}
Status PagedAppendMirror::upload(const QueueRef&queue,std::span<const PagedAppendInstruction>words)noexcept{
  if(failed_ || words.empty() || words.size()>capacity_)return Status::FailedPrecondition("paged append mirror is failed or exceeds its fixed capacity");
  if(pending_.valid() || uses_.active_batches())return Status::Unavailable("paged append mirror upload has not retired");
  if(ready_.valid()){HYLE_ASSIGN_OR_RETURN(writer_,ready_.try_write());}
  if(source_.valid()){HYLE_ASSIGN_OR_RETURN(staging_,source_.try_write());}
  HYLE_ASSIGN_OR_RETURN(auto view,staging_.view());HYLE_ASSIGN_OR_RETURN(auto mapping,view.map_host());
  HYLE_ASSIGN_OR_RETURN(auto bytes,mapping.writable_bytes());
  std::memset(bytes.data(),0,bytes.size());const PagedAppendMetadata header{static_cast<std::uint32_t>(words.size())};
  std::memcpy(bytes.data(),&header,sizeof(header));std::memcpy(bytes.data()+sizeof(header),words.data(),words.size_bytes());
  HYLE_RETURN_IF_ERROR(mapping.close());HYLE_ASSIGN_OR_RETURN(source_,staging_.publish());
  HYLE_ASSIGN_OR_RETURN(auto source,source_.view());HYLE_ASSIGN_OR_RETURN(auto target,writer_.view());
  HYLE_ASSIGN_OR_RETURN(auto use,uses_.begin());HYLE_RETURN_IF_ERROR(use.read(source_));HYLE_RETURN_IF_ERROR(use.write(writer_));
  HYLE_ASSIGN_OR_RETURN(auto sequence,QueueSequence::begin(queue));
  sequence.record(TransferAccess::enqueue_copy(sequence,target,source));
  HYLE_ASSIGN_OR_RETURN(pending_,sequence.finish());
  HYLE_FATAL_ASSERT(use.attach(pending_).ok());
  const auto terminal=pending_.wait();pending_={};
  if(!terminal.ok() || uses_.active_batches()){failed_=true;return terminal.ok()?Status::Unavailable("paged append metadata retirement unproven"):terminal;}
  HYLE_ASSIGN_OR_RETURN(ready_,writer_.publish());return Status::Ok();
}
} // namespace hyle::compute::detail
