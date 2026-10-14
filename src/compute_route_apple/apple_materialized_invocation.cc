#include "hyle/compute/detail/apple_batch_access.h"
#include "hyle/compute/detail/materialized_native_controller.h"
#include "apple_file_stream_access.h"
#include "apple_copy_internal.h"

namespace hyle::compute::apple {
namespace detail {
Status PagedBatchAccess::qualify_source(const Context&context,
    compute::detail::CompletionDependency prefix)noexcept{
  if(!context.valid())return Status::InvalidArgument("Apple pending source Context is empty");
  return qualify_recording_dependency(ContextAccess::runtime(context),prefix);
}
struct BatchTraits final {
  using Context=apple::Context;using Program=apple::Program;using Memory=MemoryResource;
  using Invocation=apple::Invocation;using CopyInvocation=CopyBatchInvocation;using View=mlx::core::array;using Submission=OwnedResult<Results>;
  using CopyUpdate=CopyBatchUpdate;using CopyAccess=CopyStreamAccess;
  static auto invocation(const Program&p,mem::AllocatorRef a){return p.create_invocation(a);}
  static auto copy_program(const Context&c,std::span<const CopyResourceSpec>s,CopyBatchOptions o){return CopyBatchProgram::prepare(c,s,o);}
  static auto copy_invocation(const CopyBatchProgram&p){return CopyBatchInvocation::create(p);}
  static auto binding(const Memory&m,ReadLease r,TensorDesc d,LeaseImportOptions o){return m.binding(std::move(r),d,o);}
  static auto paged_binding(const Memory&m,PagedArenaRead r,LeaseImportOptions o){return PagedBatchAccess::binding(m,std::move(r),o);}
  static auto paged_binding(const Memory&m,PagedArenaPendingRead r,LeaseImportOptions o){return PagedBatchAccess::binding(m,std::move(r),o);}
  static Status qualify_source(const Context&c,compute::detail::CompletionDependency p){return qualify_recording_dependency(ContextAccess::runtime(c),p);}
  static auto submit(Invocation&i,LeaseUseBatch u){return PagedBatchAccess::submit(i,std::move(u));}
  static Status wait_native(Invocation&i){return PagedBatchAccess::wait_native(i);}
  static Status reuse_gate(Invocation&i){return PagedBatchAccess::reuse_gate(i);}
  static auto prepare_program(const Context&c,const NativePagedBatchProgram&p,
      BatchLayoutVersion l,const WorkloadContext&w){return PagedBatchAccess::prepare_program(c,p,std::move(l),w);}
};
struct MaterializedInvocationData final:compute::detail::MaterializedNativeState<BatchTraits>{using MaterializedNativeState::MaterializedNativeState;};
struct PagedInvocationData final:compute::detail::MaterializedNativeState<BatchTraits,compute::detail::PagedMetadataPolicy>{using MaterializedNativeState::MaterializedNativeState;};
struct PagedReceiverData final:compute::detail::NativePagedReceiverState<BatchTraits,PagedBatchInvocation>{using NativePagedReceiverState::NativePagedReceiverState;};
void destroy(MaterializedInvocationData*p)noexcept{if(p){const auto a=p->allocator;a.destroy(p);}}
void destroy(PagedInvocationData*p)noexcept{if(p){const auto a=p->allocator;a.destroy(p);}}
void destroy(PagedReceiverData*p)noexcept{if(p){const auto a=p->allocator;a.destroy(p);}}
Result<MaterializedBatchInvocation> PagedBatchAccess::create(const Context&context,Program native,
    compute::MaterializedBatchProgram recipe,const MemoryResource&device,const MemoryResource&staging,
    MaterializedNativeBatchOptions options)noexcept{
  auto allocator=options.allocator;if(!allocator.valid())return std::unexpected(Status::InvalidArgument("native batch allocator invalid"));
  if(options.resource_budget.valid()){HYLE_ASSIGN_OR_RETURN(allocator,options.resource_budget.metadata_allocator());}
  auto*p=allocator.construct<MaterializedInvocationData>(allocator);if(!p)return std::unexpected(Status::ResourceExhausted("Apple batch controller metadata"));
  MaterializedBatchInvocation result(p);HYLE_RETURN_IF_ERROR(p->initialize(context,std::move(native),std::move(recipe),device,staging,options));return result;
}
} // namespace detail
MaterializedBatchInvocation::MaterializedBatchInvocation(MaterializedBatchInvocation&&v)noexcept:data_(std::exchange(v.data_,nullptr)){}
MaterializedBatchInvocation&MaterializedBatchInvocation::operator=(MaterializedBatchInvocation&&v)noexcept{if(this!=&v){detail::destroy(data_);data_=std::exchange(v.data_,nullptr);}return *this;}
MaterializedBatchInvocation::~MaterializedBatchInvocation(){detail::destroy(data_);}
Result<MaterializedBatchInvocation> MaterializedBatchInvocation::create(const Context&context,
    compute::MaterializedBatchProgram recipe,const MemoryResource&device,const MemoryResource&staging,
    MaterializedNativeBatchOptions options,const WorkloadContext&workload)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto native,prepare_materialized_batch(context,recipe,workload));return detail::PagedBatchAccess::create(context,std::move(native),std::move(recipe),device,staging,options);
}
Status MaterializedBatchInvocation::prepare_batch(BatchLayoutVersion v)noexcept{return data_?data_->prepare(std::move(v)):Status::FailedPrecondition("empty Apple batch controller");}
Status MaterializedBatchInvocation::bind(ReadLease q,PagedArenaRead k,PagedArenaRead v)noexcept{return data_?data_->bind(std::move(q),std::move(k),std::move(v)):Status::FailedPrecondition("empty Apple batch controller");}
Status MaterializedBatchInvocation::bind(ReadLease q,ReadLease k,ReadLease v)noexcept{return data_?data_->bind(std::move(q),std::move(k),std::move(v)):Status::FailedPrecondition("empty Apple batch controller");}
Result<OwnedResult<Results>> MaterializedBatchInvocation::submit()noexcept{if(!data_)return std::unexpected(Status::FailedPrecondition("empty Apple batch controller"));return data_->submit();}
MaterializedNativeBatchReport MaterializedBatchInvocation::report()const noexcept{return data_?data_->report:MaterializedNativeBatchReport{};}
Status MaterializedBatchInvocation::close()noexcept{return data_?data_->close():Status::Ok();}
Status MaterializedBatchInvocation::drain()noexcept{return data_?data_->drain():Status::Ok();}
PagedBatchInvocation::PagedBatchInvocation(PagedBatchInvocation&& v) noexcept
    : data_(std::exchange(v.data_,nullptr)) {}
PagedBatchInvocation& PagedBatchInvocation::operator=(PagedBatchInvocation&& v) noexcept {
  if(this!=&v){detail::destroy(data_);data_=std::exchange(v.data_,nullptr);}return *this;
}
PagedBatchInvocation::~PagedBatchInvocation(){detail::destroy(data_);}
Result<PagedBatchInvocation> PagedBatchInvocation::create(const Context& context,
    NativePagedBatchProgram recipe,const MemoryResource& device,const MemoryResource& staging,
    NativePagedInvocationOptions options,const WorkloadContext& workload) noexcept {
  if(!context.valid() || !recipe.valid() || !workload.valid())
    return std::unexpected(Status::InvalidArgument("Apple paged context/recipe/workload invalid"));
  const auto c=recipe.schema().capacity();
  if((c.query_type!=DataType::kFloat16 && c.query_type!=DataType::kBFloat16) ||
      (c.qk_head_dim!=64 && c.qk_head_dim!=128 && c.qk_head_dim!=256))
    return std::unexpected(Status::Unimplemented("Apple paged qualified flash dtype/depth"));
  auto allocator=options.allocator;
  if(!allocator.valid())return std::unexpected(Status::InvalidArgument("Apple paged allocator invalid"));
  if(options.resource_budget.valid()){HYLE_ASSIGN_OR_RETURN(allocator,options.resource_budget.metadata_allocator());}
  auto* p=allocator.construct<detail::PagedInvocationData>(allocator);
  if(!p)return std::unexpected(Status::ResourceExhausted("Apple paged controller metadata"));
  PagedBatchInvocation result(p);
  HYLE_RETURN_IF_ERROR(p->initialize(context,Program{},std::move(recipe),device,staging,options,workload));
  return result;
}
Status PagedBatchInvocation::prepare_batch(BatchLayoutVersion v) noexcept {return data_?data_->prepare(std::move(v)):Status::FailedPrecondition("empty Apple paged controller");}
Status PagedBatchInvocation::bind(ReadLease q,PagedArenaRead k,PagedArenaRead v) noexcept {return data_?data_->bind(std::move(q),std::move(k),std::move(v)):Status::FailedPrecondition("empty Apple paged controller");}
Result<OwnedResult<Results>> PagedBatchInvocation::submit() noexcept {if(!data_)return std::unexpected(Status::FailedPrecondition("empty Apple paged controller"));return data_->submit();}
NativePagedBatchReport PagedBatchInvocation::report() const noexcept {return data_?data_->report:NativePagedBatchReport{};}
Status PagedBatchInvocation::close() noexcept {return data_?data_->close():Status::Ok();}
Status PagedBatchInvocation::drain() noexcept {return data_?data_->drain():Status::Ok();}
PagedBatchReceiver::PagedBatchReceiver(PagedBatchReceiver&&v)noexcept:data_(std::exchange(v.data_,nullptr)){}
PagedBatchReceiver&PagedBatchReceiver::operator=(PagedBatchReceiver&&v)noexcept{
  if(this!=&v){detail::destroy(data_);data_=std::exchange(v.data_,nullptr);}return *this;
}
PagedBatchReceiver::~PagedBatchReceiver(){detail::destroy(data_);}
Result<PagedBatchReceiver> PagedBatchReceiver::create(PagedBatchInvocation&&owner,std::size_t capacity)noexcept{
  if(!owner.valid())return std::unexpected(Status::InvalidArgument("empty Apple paged receiver owner"));
  const auto allocator=owner.data_->allocator;
  auto*p=allocator.construct<detail::PagedReceiverData>(allocator);
  if(!p)return std::unexpected(Status::ResourceExhausted("Apple paged receiver metadata"));
  PagedBatchReceiver result(p);HYLE_RETURN_IF_ERROR(p->initialize(capacity,owner.data_->import.resource_budget));
  p->original=std::move(owner);return result;
}
Status PagedBatchReceiver::prepare_batch(BatchLayoutVersion layout)noexcept{
  if(!data_)return Status::FailedPrecondition("empty Apple paged receiver");
  HYLE_RETURN_IF_ERROR(data_->reusable());return data_->original.prepare_batch(std::move(layout));
}
Result<OwnedResult<Results>> PagedBatchReceiver::submit(ReadLease query,PagedArenaPendingRead k,
    PagedArenaPendingRead v,CopyBatchUpdate&&update)noexcept{
  if(!data_)return std::unexpected(Status::FailedPrecondition("empty Apple paged receiver"));
  return data_->submit(*data_->original.data_,std::move(query),std::move(k),std::move(v),std::move(update));
}
Result<CopyBatchUpdate> PagedBatchReceiver::take_update()noexcept{
  if(!data_)return std::unexpected(Status::FailedPrecondition("empty Apple paged receiver"));
  return data_->take_update();
}
NativePagedBatchReport PagedBatchReceiver::report()const noexcept{return data_?data_->original.report():NativePagedBatchReport{};}
Status PagedBatchReceiver::close()noexcept{return data_?data_->original.close():Status::Ok();}
Status PagedBatchReceiver::drain()noexcept{return data_?data_->drain():Status::Ok();}
} // namespace hyle::compute::apple
