#include "hyle/compute/detail/apple_batch_access.h"
#include "hyle/compute/detail/materialized_native_controller.h"

namespace hyle::compute::apple {
namespace detail {
struct BatchTraits final {
  using Context=apple::Context;using Program=apple::Program;using Memory=MemoryResource;
  using Invocation=apple::Invocation;using CopyInvocation=CopyBatchInvocation;using View=mlx::core::array;using Submission=OwnedResult<Results>;
  static auto invocation(const Program&p,mem::AllocatorRef a){return p.create_invocation(a);}
  static auto copy_program(const Context&c,std::span<const CopyResourceSpec>s,CopyBatchOptions o){return CopyBatchProgram::prepare(c,s,o);}
  static auto copy_invocation(const CopyBatchProgram&p){return CopyBatchInvocation::create(p);}
  static auto binding(const Memory&m,ReadLease r,TensorDesc d,LeaseImportOptions o){return m.binding(std::move(r),d,o);}
  static auto paged_binding(const Memory&m,PagedArenaRead r,LeaseImportOptions o){return PagedBatchAccess::binding(m,std::move(r),o);}
  static auto submit(Invocation&i,LeaseUseBatch u){return PagedBatchAccess::submit(i,std::move(u));}
  static Status wait_native(Invocation&i){return PagedBatchAccess::wait_native(i);}
  static Status reuse_gate(Invocation&i){return PagedBatchAccess::reuse_gate(i);}
};
struct MaterializedInvocationData final:compute::detail::MaterializedNativeState<BatchTraits>{using MaterializedNativeState::MaterializedNativeState;};
void destroy(MaterializedInvocationData*p)noexcept{if(p){const auto a=p->allocator;a.destroy(p);}}
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
} // namespace hyle::compute::apple
