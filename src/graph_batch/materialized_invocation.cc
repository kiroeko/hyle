#include "hyle/graph/batch.h"
#include "hyle/graph/detail/region_access.h"
#include "hyle/compute/detail/paged_arena_access.h"
#include "hyle/compute/detail/lease_access.h"
#include "hyle/util/ref_counted.h"

namespace hyle::graph::batch {
namespace c=compute;namespace cd=c::detail;
namespace detail {
struct OutputOwner final:util::RefCounted<OutputOwner>{
  OutputOwner(c::MaterializedBatchProgram p,mem::AllocatorRef a,c::ResourceBudget b):program(std::move(p)),allocator(a),budget(std::move(b)){}
  void destroy()noexcept{const auto a=allocator;this->~OutputOwner();a.deallocate(this,sizeof(*this),alignof(OutputOwner));}
  c::MaterializedBatchProgram program;mem::AllocatorRef allocator;c::ResourceBudget budget;
};
struct MaterializedInvocationData final {
  explicit MaterializedInvocationData(mem::AllocatorRef a):allocator(a){}
  mem::AllocatorRef allocator;compute::Queue*queue=nullptr;
  c::MaterializedBatchProgram program;MaterializedGraph logical;
  c::ReadLease output_ready;c::WriteLease output_writer;
  c::MaterializedBatchVersion version;c::PagedArenaRead keys,values;
  LeasedInvocation original;
  bool prepared=false,bound=false,closed=false;
};
void destroy(MaterializedInvocationData*p)noexcept{if(p){const auto a=p->allocator;a.destroy(p);}}
} // namespace detail
MaterializedInvocation::MaterializedInvocation(MaterializedInvocation&&v)noexcept:data_(std::exchange(v.data_,nullptr)){}
MaterializedInvocation&MaterializedInvocation::operator=(MaterializedInvocation&&v)noexcept{if(this!=&v){detail::destroy(data_);data_=std::exchange(v.data_,nullptr);}return *this;}
MaterializedInvocation::~MaterializedInvocation(){detail::destroy(data_);}
Result<MaterializedInvocation> MaterializedInvocation::create(const c::Context&context,
    c::MaterializedBatchProgram program,c::Queue&queue,MaterializedInvocationOptions options)noexcept{
  if(!program.valid() || !context.valid() || !queue.valid() || !options.leases.allocator.valid())
    return std::unexpected(Status::InvalidArgument("host materialized invocation needs a program/context/queue/allocator"));
  HYLE_ASSIGN_OR_RETURN(auto bytes,program.descriptions()[program.output_value()].required_bytes());
  if(bytes>SIZE_MAX-63 || bytes+63>options.max_output_bytes)
    return std::unexpected(Status::ResourceExhausted("host materialized output backing byte cap"));
  const auto cap=program.schema().capacity();if(cap.max_page_refs>(UINT32_MAX-1)/2)
    return std::unexpected(Status::ResourceExhausted("host materialized page permission capacity"));
  auto allocator=options.leases.allocator;if(options.leases.resource_budget.valid()){
    HYLE_ASSIGN_OR_RETURN(allocator,options.leases.resource_budget.metadata_allocator());
  }
  auto*p=allocator.construct<detail::MaterializedInvocationData>(allocator);
  if(!p)return std::unexpected(Status::ResourceExhausted("host materialized coordinator metadata"));
  MaterializedInvocation result(p);p->program=std::move(program);p->queue=&queue;
  HYLE_ASSIGN_OR_RETURN(p->logical,materialized_graph(p->program,{.metadata_allocator=allocator}));
  const std::array domains{StorageDomain{mem::system_resource()}};
  const std::array targets{ExecutionTarget{&context,StorageDomainId{0}}};
  HYLE_ASSIGN_OR_RETURN(auto compiled,compile(p->logical.graph,{domains,targets,{}},options.compile));
  HYLE_ASSIGN_OR_RETURN(p->original,LeasedInvocation::create(compiled.executable,options.leases));
  if(cap.max_page_refs)HYLE_RETURN_IF_ERROR(graph::detail::RegionAccess::reserve_reads(p->original,2*cap.max_page_refs,options.leases));
  HYLE_ASSIGN_OR_RETURN(auto pool,c::LeasePool::create(mem::system_resource(),bytes+63,
      {.max_blocks=1,.allocator=allocator,.resource_budget=options.leases.resource_budget}));
  HYLE_ASSIGN_OR_RETURN(p->output_writer,pool.allocate(bytes,64));
  auto*owner=allocator.construct<detail::OutputOwner>(p->program,allocator,options.leases.resource_budget);
  if(!owner)return std::unexpected(Status::ResourceExhausted("host materialized output owner"));
  cd::LeaseAccess::retain_resource_owner(pool,owner,[](void*p)noexcept{static_cast<detail::OutputOwner*>(p)->decref();});
  cd::LeaseAccess::detach_pool_root(pool);return result;
}
Status MaterializedInvocation::prepare_batch(c::BatchLayoutVersion layout)noexcept{
  if(!data_ || data_->closed)return Status::FailedPrecondition("host materialized invocation is empty/closed");
  // Validate immutable layout before invalidating the current prepared episode.
  HYLE_ASSIGN_OR_RETURN(auto version,data_->program.version(std::move(layout)));
  HYLE_ASSIGN_OR_RETURN(auto writer,graph::detail::RegionAccess::take_write(data_->original,data_->logical.output));
  HYLE_RETURN_IF_ERROR(data_->original.reset());
  data_->prepared=false;data_->bound=false;data_->keys={};data_->values={};data_->version={};
  if(writer.valid())data_->output_writer=std::move(writer);
  if(data_->output_ready.valid()){
    HYLE_ASSIGN_OR_RETURN(data_->output_writer,data_->output_ready.try_write());
  }
  HYLE_ASSIGN_OR_RETURN(auto indices,version.indices());HYLE_ASSIGN_OR_RETURN(auto mask,version.mask());
  const auto descriptions=data_->program.descriptions();
  const auto bind=[&]()noexcept->Status{
    HYLE_RETURN_IF_ERROR(data_->original.bind(data_->logical.inputs[3],descriptions[3],std::move(indices)));
    HYLE_RETURN_IF_ERROR(data_->original.bind(data_->logical.inputs[4],descriptions[4],std::move(mask)));
    return data_->original.bind(data_->logical.output,descriptions[0],std::move(data_->output_writer));
  };
  const auto status=bind();if(!status.ok())return status;
  data_->version=std::move(version);data_->prepared=true;return Status::Ok();
}
Status MaterializedInvocation::bind(c::ReadLease query,c::PagedArenaRead keys,c::PagedArenaRead values)noexcept{
  if(!data_ || !data_->prepared || data_->closed || data_->bound)
    return Status::FailedPrecondition("host materialized invocation needs an unbound prepared episode");
  if(data_->program.schema().capacity().storage!=c::BatchStorage::kPaged || keys.layout()!=data_->version.layout() || values.layout()!=data_->version.layout())
    return Status::InvalidArgument("host paged K/V require the exact prepared layout");
  HYLE_ASSIGN_OR_RETURN(auto kt,cd::PagedArenaAccess::tensor(keys));HYLE_ASSIGN_OR_RETURN(auto vt,cd::PagedArenaAccess::tensor(values));
  const auto bind=[&]()noexcept->Status{
    HYLE_RETURN_IF_ERROR(data_->original.bind(data_->logical.inputs[0],data_->program.descriptions()[0],std::move(query)));
    HYLE_RETURN_IF_ERROR(graph::detail::RegionAccess::bind_gather(data_->original,data_->logical.graph,data_->logical.inputs[1],data_->logical.inputs[3],kt,cd::PagedArenaAccess::pages(keys)));
    return graph::detail::RegionAccess::bind_gather(data_->original,data_->logical.graph,data_->logical.inputs[2],data_->logical.inputs[3],vt,cd::PagedArenaAccess::pages(values));
  };
  const auto status=bind();if(!status.ok()){data_->prepared=false;return status;}
  data_->keys=std::move(keys);data_->values=std::move(values);data_->bound=true;return Status::Ok();
}
Status MaterializedInvocation::bind(c::ReadLease query,c::ReadLease keys,c::ReadLease values)noexcept{
  if(!data_ || !data_->prepared || data_->closed || data_->bound)
    return Status::FailedPrecondition("host materialized invocation needs an unbound prepared episode");
  if(data_->program.schema().capacity().storage!=c::BatchStorage::kRagged)
    return Status::InvalidArgument("paged host reference requires exact page frames");
  const std::array inputs{query,keys,values};const auto descriptions=data_->program.descriptions();
  for(std::size_t i=0;i<3;++i){const auto status=data_->original.bind(data_->logical.inputs[i],descriptions[i],inputs[i]);if(!status.ok()){data_->prepared=false;return status;}}
  data_->bound=true;return Status::Ok();
}
Result<c::Completion> MaterializedInvocation::submit()noexcept{
  if(!data_ || !data_->prepared || !data_->bound || data_->closed)
    return std::unexpected(Status::FailedPrecondition("host materialized batch is not prepared/bound/open"));
  auto completion=data_->original.submit(*data_->queue);data_->prepared=false;data_->bound=false;return completion;
}
Result<LeasedGraphOutput> MaterializedInvocation::output()noexcept{
  if(!data_)return std::unexpected(Status::FailedPrecondition("host materialized invocation is empty"));
  HYLE_ASSIGN_OR_RETURN(auto output,data_->original.output(data_->logical.output));
  if(!data_->output_ready.valid()){HYLE_ASSIGN_OR_RETURN(data_->output_ready,output.storage_lease());}
  return output;
}
c::MaterializedBatchVersion MaterializedInvocation::version()const noexcept{return data_?data_->version:c::MaterializedBatchVersion{};}
Status MaterializedInvocation::close()noexcept{if(!data_)return Status::Ok();data_->closed=true;return data_->original.close();}
Status MaterializedInvocation::drain()noexcept{if(!data_)return Status::Ok();data_->closed=true;return data_->original.drain();}
} // namespace hyle::graph::batch
