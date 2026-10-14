#pragma once
#include "hyle/compute/detail/apple_batch_access.h"
#include "../compute_route_apple/apple_file_stream_access.h"
#include "../graph_batch/native_paged_slots.h"

namespace hyle::graph::cpu_apple::detail {
struct PageContext final {
  const compute::apple::Context* native=nullptr;
  const compute::apple::Program* initial=nullptr;
  compute::BatchLayoutVersion layout;
};
struct PageTraits final {
  using Context=PageContext;using Program=compute::apple::Program;
  using Memory=compute::apple::MemoryResource;using Invocation=compute::apple::Invocation;
  using CopyInvocation=compute::apple::CopyBatchInvocation;using View=mlx::core::array;
  using Submission=compute::OwnedResult<compute::apple::Results>;
  using CopyUpdate=compute::apple::CopyBatchUpdate;
  using CopyAccess=compute::apple::detail::CopyStreamAccess;
  static Status validate_query_resource(const Memory& memory,mem::ConstBufferView view){
    if(view.resource().properties().domain!=memory.ref().properties().domain)
      return Status::InvalidArgument("page query is not an original readonly Metal carrier");
    // The common controller still calls original MemoryResource::binding,
    // which checks real MemoryState, native handle, description and bounds.
    return Status::Ok();
  }
  static auto invocation(const Program& p,mem::AllocatorRef a){return p.create_invocation(a);}
  static auto copy_program(const Context& c,std::span<const compute::CopyResourceSpec> s,compute::CopyBatchOptions o){return compute::apple::CopyBatchProgram::prepare(*c.native,s,o);}
  static auto copy_invocation(const compute::apple::CopyBatchProgram& p){return CopyInvocation::create(p);}
  static auto binding(const Memory& m,compute::ReadLease r,compute::TensorDesc d,compute::LeaseImportOptions o){return m.binding(std::move(r),d,o);}
  static auto paged_binding(const Memory& m,compute::PagedArenaRead r,compute::LeaseImportOptions o){return compute::apple::detail::PagedBatchAccess::binding(m,std::move(r),o);}
  static auto paged_binding(const Memory& m,compute::PagedArenaPendingRead r,compute::LeaseImportOptions o){return compute::apple::detail::PagedBatchAccess::binding(m,std::move(r),o);}
  static auto qualify_source(const Context&c,compute::detail::CompletionDependency p){return compute::apple::detail::PagedBatchAccess::qualify_source(*c.native,p);}
  static auto submit(Invocation& i,compute::LeaseUseBatch u){return compute::apple::detail::PagedBatchAccess::submit(i,std::move(u));}
  static auto wait_native(Invocation& i){return compute::apple::detail::PagedBatchAccess::wait_native(i);}
  static auto reuse_gate(Invocation& i){return compute::apple::detail::PagedBatchAccess::reuse_gate(i);}
  static Result<Program> prepare_program(const Context& c,const compute::NativePagedBatchProgram& p,
      compute::BatchLayoutVersion l,const compute::WorkloadContext& w){
    if(l==c.layout)return *c.initial;
    return compute::apple::detail::PagedBatchAccess::prepare_program(*c.native,p,std::move(l),w);
  }
};
using PageSlots=batch::detail::NativePagedSlots<PageTraits>;
struct PageSlotsDelete final {
  mem::AllocatorRef allocator;
  void operator()(PageSlots* pointer)const noexcept{allocator.destroy(pointer);}
};
} // namespace hyle::graph::cpu_apple::detail
