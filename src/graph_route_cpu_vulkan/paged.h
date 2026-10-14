#pragma once
#include "hyle/compute/detail/vulkan_batch_access.h"
#include "../compute_route_vulkan/vulkan_file_stream_access.h"
#include "../graph_batch/native_paged_slots.h"

namespace hyle::graph::cpu_vulkan::detail {
struct PageContext final {
  const compute::vulkan::Context* native=nullptr;
  const compute::vulkan::Program* initial=nullptr;
  compute::BatchLayoutVersion layout;
};
struct PageTraits final {
  using Context=PageContext;using Program=compute::vulkan::Program;
  using Memory=compute::vulkan::MemoryResource;using Invocation=compute::vulkan::Invocation;
  using CopyInvocation=compute::vulkan::CopyBatchInvocation;using View=compute::vulkan::Tensor;
  using Submission=compute::OwnedResult<compute::vulkan::Results>;
  using CopyUpdate=compute::vulkan::CopyBatchUpdate;
  using CopyAccess=compute::vulkan::detail::CopyStreamAccess;
  static auto invocation(const Program& p,mem::AllocatorRef a){return p.create_invocation(a);}
  static auto copy_program(const Context& c,std::span<const compute::CopyResourceSpec> s,compute::CopyBatchOptions o){return compute::vulkan::CopyBatchProgram::prepare(*c.native,s,o);}
  static auto copy_invocation(const compute::vulkan::CopyBatchProgram& p){return CopyInvocation::create(p);}
  static auto binding(const Memory& m,compute::ReadLease r,compute::TensorDesc d,compute::LeaseImportOptions o){return m.binding(std::move(r),d,o);}
  static auto paged_binding(const Memory& m,compute::PagedArenaRead r,compute::LeaseImportOptions o){return compute::vulkan::detail::PagedBatchAccess::binding(m,std::move(r),o);}
  static auto paged_binding(const Memory& m,compute::PagedArenaPendingRead r,compute::LeaseImportOptions o){return compute::vulkan::detail::PagedBatchAccess::binding(m,std::move(r),o);}
  static auto qualify_source(const Context&c,compute::detail::CompletionDependency p){return compute::vulkan::detail::PagedBatchAccess::qualify_source(*c.native,p);}
  static auto submit(Invocation& i,compute::LeaseUseBatch u){return compute::vulkan::detail::PagedBatchAccess::submit(i,std::move(u));}
  static auto wait_native(Invocation& i){return compute::vulkan::detail::PagedBatchAccess::wait_native(i);}
  static auto reuse_gate(Invocation& i){return compute::vulkan::detail::PagedBatchAccess::reuse_gate(i);}
  static Result<Program> prepare_program(const Context& c,const compute::NativePagedBatchProgram& p,
      compute::BatchLayoutVersion l,const compute::WorkloadContext& w){
    if(l==c.layout)return *c.initial;
    return compute::vulkan::detail::PagedBatchAccess::prepare_program(*c.native,p,std::move(l),w);
  }
};
using PageSlots=batch::detail::NativePagedSlots<PageTraits>;
struct PageSlotsDelete final {
  mem::AllocatorRef allocator;
  void operator()(PageSlots* pointer)const noexcept{allocator.destroy(pointer);}
};
} // namespace hyle::graph::cpu_vulkan::detail
