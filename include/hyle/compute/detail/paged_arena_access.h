#pragma once
#include "hyle/compute/paged_arena.h"
#include "hyle/compute/detail/copy_stream_access.h"
namespace hyle::compute::detail {
struct PendingArenaPage final {
  CopyResourceId resource;
  std::uint32_t page=0,initialized_rows=0,required_rows=0;
  const void*root=nullptr;
  std::uint64_t generation=0;
  std::size_t new_offset=0,new_bytes=0;
};
// Trusted batch adapters only. Full span is an addressable capacity, not read
// permission for unused pages. Retain/admit every returned page guard, and only
// run a qualified operation whose actual reads follow this immutable layout.
struct PagedArenaAccess final {
  HYLE_COMPUTE_BATCH_API static Result<ConstTensorView> tensor(const PagedArenaRead&) noexcept;
  HYLE_COMPUTE_BATCH_API static std::span<const ReadLease> pages(const PagedArenaRead&) noexcept;
  HYLE_COMPUTE_BATCH_API static Result<ConstTensorView> tensor(const PagedArenaPendingRead&) noexcept;
  HYLE_COMPUTE_BATCH_API static std::span<const ReadLease> pages(const PagedArenaPendingRead&) noexcept;
  HYLE_COMPUTE_BATCH_API static std::span<const PendingArenaPage> pending(const PagedArenaPendingRead&) noexcept;
  HYLE_COMPUTE_BATCH_API static Status validate_page(const PagedArenaPendingRead&,
      std::size_t index,const CopyUpdateInput&) noexcept;
  // Trusted receiver only: its original source slot owns update; observations
  // cannot interleave resolve/admission. Qualify actual endpoint/native prefix
  // separately before admit_read. These functions perform no source observation.
  template<class Update,class Access>static Status validate(
      const PagedArenaPendingRead&frame,const Update&update)noexcept{
    if(!frame.valid())return Status::InvalidArgument("pending arena frame is empty");
    HYLE_ASSIGN_OR_RETURN(auto prefix,Access::dependency(update));(void)prefix;
    const auto inputs=pending(frame);
    for(std::size_t i=0;i<inputs.size();++i){const auto&page=inputs[i];
      HYLE_ASSIGN_OR_RETURN(auto input,Access::input(update,page.resource));
      HYLE_RETURN_IF_ERROR(validate_page(frame,i,input));
      if(page.new_bytes)HYLE_RETURN_IF_ERROR(Access::covers(update,page.resource,page.new_offset,page.new_bytes));
    }
    return Status::Ok();
  }
  template<class Update,class Access>static Status admit_read(LeaseUseBatch&uses,
      const PagedArenaPendingRead&frame,const Update&update)noexcept{
    HYLE_RETURN_IF_ERROR((validate<Update,Access>(frame,update)));
    for(const auto&page:pages(frame))HYLE_RETURN_IF_ERROR(uses.read(page));
    for(const auto&page:pending(frame))HYLE_RETURN_IF_ERROR(Access::admit_read(uses,update,page.resource));
    return Status::Ok();
  }
};
} // namespace hyle::compute::detail
