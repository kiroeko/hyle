#pragma once
#include "hyle/compute/paged_arena.h"
#include "hyle/compute/detail/copy_stream_access.h"
#include "hyle/mem/std_allocator.h"
#include <vector>
namespace hyle::compute::detail {
// Cold-reserved exact permission group. Entries borrow the caller's retained
// frames/update only through admit(); no pointer is used across episodes.
// Other already-admitted readonly guards may overlap this group; every added
// reference is balanced at actual retirement. Mutable permission is unchanged.
class BatchPageReads final {
public:
  explicit BatchPageReads(mem::AllocatorRef a=mem::system_allocator())noexcept:entries_(mem::StdAllocator<Entry>(a)){}
  BatchPageReads(const BatchPageReads&)=delete;
  BatchPageReads&operator=(const BatchPageReads&)=delete;
  HYLE_COMPUTE_BATCH_API Status reserve(std::size_t)noexcept;
  void reset()noexcept{entries_.clear();}
  HYLE_COMPUTE_BATCH_API Status add(const ReadLease&)noexcept;
  // Only after original update/range validation and native ordering qualification.
  HYLE_COMPUTE_BATCH_API Status add_ordered(const WriteLease&)noexcept;
  HYLE_COMPUTE_BATCH_API Status admit(LeaseUseBatch&)noexcept;
private:
  struct Entry final{const ReadLease*ready=nullptr;const WriteLease*ordered=nullptr;};
  std::vector<Entry,mem::StdAllocator<Entry>> entries_;
};
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
  HYLE_COMPUTE_BATCH_API static Status validate_storage(const PagedArenaRead&,const CacheStorage&) noexcept;
  HYLE_COMPUTE_BATCH_API static std::span<const ReadLease> pages(const PagedArenaRead&) noexcept;
  HYLE_COMPUTE_BATCH_API static Result<ConstTensorView> tensor(const PagedArenaPendingRead&) noexcept;
  HYLE_COMPUTE_BATCH_API static Status validate_storage(const PagedArenaPendingRead&,const CacheStorage&) noexcept;
  HYLE_COMPUTE_BATCH_API static std::span<const ReadLease> pages(const PagedArenaPendingRead&) noexcept;
  HYLE_COMPUTE_BATCH_API static std::span<const PendingArenaPage> pending(const PagedArenaPendingRead&) noexcept;
  HYLE_COMPUTE_BATCH_API static Status validate_page(const PagedArenaPendingRead&,
      std::size_t index,const CopyUpdateInput&) noexcept;
  // Trusted receiver only: its original source slot owns update; observations
  // cannot interleave resolve/admission. Qualify actual endpoint/native prefix
  // separately before collect_reads/admit. These functions do not observe work.
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
  template<class Update,class Access>static Status collect_reads(BatchPageReads&reads,
      const PagedArenaPendingRead&frame,const Update&update)noexcept{
    HYLE_RETURN_IF_ERROR((validate<Update,Access>(frame,update)));
    for(const auto&page:pages(frame))HYLE_RETURN_IF_ERROR(reads.add(page));
    for(const auto&page:pending(frame)){
      HYLE_ASSIGN_OR_RETURN(auto input,Access::input(update,page.resource));
      HYLE_RETURN_IF_ERROR(reads.add_ordered(*input.writer));
    }
    return Status::Ok();
  }
};
} // namespace hyle::compute::detail
