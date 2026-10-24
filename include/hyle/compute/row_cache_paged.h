#pragma once

#include "hyle/compute/paged_arena.h"
#include "hyle/compute/prepared_operation.h"

namespace hyle::compute {
struct PagedRowCacheGatherOptions final {
  mem::MemoryResourceRef resource{};
  // Optional hard cap for the automatically computed private backing.
  // ResourceBudget and actual allocation errors still apply by default.
  // Output storage stays caller-owned.
  std::size_t max_backing_bytes=SIZE_MAX;
  mem::AllocatorRef allocator=mem::system_allocator();
  ResourceBudget resource_budget{};
};
namespace detail {struct PagedRowCacheGatherState;}
// One explicit ready-frame Direct slot. IDs are dense I32[T,K], output is
// dense BF16[T,K,D]. query_indptr assigns T rows to requests. Logical IDs must
// lie within that request's owned visible history, even if an out-of-range ID
// would name valid physical storage. Checked IDs all precede output writes.
// I32 physical mapping requires max_pages*page_size <= INT32_MAX+1.
// Context/Queue/resources/allocator retain their original borrow contracts.
// No whole-arena ReadLease, hidden payload materialization or pending-page
// permission is introduced. Distinct slots may execute independently.
class PagedRowCacheGatherInvocation final {
public:
  PagedRowCacheGatherInvocation()noexcept=default;
  PagedRowCacheGatherInvocation(const PagedRowCacheGatherInvocation&)=delete;
  PagedRowCacheGatherInvocation&operator=(const PagedRowCacheGatherInvocation&)=delete;
  HYLE_COMPUTE_BATCH_API PagedRowCacheGatherInvocation(PagedRowCacheGatherInvocation&&)noexcept;
  HYLE_COMPUTE_BATCH_API PagedRowCacheGatherInvocation&operator=(PagedRowCacheGatherInvocation&&)noexcept;
  HYLE_COMPUTE_BATCH_API ~PagedRowCacheGatherInvocation();
  HYLE_COMPUTE_BATCH_API static Result<PagedRowCacheGatherInvocation> create(
      const Context&,Queue&,RowCacheBatchSchema,TensorDesc ids,PagedRowCacheGatherOptions)noexcept;
  bool valid()const noexcept{return data_!=nullptr;}
  HYLE_COMPUTE_BATCH_API TensorDesc output_description()const noexcept;
  // Explicit cold metadata copy/wait. Repeated same-version preparation reuses
  // the mirror; old actual work/completion prevents reprepare or submission.
  HYLE_COMPUTE_BATCH_API Status prepare_layout(RowCacheLayoutVersion)noexcept;
  // Mapping and the original checked codec share one admitted QueueSequence.
  // Any accepted prefix, including a later enqueue failure, owns all original
  // permissions until its actual Completion retires. Failed slots recreate.
  HYLE_COMPUTE_BATCH_API Result<Completion> submit(PagedRowCacheRead,ReadLease ids,const WriteLease&output)noexcept;
  HYLE_COMPUTE_BATCH_API Status close()noexcept;
  HYLE_COMPUTE_BATCH_API Status drain()noexcept;
private:
  explicit PagedRowCacheGatherInvocation(detail::PagedRowCacheGatherState*p)noexcept:data_(p){}
  detail::PagedRowCacheGatherState*data_=nullptr;
};
} // namespace hyle::compute
