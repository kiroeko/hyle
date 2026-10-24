#pragma once
#include "hyle/compute/apple_memory.h"
#include "hyle/compute/batch_layout.h"
#include "hyle/compute/paged_arena.h"
#if defined(HYLE_COMPUTE_APPLE)
namespace hyle::compute::apple {
struct PagedRowCacheGatherOptions final {
  std::size_t max_metadata_backing_bytes=SIZE_MAX;
  mem::AllocatorRef allocator=mem::system_allocator();ResourceBudget resource_budget{};
};
namespace detail {struct PagedRowCacheGatherState;struct PagedRowCacheGatherPaddedState;}
// Native ready-frame Direct reader. One original Program performs device
// request/page mapping and the checked row codec. Result storage is owned by
// the original native invocation, unlike the common caller-output API.
// Device/staging resources and allocator remain borrowed; the two resources
// may be the same Apple shared MemoryResource. Cold prepare copies metadata
// through original F2 and waits. IDs never undergo host observation.
class PagedRowCacheGatherInvocation final {
public:
  PagedRowCacheGatherInvocation()noexcept=default;
  PagedRowCacheGatherInvocation(const PagedRowCacheGatherInvocation&)=delete;
  PagedRowCacheGatherInvocation&operator=(const PagedRowCacheGatherInvocation&)=delete;
  HYLE_COMPUTE_ROUTE_APPLE_API PagedRowCacheGatherInvocation(PagedRowCacheGatherInvocation&&)noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API PagedRowCacheGatherInvocation&operator=(PagedRowCacheGatherInvocation&&)noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API ~PagedRowCacheGatherInvocation();
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<PagedRowCacheGatherInvocation> create(const Context&,
      const MemoryResource&device,const MemoryResource&staging,RowCacheBatchSchema,TensorDesc ids,PagedRowCacheGatherOptions={})noexcept;
  bool valid()const noexcept{return data_!=nullptr;}
  HYLE_COMPUTE_ROUTE_APPLE_API TensorDesc output_description()const noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status prepare_layout(RowCacheLayoutVersion)noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Result<OwnedResult<Results>> submit(PagedRowCacheRead,ReadLease ids)noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status close()noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status drain()noexcept;
private:
  explicit PagedRowCacheGatherInvocation(detail::PagedRowCacheGatherState*p)noexcept:data_(p){}
  detail::PagedRowCacheGatherState*data_=nullptr;
};
// Ready typed padding reader. IDs and ranges are request-local; only -1 or
// legal score columns outside [begin,end) produce zero rows and -Inf mask.
// Other invalid IDs/ranges fail checked completion. Empty histories need no
// page ReadLease and may keep every page writer live; capacity is allocated
// normally. Output0 is BF16[T,K,D], output1 F32[T,K] additive attention mask.
// Native Graph/pending page permission is not implied by this Direct owner.
class PagedRowCacheGatherPaddedInvocation final {
public:
  PagedRowCacheGatherPaddedInvocation()noexcept=default;
  PagedRowCacheGatherPaddedInvocation(const PagedRowCacheGatherPaddedInvocation&)=delete;
  PagedRowCacheGatherPaddedInvocation&operator=(const PagedRowCacheGatherPaddedInvocation&)=delete;
  HYLE_COMPUTE_ROUTE_APPLE_API PagedRowCacheGatherPaddedInvocation(PagedRowCacheGatherPaddedInvocation&&)noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API PagedRowCacheGatherPaddedInvocation&operator=(PagedRowCacheGatherPaddedInvocation&&)noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API ~PagedRowCacheGatherPaddedInvocation();
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<PagedRowCacheGatherPaddedInvocation> create(const Context&,
      const MemoryResource&device,const MemoryResource&staging,RowCacheBatchSchema,TensorDesc ids,std::uint32_t score_columns,PagedRowCacheGatherOptions={})noexcept;
  bool valid()const noexcept{return data_!=nullptr;}
  HYLE_COMPUTE_ROUTE_APPLE_API TensorDesc output_description()const noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API TensorDesc mask_description()const noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status prepare_layout(RowCacheLayoutVersion)noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Result<OwnedResult<Results>> submit(PagedRowCacheRead,ReadLease ids,ReadLease ranges)noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status close()noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status drain()noexcept;
private:
  explicit PagedRowCacheGatherPaddedInvocation(detail::PagedRowCacheGatherPaddedState*p)noexcept:data_(p){}
  detail::PagedRowCacheGatherPaddedState*data_=nullptr;
};

}
#endif
