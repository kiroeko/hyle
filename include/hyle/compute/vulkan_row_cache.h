#pragma once
#include "hyle/compute/vulkan_memory.h"
#include "hyle/compute/batch_layout.h"
#include "hyle/compute/paged_arena.h"
#if defined(HYLE_COMPUTE_VULKAN)
namespace hyle::compute::vulkan {
struct PagedRowCacheGatherOptions final {
  std::size_t max_metadata_backing_bytes=SIZE_MAX;
  mem::AllocatorRef allocator=mem::system_allocator();ResourceBudget resource_budget{};
};
namespace detail {struct PagedRowCacheGatherState;struct PagedRowCacheGatherPaddedState;}
// Native ready-frame Direct reader. One original Program/command submission
// maps request IDs then invokes the checked row codec. Results retain their
// original native storage; there is no whole-arena read lease or host ID copy.
// Same-Context device/staging resources and allocator remain borrowed. Explicit
// cold prepare copies metadata through original F2 and waits; replay does not.
class PagedRowCacheGatherInvocation final {
public:
  PagedRowCacheGatherInvocation()noexcept=default;
  PagedRowCacheGatherInvocation(const PagedRowCacheGatherInvocation&)=delete;
  PagedRowCacheGatherInvocation&operator=(const PagedRowCacheGatherInvocation&)=delete;
  HYLE_COMPUTE_ROUTE_VULKAN_API PagedRowCacheGatherInvocation(PagedRowCacheGatherInvocation&&)noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API PagedRowCacheGatherInvocation&operator=(PagedRowCacheGatherInvocation&&)noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API ~PagedRowCacheGatherInvocation();
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<PagedRowCacheGatherInvocation> create(const Context&,
      const MemoryResource&device,const MemoryResource&staging,RowCacheBatchSchema,TensorDesc ids,PagedRowCacheGatherOptions={})noexcept;
  bool valid()const noexcept{return data_!=nullptr;}
  HYLE_COMPUTE_ROUTE_VULKAN_API TensorDesc output_description()const noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status prepare_layout(RowCacheLayoutVersion)noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<OwnedResult<Results>> submit(PagedRowCacheRead,ReadLease ids)noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status close()noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status drain()noexcept;
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
  HYLE_COMPUTE_ROUTE_VULKAN_API PagedRowCacheGatherPaddedInvocation(PagedRowCacheGatherPaddedInvocation&&)noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API PagedRowCacheGatherPaddedInvocation&operator=(PagedRowCacheGatherPaddedInvocation&&)noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API ~PagedRowCacheGatherPaddedInvocation();
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<PagedRowCacheGatherPaddedInvocation> create(const Context&,
      const MemoryResource&device,const MemoryResource&staging,RowCacheBatchSchema,TensorDesc ids,std::uint32_t score_columns,PagedRowCacheGatherOptions={})noexcept;
  bool valid()const noexcept{return data_!=nullptr;}
  HYLE_COMPUTE_ROUTE_VULKAN_API TensorDesc output_description()const noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API TensorDesc mask_description()const noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status prepare_layout(RowCacheLayoutVersion)noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<OwnedResult<Results>> submit(PagedRowCacheRead,ReadLease ids,ReadLease ranges)noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status close()noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status drain()noexcept;
private:
  explicit PagedRowCacheGatherPaddedInvocation(detail::PagedRowCacheGatherPaddedState*p)noexcept:data_(p){}
  detail::PagedRowCacheGatherPaddedState*data_=nullptr;
};

}
#endif
