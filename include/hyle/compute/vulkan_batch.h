#pragma once
#include "hyle/compute/vulkan.h"

#if defined(HYLE_COMPUTE_VULKAN)
#include "hyle/compute/batch_materialize.h"
#include "hyle/compute/batch_paged_native.h"
#include "hyle/compute/vulkan_memory.h"
namespace hyle::compute::vulkan {
// Cold immutable host-known ragged layout. Three packed-capacity Q/K/V ports;
// output contains this layout's active packed Q rows. Reuse the original
// Program/Invocation and owners; another topology needs explicit preparation.
// Qualified F16 D64/128/256 reads original resident K/V request slices. Query,
// mask and packed output costs remain; no native paged/capture claim.
HYLE_COMPUTE_ROUTE_VULKAN_API Result<Program> prepare_ragged_batch(
    const Context&,BatchLayoutVersion,RaggedAttention,const WorkloadContext& = {}) noexcept;
// Explicit GPU materialization reference over the original fixed native Plan.
// Cold preparation qualifies each typed step; run uses the original Program
// and independent native slots, with no new Queue/runtime or warm planning.
// Inputs: Q/K/V, same-version uploaded I32 indices and F32 additive mask.
// GQA/gather/F32/Copy byte caps are common logical upper bounds; native scratch
// and driver storage remain additional. Not a native-paged/capture claim.
HYLE_COMPUTE_ROUTE_VULKAN_API Result<Program> prepare_materialized_batch(
    const Context&,const MaterializedBatchProgram&,const WorkloadContext& = {}) noexcept;
namespace detail {struct MaterializedInvocationData;struct PagedInvocationData;struct PagedReceiverData;struct PagedBatchAccess;}
class PagedBatchReceiver;
// Cold fixed native Program/slot and two metadata banks. Explicit prepare uses
// original F2 to upload owned same-generation indices/mask; run never plans.
// Control is externally serialized; Context/resources/allocator outlive native
// work/results. Old native results block update. Budget covers mirror/wrapper
// storage; native scratch/driver bytes retain their original unknown contract.
class MaterializedBatchInvocation final {
public:
  MaterializedBatchInvocation()noexcept=default;
  MaterializedBatchInvocation(const MaterializedBatchInvocation&)=delete;
  MaterializedBatchInvocation&operator=(const MaterializedBatchInvocation&)=delete;
  HYLE_COMPUTE_ROUTE_VULKAN_API MaterializedBatchInvocation(MaterializedBatchInvocation&&)noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API MaterializedBatchInvocation&operator=(MaterializedBatchInvocation&&)noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API ~MaterializedBatchInvocation();
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<MaterializedBatchInvocation> create(const Context&,
      compute::MaterializedBatchProgram,const MemoryResource&device,const MemoryResource&staging,
      MaterializedNativeBatchOptions,const WorkloadContext& = {})noexcept;
  bool valid()const noexcept{return data_!=nullptr;}
  HYLE_COMPUTE_ROUTE_VULKAN_API Status prepare_batch(BatchLayoutVersion)noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status bind(ReadLease,PagedArenaRead,PagedArenaRead)noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status bind(ReadLease,ReadLease,ReadLease)noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<OwnedResult<Results>> submit()noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API MaterializedNativeBatchReport report()const noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status close()noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status drain()noexcept;
private:
  friend struct detail::PagedBatchAccess;
  explicit MaterializedBatchInvocation(detail::MaterializedInvocationData*p)noexcept:data_(p){}
  detail::MaterializedInvocationData*data_=nullptr;
};

// Exact-frame native page16 attention. Preparation uploads one owned I32
// table and cold-prepares the original flash recipe for the owned topology.
// Context, resources and allocator keep stable addresses and outlive all work
// and results. Control is externally serialized; old results prevent reuse.
// Native scratch/driver storage remains additional and may be unknown.
// A layout with zero Q rows and zero page refs binds Q and two default empty
// frames; its original empty-view program reads no K/V storage.
class PagedBatchInvocation final {
public:
  PagedBatchInvocation() noexcept = default;
  PagedBatchInvocation(const PagedBatchInvocation&) = delete;
  PagedBatchInvocation& operator=(const PagedBatchInvocation&) = delete;
  HYLE_COMPUTE_ROUTE_VULKAN_API PagedBatchInvocation(PagedBatchInvocation&&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API PagedBatchInvocation& operator=(PagedBatchInvocation&&) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API ~PagedBatchInvocation();
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<PagedBatchInvocation> create(
      const Context&, NativePagedBatchProgram, const MemoryResource& device,
      const MemoryResource& staging, NativePagedInvocationOptions,
      const WorkloadContext& = {}) noexcept;
  bool valid() const noexcept { return data_ != nullptr; }
  HYLE_COMPUTE_ROUTE_VULKAN_API Status prepare_batch(BatchLayoutVersion) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status bind(ReadLease, PagedArenaRead, PagedArenaRead) noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<OwnedResult<Results>> submit() noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API NativePagedBatchReport report() const noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status close() noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status drain() noexcept;
private:
  friend class PagedBatchReceiver;
  explicit PagedBatchInvocation(detail::PagedInvocationData* p) noexcept : data_(p) {}
  detail::PagedInvocationData* data_ = nullptr;
};
// Explicit cold extension of one original paged controller. Update may contain
// several producers/resources/phases. Submit checks exact page coverage and the
// original runtime before accepting a suffix; rejection preserves Update. The
// same original result's completion/wait/result include source failures.
// Control is externally serialized. Context/resources/allocator outlive all
// controls and native work. Drop the old OwnedResult before take_update/reuse;
// take_update returns the original controls only after all actual retirement.
class PagedBatchReceiver final {
public:
  PagedBatchReceiver()noexcept=default;
  PagedBatchReceiver(const PagedBatchReceiver&)=delete;
  PagedBatchReceiver&operator=(const PagedBatchReceiver&)=delete;
  HYLE_COMPUTE_ROUTE_VULKAN_API PagedBatchReceiver(PagedBatchReceiver&&)noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API PagedBatchReceiver&operator=(PagedBatchReceiver&&)noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API ~PagedBatchReceiver();
  // Capacity counts original accepted-phase and used-producer completions;
  // allocate this extension before source submission. Failure preserves owner.
  HYLE_COMPUTE_ROUTE_VULKAN_API static Result<PagedBatchReceiver> create(
      PagedBatchInvocation&&,std::size_t max_source_completions)noexcept;
  bool valid()const noexcept{return data_!=nullptr;}
  HYLE_COMPUTE_ROUTE_VULKAN_API Status prepare_batch(BatchLayoutVersion)noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<OwnedResult<Results>> submit(ReadLease,
      PagedArenaPendingRead,PagedArenaPendingRead,CopyBatchUpdate&&)noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Result<CopyBatchUpdate> take_update()noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API NativePagedBatchReport report()const noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status close()noexcept;
  HYLE_COMPUTE_ROUTE_VULKAN_API Status drain()noexcept;
private:
  explicit PagedBatchReceiver(detail::PagedReceiverData*p)noexcept:data_(p){}
  detail::PagedReceiverData*data_=nullptr;
};
} // namespace hyle::compute::vulkan
#endif
