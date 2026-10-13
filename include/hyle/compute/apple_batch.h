#pragma once
#include "hyle/compute/apple.h"

#if defined(HYLE_COMPUTE_APPLE)
#include "hyle/compute/batch_materialize.h"
#include "hyle/compute/batch_paged_native.h"
#include "hyle/compute/apple_memory.h"
namespace hyle::compute::apple {
// Cold immutable host-known ragged layout, F16/BF16 D64/128/256. Three original
// packed-capacity Q/K/V ports and one active packed Q output. Original MLX
// views/SDPA/Program own execution; another topology needs explicit prepare.
// Native metadata/private storage remain unknown; no paged/capture claim.
HYLE_COMPUTE_ROUTE_APPLE_API Result<Program> prepare_ragged_batch(
    const Context&,BatchLayoutVersion,RaggedAttention,const WorkloadContext& = {}) noexcept;
// Explicit GPU materialization reference. Cold traces the fixed original
// typed recipe into one original MLX Program; run performs no recipe traversal
// or planning. Bind Q/K/V and same-version uploaded indices/mask in that order.
// Original native inputs/results/slots retain their ownership and close rules.
// MLX retains its immutable tracing callable and a value copy of the finite
// recipe; that native metadata is additional to the common logical byte bound.
// This is managed replay with separately unknown native storage, not capture
// or native paged. Common logical byte caps include GQA/gather/Copy costs.
HYLE_COMPUTE_ROUTE_APPLE_API Result<Program> prepare_materialized_batch(
    const Context&,const MaterializedBatchProgram&,const WorkloadContext& = {}) noexcept;
namespace detail {struct MaterializedInvocationData;struct PagedInvocationData;struct PagedBatchAccess;}
// Fixed native slots and two device metadata banks, with explicit backing cap.
// prepare uploads one owned generation; bind is cold and checks exact page
// frames. submit only runs the original native Program. Control is externally
// serialized; Context/resources/allocator outlive native work/results. Old
// results block prepare; original cancel/unknown/drain rules remain native.
class MaterializedBatchInvocation final {
public:
  MaterializedBatchInvocation()noexcept=default;
  MaterializedBatchInvocation(const MaterializedBatchInvocation&)=delete;
  MaterializedBatchInvocation&operator=(const MaterializedBatchInvocation&)=delete;
  HYLE_COMPUTE_ROUTE_APPLE_API MaterializedBatchInvocation(MaterializedBatchInvocation&&)noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API MaterializedBatchInvocation&operator=(MaterializedBatchInvocation&&)noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API ~MaterializedBatchInvocation();
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<MaterializedBatchInvocation> create(const Context&,
      compute::MaterializedBatchProgram,const MemoryResource&device,const MemoryResource&staging,
      MaterializedNativeBatchOptions,const WorkloadContext& = {})noexcept;
  bool valid()const noexcept{return data_!=nullptr;}
  HYLE_COMPUTE_ROUTE_APPLE_API Status prepare_batch(BatchLayoutVersion)noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status bind(ReadLease,PagedArenaRead,PagedArenaRead)noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status bind(ReadLease,ReadLease,ReadLease)noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Result<OwnedResult<Results>> submit()noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API MaterializedNativeBatchReport report()const noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status close()noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status drain()noexcept;
private:
  friend struct detail::PagedBatchAccess;
  explicit MaterializedBatchInvocation(detail::MaterializedInvocationData*p)noexcept:data_(p){}
  detail::MaterializedInvocationData*data_=nullptr;
};
// Exact-frame native page16 attention, F16/BF16 D64/128/256. Preparation
// uploads one owned I32 table and cold-prepares the original MLX topology.
// Context/resources/allocator keep stable addresses and outlive all work and
// results. Control is externally serialized; old results prevent reuse.
// Native/private storage remains additional and may be unknown.
class PagedBatchInvocation final {
public:
  PagedBatchInvocation() noexcept = default;
  PagedBatchInvocation(const PagedBatchInvocation&) = delete;
  PagedBatchInvocation& operator=(const PagedBatchInvocation&) = delete;
  HYLE_COMPUTE_ROUTE_APPLE_API PagedBatchInvocation(PagedBatchInvocation&&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API PagedBatchInvocation& operator=(PagedBatchInvocation&&) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API ~PagedBatchInvocation();
  HYLE_COMPUTE_ROUTE_APPLE_API static Result<PagedBatchInvocation> create(
      const Context&, NativePagedBatchProgram, const MemoryResource& device,
      const MemoryResource& staging, NativePagedInvocationOptions,
      const WorkloadContext& = {}) noexcept;
  bool valid() const noexcept { return data_ != nullptr; }
  HYLE_COMPUTE_ROUTE_APPLE_API Status prepare_batch(BatchLayoutVersion) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status bind(ReadLease, PagedArenaRead, PagedArenaRead) noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Result<OwnedResult<Results>> submit() noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API NativePagedBatchReport report() const noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status close() noexcept;
  HYLE_COMPUTE_ROUTE_APPLE_API Status drain() noexcept;
private:
  explicit PagedBatchInvocation(detail::PagedInvocationData* p) noexcept : data_(p) {}
  detail::PagedInvocationData* data_ = nullptr;
};
} // namespace hyle::compute::apple
#endif
