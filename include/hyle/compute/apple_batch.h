#pragma once
#include "hyle/compute/apple.h"

#if defined(HYLE_COMPUTE_APPLE)
#include "hyle/compute/batch_materialize.h"
#include "hyle/compute/apple_memory.h"
namespace hyle::compute::apple {
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
namespace detail {struct MaterializedInvocationData;struct PagedBatchAccess;}
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
} // namespace hyle::compute::apple
#endif
