#include "hyle/graph/apple.h"
#include "hyle/graph/batch.h"
#include "hyle/compute/detail/apple_batch_access.h"
namespace hyle::graph::apple {
Result<compute::apple::MaterializedBatchInvocation> prepare_materialized_batch(
    const compute::apple::Context&context,compute::MaterializedBatchProgram recipe,
    const compute::apple::MemoryResource&device,const compute::apple::MemoryResource&staging,
    compute::MaterializedNativeBatchOptions options,const compute::WorkloadContext&workload)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto graph,batch::materialized_graph(recipe));
  HYLE_ASSIGN_OR_RETURN(auto native,compile(context,graph.graph,{}, {.workload=workload}));
  return compute::apple::detail::PagedBatchAccess::create(context,std::move(native),std::move(recipe),device,staging,options);
}
} // namespace hyle::graph::apple
