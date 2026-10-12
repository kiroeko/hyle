#include "hyle/graph/vulkan.h"
#include "hyle/graph/batch.h"
#include "hyle/compute/detail/vulkan_batch_access.h"
namespace hyle::graph::vulkan {
Result<compute::vulkan::MaterializedBatchInvocation> prepare_materialized_batch(
    const compute::vulkan::Context&context,compute::MaterializedBatchProgram recipe,
    const compute::vulkan::MemoryResource&device,const compute::vulkan::MemoryResource&staging,
    compute::MaterializedNativeBatchOptions options,const compute::WorkloadContext&workload)noexcept{
  HYLE_ASSIGN_OR_RETURN(auto graph,batch::materialized_graph(recipe));
  HYLE_ASSIGN_OR_RETURN(auto native,compile(context,graph.graph,{}, {.workload=workload}));
  return compute::vulkan::detail::PagedBatchAccess::create(context,std::move(native),std::move(recipe),device,staging,options);
}
} // namespace hyle::graph::vulkan
