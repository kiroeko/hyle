#include "../compute_models/row_cache_graph_fixture.h"
#if defined(HYLE_ROW_CACHE_GRAPH_CUDA)
#include "hyle/graph/cuda.h"
#include "hyle/compute/backends/cuda.h"
TEST(CudaPagedRowCacheGraph, DeviceTopKFeedsSelectedPagesThenOneBatchedSink){
  auto context=hyle::graph::cuda::create_context().value();auto resource=hyle::compute::cuda::MemoryResource::create(context,hyle::compute::cuda::MemoryKind::kDevice).value();
  row_cache_graph_test::exercise(context,resource.ref(),[&](const hyle::graph::Graph&graph){return hyle::graph::cuda::compile(graph,context,resource.ref());});
  row_cache_graph_test::empty_and_private(context,resource.ref(),[&](const hyle::graph::Graph&graph){return hyle::graph::cuda::compile(graph,context,resource.ref());});
}
#else
#include "hyle/graph/cpu.h"
TEST(CpuPagedRowCacheGraph, TopKFeedsSelectedPagesThenOneBatchedSink){
  auto context=row_cache_paged_test::take(hyle::graph::cpu::create_context(),"Graph CPU context");
  row_cache_graph_test::exercise(context,hyle::mem::system_resource(),[&](const hyle::graph::Graph&graph){return hyle::graph::compile(graph,context);});
  row_cache_graph_test::empty_and_private(context,hyle::mem::system_resource(),[&](const hyle::graph::Graph&graph){return hyle::graph::compile(graph,context);});
}
#endif
