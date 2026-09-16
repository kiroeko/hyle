if(HYLE_STATIC AND HYLE_COMPUTE_CUDA_ENABLED)
  add_executable(hyle_graph_route_cuda_bench
    ../compute_route_cuda/route_p1_bench.cc)
  hyle_configure_cuda_ffn_bench(
    hyle_graph_route_cuda_bench graph_route_cuda 1 FALSE)
  add_executable(hyle_graph_route_cuda_bench_placement
    ../compute_route_cuda/route_p1_bench.cc)
  hyle_configure_cuda_ffn_bench(
    hyle_graph_route_cuda_bench_placement graph_route_cuda 1 TRUE)
  hyle_register_benches(graph_route_cuda
    hyle_graph_route_cuda_bench hyle_graph_route_cuda_bench_placement)
endif()
