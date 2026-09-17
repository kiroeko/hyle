if(HYLE_STATIC AND HYLE_COMPUTE_VULKAN_ENABLED)
  add_executable(hyle_graph_route_vulkan_bench ../compute_route_vulkan/route_p1_bench.cc)
  hyle_configure_vulkan_ffn_bench(hyle_graph_route_vulkan_bench graph_route_vulkan 1 0 0)
  add_executable(hyle_graph_route_vulkan_bench_placement
    ../compute_route_vulkan/placement.cc ../compute_route_vulkan/route_p1_bench.cc)
  hyle_configure_vulkan_ffn_bench(hyle_graph_route_vulkan_bench_placement graph_route_vulkan 1 1 0)
  add_executable(hyle_graph_vulkan_native_cost_bench ../compute_route_vulkan/route_p1_bench.cc)
  hyle_configure_vulkan_ffn_bench(hyle_graph_vulkan_native_cost_bench graph_route_vulkan 1 0 1)
  hyle_register_benches(graph_route_vulkan hyle_graph_route_vulkan_bench
    hyle_graph_route_vulkan_bench_placement hyle_graph_vulkan_native_cost_bench)
endif()
