if(HYLE_STATIC AND HYLE_COMPUTE_VULKAN_ENABLED)
  add_executable(hyle_graph_route_vulkan_bench ../compute_route_vulkan/route_p1_bench.cc)
  hyle_configure_vulkan_ffn_bench(hyle_graph_route_vulkan_bench graph_route_vulkan 1 0)
  add_executable(hyle_graph_route_vulkan_bench_placement
    ../compute_route_vulkan/placement.cc ../compute_route_vulkan/route_p1_bench.cc)
  hyle_configure_vulkan_ffn_bench(hyle_graph_route_vulkan_bench_placement graph_route_vulkan 1 1)
  hyle_register_benches(graph_route_vulkan hyle_graph_route_vulkan_bench
    hyle_graph_route_vulkan_bench_placement)
endif()

# Maintainer-only original D5 workload qualification. Current product headers
# and libraries must come from the same build; an old installed SDK cannot
# consume the expanded common fixture enum. No product/API dependency changes.
if(HYLE_COMPUTE_VULKAN_ENABLED AND TARGET GTest::gtest)
  add_executable(hyle_graph_route_vulkan_d5_bench d5_fragment_bench.cc)
  target_compile_features(hyle_graph_route_vulkan_d5_bench PRIVATE cxx_std_23)
  target_link_libraries(hyle_graph_route_vulkan_d5_bench PRIVATE
    hyle_graph_route_vulkan nanobench GTest::gtest ggml)
  hyle_apply_warnings(hyle_graph_route_vulkan_d5_bench)
  hyle_stage_runtime_dependencies(hyle_graph_route_vulkan_d5_bench)
endif()
