if(HYLE_STATIC AND HYLE_COMPUTE_APPLE_ENABLED)
  add_executable(hyle_graph_route_apple_bench ../compute_route_apple/route_p1_bench.cc)
  hyle_configure_apple_ffn_bench(hyle_graph_route_apple_bench graph_route_apple 1 FALSE)
  add_executable(hyle_graph_route_apple_bench_placement ../compute_route_apple/route_p1_bench.cc)
  hyle_configure_apple_ffn_bench(hyle_graph_route_apple_bench_placement graph_route_apple 1 TRUE)
  hyle_register_benches(graph_route_apple hyle_graph_route_apple_bench
    hyle_graph_route_apple_bench_placement)
endif()
