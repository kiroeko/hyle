if(HYLE_STATIC AND HYLE_COMPUTE_CUDA_ENABLED)
  # The retired Marlin FFN source is not a current benchmark. This opt-in
  # consumer needs the D5 local model fixture and its existing test oracle.
  option(HYLE_D5_CUDA_FRAGMENT_BENCH "Build the D5 CUDA model fragment consumer" OFF)
  if(HYLE_D5_CUDA_FRAGMENT_BENCH)
    if(NOT TARGET hyle_simd OR NOT TARGET GTest::gtest)
      message(FATAL_ERROR
        "D5 CUDA benchmark needs SIMD and GoogleTest fixture targets. Select "
        "compute,graph,simd with tests ON, or use bench/graph_route_cuda/external.")
    endif()
    add_executable(hyle_graph_route_cuda_bench d5_fragment_bench.cc)
    target_link_libraries(hyle_graph_route_cuda_bench PRIVATE
      hyle_graph_route_cuda hyle_simd nanobench GTest::gtest ggml-base)
    set_property(TARGET hyle_graph_route_cuda_bench PROPERTY
      INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_apply_warnings(hyle_graph_route_cuda_bench)
    hyle_stage_runtime_dependencies(hyle_graph_route_cuda_bench)
    hyle_register_benches(graph_route_cuda hyle_graph_route_cuda_bench)
  endif()
endif()
