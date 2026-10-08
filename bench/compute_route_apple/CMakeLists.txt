if(HYLE_STATIC AND HYLE_COMPUTE_APPLE_ENABLED)
  # Both mathematical arms use exactly one locked MLX runtime. The independent
  # peer checkout is an identity input for qualification, never a second DSO.
  if(DEFINED HYLE_APPLE_REFERENCE_SOURCE_DIR AND
     NOT EXISTS "${HYLE_APPLE_REFERENCE_SOURCE_DIR}/mlx/ops.cpp")
    message(FATAL_ERROR "The pinned MLX reference checkout is incomplete")
  endif()
  get_target_property(_mlx_source mlx HYLE_DEP_SOURCE_DIR)
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/bin/apple-mlx-source-path.txt"
    CONTENT "${_mlx_source}\n")

  function(hyle_configure_apple_ffn_bench target owner graph_mode placement)
    target_compile_definitions(${target} PRIVATE
      HYLE_APPLE_GRAPH_BENCH=${graph_mode})
    target_link_libraries(${target} PRIVATE hyle_${owner} nanobench)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_apply_warnings(${target})
    hyle_stage_apple_runtime(${target})
    if(placement)
      target_compile_options(${target} PRIVATE -falign-functions=64)
    endif()
  endfunction()
  add_executable(hyle_compute_route_apple_bench route_p1_bench.cc)
  hyle_configure_apple_ffn_bench(hyle_compute_route_apple_bench compute_route_apple 0 FALSE)
  add_executable(hyle_compute_route_apple_bench_placement route_p1_bench.cc)
  hyle_configure_apple_ffn_bench(hyle_compute_route_apple_bench_placement compute_route_apple 0 TRUE)
  hyle_register_benches(compute_route_apple hyle_compute_route_apple_bench
    hyle_compute_route_apple_bench_placement)
  add_executable(hyle_compute_route_apple_overlap_bench overlap_p1_bench.cc)
  hyle_configure_apple_ffn_bench(hyle_compute_route_apple_overlap_bench compute_route_apple 0 FALSE)
  add_executable(hyle_compute_route_apple_overlap_bench_placement overlap_p1_bench.cc)
  hyle_configure_apple_ffn_bench(hyle_compute_route_apple_overlap_bench_placement compute_route_apple 0 TRUE)
  hyle_register_benches(compute_route_apple hyle_compute_route_apple_overlap_bench
    hyle_compute_route_apple_overlap_bench_placement)
endif()
