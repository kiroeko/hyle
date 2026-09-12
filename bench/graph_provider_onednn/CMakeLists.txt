if(HYLE_STATIC AND HYLE_COMPUTE_ONEDNN_ENABLED)
  function(_hyle_configure_graph_onednn_region_bench target)
    target_link_libraries(${target} PRIVATE
      hyle_graph_provider_onednn
      hyle_compute_provider_onednn
      hyle_compute_backend_cpu
      hyle_simd
      nanobench)
    target_include_directories(${target} PRIVATE
      "${PROJECT_SOURCE_DIR}/src/compute_provider_onednn")
    set_property(TARGET ${target}
      PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_apply_warnings(${target})
  endfunction()

  add_executable(hyle_graph_onednn_region_bench region_p1_bench.cc)
  _hyle_configure_graph_onednn_region_bench(
    hyle_graph_onednn_region_bench)

  add_executable(hyle_graph_onednn_region_bench_placement
    region_p1_bench.cc)
  _hyle_configure_graph_onednn_region_bench(
    hyle_graph_onednn_region_bench_placement)
  if(MSVC)
    target_link_options(hyle_graph_onednn_region_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
      hyle_graph_onednn_region_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR
      "Graph oneDNN region placement perturbation is unsupported")
  endif()

  hyle_register_benches(graph_provider_onednn
    hyle_graph_onednn_region_bench
    hyle_graph_onednn_region_bench_placement)
endif()
