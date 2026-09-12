if(HYLE_STATIC)
  function(_hyle_configure_graph_multi_op_bench target)
    target_link_libraries(${target} PRIVATE
      hyle_graph hyle_compute_provider_native nanobench)
    set_property(TARGET ${target}
      PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_apply_warnings(${target})
  endfunction()

  add_executable(hyle_graph_multi_op_tax_bench multi_op_tax_bench.cc)
  _hyle_configure_graph_multi_op_bench(hyle_graph_multi_op_tax_bench)

  add_executable(hyle_graph_multi_op_tax_bench_placement
    multi_op_tax_bench.cc)
  _hyle_configure_graph_multi_op_bench(
    hyle_graph_multi_op_tax_bench_placement)
  if(MSVC)
    target_link_options(hyle_graph_multi_op_tax_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hyle_graph_multi_op_tax_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR
      "Graph multi-op placement perturbation is unsupported")
  endif()

  hyle_register_benches(graph
    hyle_graph_multi_op_tax_bench
    hyle_graph_multi_op_tax_bench_placement)
endif()
