add_executable(hyle_graph_one_op_tax_bench one_op_tax_bench.cc)
target_link_libraries(hyle_graph_one_op_tax_bench PRIVATE
  hyle_graph hyle_simd nanobench)
hyle_apply_warnings(hyle_graph_one_op_tax_bench)
set_property(TARGET hyle_graph_one_op_tax_bench
  PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)

# A second contract-identical placement catches short-wrapper address effects
# without changing either side's source, inputs, storage or execution order.
if(HYLE_STATIC)
  add_executable(hyle_graph_one_op_tax_bench_placement one_op_tax_bench.cc)
  target_link_libraries(hyle_graph_one_op_tax_bench_placement PRIVATE
    hyle_graph hyle_simd nanobench)
  if(MSVC)
    target_link_options(hyle_graph_one_op_tax_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hyle_graph_one_op_tax_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR
      "graph one-op placement perturbation is unsupported by this compiler")
  endif()
  hyle_apply_warnings(hyle_graph_one_op_tax_bench_placement)
  set_property(TARGET hyle_graph_one_op_tax_bench_placement
    PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
  hyle_register_bench(graph hyle_graph_one_op_tax_bench_placement)
endif()

hyle_register_bench(graph hyle_graph_one_op_tax_bench)
