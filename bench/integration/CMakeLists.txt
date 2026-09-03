# misc-module bench (gap-fill #37): util/random (xoshiro256++ vs std::mt19937_64)
# and time/clock (monotonic/realtime vs std::chrono). obs has a dedicated runner.
add_executable(hyle_perf_control_bench perf_control_bench.cc)
target_link_libraries(hyle_perf_control_bench PRIVATE hyle_core nanobench)
hyle_apply_warnings(hyle_perf_control_bench)
set_property(TARGET hyle_perf_control_bench
  PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
hyle_register_benches(integration hyle_perf_control_bench)

# Cross-factor evidence for the admitted mem_reclaim -> Container table
# hand-off. It lives here instead of pretending that generic RCU or the whole
# Container product owns both sides of the consumer journey.
option(HYLE_MEM_RECLAIM_CONTAINER_P1_BENCH
  "build the reader-safe node-table consumer evidence adapter" OFF)
if(HYLE_MEM_RECLAIM_CONTAINER_P1_BENCH)
  if(NOT HYLE_STATIC OR NOT HYLE_BENCH_TBB OR
     NOT TARGET hyle_mem_reclaim OR NOT TARGET hyle_container)
    message(FATAL_ERROR
      "HYLE_MEM_RECLAIM_CONTAINER_P1_BENCH requires a static Container "
      "closure with HYLE_BENCH_TBB=ON")
  endif()

  function(_hyle_configure_node_table_reclamation_bench target)
    target_link_libraries(${target} PRIVATE
      hyle_container TBB::tbb nanobench)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_apply_warnings(${target})
    hyle_stage_tbb_benchmark_runtime(${target})
  endfunction()

  add_executable(hyle_node_table_reclamation_bench
    node_table_reclamation_bench.cc)
  _hyle_configure_node_table_reclamation_bench(
    hyle_node_table_reclamation_bench)

  add_executable(hyle_node_table_reclamation_bench_placement
    node_table_reclamation_bench.cc)
  _hyle_configure_node_table_reclamation_bench(
    hyle_node_table_reclamation_bench_placement)
  if(MSVC)
    target_link_options(hyle_node_table_reclamation_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
      hyle_node_table_reclamation_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR
      "node-table placement perturbation is unsupported")
  endif()

  hyle_register_benches(integration
    hyle_node_table_reclamation_bench
    hyle_node_table_reclamation_bench_placement)
endif()

if(TARGET hyle_time AND TARGET hyle_util)
  add_executable(hyle_misc_bench "${CMAKE_CURRENT_SOURCE_DIR}/../misc/misc_bench.cc")
  target_link_libraries(hyle_misc_bench PRIVATE hyle_time hyle_util nanobench)
  hyle_apply_warnings(hyle_misc_bench)
  hyle_register_benches(integration hyle_misc_bench)
endif()
