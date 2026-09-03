function(_hyle_configure_mem_reclaim_bench target)
  target_link_libraries(${target} PRIVATE hyle_mem_reclaim nanobench)
  set_property(TARGET ${target}
    PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
  hyle_apply_warnings(${target})
  if(HYLE_BENCH_FOLLY)
    target_compile_definitions(${target} PRIVATE HYLE_BENCH_FOLLY)
    target_link_libraries(${target} PRIVATE hyle_bench_folly)
    hyle_stage_folly_benchmark_runtime(${target})
  endif()
endfunction()

add_executable(hyle_mem_reclaim_consumer_bench consumer_bench.cc)
_hyle_configure_mem_reclaim_bench(hyle_mem_reclaim_consumer_bench)

set(_hyle_mem_reclaim_benches hyle_mem_reclaim_consumer_bench)
if(HYLE_STATIC)
  add_executable(hyle_mem_reclaim_consumer_bench_placement consumer_bench.cc)
  _hyle_configure_mem_reclaim_bench(
    hyle_mem_reclaim_consumer_bench_placement)
  if(MSVC)
    target_link_options(hyle_mem_reclaim_consumer_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hyle_mem_reclaim_consumer_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR
      "mem_reclaim placement perturbation is unsupported")
  endif()
  list(APPEND _hyle_mem_reclaim_benches
    hyle_mem_reclaim_consumer_bench_placement)
endif()

hyle_register_benches(mem_reclaim ${_hyle_mem_reclaim_benches})
unset(_hyle_mem_reclaim_benches)
