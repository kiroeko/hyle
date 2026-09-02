add_executable(hyle_mem_reclaim_consumer_bench consumer_bench.cc)
target_link_libraries(hyle_mem_reclaim_consumer_bench PRIVATE
  hyle_mem_reclaim nanobench)
set_property(TARGET hyle_mem_reclaim_consumer_bench
  PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
hyle_apply_warnings(hyle_mem_reclaim_consumer_bench)

set(_hyle_mem_reclaim_benches hyle_mem_reclaim_consumer_bench)
if(HYLE_STATIC)
  add_executable(hyle_mem_reclaim_consumer_bench_placement consumer_bench.cc)
  target_link_libraries(hyle_mem_reclaim_consumer_bench_placement PRIVATE
    hyle_mem_reclaim nanobench)
  set_property(TARGET hyle_mem_reclaim_consumer_bench_placement
    PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
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
  hyle_apply_warnings(hyle_mem_reclaim_consumer_bench_placement)
  list(APPEND _hyle_mem_reclaim_benches
    hyle_mem_reclaim_consumer_bench_placement)
endif()

hyle_register_benches(mem_reclaim ${_hyle_mem_reclaim_benches})
unset(_hyle_mem_reclaim_benches)
