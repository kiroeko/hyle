add_executable(hyle_platform_bench platform_bench.cc)
target_link_libraries(hyle_platform_bench PRIVATE hyle_platform nanobench)
hyle_apply_warnings(hyle_platform_bench)

if(HYLE_BENCH_TBB)
  if(TARGET hyle_platform_bench)
    target_compile_definitions(hyle_platform_bench PRIVATE
                               HYLE_BENCH_TBB TBB_PREVIEW_NUMA_ALLOCATION=1)
    target_link_libraries(hyle_platform_bench PRIVATE TBB::tbb)
  endif()
endif()

hyle_register_benches(platform hyle_platform_bench)
