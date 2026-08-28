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

if(HYLE_STATIC AND HYLE_BENCH_TBB AND HYLE_BENCH_BOOST)
  function(_hyle_configure_platform_consumer_bench target)
    target_link_libraries(${target} PRIVATE hyle_platform nanobench TBB::tbb
                          hyle_bench_boost_headers)
    target_include_directories(${target} SYSTEM PRIVATE
      "${HYLE_TBB_DIR}/src/tbb")
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_apply_warnings(${target})
  endfunction()

  add_executable(hyle_platform_consumer_bench consumer_bench.cc)
  _hyle_configure_platform_consumer_bench(hyle_platform_consumer_bench)

  add_executable(hyle_platform_consumer_bench_placement consumer_bench.cc)
  _hyle_configure_platform_consumer_bench(
    hyle_platform_consumer_bench_placement)
  if(MSVC)
    target_link_options(hyle_platform_consumer_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hyle_platform_consumer_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR
      "platform consumer placement perturbation is unsupported")
  endif()
  hyle_register_benches(platform
    hyle_platform_consumer_bench
    hyle_platform_consumer_bench_placement)
endif()

hyle_register_benches(platform hyle_platform_bench)
