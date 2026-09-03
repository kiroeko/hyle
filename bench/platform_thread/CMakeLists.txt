if(HYLE_STATIC AND HYLE_BENCH_TBB)
  function(_hyle_configure_platform_thread_consumer_bench target)
    target_link_libraries(${target} PRIVATE hyle_platform_thread nanobench
                          TBB::tbb)
    target_include_directories(${target} SYSTEM PRIVATE
      "${HYLE_TBB_DIR}/src/tbb")
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_stage_tbb_benchmark_runtime(${target})
    hyle_apply_warnings(${target})
  endfunction()

  add_executable(hyle_platform_thread_consumer_bench
    thread_consumer_bench.cc)
  _hyle_configure_platform_thread_consumer_bench(
    hyle_platform_thread_consumer_bench)

  add_executable(hyle_platform_thread_consumer_bench_placement
    thread_consumer_bench.cc)
  _hyle_configure_platform_thread_consumer_bench(
    hyle_platform_thread_consumer_bench_placement)
  if(MSVC)
    target_link_options(hyle_platform_thread_consumer_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hyle_platform_thread_consumer_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR
      "platform_thread consumer placement perturbation is unsupported")
  endif()

  hyle_register_benches(platform_thread
    hyle_platform_thread_consumer_bench
    hyle_platform_thread_consumer_bench_placement)
endif()
