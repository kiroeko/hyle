if(HYLE_STATIC)
  function(_hyle_configure_platform_fs_consumer_bench target)
    target_link_libraries(${target} PRIVATE hyle_platform_fs nanobench)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_apply_warnings(${target})
  endfunction()

  add_executable(hyle_platform_fs_consumer_bench fs_consumer_bench.cc)
  _hyle_configure_platform_fs_consumer_bench(
    hyle_platform_fs_consumer_bench)

  add_executable(hyle_platform_fs_consumer_bench_placement
    fs_consumer_bench.cc)
  _hyle_configure_platform_fs_consumer_bench(
    hyle_platform_fs_consumer_bench_placement)
  if(MSVC)
    target_link_options(hyle_platform_fs_consumer_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hyle_platform_fs_consumer_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR
      "platform_fs consumer placement perturbation is unsupported")
  endif()

  hyle_register_benches(platform_fs
    hyle_platform_fs_consumer_bench
    hyle_platform_fs_consumer_bench_placement)
endif()
