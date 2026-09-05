if(WIN32 AND HYLE_STATIC)
  function(_hyle_configure_platform_host_bench target)
    target_link_libraries(${target} PRIVATE hyle_platform nanobench)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_apply_warnings(${target})
  endfunction()

  add_executable(hyle_platform_host_bench host_bench.cc)
  _hyle_configure_platform_host_bench(hyle_platform_host_bench)

  add_executable(hyle_platform_host_bench_placement host_bench.cc)
  _hyle_configure_platform_host_bench(hyle_platform_host_bench_placement)
  if(MSVC)
    target_link_options(hyle_platform_host_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hyle_platform_host_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR
      "platform host placement perturbation is unsupported")
  endif()

  hyle_register_benches(platform
    hyle_platform_host_bench
    hyle_platform_host_bench_placement)
endif()
