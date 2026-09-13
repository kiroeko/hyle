if(HYLE_STATIC AND HYLE_COMPUTE_ONEDNN_ENABLED)
  function(_hyle_configure_compute_route_bench target)
    # This product-level route ledger compares two public Compute contexts.
    # The optional provider and SIMD leaf remain separately owned factors.
    target_link_libraries(${target} PRIVATE
      hyle_compute_provider_onednn hyle_compute_provider_native
      hyle_simd nanobench)
    set_property(TARGET ${target}
      PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_apply_warnings(${target})
  endfunction()

  add_executable(hyle_compute_route_bench route_p1_bench.cc)
  _hyle_configure_compute_route_bench(hyle_compute_route_bench)

  add_executable(hyle_compute_route_bench_placement route_p1_bench.cc)
  _hyle_configure_compute_route_bench(hyle_compute_route_bench_placement)
  if(MSVC)
    target_link_options(hyle_compute_route_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hyle_compute_route_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR "Compute route placement perturbation is unsupported")
  endif()

  hyle_register_benches(compute
    hyle_compute_route_bench
    hyle_compute_route_bench_placement)
endif()

if(HYLE_STATIC AND HYLE_COMPUTE_CUDA_ENABLED AND
   TARGET hyle_compute_backend_cuda)
  function(_hyle_configure_compute_cuda_runtime_bench target)
    target_link_libraries(${target} PRIVATE
      hyle_compute_backend_cuda nanobench)
    set_property(TARGET ${target}
      PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_apply_warnings(${target})
  endfunction()

  add_executable(hyle_compute_cuda_runtime_bench cuda_runtime_p1_bench.cc)
  _hyle_configure_compute_cuda_runtime_bench(
    hyle_compute_cuda_runtime_bench)

  add_executable(hyle_compute_cuda_runtime_bench_placement
    cuda_runtime_p1_bench.cc)
  _hyle_configure_compute_cuda_runtime_bench(
    hyle_compute_cuda_runtime_bench_placement)
  if(MSVC)
    target_link_options(hyle_compute_cuda_runtime_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hyle_compute_cuda_runtime_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR
      "Compute CUDA runtime placement perturbation is unsupported")
  endif()

  hyle_register_benches(compute
    hyle_compute_cuda_runtime_bench
    hyle_compute_cuda_runtime_bench_placement)
endif()
