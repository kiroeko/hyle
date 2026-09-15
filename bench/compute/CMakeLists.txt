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
