if(HYLE_STATIC AND HYLE_COMPUTE_CUDA_ENABLED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 80 89-real)
  endif()
  enable_language(CUDA)
  # The peer changes only the actual activation implementation. It retains the
  # same HYLE Provider ownership, errors, admission and Completion contract.
  add_library(hyle_cuda_sglang_reference STATIC
    reference/reference_provider.cc reference/native_kernel.cu)
  set_target_properties(hyle_cuda_sglang_reference PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "80;89-real"
    INTERPROCEDURAL_OPTIMIZATION FALSE)
  target_include_directories(hyle_cuda_sglang_reference PRIVATE
    "${PROJECT_SOURCE_DIR}/src/compute"
    "${PROJECT_SOURCE_DIR}/src/compute_backend_cuda")
  target_link_libraries(hyle_cuda_sglang_reference PRIVATE
    hyle_compute hyle_compute_backend_cuda CUDA::cudart)
  target_compile_options(hyle_cuda_sglang_reference PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
  if(MSVC)
    target_compile_options(hyle_cuda_sglang_reference PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8,/wd4211>")
  endif()

  # Both owning fragments use the same fixture and peer, with separate targets.
  function(hyle_configure_cuda_ffn_bench target owner graph_mode placement)
    target_compile_definitions(${target} PRIVATE
      HYLE_CUDA_GRAPH_BENCH=${graph_mode})
    target_link_libraries(${target} PRIVATE
      hyle_${owner} hyle_compute_route_cuda hyle_compute_backend_cuda
      hyle_compute_provider_cublas hyle_compute_provider_cuda
      hyle_cuda_sglang_reference CUDA::cudart nanobench)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_apply_warnings(${target})
    if(placement)
      if(MSVC)
        target_link_options(${target} PRIVATE /FUNCTIONPADMIN:64)
      elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -falign-functions=64)
      else()
        message(FATAL_ERROR "CUDA route placement perturbation is unsupported")
      endif()
    endif()
  endfunction()

  add_executable(hyle_compute_route_cuda_bench route_p1_bench.cc)
  hyle_configure_cuda_ffn_bench(
    hyle_compute_route_cuda_bench compute_route_cuda 0 FALSE)
  add_executable(hyle_compute_route_cuda_bench_placement route_p1_bench.cc)
  hyle_configure_cuda_ffn_bench(
    hyle_compute_route_cuda_bench_placement compute_route_cuda 0 TRUE)
  hyle_register_benches(compute_route_cuda
    hyle_compute_route_cuda_bench hyle_compute_route_cuda_bench_placement)
endif()
