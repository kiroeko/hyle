if(_HYLE_COMPUTE_HAVE_CUDA)
  hyle_audit_in_tree_dependencies(sglang_activation deepseek_fp8 deepseek_quantization mlx_cuda_quant)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 80 89-real)
  endif()
  enable_language(CUDA)
  hyle_add_module(compute_provider_cuda SOURCES
    collectives_provider.cc collectives.cu sampling_provider.cc sampling_kernel.cu
    cuda_provider.cc native_kernel.cu sglang_kernel.cu sglang_norm_kernel.cu
    artifact_provider.cc artifact_kernel.cu
    router_provider.cc sglang_router_kernel.cu fp8_provider.cc grouped_fp8_provider.cc
    deepseek_fp8_provider.cc deepseek_grouped_fp4_provider.cc grouped_fp4_f32_kernel.cu
    tensor_provider.cc tensor_kernel.cu norm_provider.cc norm_kernel.cu
    nvfp4_provider.cc nvfp4_kernel.cu
    blockwise_w4a8_provider.cc blockwise_w4a8_kernel.cu
    reduction_provider.cc reduction_kernel.cu
    concat_provider.cc concat_kernel.cu
    causal_conv_provider.cc causal_conv_kernel.cu
    sequence_provider.cc sequence_kernel.cu
    scatter_provider.cc scatter_kernel.cu
    row_cache_provider.cc row_cache_kernel.cu
    float8_quantize_provider.cc float8_quantize_kernel.cu
    row_cache_indexer_provider.cc row_cache_indexer_kernel.cu)
  set_target_properties(hyle_compute_provider_cuda PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Shared)
  # CUDA language variables belong to this enabled-language directory, not
  # necessarily to the sibling compiled route directory.
  set_property(TARGET hyle_compute_provider_cuda PROPERTY
    HYLE_CUDA_COMPILER_VERSION "${CMAKE_CUDA_COMPILER_VERSION}")
  target_include_directories(hyle_compute_provider_cuda PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../compute"
    "${CMAKE_CURRENT_SOURCE_DIR}/../compute_backend_cuda"
    "${PROJECT_SOURCE_DIR}/third_party/sglang_activation"
    "${PROJECT_SOURCE_DIR}/third_party/deepseek_fp8"
    "${PROJECT_SOURCE_DIR}/third_party/deepseek_quantization"
    "${PROJECT_SOURCE_DIR}/third_party/mlx_cuda_quant")
  target_link_libraries(hyle_compute_provider_cuda PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::cublasLt)
  if(TARGET HyleCudaSdk::cublasLt_runtime)
    target_link_libraries(hyle_compute_provider_cuda PRIVATE HyleCudaSdk::cublasLt_runtime)
  endif()
  if(WIN32 AND (MSVC OR CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC"))
    # Same CUDA 13.3 driver-loader CRT directive as compute_backend_cuda.
    target_link_options(hyle_compute_provider_cuda PRIVATE /NODEFAULTLIB:LIBCMT)
    if(HYLE_STATIC)
      target_link_options(hyle_compute_provider_cuda INTERFACE /NODEFAULTLIB:LIBCMT)
    endif()
  endif()
  target_compile_options(hyle_compute_provider_cuda PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
  if(NOT MSVC AND NOT HYLE_SANITIZER STREQUAL "off")
    # Sanitizer flags instrument the CUDA TU's host adapter, not device code.
    # nvcc forwards the ordinary HYLE host flags during compilation and link.
    target_compile_options(hyle_compute_provider_cuda PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:--forward-unknown-to-host-compiler>")
    target_link_options(hyle_compute_provider_cuda PRIVATE
      "$<$<LINK_LANGUAGE:CUDA>:--forward-unknown-to-host-compiler>")
  endif()
  hyle_target_compute_capability(hyle_compute_provider_cuda cuda)
  if(MSVC)
    # SDK collectives and CUDA atomics include CCCL headers, which require the conforming
    # preprocessor in nvcc's MSVC host preprocessing phase.
    set_property(SOURCE collectives.cu sampling_kernel.cu norm_kernel.cu reduction_kernel.cu scatter_kernel.cu nvfp4_kernel.cu blockwise_w4a8_kernel.cu APPEND PROPERTY
      COMPILE_OPTIONS -Xcompiler=/Zc:preprocessor)
    target_compile_options(hyle_compute_provider_cuda PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8,/wd4211>")
  endif()
  # CCCL is supplied by the user-installed CUDA SDK, not a FetchContent or
  # vendored source dependency. Ship its original aggregate license with the
  # instantiated device code in both SDK forms; the source form includes it.
  # Unlike locked source dependencies, this license belongs to the installed
  # machine SDK. Respect both install opt-in and the production package closure.
  if(HYLE_INSTALL AND "compute_provider_cuda" IN_LIST HYLE_PRODUCTION_COMPONENT_LIST)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/CCCL-LICENSE.txt"
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Hyle"
      RENAME cuda-cccl-LICENSE.txt COMPONENT Hyle_Runtime)
  endif()
  hyle_register_package_dependency(
    NAME sglang_activation
    OWNER compute_provider_cuda
    LOCKED_DEPENDENCIES sglang_activation
    RUNTIME_FILE_SPECS
      "${PROJECT_SOURCE_DIR}/third_party/sglang_activation/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|sglang-activation-LICENSE"
      "${PROJECT_SOURCE_DIR}/third_party/sglang_activation/NOTICE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|sglang-activation-NOTICE"
      "${PROJECT_SOURCE_DIR}/third_party/sglang_activation/TRITON-LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|sglang-triton-LICENSE"
      "${PROJECT_SOURCE_DIR}/third_party/sglang_activation/FLASHINFER-LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|sglang-flashinfer-LICENSE")
  hyle_register_package_dependency(
    NAME deepseek_fp8
    OWNER compute_provider_cuda
    LOCKED_DEPENDENCIES deepseek_fp8
    RUNTIME_FILE_SPECS
      "${PROJECT_SOURCE_DIR}/third_party/deepseek_fp8/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|deepseek-fp8-LICENSE"
      "${PROJECT_SOURCE_DIR}/third_party/deepseek_fp8/TRITON-LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|deepseek-fp8-TRITON-LICENSE")
  hyle_register_package_dependency(
    NAME deepseek_quantization
    OWNER compute_provider_cuda
    LOCKED_DEPENDENCIES deepseek_quantization
    RUNTIME_FILE_SPECS
      "${PROJECT_SOURCE_DIR}/third_party/deepseek_quantization/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|deepseek-quantization-LICENSE"
      "${PROJECT_SOURCE_DIR}/third_party/deepseek_quantization/TILELANG-LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|deepseek-quantization-TILELANG-LICENSE"
      "${PROJECT_SOURCE_DIR}/third_party/deepseek_quantization/CUTLASS-LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|deepseek-quantization-CUTLASS-LICENSE")
  # Only the native packed kernels use this codec; the optional ATen adapter
  # has no codec dependency. Keep its original installed license filenames.
  hyle_register_package_dependency(
    NAME mlx_cuda_quant OWNER compute_provider_cuda
    LOCKED_DEPENDENCIES mlx_cuda_quant
    RUNTIME_FILE_SPECS
      "${HYLE_SOURCE_DIR}/third_party/mlx_cuda_quant/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|mlx_cuda_quant-LICENSE"
      "${HYLE_SOURCE_DIR}/third_party/mlx_cuda_quant/NOTICE.md|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|mlx_cuda_quant-NOTICE.md")
else()
  hyle_add_module(compute_provider_cuda OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_provider_cuda PROPERTY
    HYLE_INTERNAL_ONLY_ABSENT TRUE)
endif()
