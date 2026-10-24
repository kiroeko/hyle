if(NOT _HYLE_COMPUTE_HAVE_CUDA)
  hyle_add_module(compute_provider_ggml_cuda OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_provider_ggml_cuda PROPERTY HYLE_INTERNAL_ONLY_ABSENT TRUE)
  return()
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES 80 89-real)
endif()
enable_language(CUDA)
hyle_add_module(compute_provider_ggml_cuda SOURCES
  ggml_cuda_provider.cc tensor_program.cc index_guard.cu)
set_target_properties(hyle_compute_provider_ggml_cuda PROPERTIES
  CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Shared)
target_include_directories(hyle_compute_provider_ggml_cuda PRIVATE
  "${HYLE_SOURCE_DIR}/src/compute"
  "${HYLE_SOURCE_DIR}/src/compute_backend_cuda"
  "${HYLE_SOURCE_DIR}/src/compute_ggml_runtime")
target_include_directories(hyle_compute_provider_ggml_cuda SYSTEM PRIVATE
  "${HYLE_GGML_SOURCE_DIR}/src")
target_link_libraries(hyle_compute_provider_ggml_cuda PRIVATE ggml CUDA::cudart)
if(TARGET HyleCudaSdk::cublas_runtime)
  # ggml is static here; its Windows cuBLAS imports belong to this DLL.
  target_link_libraries(hyle_compute_provider_ggml_cuda PRIVATE HyleCudaSdk::cublas_runtime)
endif()
hyle_target_compute_capability(hyle_compute_provider_ggml_cuda ggml_cuda)
target_compile_definitions(hyle_compute_provider_ggml_cuda PRIVATE
  "HYLE_GGML_CUDA_PROVIDER_VERSION=\"${HYLE_GGML_CUDA_PROVIDER_VERSION}\"")
