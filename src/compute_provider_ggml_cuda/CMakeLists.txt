if(NOT HYLE_COMPUTE_CUDA_ENABLED)
  hyle_add_module(compute_provider_ggml_cuda OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_provider_ggml_cuda PROPERTY HYLE_INTERNAL_ONLY_ABSENT TRUE)
  return()
endif()
hyle_add_module(compute_provider_ggml_cuda SOURCES ggml_cuda_provider.cc tensor_program.cc)
target_include_directories(hyle_compute_provider_ggml_cuda PRIVATE
  "${HYLE_SOURCE_DIR}/src/compute"
  "${HYLE_SOURCE_DIR}/src/compute_backend_cuda"
  "${HYLE_SOURCE_DIR}/src/compute_ggml_runtime")
target_link_libraries(hyle_compute_provider_ggml_cuda PRIVATE ggml CUDA::cudart)
if(TARGET HyleCudaSdk::cublas_runtime)
  # ggml is static here; its Windows cuBLAS imports belong to this DLL.
  target_link_libraries(hyle_compute_provider_ggml_cuda PRIVATE HyleCudaSdk::cublas_runtime)
endif()
target_compile_definitions(hyle_compute_provider_ggml_cuda PUBLIC HYLE_COMPUTE_GGML_CUDA=1)
target_compile_definitions(hyle_compute_provider_ggml_cuda PRIVATE
  "HYLE_GGML_CUDA_PROVIDER_VERSION=\"${HYLE_GGML_CUDA_PROVIDER_VERSION}\"")
