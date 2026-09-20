hyle_add_header_self_contained_test(compute_provider_ggml_cuda
  HEADER_DIR compute/providers HEADERS ggml_cuda.h)
if(HYLE_COMPUTE_CUDA_ENABLED)
  hyle_add_test(compute_provider_ggml_cuda SOURCES provider_test.cc allocation_test.cc
    DEPS hyle_compute_backend_cuda)
  if(HYLE_STATIC AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(hyle_compute_provider_ggml_cuda_test PRIVATE
      HYLE_GGML_CUDA_PROVIDER_FAULTS=1)
    target_link_options(hyle_compute_provider_ggml_cuda_test PRIVATE
      LINKER:--wrap=cudaStreamBeginCapture)
  endif()
  hyle_stage_runtime_dependencies(hyle_compute_provider_ggml_cuda_test)
endif()
