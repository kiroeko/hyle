hyle_add_header_self_contained_test(compute_provider_flashinfer
  HEADER_DIR compute/providers HEADERS flashinfer.h)
if(HYLE_COMPUTE_CUDA_ENABLED AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  hyle_add_test(compute_provider_flashinfer SOURCES flashinfer_provider_test.cc failure_test.cc
    DEPS hyle_compute_backend_cuda)
  if(HYLE_STATIC)
    target_compile_definitions(hyle_compute_provider_flashinfer_test PRIVATE
      HYLE_FLASHINFER_FAULT_INJECTION=1)
    target_link_options(hyle_compute_provider_flashinfer_test PRIVATE
      LINKER:--wrap=cudaFuncSetAttribute)
  endif()
endif()
