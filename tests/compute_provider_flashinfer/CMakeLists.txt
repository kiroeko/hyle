hyle_add_header_self_contained_test(compute_provider_flashinfer
  HEADER_DIR compute/providers HEADERS flashinfer.h)
if(_HYLE_COMPUTE_HAVE_CUDA AND CMAKE_SYSTEM_NAME MATCHES "^(Windows|Linux)$")
  hyle_add_test(compute_provider_flashinfer SOURCES flashinfer_provider_test.cc failure_test.cc native_cache_test.cc probability_precision_test.cc ragged_mla_test.cc gqa16_test.cc
    DEPS hyle_compute_backend_cuda hyle_compute_batch hyle_simd hyle_graph_batch hyle_compute_route_cuda)
  if(HYLE_STATIC AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(hyle_compute_provider_flashinfer_test PRIVATE
      HYLE_FLASHINFER_FAULT_INJECTION=1)
    target_link_options(hyle_compute_provider_flashinfer_test PRIVATE
      LINKER:--wrap=cudaFuncSetAttribute LINKER:--wrap=cudaGraphLaunch)
  endif()
endif()
