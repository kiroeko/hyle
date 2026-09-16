hyle_add_header_self_contained_test(compute_provider_cublas
  HEADER_DIR compute/providers HEADERS cublas.h)
if(HYLE_COMPUTE_CUDA_ENABLED)
  hyle_add_test(compute_provider_cublas SOURCES cublas_provider_test.cc
    DEPS hyle_compute_backend_cuda)
endif()
