hyle_add_header_self_contained_test(compute_provider_cublas
  HEADER_DIR compute/providers HEADERS cublas.h)
if(HYLE_COMPUTE_CUDA_ENABLED)
  hyle_add_test(compute_provider_cublas SOURCES cublas_provider_test.cc
    DEPS hyle_compute_backend_cuda)
  # Native owned-import fixtures allocate an original driver buffer directly.
  target_link_libraries(hyle_compute_provider_cublas_test PRIVATE CUDA::cuda_driver)
endif()
