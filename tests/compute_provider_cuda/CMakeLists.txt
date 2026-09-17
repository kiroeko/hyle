hyle_add_header_self_contained_test(compute_provider_cuda
  HEADER_DIR compute/providers HEADERS cuda.h)
if(HYLE_COMPUTE_CUDA_ENABLED)
  hyle_add_test(compute_provider_cuda SOURCES cuda_provider_test.cc tensor_test.cc gather_test.cc scatter_test.cc
    DEPS hyle_compute_backend_cuda)
endif()
