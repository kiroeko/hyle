hyle_add_header_self_contained_test(compute_route_cuda
  HEADER_DIR compute HEADERS cuda.h)
if(HYLE_COMPUTE_CUDA_ENABLED)
  hyle_add_test(compute_route_cuda SOURCES context_test.cc ggml_test.cc
    DEPS hyle_compute_backend_cuda)
endif()
