hyle_add_header_self_contained_test(graph_route_cuda
  HEADER_DIR graph HEADERS cuda.h)
if(HYLE_COMPUTE_CUDA_ENABLED)
  hyle_add_test(graph_route_cuda SOURCES graph_test.cc tensor_test.cc norm_test.cc
    reduction_test.cc topk_test.cc
    DEPS hyle_compute_backend_cuda)
endif()
