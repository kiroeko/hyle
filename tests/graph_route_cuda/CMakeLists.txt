hyle_add_header_self_contained_test(graph_route_cuda
  HEADER_DIR graph HEADERS cuda.h)
if(HYLE_COMPUTE_CUDA_ENABLED)
  hyle_add_test(graph_route_cuda SOURCES graph_test.cc tensor_test.cc norm_test.cc sglang_primitives_test.cc
    reduction_test.cc topk_test.cc concat_test.cc gather_test.cc scatter_test.cc scan_test.cc
    DEPS hyle_compute_backend_cuda)
endif()
