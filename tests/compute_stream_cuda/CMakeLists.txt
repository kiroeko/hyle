if(_HYLE_COMPUTE_HAVE_CUDA)
  hyle_add_header_self_contained_test(compute_stream_cuda
    HEADER_DIR compute HEADERS cuda_file_stream.h)
  hyle_add_test(compute_stream_cuda SOURCES ../compute_route_cuda/file_stream_test.cc)
  hyle_stage_runtime(hyle_compute_stream_cuda_test)
endif()
