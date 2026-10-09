hyle_add_header_self_contained_test(graph_provider_ggml_cuda
  HEADER_DIR graph/providers HEADERS ggml_cuda.h)
if(HYLE_COMPUTE_CUDA_ENABLED)
  hyle_add_test(graph_provider_ggml_cuda SOURCES region_test.cc
    DEPS hyle_graph hyle_compute_backend_cuda hyle_compute_provider_ggml_cuda)
  target_link_libraries(hyle_graph_provider_ggml_cuda_test PRIVATE ggml)
  hyle_stage_runtime(hyle_graph_provider_ggml_cuda_test)
endif()
