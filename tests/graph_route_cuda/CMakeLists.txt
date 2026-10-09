hyle_add_header_self_contained_test(graph_route_cuda
  HEADER_DIR graph HEADERS cuda.h)
if(HYLE_COMPUTE_CUDA_ENABLED)
  hyle_add_test(graph_route_cuda SOURCES graph_test.cc artifact_test.cc d5_qwen35_profile_test.cc d5_qwen35_gguf_fragment_test.cc d5_qwen38_official_test.cc d5_qwen_gguf_vocabulary_test.cc tensor_test.cc norm_test.cc sglang_primitives_test.cc
    reduction_test.cc topk_test.cc concat_test.cc gather_test.cc scatter_test.cc scan_test.cc ggml_test.cc attention_test.cc collectives_test.cc
    DEPS hyle_compute_backend_cuda)
  target_link_libraries(hyle_graph_route_cuda_test PRIVATE ggml)
  target_include_directories(hyle_graph_route_cuda_test PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
  hyle_stage_runtime(hyle_graph_route_cuda_test)
endif()
