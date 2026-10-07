hyle_add_header_self_contained_test(compute_route_cuda
  HEADER_DIR compute HEADERS cuda.h)
if(HYLE_COMPUTE_CUDA_ENABLED)
  hyle_add_test(compute_route_cuda SOURCES context_test.cc ggml_test.cc
    d5_qwen35_profile_test.cc d5_qwen35_execution_test.cc
    d5_qwen_gguf_vocabulary_test.cc
    d5_qwen38_official_test.cc
    d5_qwen35_gguf_fragment_test.cc
    DEPS hyle_compute_backend_cuda)
  target_link_libraries(hyle_compute_route_cuda_test PRIVATE ggml)
  target_include_directories(hyle_compute_route_cuda_test PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
  hyle_stage_runtime_dependencies(hyle_compute_route_cuda_test)
endif()
