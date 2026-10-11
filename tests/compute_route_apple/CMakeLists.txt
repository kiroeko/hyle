hyle_add_header_self_contained_test(compute_route_apple
  HEADER_DIR compute HEADERS apple.h detail/apple_plan.h)
if(HYLE_COMPUTE_APPLE_ENABLED)
  hyle_add_test(compute_route_apple NAME file_stream SOURCES file_stream_test.cc)
  target_include_directories(hyle_compute_route_apple_file_stream_test PRIVATE
    "${HYLE_SOURCE_DIR}/src/compute_route_apple")
  hyle_stage_runtime(hyle_compute_route_apple_file_stream_test)
  hyle_add_test(compute_route_apple SOURCES copy_program_test.cc copy_recording_test.cc context_test.cc operations_test.cc
    d5_qwen35_execution_test.cc tensor_test.cc reduction_test.cc indexing_test.cc
    scatter_test.cc lifecycle_test.cc recurrent_test.cc mlx_quantized_test.cc mature_overlap_test.cc)
  target_link_libraries(hyle_compute_route_apple_test PRIVATE ggml)
  target_include_directories(hyle_compute_route_apple_test PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
  target_include_directories(hyle_compute_route_apple_test PRIVATE "${HYLE_SOURCE_DIR}/src/compute_route_apple")
endif()
