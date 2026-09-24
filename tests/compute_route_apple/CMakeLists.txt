hyle_add_header_self_contained_test(compute_route_apple
  HEADER_DIR compute HEADERS apple.h detail/apple_plan.h)
if(HYLE_COMPUTE_APPLE_ENABLED)
  hyle_add_test(compute_route_apple SOURCES context_test.cc operations_test.cc
    d5_qwen35_execution_test.cc tensor_test.cc reduction_test.cc indexing_test.cc
    scatter_test.cc lifecycle_test.cc recurrent_test.cc)
  target_link_libraries(hyle_compute_route_apple_test PRIVATE ggml)
endif()
