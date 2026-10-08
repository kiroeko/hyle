hyle_add_header_self_contained_test(graph_route_apple
  HEADER_DIR graph HEADERS apple.h)
if(HYLE_COMPUTE_APPLE_ENABLED)
  hyle_add_test(graph_route_apple SOURCES graph_test.cc tensor_test.cc reduction_test.cc indexing_test.cc scatter_test.cc recurrent_test.cc model_composition_test.cc d5_qwen35_execution_test.cc mlx_quantized_test.cc)
  target_link_libraries(hyle_graph_route_apple_test PRIVATE ggml)
endif()
