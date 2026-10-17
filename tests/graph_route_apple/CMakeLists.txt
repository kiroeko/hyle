hyle_add_header_self_contained_test(graph_route_apple
  HEADER_DIR graph HEADERS apple.h)
if(_HYLE_COMPUTE_HAVE_APPLE)
  hyle_add_test(graph_route_apple SOURCES graph_test.cc lease_binding_test.cc tensor_test.cc reduction_test.cc indexing_test.cc scatter_test.cc recurrent_test.cc model_composition_test.cc d5_qwen35_execution_test.cc mlx_quantized_test.cc)
  target_link_libraries(hyle_graph_route_apple_test PRIVATE ggml)
  target_include_directories(hyle_graph_route_apple_test PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
  hyle_add_test(graph_route_apple NAME packed_consumer
    SOURCES packed_resident_consumer.cc)
  hyle_stage_runtime(hyle_graph_route_apple_packed_consumer_test)
endif()

if(_HYLE_COMPUTE_HAVE_APPLE)

hyle_add_test(graph_route_apple NAME sampling_consumer
  SOURCES ../compute_route_apple/sampling_consumer.cc)
target_compile_definitions(hyle_graph_route_apple_sampling_consumer_test PRIVATE HYLE_G08_GRAPH=1)
hyle_stage_runtime(hyle_graph_route_apple_sampling_consumer_test)
endif()
