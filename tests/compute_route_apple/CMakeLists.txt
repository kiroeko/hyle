hyle_add_header_self_contained_test(compute_route_apple
  HEADER_DIR compute HEADERS apple.h apple_batch.h apple_row_cache.h detail/apple_plan.h)
if(_HYLE_COMPUTE_HAVE_APPLE)
  hyle_add_test(compute_route_apple NAME gdn_output_scale
    SOURCES ../compute_provider_native/gdn_output_scale_test.cc)
  target_compile_definitions(hyle_compute_route_apple_gdn_output_scale_test PRIVATE HYLE_GDN_SCALE_APPLE=1)
  hyle_stage_runtime(hyle_compute_route_apple_gdn_output_scale_test)
  hyle_add_test(compute_route_apple NAME row_cache_paged SOURCES ../compute_route_vulkan/row_cache_paged_test.cc DEPS hyle_compute_batch)
  target_compile_definitions(hyle_compute_route_apple_row_cache_paged_test PRIVATE HYLE_ROW_CACHE_PAGED_APPLE=1)
  hyle_add_test(compute_route_apple NAME packed_consumer SOURCES packed_resident_consumer.cc)
  hyle_stage_runtime(hyle_compute_route_apple_packed_consumer_test)
  hyle_add_test(compute_route_apple SOURCES copy_program_test.cc copy_recording_test.cc context_test.cc operations_test.cc
    d5_qwen35_execution_test.cc fp8_tile_test.cc tensor_test.cc attention_test.cc reduction_test.cc indexing_test.cc
    scatter_test.cc lifecycle_test.cc late_input_test.cc recurrent_test.cc mlx_quantized_test.cc mature_overlap_test.cc native_ragged_test.cc native_cache_test.cc)
  target_link_libraries(hyle_compute_route_apple_test PRIVATE ggml)
  target_include_directories(hyle_compute_route_apple_test PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
  target_include_directories(hyle_compute_route_apple_test PRIVATE "${HYLE_SOURCE_DIR}/src/compute_route_apple")
endif()

if(_HYLE_COMPUTE_HAVE_APPLE)

hyle_add_test(compute_route_apple NAME sampling_consumer
  SOURCES sampling_consumer.cc)
hyle_stage_runtime(hyle_compute_route_apple_sampling_consumer_test)
endif()

if(_HYLE_COMPUTE_HAVE_APPLE)
  hyle_add_test(compute_route_apple NAME sampling_math SOURCES sampling_math_test.cc)
  hyle_stage_runtime(hyle_compute_route_apple_sampling_math_test)
endif()

if(_HYLE_COMPUTE_HAVE_APPLE AND TARGET hyle_graph_route_apple)
  hyle_add_test(compute_route_apple NAME grouped_shared_activation
    SOURCES grouped_shared_activation/main.cc DEPS hyle_graph_route_apple)
  hyle_stage_runtime(hyle_compute_route_apple_grouped_shared_activation_test)
  hyle_add_test(compute_route_apple NAME gguf_nvfp4
    SOURCES gguf_nvfp4_test.cc DEPS hyle_graph_route_apple)
  hyle_stage_runtime(hyle_compute_route_apple_gguf_nvfp4_test)
  hyle_add_test(compute_route_apple NAME row_cache_padded
    SOURCES ../compute_route_vulkan/row_cache_padded_test.cc
    DEPS hyle_graph_route_apple hyle_compute_batch hyle_simd)
  target_compile_definitions(hyle_compute_route_apple_row_cache_padded_test PRIVATE HYLE_NATIVE_ROW_PADDED_APPLE=1)
  hyle_stage_runtime(hyle_compute_route_apple_row_cache_padded_test)
  hyle_add_test(compute_route_apple NAME row_cache_indexer
    SOURCES ../compute_route_vulkan/row_cache_indexer_test.cc DEPS hyle_graph_route_apple hyle_simd)
  target_compile_definitions(hyle_compute_route_apple_row_cache_indexer_test PRIVATE HYLE_ROW_INDEXER_APPLE=1)
  hyle_stage_runtime(hyle_compute_route_apple_row_cache_indexer_test)
  hyle_add_test(compute_route_apple NAME gather_elements
    SOURCES gather_elements_test.cc DEPS hyle_graph_route_apple hyle_simd)
  hyle_stage_runtime(hyle_compute_route_apple_gather_elements_test)
  hyle_add_test(compute_route_apple NAME row_cache
    SOURCES row_cache_test.cc DEPS hyle_graph_route_apple hyle_simd)
  hyle_stage_runtime(hyle_compute_route_apple_row_cache_test)
endif()
