hyle_add_header_self_contained_test(graph_route_vulkan
  HEADER_DIR graph HEADERS vulkan.h)
if(HYLE_COMPUTE_VULKAN_ENABLED)
  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/affine.spv"
    COMMAND "${HYLE_VULKAN_GLSLANG}" -V --target-env vulkan1.2 -S comp
      -o "${CMAKE_CURRENT_BINARY_DIR}/affine.spv"
      "${HYLE_SOURCE_DIR}/tests/compute_route_vulkan/affine.comp"
    DEPENDS "${HYLE_SOURCE_DIR}/tests/compute_route_vulkan/affine.comp" VERBATIM)
  hyle_add_test(graph_route_vulkan SOURCES graph_test.cc tensor_test.cc view_test.cc gate_norm_test.cc concat_test.cc reduction_test.cc managed_region_test.cc d5_qwen35_execution_test.cc
    topk_test.cc gather_test.cc scan_test.cc scatter_test.cc recurrent_test.cc
    mxfp4_test.cc gguf_test.cc iree_test.cc mature_tensor_test.cc catalogue_test.cc "${CMAKE_CURRENT_BINARY_DIR}/affine.spv")
  target_compile_definitions(hyle_graph_route_vulkan_test PRIVATE
    HYLE_VULKAN_TEST_SHADER="${CMAKE_CURRENT_BINARY_DIR}/affine.spv")
  target_link_libraries(hyle_graph_route_vulkan_test PRIVATE
    hyle_compute_provider_ggml_vulkan hyle_compute_ggml_runtime ggml)
endif()
