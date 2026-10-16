hyle_add_header_self_contained_test(compute_route_vulkan
  HEADER_DIR compute HEADERS vulkan.h vulkan_batch.h detail/vulkan_plan.h)
if(HYLE_COMPUTE_VULKAN_ENABLED)
  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/affine.spv"
    COMMAND "${HYLE_VULKAN_GLSLANG}" -V --target-env vulkan1.2 -S comp
      -o "${CMAKE_CURRENT_BINARY_DIR}/affine.spv" "${CMAKE_CURRENT_SOURCE_DIR}/affine.comp"
    DEPENDS affine.comp VERBATIM)
  hyle_add_test(compute_route_vulkan NAME file_stream SOURCES file_stream_test.cc)
  target_include_directories(hyle_compute_route_vulkan_file_stream_test PRIVATE
    "${HYLE_SOURCE_DIR}/src/compute_route_vulkan")
  hyle_stage_runtime(hyle_compute_route_vulkan_file_stream_test)
  hyle_add_test(compute_route_vulkan SOURCES copy_program_test.cc copy_recording_test.cc native_test.cc lifecycle_test.cc tensor_test.cc view_test.cc gate_norm_test.cc concat_test.cc reduction_test.cc provider_bridge_test.cc managed_region_test.cc
    d5_qwen35_profile_test.cc d5_qwen35_execution_test.cc topk_test.cc gather_test.cc scan_test.cc scatter_test.cc recurrent_test.cc
    mxfp4_test.cc gguf_test.cc iree_test.cc mature_tensor_test.cc catalogue_test.cc native_ragged_test.cc native_ragged_cost_test.cc native_cache_test.cc "${CMAKE_CURRENT_BINARY_DIR}/affine.spv")
  target_compile_definitions(hyle_compute_route_vulkan_test PRIVATE
    HYLE_VULKAN_TEST_SHADER="${CMAKE_CURRENT_BINARY_DIR}/affine.spv")
  target_include_directories(hyle_compute_route_vulkan_test PRIVATE
    "${HYLE_SOURCE_DIR}/src/compute_route_vulkan"
    "${HYLE_SOURCE_DIR}/src/compute")
  target_include_directories(hyle_compute_route_vulkan_test SYSTEM PRIVATE
    "${HYLE_GGML_SOURCE_DIR}/src")
  target_link_libraries(hyle_compute_route_vulkan_test PRIVATE
    hyle_compute_provider_ggml_vulkan hyle_compute_ggml_runtime ggml)
  hyle_stage_runtime(hyle_compute_route_vulkan_test)
endif()

if(HYLE_COMPUTE_VULKAN_ENABLED)

hyle_add_test(compute_route_vulkan NAME sampling_consumer
  SOURCES sampling_consumer.cc)
hyle_stage_runtime(hyle_compute_route_vulkan_sampling_consumer_test)
endif()

if(HYLE_COMPUTE_VULKAN_ENABLED)
  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/sampling_math.spv"
    COMMAND "${HYLE_VULKAN_GLSLANG}" -V --target-env vulkan1.2 -S comp
      "-I${HYLE_SOURCE_DIR}/src/compute_route_vulkan/shaders"
      -o "${CMAKE_CURRENT_BINARY_DIR}/sampling_math.spv"
      "${CMAKE_CURRENT_SOURCE_DIR}/sampling_math.comp"
    DEPENDS sampling_math.comp "${HYLE_SOURCE_DIR}/src/compute_route_vulkan/shaders/sampling_math.glsl"
    VERBATIM)
  hyle_add_test(compute_route_vulkan NAME sampling_math
    SOURCES sampling_math_test.cc "${CMAKE_CURRENT_BINARY_DIR}/sampling_math.spv")
  target_compile_definitions(hyle_compute_route_vulkan_sampling_math_test PRIVATE
    HYLE_G08_VULKAN_MATH_SHADER="${CMAKE_CURRENT_BINARY_DIR}/sampling_math.spv")
  hyle_stage_runtime(hyle_compute_route_vulkan_sampling_math_test)
endif()
