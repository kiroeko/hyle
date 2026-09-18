hyle_add_header_self_contained_test(compute_route_vulkan
  HEADER_DIR compute HEADERS vulkan.h detail/vulkan_plan.h)
if(HYLE_COMPUTE_VULKAN_ENABLED)
  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/affine.spv"
    COMMAND "${HYLE_VULKAN_GLSLANG}" -V --target-env vulkan1.2 -S comp
      -o "${CMAKE_CURRENT_BINARY_DIR}/affine.spv" "${CMAKE_CURRENT_SOURCE_DIR}/affine.comp"
    DEPENDS affine.comp VERBATIM)
  hyle_add_test(compute_route_vulkan SOURCES native_test.cc lifecycle_test.cc tensor_test.cc view_test.cc gate_norm_test.cc
    "${CMAKE_CURRENT_BINARY_DIR}/affine.spv")
  target_compile_definitions(hyle_compute_route_vulkan_test PRIVATE
    HYLE_VULKAN_TEST_SHADER="${CMAKE_CURRENT_BINARY_DIR}/affine.spv")
  target_include_directories(hyle_compute_route_vulkan_test PRIVATE
    "${HYLE_SOURCE_DIR}/src/compute_route_vulkan"
    "${HYLE_SOURCE_DIR}/src/compute")
endif()
