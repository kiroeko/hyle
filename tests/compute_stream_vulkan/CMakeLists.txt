hyle_add_header_self_contained_test(compute_stream_vulkan
  HEADER_DIR compute HEADERS vulkan_file_stream.h)
if(_HYLE_COMPUTE_HAVE_VULKAN)
  hyle_add_test(compute_stream_vulkan SOURCES ../compute_route_vulkan/file_stream_test.cc)
  target_include_directories(hyle_compute_stream_vulkan_test PRIVATE
    "${HYLE_SOURCE_DIR}/src/compute_route_vulkan")
  hyle_stage_runtime(hyle_compute_stream_vulkan_test)
endif()
