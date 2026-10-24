hyle_add_header_self_contained_test(compute_stream_apple
  HEADER_DIR compute HEADERS apple_file_stream.h)
if(_HYLE_COMPUTE_HAVE_APPLE)
  hyle_add_test(compute_stream_apple SOURCES ../compute_route_apple/file_stream_test.cc)
  target_include_directories(hyle_compute_stream_apple_test PRIVATE
    "${HYLE_SOURCE_DIR}/src/compute_route_apple")
  hyle_stage_runtime(hyle_compute_stream_apple_test)
endif()
