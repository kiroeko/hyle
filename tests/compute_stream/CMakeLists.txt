hyle_add_header_self_contained_test(compute_stream HEADER_DIR compute
  HEADERS file_stream.h detail/file_stream_access.h detail/file_stream_copy.h)
hyle_add_test(compute_stream SOURCES file_stream_test.cc DEPS hyle_compute_route_cpu)
