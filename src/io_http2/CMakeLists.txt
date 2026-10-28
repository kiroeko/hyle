hyle_add_module(io_http2 SOURCES feature.cc http2.cc)

target_link_libraries(hyle_io_http2 PRIVATE nghttp2)
target_compile_definitions(hyle_io_http2 PUBLIC HYLE_IO_HTTP2)
# MSVC has no POSIX ssize_t. Keep nghttp2's modern nghttp2_ssize API while
# hiding only its deprecated ssize_t overloads.
if(WIN32)
  target_compile_definitions(hyle_io_http2 PRIVATE NGHTTP2_NO_SSIZE_T)
endif()
