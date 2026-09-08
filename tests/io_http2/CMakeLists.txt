hyle_ensure_module_test_target(io_http2)
hyle_add_header_self_contained_test(io_http2 HEADER_DIR io HEADERS http2.h)

if(HYLE_IO_HTTP2)
  hyle_add_test(io_http2 SOURCES http2_test.cc DEPS hyle_sched)
  if(WIN32)
    target_link_libraries(hyle_io_http2_test PRIVATE ws2_32)
  endif()
endif()
