hyle_add_test(io_http SOURCES http_test.cc DEPS hyle_sched)
hyle_add_test(io_http NAME cancel SOURCES http_cancel_test.cc DEPS hyle_sched)
hyle_add_header_self_contained_test(io_http HEADER_DIR io HEADERS http.h)

if(WIN32)
  target_link_libraries(hyle_io_http_test PRIVATE ws2_32)
endif()
