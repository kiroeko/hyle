hyle_add_header_self_contained_test(io_websocket
  HEADER_DIR io
  HEADERS websocket.h)
hyle_add_test(io_websocket SOURCES websocket_test.cc DEPS hyle_sched)
target_include_directories(hyle_io_websocket_test PRIVATE
  "${PROJECT_SOURCE_DIR}/src/io_websocket")

if(WIN32)
  target_link_libraries(hyle_io_websocket_test PRIVATE ws2_32)
endif()
