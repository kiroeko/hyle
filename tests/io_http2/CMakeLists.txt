hyle_ensure_module_test_target(io_http2)
hyle_add_header_self_contained_test(io_http2 HEADER_DIR io HEADERS http2.h)
hyle_add_test(io_http2 NAME feature SOURCES feature_test.cc)

if(HYLE_IO_HTTP2)
  hyle_add_test(io_http2 SOURCES http2_test.cc DEPS hyle_sched)
  if(WIN32)
    target_link_libraries(hyle_io_http2_test PRIVATE ws2_32)
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # The ordinary suite covers the preferred io_uring path.  Replay the
    # bounded-buffer flow-control case in an isolated process on the readiness
    # fallback so a write/write progress stall cannot hide behind large kernel
    # socket buffers or a different backend.
    add_test(NAME hyle_io_http2_epoll_duplex_progress_test
      COMMAND hyle_io_http2_test
        --gtest_filter=IoHttp2.DuplexProgressSurvivesSmallSocketAndFlowControl)
    set_tests_properties(hyle_io_http2_epoll_duplex_progress_test PROPERTIES
      ENVIRONMENT "HYLE_IO_FORCE_EPOLL=1"
      LABELS "io_http2"
      TIMEOUT 20)
  endif()
endif()
