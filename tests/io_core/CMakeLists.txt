hyle_add_test(io_core SOURCES proactor_test.cc)
hyle_add_test(io_core NAME endpoint SOURCES endpoint_test.cc)
hyle_add_test(io_core NAME buffer SOURCES buffer_test.cc)
if(TARGET hyle_io_core_test)
  target_include_directories(hyle_io_core_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src/io_core")
endif()
if(WIN32 AND TARGET hyle_io_core_test)
  # The io test itself calls Winsock directly (WSASocketW/bind/getsockname/... in
  # the IOCP TCP round-trip test), so it needs ws2_32 linked into the test exe.
  target_link_libraries(hyle_io_core_test PRIVATE ws2_32)
endif()
if(TARGET hyle_sched)
  # io-04 net: the co_await socket layer needs BOTH io (proactor + IoContext) and
  # sched (scheduler drives the reap). Only buildable where sched exists.
  hyle_add_test(io_core NAME io_context SOURCES io_context_test.cc DEPS hyle_sched)
  hyle_add_test(io_core NAME net SOURCES net_test.cc DEPS hyle_sched)
  hyle_add_test(io_core NAME tcp SOURCES tcp_test.cc DEPS hyle_sched)
  hyle_add_test(io_core NAME udp SOURCES udp_test.cc DEPS hyle_sched)
  # io-05 file io + io-09 async DNS / signal_set: co_await file ops, getaddrinfo
  # offload to the blocking pool, and signal waits -- all need io + sched.
  hyle_add_test(io_core NAME file SOURCES file_test.cc DEPS hyle_sched)
  hyle_add_test(io_core NAME file_region SOURCES file_region_test.cc DEPS hyle_sched)
  hyle_add_test(io_core NAME local SOURCES local_test.cc DEPS hyle_sched)
  hyle_add_test(io_core NAME dns SOURCES dns_test.cc DEPS hyle_sched)
  hyle_add_test(io_core NAME signal SOURCES signal_test.cc DEPS hyle_sched)
  hyle_add_test(io_core NAME timeout SOURCES timeout_test.cc DEPS hyle_sched)
  # sched-07 cancellable io: token-bound recv/accept, submit_cancel on cancel.
  hyle_add_test(io_core NAME net_cancel SOURCES net_cancel_test.cc DEPS hyle_sched hyle_time)
  if(WIN32)
    target_link_libraries(hyle_io_core_net_test PRIVATE ws2_32)  # test builds the loopback pair
    target_link_libraries(hyle_io_core_net_cancel_test PRIVATE ws2_32)  # loopback TCP pair
    target_link_libraries(hyle_io_core_tcp_test PRIVATE ws2_32)  # sockaddr / winsock types
    target_link_libraries(hyle_io_core_udp_test PRIVATE ws2_32)  # sockaddr / winsock types
    target_link_libraries(hyle_io_core_file_region_test PRIVATE ws2_32)
    target_link_libraries(hyle_io_core_local_test PRIVATE ws2_32)  # IoContext initializes Winsock
    target_link_libraries(hyle_io_core_dns_test PRIVATE ws2_32)  # getaddrinfo + WSAStartup
    target_link_libraries(hyle_io_core_timeout_test PRIVATE ws2_32)  # loopback TCP pair
  endif()
endif()
