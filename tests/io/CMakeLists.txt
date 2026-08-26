hyle_add_test(io SOURCES proactor_test.cc)
if(WIN32 AND TARGET hyle_io_test)
  # The io test itself calls Winsock directly (WSASocketW/bind/getsockname/... in
  # the IOCP TCP round-trip test), so it needs ws2_32 linked into the test exe.
  target_link_libraries(hyle_io_test PRIVATE ws2_32)
endif()
if(TARGET hyle_sched)
  # io-04 net: the co_await socket layer needs BOTH io (proactor + IoContext) and
  # sched (scheduler drives the reap). Only buildable where sched exists.
  hyle_add_test(io NAME net SOURCES net_test.cc DEPS hyle_sched)
  hyle_add_test(io NAME tcp SOURCES tcp_test.cc DEPS hyle_sched)
  hyle_add_test(io NAME udp SOURCES udp_test.cc DEPS hyle_sched)
  # io-05 file io + io-09 async DNS / signal_set: co_await file ops, getaddrinfo
  # offload to the blocking pool, and signal waits -- all need io + sched.
  hyle_add_test(io NAME file SOURCES file_test.cc DEPS hyle_sched)
  hyle_add_test(io NAME dns SOURCES dns_test.cc DEPS hyle_sched)
  hyle_add_test(io NAME signal SOURCES signal_test.cc DEPS hyle_sched)
  hyle_add_test(io NAME timeout SOURCES timeout_test.cc DEPS hyle_sched hyle_time)
  # io-10 HTTP/1.1: server+client integration over loopback TCP (borrowed parser).
  hyle_add_test(io NAME http SOURCES http_test.cc DEPS hyle_sched)
  if(HYLE_IO_HTTP2)
    # io-11: real h2c loopback with two interleaved streams and reverse-order
    # responses (HPACK + multiplex + flow-control session path).
    hyle_add_test(io NAME http2 SOURCES http2_test.cc DEPS hyle_sched)
  endif()
  # sched-07 cancellable io: token-bound recv/accept, submit_cancel on cancel.
  hyle_add_test(io NAME net_cancel SOURCES net_cancel_test.cc DEPS hyle_sched hyle_time)
  # sched-07 consumer: HTTP request fan-out connected-cancel (request -> subtask tree).
  hyle_add_test(io NAME http_cancel SOURCES http_cancel_test.cc DEPS hyle_sched)
  # io-08 TLS: only when OpenSSL is available (matches the io module's optional TLS).
  # The test itself calls SSL_CTX_set_verify (client skips verify for the self-signed
  # cert), so it links OpenSSL directly too.
  if(HYLE_IO_TLS_ENABLED)
    hyle_add_test(io NAME tls SOURCES tls_test.cc DEPS hyle_sched)
    target_link_libraries(hyle_io_tls_test PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    if(WIN32)
      target_link_libraries(hyle_io_tls_test PRIVATE ws2_32)
    endif()
  endif()
  if(WIN32)
    target_link_libraries(hyle_io_net_test PRIVATE ws2_32)  # test builds the loopback pair
    target_link_libraries(hyle_io_net_cancel_test PRIVATE ws2_32)  # loopback TCP pair
    target_link_libraries(hyle_io_tcp_test PRIVATE ws2_32)  # sockaddr / winsock types
    target_link_libraries(hyle_io_udp_test PRIVATE ws2_32)  # sockaddr / winsock types
    target_link_libraries(hyle_io_dns_test PRIVATE ws2_32)  # getaddrinfo + WSAStartup
    target_link_libraries(hyle_io_timeout_test PRIVATE ws2_32)  # loopback TCP pair
    if(HYLE_IO_HTTP2)
      target_link_libraries(hyle_io_http2_test PRIVATE ws2_32)
    endif()
  endif()
endif()
