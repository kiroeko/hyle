if(HYLE_IO_TLS_ENABLED)
  # Same cipher, certificate and payload isolate the HYLE coroutine/memory-BIO
  # integration cost from OpenSSL's common crypto floor.
  add_executable(hyle_io_tls_bench tls_bench.cc)
  target_link_libraries(hyle_io_tls_bench PRIVATE
    hyle_sched hyle_io_tls nanobench OpenSSL::SSL OpenSSL::Crypto)
  hyle_apply_warnings(hyle_io_tls_bench)
  if(WIN32)
    target_link_libraries(hyle_io_tls_bench PRIVATE ws2_32)
  endif()
  hyle_register_benches(io_tls hyle_io_tls_bench)
endif()
