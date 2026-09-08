if(HYLE_IO_TLS_ENABLED)
  # Same cipher, certificate and payload isolate the HYLE coroutine/memory-BIO
  # integration cost from OpenSSL's common crypto floor.
  add_executable(hyle_io_tls_bench tls_bench.cc)
  target_link_libraries(hyle_io_tls_bench PRIVATE
    hyle_sched hyle_io_tls nanobench OpenSSL::SSL OpenSSL::Crypto)
  hyle_apply_warnings(hyle_io_tls_bench)
  option(HYLE_BENCH_ASIO_TLS
    "build the io_tls consumer benchmark against standalone Asio SSL" OFF)
  if(HYLE_BENCH_ASIO_TLS)
    set(HYLE_ASIO_DIR "${HYLE_SOURCE_DIR}/../asio" CACHE PATH
        "clean standalone Asio checkout used only by TLS benchmarks")
    if(NOT EXISTS "${HYLE_ASIO_DIR}/asio/include/asio.hpp" OR
       NOT EXISTS "${HYLE_ASIO_DIR}/asio/include/asio/ssl.hpp")
      message(FATAL_ERROR
        "HYLE_BENCH_ASIO_TLS=ON requires standalone Asio at "
        "${HYLE_ASIO_DIR}/asio/include")
    endif()
    target_compile_definitions(hyle_io_tls_bench PRIVATE
      HYLE_BENCH_ASIO_TLS ASIO_STANDALONE)
    target_include_directories(hyle_io_tls_bench SYSTEM PRIVATE
      "${HYLE_ASIO_DIR}/asio/include")
  endif()
  if(WIN32)
    target_link_libraries(hyle_io_tls_bench PRIVATE ws2_32)
  endif()
  hyle_register_benches(io_tls hyle_io_tls_bench)
endif()
