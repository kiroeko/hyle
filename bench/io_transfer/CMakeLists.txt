if(HYLE_BUILD_IO_TRANSFER)
  option(HYLE_BENCH_CURL
    "build io_transfer P1 benchmarks against a source-pinned libcurl" OFF)
  set(HYLE_CURL_DIR "" CACHE PATH
    "clean source-pinned curl checkout used only by io_transfer P1")
else()
  set(HYLE_BENCH_CURL OFF)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND HYLE_STATIC AND HYLE_BENCH_CURL)
  if(NOT IS_DIRECTORY "${HYLE_CURL_DIR}" OR
     NOT EXISTS "${HYLE_CURL_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "HYLE_BENCH_CURL requires -DHYLE_CURL_DIR=<clean curl checkout>")
  endif()

  # This is an explicit benchmark-only peer build. Keep the ordinary HYLE
  # product graph free of libcurl and build one exact static public API route.
  set(BUILD_CURL_EXE OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(BUILD_LIBCURL_DOCS OFF CACHE BOOL "" FORCE)
  set(BUILD_MISC_DOCS OFF CACHE BOOL "" FORCE)
  set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(CURL_DISABLE_INSTALL ON CACHE BOOL "" FORCE)
  set(CURL_ENABLE_SSL OFF CACHE BOOL "" FORCE)
  set(CURL_USE_PKGCONFIG OFF CACHE BOOL "" FORCE)
  set(CURL_ZLIB OFF CACHE STRING "" FORCE)
  set(CURL_BROTLI OFF CACHE STRING "" FORCE)
  set(CURL_ZSTD OFF CACHE STRING "" FORCE)
  set(CURL_USE_LIBPSL OFF CACHE BOOL "" FORCE)
  set(CURL_USE_LIBSSH2 OFF CACHE BOOL "" FORCE)
  set(CURL_USE_LIBSSH OFF CACHE BOOL "" FORCE)
  set(USE_LIBIDN2 OFF CACHE BOOL "" FORCE)
  set(USE_NGHTTP2 OFF CACHE BOOL "" FORCE)
  add_subdirectory("${HYLE_CURL_DIR}"
    "${CMAKE_BINARY_DIR}/_deps/curl-build" EXCLUDE_FROM_ALL)

  function(_hyle_configure_io_transfer_p1_bench target)
    target_link_libraries(${target} PRIVATE
      hyle_io_transfer nanobench Threads::Threads CURL::libcurl)
    set_property(TARGET ${target} PROPERTY
      INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_apply_warnings(${target})
  endfunction()

  add_executable(hyle_io_transfer_local_p1_bench
    local_p1_bench.cc)
  _hyle_configure_io_transfer_p1_bench(
    hyle_io_transfer_local_p1_bench)

  hyle_register_benches(io_transfer
    hyle_io_transfer_local_p1_bench)

  add_executable(hyle_io_transfer_resume_p1_bench
    resume_p1_bench.cc)
  _hyle_configure_io_transfer_p1_bench(
    hyle_io_transfer_resume_p1_bench)
  hyle_register_benches(io_transfer
    hyle_io_transfer_resume_p1_bench)
endif()
