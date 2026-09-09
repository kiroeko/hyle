option(HYLE_IO_WATCH_P1_BENCH
  "build the IO Watch v1 same-contract libuv P1 benchmark" OFF)

if(HYLE_IO_WATCH_P1_BENCH)
  if(NOT HYLE_STATIC)
    message(FATAL_ERROR "HYLE_IO_WATCH_P1_BENCH requires HYLE_STATIC")
  endif()
  set(HYLE_LIBUV_DIR "${HYLE_SOURCE_DIR}/../libuv" CACHE PATH
    "source-pinned libuv checkout used only by the IO Watch P1 peer")
  if(NOT EXISTS "${HYLE_LIBUV_DIR}/include/uv.h")
    message(FATAL_ERROR
      "HYLE_IO_WATCH_P1_BENCH requires libuv at ${HYLE_LIBUV_DIR}")
  endif()

  if(NOT TARGET uv_a)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(LIBUV_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(LIBUV_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LIBUV_BUILD_BENCH OFF CACHE BOOL "" FORCE)
    add_subdirectory(
      "${HYLE_LIBUV_DIR}"
      "${HYLE_BINARY_DIR}/bench/peers/io-watch-libuv"
      EXCLUDE_FROM_ALL)
  endif()

  function(_hyle_configure_io_watch_p1 target placement)
    target_link_libraries(${target} PRIVATE
      hyle_io_watch hyle_sched nanobench uv_a Threads::Threads)
    target_include_directories(${target} SYSTEM PRIVATE
      "${HYLE_LIBUV_DIR}/include")
    target_compile_definitions(${target} PRIVATE
      HYLE_IO_WATCH_P1_PLACEMENT=${placement})
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_apply_warnings(${target})
  endfunction()

  add_executable(hyle_io_watch_p1_bench watch_p1_bench.cc)
  _hyle_configure_io_watch_p1(hyle_io_watch_p1_bench 0)

  add_executable(hyle_io_watch_p1_placement_bench watch_p1_bench.cc)
  _hyle_configure_io_watch_p1(hyle_io_watch_p1_placement_bench 1)

  hyle_register_benches(io_watch
    hyle_io_watch_p1_bench
    hyle_io_watch_p1_placement_bench)
endif()
