# Linux AF_UNIX benchmark: the default keeps a manual-loop nghttp2 reference;
# --paired uses a shared Scheduler and two endpoint Tasks on both sides.
# Both modes validate the complete protocol payload and resource lifetime.
if(HYLE_IO_HTTP2 AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(hyle_io_http2_p1_bench http2_p1_bench.cc)
  target_link_libraries(hyle_io_http2_p1_bench PRIVATE
    hyle_sched hyle_io_http2 nanobench nghttp2 Threads::Threads)
  hyle_apply_warnings(hyle_io_http2_p1_bench)
  hyle_register_benches(io_http2 hyle_io_http2_p1_bench)
endif()
