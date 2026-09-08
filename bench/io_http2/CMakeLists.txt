# HTTP/2 P1 compares the public HYLE route with the pinned nghttp2 session
# mechanism under the same protocol, transport, worker, stream and byte
# contract.  This target is intentionally Linux-only: closure-perf-v1 is bound
# to W1-WSL, while Windows/macOS remain correctness and delivery gates.
if(HYLE_IO_HTTP2 AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(hyle_io_http2_p1_bench http2_p1_bench.cc)
  target_link_libraries(hyle_io_http2_p1_bench PRIVATE
    hyle_sched hyle_io_http2 nanobench nghttp2 Threads::Threads)
  hyle_apply_warnings(hyle_io_http2_p1_bench)
  hyle_register_benches(io_http2 hyle_io_http2_p1_bench)
endif()
