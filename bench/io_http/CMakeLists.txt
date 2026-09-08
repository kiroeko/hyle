# Full HTTP/1 request/response path over the same coroutine transport. This is
# factor-owned evidence; raw TCP remains a cost decomposition row, not a claim
# that different framing/body contracts are interchangeable.
add_executable(hyle_http_bench http_bench.cc)
target_link_libraries(hyle_http_bench PRIVATE
  hyle_sched hyle_io_http nanobench)
hyle_apply_warnings(hyle_http_bench)
if(WIN32)
  target_link_libraries(hyle_http_bench PRIVATE ws2_32 psapi)
endif()
hyle_register_benches(io_http hyle_http_bench)
