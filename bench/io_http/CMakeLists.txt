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

# Formal HTTP evidence uses a separate Linux-only executable so the ordinary
# product-shaped smoke benchmark does not acquire Boost. Both HYLE and Beast
# keep one event worker alive and charge the same keep-alive REST + chunked
# response episode; the central peer target owns modular Boost discovery.
if(HYLE_BENCH_BOOST AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(hyle_io_http_p1_bench http_p1_bench.cc)
  target_compile_definitions(hyle_io_http_p1_bench PRIVATE
    BOOST_ERROR_CODE_HEADER_ONLY)
  target_link_libraries(hyle_io_http_p1_bench PRIVATE
    hyle_sched hyle_io_http nanobench hyle_bench_boost_headers Threads::Threads)
  hyle_apply_warnings(hyle_io_http_p1_bench)
  hyle_register_benches(io_http hyle_io_http_p1_bench)
endif()
