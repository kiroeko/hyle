# Same-contract RFC 6455 mask primitive. Boost.Beast remains a benchmark-only
# header peer; the product neither links Asio nor acquires a second IO runtime.
if(HYLE_BENCH_BOOST)
  add_executable(hyle_io_websocket_mask_p1_bench mask_p1_bench.cc)
  target_compile_definitions(hyle_io_websocket_mask_p1_bench PRIVATE
    BOOST_BEAST_HEADER_ONLY BOOST_ERROR_CODE_HEADER_ONLY)
  if(WIN32)
    target_compile_definitions(hyle_io_websocket_mask_p1_bench PRIVATE
      _WIN32_WINNT=0x0A00)
  endif()
  target_include_directories(hyle_io_websocket_mask_p1_bench PRIVATE
    "${PROJECT_SOURCE_DIR}/src/io_websocket")
  target_link_libraries(hyle_io_websocket_mask_p1_bench PRIVATE
    hyle_io_websocket nanobench hyle_bench_boost_headers)
  hyle_apply_warnings(hyle_io_websocket_mask_p1_bench)
  hyle_register_benches(io_websocket hyle_io_websocket_mask_p1_bench)
endif()
