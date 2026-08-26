cmake_minimum_required(VERSION 3.25)
project(hyle_sample_installed_io_http2 LANGUAGES CXX)

find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS io)
if(NOT io_http2 IN_LIST Hyle_AVAILABLE_FEATURES)
  message(FATAL_ERROR
    "This sample requires an HYLE package built with HYLE_IO_HTTP2=ON; "
    "available features: ${Hyle_AVAILABLE_FEATURES}")
endif()

add_executable(installed_io_http2_consumer main.cc)
target_link_libraries(installed_io_http2_consumer PRIVATE hyle::io)
hyle_stage_runtime_dependencies(installed_io_http2_consumer)

enable_testing()
add_test(NAME installed_io_http2_consumer COMMAND installed_io_http2_consumer)
