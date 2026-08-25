cmake_minimum_required(VERSION 3.25)
project(hyle_sample_installed_io_http2_ffi_schema LANGUAGES C)

find_package(Hyle 0.1 REQUIRED COMPONENTS io)
if(NOT io_http2 IN_LIST Hyle_AVAILABLE_FEATURES)
  message(FATAL_ERROR
    "This sample requires an HYLE package built with HYLE_IO_HTTP2=ON; "
    "available features: ${Hyle_AVAILABLE_FEATURES}")
endif()

add_executable(installed_io_http2_schema_fixture main.c)
target_link_libraries(installed_io_http2_schema_fixture PRIVATE hyle::io)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(installed_io_http2_schema_fixture PRIVATE -fshort-enums)
endif()
hyle_stage_runtime_dependencies(installed_io_http2_schema_fixture)

enable_testing()
add_test(NAME installed_io_http2_schema_fixture COMMAND installed_io_http2_schema_fixture)
