cmake_minimum_required(VERSION 3.25)
project(hyle_package_error_c LANGUAGES C)

find_package(Hyle 0.1 REQUIRED COMPONENTS error)

add_executable(hyle_package_error_c main.c)
target_link_libraries(hyle_package_error_c PRIVATE hyle::error)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(hyle_package_error_c PRIVATE -fshort-enums)
endif()
hyle_stage_runtime_dependencies(hyle_package_error_c)

enable_testing()
add_test(NAME hyle_package_error_c COMMAND hyle_package_error_c)
