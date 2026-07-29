cmake_minimum_required(VERSION 3.25)
project(hyle_package_error_cpp LANGUAGES CXX)

find_package(Hyle 0.1 REQUIRED COMPONENTS error)

add_executable(hyle_package_error_cpp main.cc)
target_link_libraries(hyle_package_error_cpp PRIVATE hyle::error)
hyle_stage_runtime_dependencies(hyle_package_error_cpp)

enable_testing()
add_test(NAME hyle_package_error_cpp COMMAND hyle_package_error_cpp)
