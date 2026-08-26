cmake_minimum_required(VERSION 3.25)
project(hyle_package_util_core_cpp LANGUAGES CXX)

find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS util_core)

add_executable(hyle_package_util_core_cpp main.cc)
target_link_libraries(hyle_package_util_core_cpp PRIVATE hyle::util_core)

enable_testing()
add_test(NAME hyle_package_util_core_cpp COMMAND hyle_package_util_core_cpp)
