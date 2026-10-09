cmake_minimum_required(VERSION 3.25)
project(hyle_package_io_cpp LANGUAGES CXX)

find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS io)

add_executable(hyle_package_io_cpp main.cc)
target_compile_features(hyle_package_io_cpp PRIVATE cxx_std_23)
target_link_libraries(hyle_package_io_cpp PRIVATE hyle::io)
hyle_stage_runtime(hyle_package_io_cpp)

enable_testing()
add_test(NAME hyle_package_io_cpp COMMAND hyle_package_io_cpp)
