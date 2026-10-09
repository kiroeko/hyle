cmake_minimum_required(VERSION 3.25)
project(hyle_package_mem_cpp LANGUAGES CXX)

find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS mem)

add_executable(hyle_package_mem_cpp main.cc)
target_compile_features(hyle_package_mem_cpp PRIVATE cxx_std_23)
target_link_libraries(hyle_package_mem_cpp PRIVATE hyle::mem)
hyle_stage_runtime(hyle_package_mem_cpp)

enable_testing()
add_test(NAME hyle_package_mem_cpp COMMAND hyle_package_mem_cpp)
