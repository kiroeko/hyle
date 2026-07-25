cmake_minimum_required(VERSION 3.25)
project(hyle_sample_installed_c LANGUAGES C CXX)

find_package(Hyle 0.1 REQUIRED COMPONENTS mem)

add_executable(installed_c_consumer main.c)
target_link_libraries(installed_c_consumer PRIVATE hyle::mem)
hyle_stage_runtime_dependencies(installed_c_consumer)

add_executable(installed_mem_cpp_consumer system_allocator.cc)
target_link_libraries(installed_mem_cpp_consumer PRIVATE hyle::mem)
hyle_stage_runtime_dependencies(installed_mem_cpp_consumer)

enable_testing()
add_test(NAME installed_c_consumer COMMAND installed_c_consumer)
add_test(NAME installed_mem_cpp_consumer COMMAND installed_mem_cpp_consumer)
