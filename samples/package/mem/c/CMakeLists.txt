cmake_minimum_required(VERSION 3.25)
project(hyle_package_mem_c LANGUAGES C)

find_package(Hyle 0.1 REQUIRED COMPONENTS mem)

add_executable(hyle_package_mem_c main.c)
target_link_libraries(hyle_package_mem_c PRIVATE hyle::mem)
hyle_stage_runtime_dependencies(hyle_package_mem_c)

enable_testing()
add_test(NAME hyle_package_mem_c COMMAND hyle_package_mem_c)
