cmake_minimum_required(VERSION 3.25)
project(hyle_sample_installed_c LANGUAGES C)

find_package(Hyle 0.1 REQUIRED COMPONENTS error)

add_executable(installed_c_consumer main.c)
target_link_libraries(installed_c_consumer PRIVATE hyle::error)
hyle_stage_runtime_dependencies(installed_c_consumer)
