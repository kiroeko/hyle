cmake_minimum_required(VERSION 3.25)
project(hyle_sample_installed_compute LANGUAGES CXX)

find_package(Hyle 0.1 REQUIRED COMPONENTS compute)

add_executable(installed_compute_consumer main.cc)
target_link_libraries(installed_compute_consumer PRIVATE hyle::compute)
hyle_stage_runtime_dependencies(installed_compute_consumer)

enable_testing()
add_test(NAME installed_compute_consumer COMMAND installed_compute_consumer)
