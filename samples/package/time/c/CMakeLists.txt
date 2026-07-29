cmake_minimum_required(VERSION 3.25)
project(hyle_time_c_package_probe LANGUAGES C)

find_package(Hyle 0.1 REQUIRED COMPONENTS time)

add_executable(hyle_time_c_package_probe main.c)
target_link_libraries(hyle_time_c_package_probe PRIVATE hyle::time)
hyle_stage_runtime_dependencies(hyle_time_c_package_probe)

enable_testing()
add_test(NAME hyle_time_c_package_probe COMMAND hyle_time_c_package_probe)
