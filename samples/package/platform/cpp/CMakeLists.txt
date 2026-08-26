cmake_minimum_required(VERSION 3.25)
project(hyle_package_platform_cpp LANGUAGES CXX)

find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS platform)

add_executable(hyle_package_platform_cpp main.cc)
target_link_libraries(hyle_package_platform_cpp PRIVATE hyle::platform)
hyle_stage_runtime_dependencies(hyle_package_platform_cpp)

enable_testing()
add_test(NAME hyle_package_platform_cpp COMMAND hyle_package_platform_cpp)
