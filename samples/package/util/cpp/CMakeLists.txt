cmake_minimum_required(VERSION 3.25)
project(hyle_util_cpp_package_probe LANGUAGES CXX)

find_package(Hyle 0.1 REQUIRED COMPONENTS util)

add_executable(hyle_util_cpp_package_probe main.cc)
target_link_libraries(hyle_util_cpp_package_probe PRIVATE hyle::util)
target_compile_features(hyle_util_cpp_package_probe PRIVATE cxx_std_23)
hyle_stage_runtime_dependencies(hyle_util_cpp_package_probe)

enable_testing()
add_test(NAME hyle_util_cpp_package_probe COMMAND hyle_util_cpp_package_probe)
