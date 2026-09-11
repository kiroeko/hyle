cmake_minimum_required(VERSION 3.25)
project(hyle_package_process_cpp LANGUAGES CXX)

find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS process)

add_executable(hyle_package_process_cpp main.cc)
target_compile_features(hyle_package_process_cpp PRIVATE cxx_std_23)
target_link_libraries(hyle_package_process_cpp PRIVATE hyle::process)
hyle_stage_runtime_dependencies(hyle_package_process_cpp)

enable_testing()
add_test(NAME hyle_package_process_cpp COMMAND hyle_package_process_cpp)
