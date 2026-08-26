cmake_minimum_required(VERSION 3.25)
project(hyle_package_core_cpp LANGUAGES CXX)

find_package(Hyle 0.1 REQUIRED COMPONENTS core)

add_executable(hyle_package_core_cpp main.cc)
target_compile_features(hyle_package_core_cpp PRIVATE cxx_std_23)
target_link_libraries(hyle_package_core_cpp PRIVATE hyle::core)

enable_testing()
add_test(NAME hyle_package_core_cpp COMMAND hyle_package_core_cpp)
