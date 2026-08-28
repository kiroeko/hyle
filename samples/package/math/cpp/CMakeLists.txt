cmake_minimum_required(VERSION 3.25)
project(hyle_package_math_cpp LANGUAGES CXX)

find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS math)

add_executable(hyle_package_math_cpp main.cc)
target_link_libraries(hyle_package_math_cpp PRIVATE hyle::math)

enable_testing()
add_test(NAME hyle_package_math_cpp COMMAND hyle_package_math_cpp)
