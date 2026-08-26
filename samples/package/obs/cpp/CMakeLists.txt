cmake_minimum_required(VERSION 3.25)
project(hyle_package_obs_cpp LANGUAGES CXX)

find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS obs)

add_executable(hyle_package_obs_cpp main.cc)
target_link_libraries(hyle_package_obs_cpp PRIVATE hyle::obs)
hyle_stage_runtime_dependencies(hyle_package_obs_cpp)

enable_testing()
add_test(NAME hyle_package_obs_cpp COMMAND hyle_package_obs_cpp)
