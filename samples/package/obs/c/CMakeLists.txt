cmake_minimum_required(VERSION 3.25)
project(hyle_package_obs_c LANGUAGES C)

find_package(Hyle 0.1 REQUIRED COMPONENTS obs)

add_executable(hyle_package_obs_c main.c)
target_link_libraries(hyle_package_obs_c PRIVATE hyle::obs)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(hyle_package_obs_c PRIVATE -fshort-enums)
endif()
hyle_stage_runtime_dependencies(hyle_package_obs_c)

enable_testing()
add_test(NAME hyle_package_obs_c COMMAND hyle_package_obs_c)
