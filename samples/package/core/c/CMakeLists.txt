cmake_minimum_required(VERSION 3.25)
project(hyle_package_core_c LANGUAGES C)

find_package(Hyle 0.1 REQUIRED COMPONENTS core)

add_executable(hyle_package_core_c main.c)
# Deliberately request only C11 locally. Linking hyle::core must raise the
# consumer to the public C17 floor through hyle::_build_contract.
set_target_properties(hyle_package_core_c PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF)
target_link_libraries(hyle_package_core_c PRIVATE hyle::core)
hyle_stage_runtime_dependencies(hyle_package_core_c)

enable_testing()
add_test(NAME hyle_package_core_c COMMAND hyle_package_core_c)
