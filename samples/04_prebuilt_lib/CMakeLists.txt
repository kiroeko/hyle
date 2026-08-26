cmake_minimum_required(VERSION 3.25)
project(hyle_sample_installed_package LANGUAGES CXX)

# CMAKE_PREFIX_PATH (or Hyle_DIR) points at an installed HYLE prefix. The package
# supplies headers, linkage mode, transitive module dependencies and C++23.
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS simd mem)

add_executable(installed_package_consumer consumer.cc)
target_link_libraries(installed_package_consumer PRIVATE hyle::simd hyle::mem)

# No-op for static/Unix; stages the complete imported DLL closure on Windows.
hyle_stage_runtime_dependencies(installed_package_consumer)

enable_testing()
add_test(NAME installed_package_consumer COMMAND installed_package_consumer)
