cmake_minimum_required(VERSION 3.25)
project(hyle_simd_cpp_package_probe LANGUAGES CXX)

find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS simd)

get_target_property(_hyle_simd_ordinary_links hyle::simd
  INTERFACE_LINK_LIBRARIES)
get_target_property(_hyle_simd_ordinary_includes hyle::simd
  INTERFACE_INCLUDE_DIRECTORIES)
if("${_hyle_simd_ordinary_links}" MATCHES "simd_expert" OR
   "${_hyle_simd_ordinary_includes}" MATCHES "dependencies/(highway|xsimd)")
  message(FATAL_ERROR
    "ordinary hyle::simd unexpectedly propagates the opt-in expert SDK")
endif()

add_executable(hyle_simd_cpp_package_probe main.cc)
target_link_libraries(hyle_simd_cpp_package_probe PRIVATE hyle::simd)
target_compile_features(hyle_simd_cpp_package_probe PRIVATE cxx_std_23)
hyle_stage_runtime_dependencies(hyle_simd_cpp_package_probe)

# The same expert journey must work for source embedding and for relocated
# static/shared SDKs. Keep provider-native types inside this object TU.
add_library(hyle_simd_cpp_provider_probe OBJECT expert_probe.cc)
hyle_target_simd(TARGET hyle_simd_cpp_provider_probe ISA SCALAR)
target_sources(hyle_simd_cpp_package_probe PRIVATE
  $<TARGET_OBJECTS:hyle_simd_cpp_provider_probe>)

# A separately compiled pointer/POD kernel may still target unknown customer
# CPUs. The helper emits only the current platform's canonical variants.
hyle_add_fat_kernel(
  TARGET hyle_simd_cpp_package_probe
  NAME package_bias_relu
  IMPL "${CMAKE_CURRENT_SOURCE_DIR}/fat_kernel-inl.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)

enable_testing()
add_test(NAME hyle_simd_cpp_package_probe COMMAND hyle_simd_cpp_package_probe)
