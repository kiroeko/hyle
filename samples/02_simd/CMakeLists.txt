cmake_minimum_required(VERSION 3.25)
project(hyle_sample_simd LANGUAGES CXX)

option(HYLE_SAMPLE_SIMD_USE_INSTALLED "Consume an installed HYLE SIMD package" OFF)
option(HYLE_SAMPLE_SIMD_FAT
  "Build a direct-Highway kernel for unknown customer CPUs" OFF)
if(HYLE_SAMPLE_SIMD_USE_INSTALLED)
  find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS simd)
else()
  set(HYLE_BUILD_TESTS OFF)
  set(HYLE_BUILD_BENCH OFF)
  set(HYLE_COMPONENTS simd)
  add_subdirectory(../.. hyle_build)
endif()

add_executable(simd_sample main.cc)
target_link_libraries(simd_sample PRIVATE hyle::simd)
hyle_stage_runtime_dependencies(simd_sample)

# OPTIONAL exact target for this consumer TU. It exposes the locked
# Highway headers and compiles the expert kernel without CPUID/function pointers.
# To bind all ordinary HYLE batch APIs too, also configure HYLE_SIMD_EXACT_TARGET=AVX2.
set(HYLE_SAMPLE_SIMD_ISA "" CACHE STRING "Exact target ISA for the expert path")
if(HYLE_SAMPLE_SIMD_ISA AND HYLE_SAMPLE_SIMD_FAT)
  message(FATAL_ERROR
    "HYLE_SAMPLE_SIMD_ISA and HYLE_SAMPLE_SIMD_FAT are separate deployment journeys; "
    "select only one")
endif()
if(HYLE_SAMPLE_SIMD_ISA)
  hyle_target_simd(TARGET simd_sample ISA "${HYLE_SAMPLE_SIMD_ISA}")
endif()

if(HYLE_SAMPLE_SIMD_FAT)
  target_compile_definitions(simd_sample PRIVATE HYLE_SAMPLE_SIMD_FAT=1)
  hyle_add_fat_kernel(
    TARGET simd_sample
    NAME sample_bias_relu
    IMPL "${CMAKE_CURRENT_SOURCE_DIR}/fat_kernel-inl.h"
    TARGETS scalar sse4 avx2 avx512 neon sve rvv)
endif()
