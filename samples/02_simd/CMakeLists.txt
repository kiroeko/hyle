cmake_minimum_required(VERSION 3.25)
project(hyle_sample_simd LANGUAGES CXX)

option(HYLE_SAMPLE_SIMD_USE_INSTALLED "Consume an installed HYLE SIMD package" OFF)
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

# OPTIONAL source-only exact target for this consumer TU. It exposes the locked
# Highway headers and compiles the expert kernel without CPUID/function pointers.
# To bind all ordinary HYLE batch APIs too, also configure HYLE_SIMD_EXACT_TARGET=AVX2.
set(HYLE_SAMPLE_SIMD_ISA "" CACHE STRING "Exact target ISA for the source expert path")
if(HYLE_SAMPLE_SIMD_ISA)
  if(HYLE_SAMPLE_SIMD_USE_INSTALLED)
    message(FATAL_ERROR
      "HYLE_SAMPLE_SIMD_ISA is source-only; installed SDK consumers use whole-buffer APIs")
  endif()
  hyle_target_simd(TARGET simd_sample ISA "${HYLE_SAMPLE_SIMD_ISA}")
endif()
