cmake_minimum_required(VERSION 3.25)
project(hyle_sample_simd LANGUAGES CXX)

set(HYLE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(HYLE_BUILD_BENCH OFF CACHE BOOL "" FORCE)
add_subdirectory(../.. hyle_build)

add_executable(simd_sample main.cc)
target_link_libraries(simd_sample PRIVATE hyle::simd)

# OPTIONAL exact target for this consumer TU. Native calls then have no CPUID or function pointer.
# To bind all ordinary HYLE batch APIs too, also configure HYLE_SIMD_EXACT_TARGET=AVX2.
set(HYLE_SAMPLE_SIMD_ISA "" CACHE STRING "Exact target ISA for the native sample path")
if(HYLE_SAMPLE_SIMD_ISA)
  hyle_target_simd(TARGET simd_sample ISA "${HYLE_SAMPLE_SIMD_ISA}")
endif()
