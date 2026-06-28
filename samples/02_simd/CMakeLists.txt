cmake_minimum_required(VERSION 3.25)
project(hyle_sample_simd LANGUAGES CXX)

set(HYLE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(HYLE_BUILD_BENCH OFF CACHE BOOL "" FORCE)
add_subdirectory(../.. hyle_build)

add_executable(simd_sample main.cc)
target_link_libraries(simd_sample PRIVATE hyle::simd)

# OPTIONAL: build this sample for your own machine to make the DIRECT vec<T,N> kernel
# (the one written in main.cc, not the dispatched ops) use wide SIMD. Without it, the
# direct kernel compiles at the SSE2 baseline. The DISPATCHED dot_f32/saxpy_f32 pick the
# best ISA at RUNTIME regardless -- that's the whole point. Uncomment to self-build:
#   if(NOT MSVC)
#     target_compile_options(simd_sample PRIVATE -march=native)
#   else()
#     target_compile_options(simd_sample PRIVATE /arch:AVX2)
#   endif()
