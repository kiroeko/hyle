cmake_minimum_required(VERSION 3.25)
project(hyle_custom_kernel_sample LANGUAGES CXX)

option(HYLE_SAMPLE_CUDA "Build the custom CUDA kernel journey" OFF)
if(HYLE_SAMPLE_SOURCE)
  set(HYLE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(HYLE_BUILD_BENCH OFF CACHE BOOL "" FORCE)
  set(HYLE_COMPONENTS graph)
  if(HYLE_SAMPLE_CUDA)
    set(HYLE_COMPUTE_CUDA ON CACHE STRING "" FORCE)
  endif()
  add_subdirectory("${HYLE_SAMPLE_SOURCE}" hyle_build)
else()
  find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS graph)
endif()

# Deliberately put the factory and schema decode in another code image. This
# is a C++ same-toolchain library, not a stable plugin ABI or Native FFI.
add_library(hyle_custom_example SHARED kernel.cc)
target_compile_features(hyle_custom_example PRIVATE cxx_std_23)
target_compile_definitions(hyle_custom_example PRIVATE HYLE_CUSTOM_EXAMPLE_BUILD)
target_link_libraries(hyle_custom_example PRIVATE hyle::compute)
set_target_properties(hyle_custom_example PROPERTIES
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)

add_executable(hyle_custom_kernel_sample main.cc)
target_compile_features(hyle_custom_kernel_sample PRIVATE cxx_std_23)
target_link_libraries(hyle_custom_kernel_sample PRIVATE hyle_custom_example hyle::graph)
if(COMMAND hyle_stage_runtime_dependencies)
  hyle_stage_runtime_dependencies(hyle_custom_kernel_sample)
endif()

enable_testing()
add_test(NAME hyle_custom_kernel_sample COMMAND hyle_custom_kernel_sample)

if(HYLE_SAMPLE_CUDA)
  if(NOT HYLE_SAMPLE_SOURCE AND NOT "compute_cuda" IN_LIST Hyle_AVAILABLE_FEATURES)
    message(FATAL_ERROR "The installed Hyle SDK does not contain compute_cuda")
  endif()
  find_package(CUDAToolkit 13.3 REQUIRED)
  enable_language(CUDA)
  # nvcc compiles only the native kernel. HYLE's C++23 factory stays in the
  # consumer, with one HYLE runtime owner even when using static archives.
  add_library(hyle_custom_cuda_kernel SHARED cuda_kernel.cu)
  target_compile_definitions(hyle_custom_cuda_kernel PRIVATE HYLE_CUSTOM_CUDA_KERNEL_BUILD)
  target_link_libraries(hyle_custom_cuda_kernel PRIVATE CUDA::cudart)
  set_target_properties(hyle_custom_cuda_kernel PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES native
    CUDA_RUNTIME_LIBRARY Shared CXX_VISIBILITY_PRESET hidden)
  if(MSVC)
    target_compile_options(hyle_custom_cuda_kernel PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>"
      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4211>")
  endif()
  add_executable(hyle_custom_cuda_sample cuda_main.cc)
  target_compile_features(hyle_custom_cuda_sample PRIVATE cxx_std_23)
  target_link_libraries(hyle_custom_cuda_sample PRIVATE hyle_custom_cuda_kernel hyle::graph)
  hyle_stage_runtime_dependencies(hyle_custom_cuda_sample)
  add_test(NAME hyle_custom_cuda_sample COMMAND hyle_custom_cuda_sample)
  set_tests_properties(hyle_custom_cuda_sample PROPERTIES TIMEOUT 60)
endif()
