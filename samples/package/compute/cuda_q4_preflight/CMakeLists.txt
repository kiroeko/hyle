cmake_minimum_required(VERSION 3.25)
project(hyle_cuda_q4_preflight LANGUAGES CXX)

if(HYLE_SAMPLE_SOURCE)
  set(HYLE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(HYLE_BUILD_BENCH OFF CACHE BOOL "" FORCE)
  set(HYLE_COMPONENTS compute)
  set(HYLE_COMPUTE_CUDA ON CACHE STRING "" FORCE)
  add_subdirectory("${HYLE_SAMPLE_SOURCE}" hyle_build)
else()
  find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS compute)
  if(NOT "compute_cuda" IN_LIST Hyle_AVAILABLE_FEATURES)
    message(FATAL_ERROR "The installed Hyle SDK must contain compute_cuda")
  endif()
endif()

find_package(CUDAToolkit 13.3 REQUIRED)
enable_language(CUDA)
add_library(q4_preflight_kernel SHARED kernel.cu)
target_compile_definitions(q4_preflight_kernel PRIVATE Q4_PREFLIGHT_KERNEL_BUILD)
target_link_libraries(q4_preflight_kernel PRIVATE CUDA::cudart)
set_target_properties(q4_preflight_kernel PROPERTIES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES native
  CUDA_RUNTIME_LIBRARY Shared CXX_VISIBILITY_PRESET hidden)
if(MSVC)
  target_compile_options(q4_preflight_kernel PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4211>")
endif()

add_executable(hyle_cuda_q4_preflight main.cc)
target_compile_features(hyle_cuda_q4_preflight PRIVATE cxx_std_23)
target_link_libraries(hyle_cuda_q4_preflight PRIVATE
  q4_preflight_kernel hyle::compute CUDA::cublasLt)
set_target_properties(hyle_cuda_q4_preflight PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/app")
hyle_stage_runtime_dependencies(hyle_cuda_q4_preflight)
enable_testing()
add_test(NAME hyle_cuda_q4_preflight COMMAND hyle_cuda_q4_preflight)
set_tests_properties(hyle_cuda_q4_preflight PROPERTIES TIMEOUT 60)
