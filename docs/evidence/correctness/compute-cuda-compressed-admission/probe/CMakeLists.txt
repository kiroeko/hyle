cmake_minimum_required(VERSION 3.25)
project(hyle_marlin_admission LANGUAGES CXX)

# Research fixture: use a pinned, unmodified reference checkout, not a product dependency.
if(NOT MARLIN_REFERENCE)
  message(FATAL_ERROR "Set MARLIN_REFERENCE to the original Marlin reference checkout")
endif()
find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${MARLIN_REFERENCE}" rev-parse HEAD
  OUTPUT_VARIABLE marlin_revision OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
if(NOT marlin_revision STREQUAL "1f25790bdd49fba53106164a24666dade68d7c90")
  message(FATAL_ERROR "This research fixture requires its recorded Marlin revision")
endif()
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS compute)
if(NOT "compute_cuda" IN_LIST Hyle_AVAILABLE_FEATURES)
  message(FATAL_ERROR "Use an installed CUDA-enabled Hyle SDK")
endif()
find_package(CUDAToolkit 13.3 REQUIRED)
enable_language(CUDA)
add_library(marlin_reference STATIC "${MARLIN_REFERENCE}/marlin/marlin_cuda_kernel.cu")
set_target_properties(marlin_reference PROPERTIES CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES 89 CUDA_RUNTIME_LIBRARY Shared)
target_link_libraries(marlin_reference PRIVATE CUDA::cudart)
target_compile_options(marlin_reference PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
if(MSVC)
  target_compile_options(marlin_reference PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()

add_executable(hyle_marlin_admission main.cc)
target_compile_features(hyle_marlin_admission PRIVATE cxx_std_23)
target_link_libraries(hyle_marlin_admission PRIVATE marlin_reference hyle::compute CUDA::cudart)
set_target_properties(hyle_marlin_admission PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/app")
hyle_stage_runtime_dependencies(hyle_marlin_admission)
enable_testing()
add_test(NAME hyle_marlin_admission COMMAND hyle_marlin_admission)
set_tests_properties(hyle_marlin_admission PROPERTIES TIMEOUT 90)
