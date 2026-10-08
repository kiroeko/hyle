cmake_minimum_required(VERSION 3.25)
project(hyle_cuda_matrix_qualified LANGUAGES CXX)
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS core)
get_filename_component(_root "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
if(NOT EXISTS "${CUDA_MATRIX_IMAGE}/hyle_cuda_matrix_bench.exe")
  message(FATAL_ERROR "Provide the frozen, AA-qualified matrix image directory")
endif()
# Preserve the exact measured image and staged DLL bytes. Product math is not
# rebuilt by the instrumentation/control project.
file(GLOB _matrix_runtime "${CUDA_MATRIX_IMAGE}/*.dll")
add_custom_target(hyle_cuda_matrix_bench
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${CUDA_MATRIX_IMAGE}/hyle_cuda_matrix_bench.exe" "${CMAKE_BINARY_DIR}/hyle_cuda_matrix_bench.exe"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${_matrix_runtime} "${CMAKE_BINARY_DIR}"
  VERBATIM)
find_package(Threads REQUIRED)
add_library(matrix_nanobench STATIC "${_root}/bench/nanobench_impl.cc")
target_include_directories(matrix_nanobench SYSTEM PUBLIC "${_root}/third_party/nanobench")
target_link_libraries(matrix_nanobench PUBLIC Threads::Threads)
add_executable(hyle_perf_control_bench "${_root}/bench/integration/perf_control_bench.cc")
target_compile_features(hyle_perf_control_bench PRIVATE cxx_std_23)
target_compile_definitions(hyle_perf_control_bench PRIVATE _CRT_SECURE_NO_WARNINGS)
target_link_libraries(hyle_perf_control_bench PRIVATE hyle::core matrix_nanobench)
