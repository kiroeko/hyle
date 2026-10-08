cmake_minimum_required(VERSION 3.25)
project(hyle_cuda_best_of_consumer LANGUAGES CXX)
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS compute graph simd)
if(NOT EXISTS "${C3_GTEST_SOURCE}/googletest/src/gtest-all.cc")
  message(FATAL_ERROR "C3 requires the existing locked GoogleTest source")
endif()
block()
  set(BUILD_SHARED_LIBS OFF)
  set(gtest_force_shared_crt ON)
  set(BUILD_GMOCK OFF)
  set(INSTALL_GTEST OFF)
  add_subdirectory("${C3_GTEST_SOURCE}" fixture-gtest EXCLUDE_FROM_ALL)
endblock()
get_filename_component(_root "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
add_executable(cuda_c3_qualification
  "${_root}/tests/compute_provider_cuda/cuda_provider_test.cc"
  "${_root}/tests/compute_provider_cuda/tensor_test.cc"
  "${_root}/tests/compute_route_cuda/best_of_test.cc")
target_compile_features(cuda_c3_qualification PRIVATE cxx_std_23)
target_link_libraries(cuda_c3_qualification PRIVATE hyle::compute hyle::simd
  hyle::_factor_compute_provider_cuda GTest::gtest_main)
hyle_stage_runtime_dependencies(cuda_c3_qualification)
add_executable(cuda_c3_ordinary
  "${_root}/evidence/correctness/cuda-u4-defaults/sdk-consumer/main.cc")
target_compile_features(cuda_c3_ordinary PRIVATE cxx_std_23)
target_link_libraries(cuda_c3_ordinary PRIVATE hyle::compute hyle::graph)
hyle_stage_runtime_dependencies(cuda_c3_ordinary)
