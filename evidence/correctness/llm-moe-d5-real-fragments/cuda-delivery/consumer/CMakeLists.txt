cmake_minimum_required(VERSION 3.25)
project(d5_cuda_delivery LANGUAGES C CXX)

if(D5_SOURCE_ROOT)
  set(HYLE_COMPONENTS "compute,graph,simd" CACHE STRING "" FORCE)
  set(HYLE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(HYLE_BUILD_BENCH OFF CACHE BOOL "" FORCE)
  set(HYLE_TEST_INSTRUMENTATION OFF CACHE BOOL "" FORCE)
  set(HYLE_INSTALL ON CACHE BOOL "" FORCE)
  set(HYLE_STATIC ON CACHE BOOL "" FORCE)
  set(HYLE_COMPUTE_CPU OFF CACHE STRING "" FORCE)
  set(HYLE_COMPUTE_CUDA ON CACHE STRING "" FORCE)
  set(HYLE_COMPUTE_ONEDNN OFF CACHE STRING "" FORCE)
  set(HYLE_COMPUTE_APPLE OFF CACHE STRING "" FORCE)
  set(HYLE_COMPUTE_VULKAN OFF CACHE STRING "" FORCE)
  add_subdirectory("${D5_SOURCE_ROOT}" hyle)
else()
  find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS compute graph simd)
endif()

# Fixture dependencies are independent of the SDK: no producer include/library
# directories are used by an installed consumer. The quantizer prepares bytes;
# all observed model execution goes through ordinary HYLE CUDA APIs.
if(NOT TARGET ggml-base)
  block()
    set(BUILD_SHARED_LIBS OFF)
    set(GGML_BUILD_TESTS OFF)
    set(GGML_BUILD_EXAMPLES OFF)
    set(GGML_CPU OFF)
    set(GGML_CUDA OFF)
    set(GGML_VULKAN OFF)
    set(GGML_METAL OFF)
    set(GGML_NATIVE OFF)
    set(GGML_OPENMP OFF)
    add_subdirectory("${D5_GGML_SOURCE}" fixture-ggml EXCLUDE_FROM_ALL)
  endblock()
endif()
if(NOT TARGET GTest::gtest_main)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  add_subdirectory("${D5_GTEST_SOURCE}" fixture-gtest EXCLUDE_FROM_ALL)
endif()

add_executable(d5_cuda_delivery
  tests/compute_route_cuda/d5_qwen35_execution_test.cc
  tests/compute_route_cuda/d5_qwen38_official_test.cc
  tests/graph_route_cuda/d5_qwen35_profile_test.cc
  tests/graph_route_cuda/d5_qwen38_official_test.cc)
target_compile_features(d5_cuda_delivery PRIVATE cxx_std_23)
target_compile_options(d5_cuda_delivery PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
target_link_libraries(d5_cuda_delivery PRIVATE
  hyle::compute hyle::graph hyle::simd GTest::gtest_main ggml-base)
hyle_stage_runtime_dependencies(d5_cuda_delivery)
enable_testing()
add_test(NAME d5_cuda_delivery COMMAND d5_cuda_delivery
  "--gtest_filter=*D5Qwen35OfficialFfn*:*.D5Qwen35OfficialCausalConvAndGdn*:*.D5DeepSeekOfficialMoe:*.D5DeepSeekQuantizationRoundtrip")
set_tests_properties(d5_cuda_delivery PROPERTIES TIMEOUT 1800)
