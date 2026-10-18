cmake_minimum_required(VERSION 3.25)
project(d5_cpu_delivery LANGUAGES C CXX)

if(D5_SOURCE_ROOT)
  set(HYLE_COMPONENTS "compute,graph,simd" CACHE STRING "" FORCE)
  set(HYLE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(HYLE_BUILD_BENCH OFF CACHE BOOL "" FORCE)
  set(HYLE_TEST_INSTRUMENTATION OFF CACHE BOOL "" FORCE)
  set(HYLE_INSTALL ON CACHE BOOL "" FORCE)
  set(HYLE_STATIC ON CACHE BOOL "" FORCE)
  set(HYLE_COMPUTE_CPU ON CACHE BOOL "" FORCE)
  set(HYLE_COMPUTE_CUDA OFF CACHE STRING "" FORCE)
  set(HYLE_COMPUTE_ONEDNN ON CACHE STRING "" FORCE)
  set(HYLE_COMPUTE_APPLE OFF CACHE STRING "" FORCE)
  set(HYLE_COMPUTE_VULKAN OFF CACHE STRING "" FORCE)
  add_subdirectory("${D5_SOURCE_ROOT}" hyle)
else()
  find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS compute graph simd)
endif()

# Fixture dependencies are independent of the SDK: no producer include/library
# directories are used by an installed consumer. The quantizer prepares bytes;
# all observed model execution goes through ordinary HYLE CPU APIs.
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

add_executable(d5_cpu_direct
  tests/compute_route_cpu/d5_qwen35_profile_test.cc
  tests/compute_route_cpu/d5_qwen38_official_test.cc
  tests/compute_route_cpu/d5_qwen_gguf_vocabulary_test.cc)
add_executable(d5_cpu_graph
  tests/graph_route_cpu/d5_qwen35_profile_test.cc
  tests/graph_route_cpu/d5_qwen38_official_test.cc
  tests/graph_route_cpu/d5_qwen_gguf_vocabulary_test.cc)
target_link_libraries(d5_cpu_direct PRIVATE hyle::compute)
target_link_libraries(d5_cpu_graph PRIVATE hyle::graph)
add_custom_target(d5_cpu_delivery DEPENDS d5_cpu_direct d5_cpu_graph)
enable_testing()
foreach(_target IN ITEMS d5_cpu_direct d5_cpu_graph)
  target_compile_features(${_target} PRIVATE cxx_std_23)
  target_compile_options(${_target} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
  target_link_libraries(${_target} PRIVATE hyle::simd GTest::gtest_main ggml-base)
  hyle_stage_runtime_dependencies(${_target})
  add_test(NAME ${_target} COMMAND ${_target}
    "--gtest_filter=*D5Qwen35Official*Ffn*:*D5Qwen35OfficialCausalConvAndGdn*:*D5Qwen35OfficialEmbeddingNormHeadAndFailure*:*D5DeepSeekHashRoutingFeedsMixedWeightMoe")
  set_tests_properties(${_target} PROPERTIES TIMEOUT 1800)
endforeach()
