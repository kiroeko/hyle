cmake_minimum_required(VERSION 3.25)
project(hyle_d5_cuda_fragment_consumer LANGUAGES C CXX)

# Reuse the relocated A5 SDK: this target never configures/rebuilds CUDA
# product libraries. The pinned host-only oracle dependencies are explicit.
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS core compute graph simd)
get_filename_component(_hyle_fixture_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
if(NOT EXISTS "${D5_GGML_SOURCE}/src/ggml.c" OR
   NOT EXISTS "${D5_GTEST_SOURCE}/googletest/src/gtest-all.cc")
  message(FATAL_ERROR "Provide the locked D5_GGML_SOURCE and D5_GTEST_SOURCE fixture checkouts")
endif()
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
block()
  set(gtest_force_shared_crt ON)
  set(BUILD_GMOCK OFF)
  set(INSTALL_GTEST OFF)
  add_subdirectory("${D5_GTEST_SOURCE}" fixture-gtest EXCLUDE_FROM_ALL)
endblock()
find_package(Threads REQUIRED)
add_library(d5_nanobench STATIC "${_hyle_fixture_root}/bench/nanobench_impl.cc")
target_include_directories(d5_nanobench SYSTEM PUBLIC
  "${_hyle_fixture_root}/third_party/nanobench")
target_link_libraries(d5_nanobench PUBLIC Threads::Threads)
add_executable(hyle_graph_route_cuda_bench ../d5_fragment_bench.cc)
target_compile_features(hyle_graph_route_cuda_bench PRIVATE cxx_std_23)
target_compile_options(hyle_graph_route_cuda_bench PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/utf-8;/W4;/WX>)
target_link_libraries(hyle_graph_route_cuda_bench PRIVATE hyle::graph hyle::compute
  hyle::simd d5_nanobench GTest::gtest ggml-base)
hyle_stage_runtime_dependencies(hyle_graph_route_cuda_bench)
# The canonical control executable is built unchanged, with the same compiler
# and nanobench as the consumer. No module-local statistics or controls.
add_executable(hyle_perf_control_bench
  "${_hyle_fixture_root}/bench/integration/perf_control_bench.cc")
target_compile_features(hyle_perf_control_bench PRIVATE cxx_std_23)
target_compile_definitions(hyle_perf_control_bench PRIVATE _CRT_SECURE_NO_WARNINGS)
target_link_libraries(hyle_perf_control_bench PRIVATE hyle::core d5_nanobench)
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/d5-input-paths.json" CONTENT
  "{\"sdk\":\"${Hyle_DIR}/../../..\",\"reference\":\"${D5_DIRECT_REFERENCE}\"}\n")
enable_testing()
add_test(NAME d5_cuda_fragment_verify COMMAND hyle_graph_route_cuda_bench)
set_tests_properties(d5_cuda_fragment_verify PROPERTIES TIMEOUT 600
  ENVIRONMENT "HYLE_BENCH_VERIFY_ONLY=1")
