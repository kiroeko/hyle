cmake_minimum_required(VERSION 3.25)
project(hyle_compute_stream_sample LANGUAGES CXX)

if(HYLE_SAMPLE_SOURCE)
  set(HYLE_COMPONENTS compute_stream)
  add_subdirectory("${HYLE_SAMPLE_SOURCE}" hyle_build)
else()
  find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS compute_stream)
endif()

add_executable(hyle_compute_stream_sample main.cc)
target_compile_features(hyle_compute_stream_sample PRIVATE cxx_std_23)
target_link_libraries(hyle_compute_stream_sample PRIVATE hyle::compute_stream)
hyle_stage_runtime(hyle_compute_stream_sample)

enable_testing()
add_test(NAME hyle_compute_stream_sample COMMAND hyle_compute_stream_sample)
set_tests_properties(hyle_compute_stream_sample PROPERTIES TIMEOUT 180)
