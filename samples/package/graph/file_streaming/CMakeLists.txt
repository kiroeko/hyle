cmake_minimum_required(VERSION 3.25)
project(hyle_file_streaming_sample LANGUAGES CXX)

if(HYLE_SAMPLE_SOURCE)
  set(HYLE_COMPONENTS graph)
  add_subdirectory("${HYLE_SAMPLE_SOURCE}" hyle_build)
else()
  find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS graph)
endif()

add_executable(hyle_file_streaming_sample main.cc)
target_compile_features(hyle_file_streaming_sample PRIVATE cxx_std_23)
target_link_libraries(hyle_file_streaming_sample PRIVATE hyle::graph)
hyle_stage_runtime(hyle_file_streaming_sample)

# The caller supplies an official-tool checked range manifest. This is an
# explicit local-weight journey, never an implicit checkpoint download/read.
add_executable(hyle_file_streaming_qwen qwen_main.cc)
target_compile_features(hyle_file_streaming_qwen PRIVATE cxx_std_23)
target_link_libraries(hyle_file_streaming_qwen PRIVATE hyle::graph)
hyle_stage_runtime(hyle_file_streaming_qwen)
enable_testing()
add_test(NAME hyle_file_streaming_sample COMMAND hyle_file_streaming_sample)
set_tests_properties(hyle_file_streaming_sample PROPERTIES TIMEOUT 180)
