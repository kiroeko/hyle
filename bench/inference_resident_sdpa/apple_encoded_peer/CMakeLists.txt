cmake_minimum_required(VERSION 3.25)
project(g04_apple_encoded_peer LANGUAGES C CXX)
if(NOT APPLE)
  message(FATAL_ERROR "The original Metal peer requires the actual MacBook Air")
endif()
get_filename_component(HYLE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(HYLE_RESIDENT_REFERENCE_SOURCE "" CACHE PATH "Authenticated original ggml456 source")
find_package(Hyle CONFIG REQUIRED COMPONENTS compute graph)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
if(NOT Hyle_STATIC)
  message(FATAL_ERROR "Use the independently qualified static SDK")
endif()
set(_source "${HYLE_SOURCE_DIR}/bench/inference_resident_sdpa")
set(_generated "${CMAKE_CURRENT_BINARY_DIR}/original-metal-q8")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/prepare_source.py"
  --original "${HYLE_RESIDENT_REFERENCE_SOURCE}" --output "${_generated}"
  RESULT_VARIABLE _prepared OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/source-generation.log")
if(NOT _prepared EQUAL 0)
  message(FATAL_ERROR "Original Metal Q8 source identity/materialization failed")
endif()
add_library(g04_apple_encoded_nanobench STATIC "${HYLE_SOURCE_DIR}/bench/nanobench_impl.cc")
target_include_directories(g04_apple_encoded_nanobench SYSTEM PUBLIC "${HYLE_SOURCE_DIR}/third_party/nanobench")
add_executable(g04_apple_encoded_bench "${_source}/apple_encoded_page_bench.cc"
  "${_source}/apple_encoded_page_reference.cc")
target_compile_features(g04_apple_encoded_bench PRIVATE cxx_std_23)
target_compile_options(g04_apple_encoded_bench PRIVATE -Wall -Wextra -Wpedantic -Werror)
# Original Metal source is embedded byte-for-byte (230690 bytes). Clang accepts
# this literal extension; retain its minimum-standard-limit warning in the log.
target_compile_options(g04_apple_encoded_bench PRIVATE -Wno-error=overlength-strings)
target_include_directories(g04_apple_encoded_bench PRIVATE "${_generated}"
  "${HYLE_RESIDENT_REFERENCE_SOURCE}/src/ggml-metal" "${HYLE_SOURCE_DIR}/bench/common")
target_link_libraries(g04_apple_encoded_bench PRIVATE hyle::compute g04_apple_encoded_nanobench)
set_property(TARGET g04_apple_encoded_bench PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
hyle_stage_runtime(g04_apple_encoded_bench)
