cmake_minimum_required(VERSION 3.25)
project(hyle_sample_inference_service LANGUAGES CXX)

# This source-embedding consumer exercises HTTP/1. Optional TLS/HTTP2 stay at
# HYLE's default OFF unless the caller explicitly enables them.
set(HYLE_COMPONENTS "container;util;sched;io;simd;obs;platform;mem")
add_subdirectory(../.. hyle_build)

add_executable(hyle_inference_service main.cc)
target_compile_features(hyle_inference_service PRIVATE cxx_std_23)
target_link_libraries(hyle_inference_service PRIVATE
  hyle::container
  hyle::io
  hyle::mem
  hyle::obs
  hyle::platform
  hyle::sched
  hyle::simd
  hyle::util)
hyle_stage_runtime(hyle_inference_service)
