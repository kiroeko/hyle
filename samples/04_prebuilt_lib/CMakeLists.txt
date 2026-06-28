cmake_minimum_required(VERSION 3.25)
project(hyle_sample_prebuilt LANGUAGES CXX)

# Consume HYLE as a PREBUILT library -- the opposite of samples 01-03 (which build HYLE
# from source via add_subdirectory). Here HYLE is already compiled into shared/static
# artifacts; we only link them. This validates the HYLE_<MOD>_API export boundary.
#
# Two paths the caller supplies (see build_and_run.sh / .bat in this dir):
#   HYLE_REPO_DIR     : the HYLE repo root (for headers under include/)
#   HYLE_PREBUILT_DIR : the build dir of a prebuilt HYLE (libs land in <build>/bin)
#   HYLE_LINK_MODE    : "shared" or "static" (doc/diagnostic only; find_library picks
#                       whatever artifact exists in the prebuilt dir)
if(NOT HYLE_REPO_DIR OR NOT HYLE_PREBUILT_DIR)
  message(FATAL_ERROR "pass -DHYLE_REPO_DIR=<repo> -DHYLE_PREBUILT_DIR=<build>")
endif()

# Find the prebuilt hyle_simd artifact (.so/.dll-import-lib/.a/.lib) in the build's bin/.
find_library(HYLE_SIMD_LIB NAMES hyle_simd
  PATHS "${HYLE_PREBUILT_DIR}/bin" "${HYLE_PREBUILT_DIR}/lib" "${HYLE_PREBUILT_DIR}"
  REQUIRED NO_DEFAULT_PATH)
message(STATUS "prebuilt hyle_simd (${HYLE_LINK_MODE}): ${HYLE_SIMD_LIB}")

add_executable(prebuilt_consumer consumer.cc)
target_compile_features(prebuilt_consumer PRIVATE cxx_std_23)
target_include_directories(prebuilt_consumer PRIVATE "${HYLE_REPO_DIR}/include")
target_link_libraries(prebuilt_consumer PRIVATE "${HYLE_SIMD_LIB}")

# A static HYLE build needs HYLE_STATIC defined in the consumer too, so HYLE_SIMD_API
# expands to nothing (matching how the .a was compiled) instead of dllimport.
if(HYLE_LINK_MODE STREQUAL "static")
  target_compile_definitions(prebuilt_consumer PRIVATE HYLE_STATIC)
endif()

# Shared: let the runtime loader find the .so next to the prebuilt build (Linux RPATH).
# Windows finds the .dll via PATH / same-dir at run time (the run script handles it).
set_target_properties(prebuilt_consumer PROPERTIES
  BUILD_RPATH "${HYLE_PREBUILT_DIR}/bin")
