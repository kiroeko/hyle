cmake_minimum_required(VERSION 3.25)

project(hyle
  VERSION 0.1.0
  DESCRIPTION "HYbrid LEvels - general-purpose high-performance C++ infra"
  LANGUAGES C CXX ASM)

# ---- C++23 baseline + toolchain gate ----------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# C baseline. HYLE's own C-ABI surface (extern "C" vtables) is compiled as C++;
# this only governs real .c/.S translation units (vendored C deps like liburing/
# XXH3, fcontext asm shims) so they don't fall back to compiler-default C.
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Toolchain gate (charter: GCC14+ / Clang18+ / MSVC19.43+ for C++23 std::expected etc.)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
  message(FATAL_ERROR "HYLE requires GCC 14+ (got ${CMAKE_CXX_COMPILER_VERSION})")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18)
  message(FATAL_ERROR "HYLE requires Clang 18+ (got ${CMAKE_CXX_COMPILER_VERSION})")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.43)
  message(FATAL_ERROR "HYLE requires MSVC 19.43+ / VS 17.13+ (got ${CMAKE_CXX_COMPILER_VERSION})")
endif()

# ---- global options ---------------------------------------------------------
option(HYLE_BUILD_TESTS "build unit tests (GoogleTest)" ON)
option(HYLE_BUILD_BENCH "build microbenchmarks (nanobench)" OFF)
option(HYLE_STATIC      "build static lib (default SHARED)" OFF)
option(HYLE_WERROR      "treat warnings as errors (CI on, local off)" OFF)
set(HYLE_SANITIZER "off" CACHE STRING "off|asan|tsan|ubsan")
set_property(CACHE HYLE_SANITIZER PROPERTY STRINGS off asan tsan ubsan)

# Build configurations. Multi-config generators (VS, Xcode, Ninja Multi-Config)
# pick per-build via --config; single-config (plain Ninja, Makefiles) via
# CMAKE_BUILD_TYPE. RelWithDebInfo kept: a perf lib needs an optimized build that
# still profiles/symbolicates. MinSizeRel dropped (irrelevant for infra).
get_property(_hyle_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(_hyle_multi_config)
  set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "" FORCE)
elseif(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)  # perf-first default
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo)
endif()

if(HYLE_STATIC)
  set(BUILD_SHARED_LIBS OFF)
else()
  set(BUILD_SHARED_LIBS ON)
endif()

# Unify runtime artifacts (.exe + .dll on Windows) into one dir so test/bench
# exes find the shared libs they link against without PATH juggling.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(HyleWarnings)
include(HyleSanitizers)
include(HyleModule)

# ---- global correctness flags (apply to ALL targets incl. fetched deps) ------
# These change language semantics, not just diagnostics, so they must be uniform
# across the whole build (gtest included) to avoid ODR/ABI skew.
if(MSVC)
  # /Zc:__cplusplus  : report the true __cplusplus value (else MSVC pins 199711L
  #                    and every C++23 feature-test macro misfires).
  # /utf-8           : treat source AND execution charset as UTF-8 (kills C4819).
  add_compile_options(/Zc:__cplusplus /utf-8)
endif()

# ---- modules (added per dependency layer, roadmap P-1..P6) ------------------
add_subdirectory(src/core)   # 00-scaffolding: empty core + macro skeleton
add_subdirectory(src/error)  # error-01: Status + Result + propagation macros

if(HYLE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
if(HYLE_BUILD_BENCH)
  add_subdirectory(bench)
endif()

# F5 in a generated VS solution should launch the smoke test, not ALL_BUILD.
if(_hyle_multi_config AND HYLE_BUILD_TESTS AND TARGET hyle_core_test)
  set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT hyle_core_test)
endif()

message(STATUS "HYLE ${PROJECT_VERSION} | ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} | C++${CMAKE_CXX_STANDARD} | sanitizer=${HYLE_SANITIZER} | shared=${BUILD_SHARED_LIBS}")
