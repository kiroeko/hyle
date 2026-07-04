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
# VALIDATE + NORMALIZE the value. set_property(STRINGS) only fills the GUI dropdown; it does
# NOT reject a bad -D value, so `-DHYLE_SANITIZER=thread` (a natural mistake -- it's the
# compiler's own -fsanitize=THREAD name) used to fall through every STREQUAL and silently
# build with NO instrumentation -> a green test run that proved nothing (a FALSE pass). Accept
# the compiler-native aliases (address/thread/undefined/none) AND hard-FAIL on anything else,
# so a typo can never again degrade to "off" unnoticed.
string(TOLOWER "${HYLE_SANITIZER}" _san_lc)
if(_san_lc STREQUAL "address")
  set(_san_lc "asan")
elseif(_san_lc STREQUAL "thread")
  set(_san_lc "tsan")
elseif(_san_lc STREQUAL "undefined")
  set(_san_lc "ubsan")
elseif(_san_lc STREQUAL "none" OR _san_lc STREQUAL "")
  set(_san_lc "off")
endif()
if(NOT _san_lc MATCHES "^(off|asan|tsan|ubsan)$")
  message(FATAL_ERROR
    "HYLE_SANITIZER='${HYLE_SANITIZER}' is not valid. Use one of: off | asan | tsan | ubsan "
    "(aliases address/thread/undefined/none also accepted). Refusing to silently build "
    "WITHOUT instrumentation -- that would make a sanitizer run a false pass.")
endif()
set(HYLE_SANITIZER "${_san_lc}" CACHE STRING "off|asan|tsan|ubsan" FORCE)
message(STATUS "HYLE_SANITIZER = ${HYLE_SANITIZER}")

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
# Static archives (.a/.lib) land in the SAME bin/ as shared libs, so a prebuilt-library
# consumer (samples/04) finds the artifact in one predictable place regardless of
# HYLE_STATIC. Without this, static .a defaults to per-target build subdirs.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(HyleWarnings)
include(HyleSanitizers)
include(HyleModule)
include(HyleTrace)  # obs-02: HYLE_TRACE gate + optional tracy backend (default OFF)

# ---- global correctness flags (apply to ALL targets incl. fetched deps) ------
# These change language semantics, not just diagnostics, so they must be uniform
# across the whole build (gtest included) to avoid ODR/ABI skew. Scoped to C/CXX
# via a generator expression so they are NOT passed to the assembler (ml64 would
# choke on /Zc:__cplusplus etc. when we compile fcontext .asm).
if(MSVC)
  # /Zc:__cplusplus  : report the true __cplusplus value (else MSVC pins 199711L
  #                    and every C++23 feature-test macro misfires).
  # /utf-8           : treat source AND execution charset as UTF-8 (kills C4819).
  # One quoted genex PER flag: a ';' inside $<...> is list-split by
  # add_compile_options before evaluation, which would mangle the flags.
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:C,CXX>:/Zc:__cplusplus>"
    "$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>")
  # fcontext fiber switching needs an assembler. On MSVC x64 that is MASM (ml64).
  enable_language(ASM_MASM)
endif()

# ---- x86-64-v2 baseline: SSSE3 everywhere (GCC/Clang) ------------------------
# SwissTable's GroupSse2::match_empty (probe.h, a HEADER-inline member) has a 1-insn
# SSSE3 fast path gated on HYLE_HAVE_SSSE3. MSVC exposes SSSE3 intrinsics on x64
# unconditionally; GCC/Clang gate __SSSE3__ behind -mssse3. This MUST be GLOBAL (not
# per-module): probe.h is inlined into container TUs AND test/bench TUs (via registry.h
# etc.), so a per-module flag would compile match_empty two different ways in one binary
# = ODR violation. Setting it project-wide makes every TU agree. SSSE3 is the x86-64-v2
# baseline (universal since ~2009); a distributable binary requiring it is standard. This
# does NOT touch the wide AVX2/AVX512 SIGILL contract -- those stay confined to the
# per-ISA kernel TUs in simd/ (which add their own -mavx2 on top).
if(NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mssse3" HYLE_CXX_HAS_MSSSE3)
  if(HYLE_CXX_HAS_MSSSE3)
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-mssse3>")
  endif()
endif()

# ---- modules (added per dependency layer, roadmap P-1..P6) ------------------
add_subdirectory(src/core)   # 00-scaffolding: empty core + macro skeleton
add_subdirectory(src/error)  # error-01: Status + Result + propagation macros
add_subdirectory(src/obs)    # obs-01: structured log production (sink registry)
add_subdirectory(src/platform)  # platform-01: thread + park (wait_on_address)
add_subdirectory(src/mem)    # mem-01: Allocator vtable + alignment fix
add_subdirectory(src/time)   # time-01: monotonic + realtime clock + sleep
add_subdirectory(src/util)   # util-01: ref_counted (Ref<T>) + scope_guard (DEFER)
add_subdirectory(src/math)   # math: scalar helpers (div_ceil/sgn/approx_eq)
add_subdirectory(src/container)  # container-01: bounded MPMC ring (Vyukov)
add_subdirectory(src/simd)   # simd-01: runtime CPUID dispatch + vec<T,N> D-tag + scalar

# sched fiber switching needs platform fcontext asm. We have vendored x86_64 asm
# for MS/PE (MASM) and SysV/ELF (GAS); wire those two. arm64 + macho asm are
# vendored but not yet wired (no test machine). Gate the module on having a
# usable asm flavor for this (platform, arch).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  if(MSVC)
    set(HYLE_HAS_FCONTEXT TRUE)
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(HYLE_HAS_FCONTEXT TRUE)  # GAS .S via the ASM language
  endif()
endif()
if(HYLE_HAS_FCONTEXT)
  add_subdirectory(src/sched)  # sched-01/02: fcontext fiber + run-queue/worker
endif()
add_subdirectory(src/io)     # io-01: proactor completion seam (IOCP; io_uring next)

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
