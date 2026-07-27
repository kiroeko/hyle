cmake_minimum_required(VERSION 3.25)

# The checked-in C/C++ header is the single version source, including for a
# component package that does not build hyle_core.  Read it before project() so
# package metadata, loader identities and runtime assertions cannot drift from
# the public macros.
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/hyle/core/version_defs.h"
  _hyle_version_lines REGEX
  "^#define HYLE_VERSION_(MAJOR|MINOR|PATCH) [0-9]+$")
foreach(_hyle_version_part IN ITEMS MAJOR MINOR PATCH)
  foreach(_hyle_version_line IN LISTS _hyle_version_lines)
    if(_hyle_version_line MATCHES
        "^#define HYLE_VERSION_${_hyle_version_part} ([0-9]+)$")
      set(_hyle_version_${_hyle_version_part} "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  if(NOT DEFINED _hyle_version_${_hyle_version_part})
    message(FATAL_ERROR
      "Cannot read HYLE_VERSION_${_hyle_version_part} from version_defs.h")
  endif()
endforeach()
set(_hyle_project_version
  "${_hyle_version_MAJOR}.${_hyle_version_MINOR}.${_hyle_version_PATCH}")

project(hyle
  VERSION "${_hyle_project_version}"
  DESCRIPTION "HYbrid LEvels - general-purpose high-performance C++ infra"
  LANGUAGES C CXX)

unset(_hyle_project_version)
unset(_hyle_version_lines)
unset(_hyle_version_line)
unset(_hyle_version_part)
unset(_hyle_version_MAJOR)
unset(_hyle_version_MINOR)
unset(_hyle_version_PATCH)

# A pre-1.0 minor release may change the C and C++ ABI.  Give each 0.x minor
# its own loader identity instead of claiming that every 0.x package is binary
# compatible.  Once the project reaches 1.0, the conventional major-only ABI
# identity applies.
if(PROJECT_VERSION_MAJOR EQUAL 0)
  set(HYLE_ABI_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")
else()
  set(HYLE_ABI_VERSION "${PROJECT_VERSION_MAJOR}")
endif()

set(HYLE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(HYLE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
list(APPEND CMAKE_MODULE_PATH "${HYLE_SOURCE_DIR}/cmake")
include(HyleComponents)
include(HyleFeatureOption)

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
option(HYLE_BUILD_TESTS "build unit tests (GoogleTest)" ${PROJECT_IS_TOP_LEVEL})
option(HYLE_TEST_INSTRUMENTATION
  "compile test-only fault injection and structural diagnostics" ON)
option(HYLE_BUILD_BENCH "build microbenchmarks (nanobench)" OFF)
option(HYLE_INSTALL "generate install rules and a relocatable CMake package" OFF)
if(HYLE_INSTALL AND HYLE_BUILD_TESTS)
  message(FATAL_ERROR
    "HYLE_INSTALL=ON requires HYLE_BUILD_TESTS=OFF. Test configurations are validation "
    "artifacts; the instrumented form may change layout, and neither form is an SDK.")
endif()
set(HYLE_INSTALL_INCLUDEDIR "include")
if(HYLE_INSTALL)
  include(GNUInstallDirs)
  set(HYLE_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}")
  foreach(_hyle_install_dir IN ITEMS BINDIR LIBDIR INCLUDEDIR DATADIR)
    if(IS_ABSOLUTE "${CMAKE_INSTALL_${_hyle_install_dir}}")
      message(FATAL_ERROR
        "HYLE's relocatable package requires relative GNUInstallDirs values; "
        "CMAKE_INSTALL_${_hyle_install_dir}='${CMAKE_INSTALL_${_hyle_install_dir}}' "
        "is absolute. Set CMAKE_INSTALL_PREFIX for the destination instead.")
    endif()
  endforeach()
  unset(_hyle_install_dir)
endif()
set(HYLE_SIMD_EXACT_TARGET "" CACHE STRING
    "Build hyle_simd for one required ISA; empty enables runtime dispatch")
set_property(CACHE HYLE_SIMD_EXACT_TARGET PROPERTY STRINGS
             "" SCALAR SSE4 AVX2 AVX512 NEON SVE RVV)
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
if(PROJECT_IS_TOP_LEVEL)
  if(_hyle_multi_config)
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "" FORCE)
  elseif(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)  # perf-first top-level default
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo)
  endif()
endif()

# Unify runtime artifacts (.exe + .dll on Windows) into one dir so test/bench
# exes find the shared libs they link against without PATH juggling.
if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
# Static archives (.a/.lib) land in the SAME bin/ as shared libs, so a prebuilt-library
# consumer (samples/04) finds the artifact in one predictable place regardless of
# HYLE_STATIC. Without this, static .a defaults to per-target build subdirs.
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

include(HyleWarnings)
include(HyleSanitizers)
include(HyleModule)
include(HyleTrace)    # obs-02: HYLE_TRACE gate + optional tracy backend (default OFF)
include(HyleTesting)  # HYLE_TESTING gate: optional test-only instrumentation

# ---- optional OpenSSL (io-08 TLS) -------------------------------------------
# AUTO preserves convenient source builds; ON makes TLS part of the requested
# product contract; OFF is deterministic and does not even probe OpenSSL.
hyle_auto_feature_option(HYLE_IO_TLS "Build the io TLS capability using OpenSSL")
set(HYLE_IO_TLS_ENABLED OFF)
if(HYLE_BUILD_IO AND NOT HYLE_IO_TLS STREQUAL "OFF")
  find_package(OpenSSL QUIET)
  if(OpenSSL_FOUND)
    set(HYLE_IO_TLS_ENABLED ON)
    message(STATUS
      "HYLE: io TLS ${HYLE_IO_TLS} -- enabled with OpenSSL ${OPENSSL_VERSION}")
  elseif(HYLE_IO_TLS STREQUAL "ON")
    message(FATAL_ERROR
      "HYLE_IO_TLS=ON requires OpenSSL, but it was not found. Set OPENSSL_ROOT_DIR "
      "or CMAKE_TOOLCHAIN_FILE, or select HYLE_IO_TLS=AUTO/OFF.")
  else()
    message(STATUS "HYLE: io TLS AUTO -- OpenSSL not found, capability disabled")
  endif()
elseif(HYLE_BUILD_IO)
  message(STATUS "HYLE: io TLS OFF -- dependency probe skipped")
endif()

# ---- HTTP/2 protocol core (io-11) ------------------------------------------
# nghttp2 owns RFC 9113 framing, HPACK, stream state and flow control. HYLE owns
# the coroutine transport over TcpStream/TlsStream and never imports nghttp2's
# event loop, threads, applications or examples. Keep the protocol dependency
# pinned: HTTP/2 parsers are security-sensitive and an uncontrolled system
# version would make identical HYLE builds behave differently.
option(HYLE_IO_HTTP2 "io-11: HTTP/2 via nghttp2 protocol core" ON)
set(HYLE_DEPS_DIR "${HYLE_BINARY_DIR}/_deps" CACHE PATH
    "Build-tree directory for dependencies fetched by HYLE")
if(HYLE_BUILD_IO AND HYLE_IO_HTTP2)
  set(FETCHCONTENT_BASE_DIR "${HYLE_DEPS_DIR}")
  include(FetchContent)
  set(NGHTTP2_PIN_COMMIT "68cb6900fde14c77f0cd7add0e094a862960eb99") # v1.69.0
  FetchContent_Declare(nghttp2
    GIT_REPOSITORY https://github.com/nghttp2/nghttp2.git
    GIT_TAG        ${NGHTTP2_PIN_COMMIT}
    GIT_SHALLOW    TRUE)
  # nghttp2 is a pinned PRIVATE protocol kernel, not a second public/runtime ABI.
  # Always build one PIC archive: a shared hyle_io absorbs it; a static hyle_io
  # exports the archive as LINK_ONLY. block() keeps nghttp2's generic options from
  # changing an embedding parent's BUILD_{SHARED,STATIC}_LIBS contract.
  block()
    set(ENABLE_LIB_ONLY ON)
    set(ENABLE_APP OFF)
    set(ENABLE_EXAMPLES OFF)
    set(ENABLE_HPACK_TOOLS OFF)
    set(ENABLE_DOC OFF)
    set(ENABLE_HTTP3 OFF)
    set(ENABLE_FAILMALLOC OFF)
    set(ENABLE_THREADS OFF)
    set(BUILD_TESTING OFF)
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_STATIC_LIBS ON)
    FetchContent_MakeAvailable(nghttp2)
  endblock()
  FetchContent_GetProperties(nghttp2)
  # The upstream subproject owns no part of HYLE's install layout. Ignore its
  # generic install rules; HylePackage installs the exact target/header/license
  # closure with HYLE component ownership.
  set_property(DIRECTORY "${nghttp2_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)
  set_target_properties(nghttp2_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
  message(STATUS
    "HYLE: nghttp2 v1.69.0 (${NGHTTP2_PIN_COMMIT}) -- io-11 HTTP/2 enabled "
    "as a private static protocol core")
endif()

# ---- optional CUDA (compute-01 CUDA backend) --------------------------------
# CUDA follows the same deterministic AUTO/ON/OFF contract as TLS.  Keep the
# public compile definition HYLE_COMPUTE_CUDA; the CMake-side result has the
# distinct *_ENABLED name so a string value such as "OFF" is never truth-tested.
hyle_auto_feature_option(HYLE_COMPUTE_CUDA "Build the compute CUDA backend")
set(HYLE_COMPUTE_CUDA_ENABLED OFF)
if(HYLE_BUILD_COMPUTE AND NOT HYLE_COMPUTE_CUDA STREQUAL "OFF")
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR
       CMAKE_CUDA_ARCHITECTURES STREQUAL "")
      set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING
        "CUDA architectures for HYLE and fetched CUDA kernels")
    endif()
    if(HYLE_INSTALL AND "native" IN_LIST CMAKE_CUDA_ARCHITECTURES)
      message(FATAL_ERROR
        "A CUDA-enabled binary SDK cannot use CMAKE_CUDA_ARCHITECTURES=native; "
        "that would bind the package to the build GPU. Set an explicit list "
        "(for example 80;86;89) or all-major, or select HYLE_COMPUTE_CUDA=OFF.")
    endif()
    enable_language(CUDA)
    set(HYLE_COMPUTE_CUDA_ENABLED ON)
    message(STATUS
      "HYLE: compute CUDA ${HYLE_COMPUTE_CUDA} -- enabled with Toolkit ${CUDAToolkit_VERSION}")
  elseif(HYLE_COMPUTE_CUDA STREQUAL "ON")
    message(FATAL_ERROR
      "HYLE_COMPUTE_CUDA=ON requires CUDAToolkit, but it was not found. Set "
      "CUDAToolkit_ROOT or CMAKE_TOOLCHAIN_FILE, or select HYLE_COMPUTE_CUDA=AUTO/OFF.")
  else()
    message(STATUS "HYLE: compute CUDA AUTO -- Toolkit not found, host backend only")
  endif()
elseif(HYLE_BUILD_COMPUTE)
  message(STATUS "HYLE: compute CUDA OFF -- dependency probe and CUDA language skipped")
endif()

# ---- optional ggml operator library (compute-05 ggml_bridge) -----------------
# ggml is BORROWED as an operator-kernel supplier (compute-05: matmul/softmax/... on
# HYLE-managed VRAM + completion + scheduling). Same "opt-in capability = opt-in
# dependency" rule as CUDA/OpenSSL: HYLE_COMPUTE_GGML gates ggml_bridge.{h,cc}; without
# it compute still builds (01-04) with zero ggml dependency.
#
# INTEGRATION = FetchContent PINNED BY COMMIT (NOT find_package(ggml)): a system-installed
# ggml has UNCONTROLLED build options (OpenMP on/off, backends, version), but the
# n_threads=1 leaf contract (compute-design 发现 10) REQUIRES OpenMP OFF and zero-copy
# REQUIRES a pinned version. So we own the build via the knobs below, pinned to a commit.
#
# Fetches live under HYLE_DEPS_DIR (build-tree `_deps` by default), never in the source
# tree. A developer who wants one cache shared by several disposable build trees can set
# HYLE_DEPS_DIR explicitly; a source-embedded consumer otherwise gets an isolated cache.
option(HYLE_COMPUTE_GGML "compute-05: borrow ggml operators via ggml_bridge" ON)
set(_hyle_ggml_native_default ON)
if(HYLE_INSTALL)
  set(_hyle_ggml_native_default OFF)
endif()
option(HYLE_COMPUTE_GGML_NATIVE
  "Tune ggml CPU kernels for the build CPU (not portable across CPUs)"
  ${_hyle_ggml_native_default})
unset(_hyle_ggml_native_default)
if(HYLE_BUILD_COMPUTE AND HYLE_COMPUTE_GGML)
  set(FETCHCONTENT_BASE_DIR "${HYLE_DEPS_DIR}")
  include(FetchContent)
  set(GGML_PIN_COMMIT "eced84c86f8b012c752c016f7fe789adea168e1e")  # v0.15.3

  FetchContent_Declare(ggml
    GIT_REPOSITORY https://github.com/ggml-org/ggml.git
    GIT_TAG        ${GGML_PIN_COMMIT}
    GIT_SHALLOW    TRUE)
  # ggml is a pinned implementation kernel, not a second public/runtime ABI.
  # Always build a PIC static closure: shared hyle_compute absorbs it; static
  # hyle_compute exports it through LINK_ONLY.  Normal variables + block() honor
  # upstream option() without FORCE-writing an embedding parent's cache.
  block()
    # The hook runs inside ggml's project() before language detection.  Its own
    # cmake_minimum_required(...3.28) would otherwise reset a parent policy.
    set(HYLE_GGML_PARENT_PROJECT_INCLUDE_BEFORE
      "${CMAKE_PROJECT_ggml_INCLUDE_BEFORE}")
    set(CMAKE_PROJECT_ggml_INCLUDE_BEFORE
      "${HYLE_SOURCE_DIR}/cmake/HyleGgmlProjectPolicy.cmake")
    set(BUILD_SHARED_LIBS OFF)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(GGML_STANDALONE OFF)
    set(GGML_BUILD_TESTS OFF)
    set(GGML_BUILD_EXAMPLES OFF)
    set(GGML_BUILD_SERVER OFF)
    set(GGML_BACKEND_DL OFF)
    set(GGML_OPENMP OFF)       # n_threads=1 leaf; HYLE owns outer parallelism
    set(GGML_CCACHE OFF)
    set(GGML_NATIVE ${HYLE_COMPUTE_GGML_NATIVE})
    set(GGML_CPU ON)
    set(GGML_CPU_ALL_VARIANTS OFF)
    set(GGML_ACCELERATE OFF)
    set(GGML_BLAS OFF)
    set(GGML_LLAMAFILE OFF)
    set(GGML_METAL OFF)
    set(GGML_RPC OFF)
    set(GGML_VULKAN OFF)
    set(GGML_CUDA ${HYLE_COMPUTE_CUDA_ENABLED})
    set(GGML_CUDA_FA OFF)      # bridge uses GEMM/elementwise, not FlashAttention
    set(GGML_CUDA_NCCL OFF)    # no accidental machine-local multi-GPU dependency
    # A binary SDK defaults to the x86-64-v2/SSE4 contract. A deliberately exact
    # HYLE ISA package raises ggml to the same floor; a source/local build may
    # explicitly retain upstream's build-CPU tuning via *_GGML_NATIVE.
    if(NOT HYLE_COMPUTE_GGML_NATIVE AND
       CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
      set(GGML_SSE42 ON)
      set(GGML_AVX OFF)
      set(GGML_AVX2 OFF)
      set(GGML_BMI2 OFF)
      set(GGML_AVX512 OFF)
      set(GGML_AVX512_VBMI OFF)
      set(GGML_AVX512_VNNI OFF)
      set(GGML_AVX512_BF16 OFF)
      if(NOT MSVC)
        set(GGML_FMA OFF)
        set(GGML_F16C OFF)
      endif()
      if(HYLE_SIMD_EXACT_TARGET STREQUAL "AVX2" OR
         HYLE_SIMD_EXACT_TARGET STREQUAL "AVX512")
        set(GGML_AVX ON)
        set(GGML_AVX2 ON)
        set(GGML_BMI2 ON)
        if(NOT MSVC)
          set(GGML_FMA ON)
          set(GGML_F16C ON)
        endif()
      endif()
      if(HYLE_SIMD_EXACT_TARGET STREQUAL "AVX512")
        set(GGML_AVX512 ON)
      endif()
    endif()
    FetchContent_MakeAvailable(ggml)
  endblock()
  FetchContent_GetProperties(ggml)
  set_property(DIRECTORY "${ggml_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)
  foreach(_hyle_ggml_target IN ITEMS ggml ggml-base ggml-cpu ggml-cuda)
    if(TARGET ${_hyle_ggml_target})
      set_target_properties(${_hyle_ggml_target} PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    endif()
  endforeach()
  unset(_hyle_ggml_target)
  message(STATUS
    "HYLE: ggml ${GGML_PIN_COMMIT} -- private PIC static CPU kernel "
    "(native=${HYLE_COMPUTE_GGML_NATIVE}, exact=${HYLE_SIMD_EXACT_TARGET}, "
    "CUDA=${HYLE_COMPUTE_CUDA_ENABLED}, cache: ${HYLE_DEPS_DIR})")
endif()


# ---- optional Vulkan compute (compute-06 self-authored SPIR-V kernels) --------
# compute-06 gives HYLE its OWN cross-vendor GPU compute path (SPIR-V), the Vulkan
# twin of compute-04's CUDA NVRTC "any kernel". Same "opt-in capability = opt-in
# dependency" rule as CUDA/ggml: HYLE_COMPUTE_VULKAN gates the Vulkan backend +
# vulkan_bridge; without it compute still builds (01-05) with zero Vulkan dependency.
#
# WHY volk (NOT find_package(Vulkan)): the Vulkan LOADER is the one piece we can't just
# FetchContent -- but linking it the classic way needs vulkan-1.lib, which only ships
# with an installed SDK (this machine has the runtime vulkan-1.dll from the driver but
# NO .lib -- find_package(Vulkan) fails). volk (zeux/volk, the industry-standard Vulkan
# meta-loader -- NOT a glad-style code generator, a fixed 2-file lib) sidesteps this: it
# dlopen's the runtime loader at startup (volkInitialize -> vkGetInstanceProcAddr) and
# resolves all vk* pointers itself, so NO import lib / NO SDK install is needed -- only
# the driver's runtime loader (always present). Bonus: device-level function resolution
# skips the loader trampoline (one less indirection per call = faster, matches HYLE's
# perf goal). kompute uses the same dynamic-dispatch idea (VULKAN_HPP_DEFAULT_DISPATCHER).
#
# THREE PINNED DEPS at vulkan-sdk-1.4.341.0 (matches the local runtime
# loader 1.4.341 -- zero version skew), all Khronos/zeux source repos (NOT the LunarG SDK
# packaging layer), landing in HYLE_DEPS_DIR when their owning configuration needs them:
#   1. Vulkan-Headers (Khronos)  -- pure headers, version OWNED by HYLE not the machine.
#   2. volk (zeux)               -- meta-loader; runtime dlopen, no .lib/SDK needed.
#   3. glslang (Khronos)         -- GLSL->SPIR-V: ENABLE_OPT=OFF (no SPIRV-Tools dep),
#      tests/install/HLSL off; fetched only for HYLE tests/bench shader generation.
# The slate-bundled Vulkan (headers v270, stale) is NOT used.
option(HYLE_COMPUTE_VULKAN "compute-06: self-authored Vulkan compute SPIR-V kernels" ON)
set(HYLE_COMPUTE_VULKAN_ENABLED OFF)
set(HYLE_COMPUTE_SHADER_TOOLS_ENABLED OFF)
if(HYLE_BUILD_COMPUTE AND HYLE_COMPUTE_VULKAN)
  set(FETCHCONTENT_BASE_DIR "${HYLE_DEPS_DIR}")
  include(FetchContent)
  # All pinned to vulkan-sdk-1.4.341.0 (== local loader 1.4.341, no skew).
  set(VULKAN_HEADERS_PIN "b5c8f996196ba4aa6d8f97e52b5d3b6e70f7e4e2")  # Vulkan-Headers 1.4.341.0
  set(VOLK_PIN           "d979819fd03de3a7606d6e23d6ff4968942599da")  # volk 1.4.341.0
  set(GLSLANG_PIN        "f0bd0257c308b9a26562c1a30c4748a0219cc951")  # glslang 1.4.341.0

  # 1. Vulkan-Headers first -- creates the Vulkan::Headers target volk can see.
  set(VULKAN_HEADERS_ENABLE_TESTS OFF)
  set(VULKAN_HEADERS_ENABLE_INSTALL OFF)
  set(VULKAN_HEADERS_ENABLE_MODULE OFF)
  FetchContent_Declare(vulkan_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG        ${VULKAN_HEADERS_PIN}
    GIT_SHALLOW    TRUE)
  FetchContent_MakeAvailable(vulkan_headers)

  # 2. volk -- point it at the fetched headers (VOLK_PULL_IN_VULKAN tries this var first).
  set(VULKAN_HEADERS_INSTALL_DIR "${vulkan_headers_SOURCE_DIR}")
  set(VOLK_PULL_IN_VULKAN ON)
  set(VOLK_INSTALL OFF)
  FetchContent_Declare(volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG        ${VOLK_PIN}
    GIT_SHALLOW    TRUE)
  FetchContent_MakeAvailable(volk)
  FetchContent_GetProperties(volk)
  set_property(DIRECTORY "${vulkan_headers_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)
  set_property(DIRECTORY "${volk_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)
  set_target_properties(volk PROPERTIES POSITION_INDEPENDENT_CODE ON)
  # Replace volk's raw fetched-source include paths with a relocatable two-sided
  # contract.  The install root contains volk.h plus vulkan/ and vk_video/.
  set_property(TARGET volk PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    "$<BUILD_INTERFACE:${volk_SOURCE_DIR}>;$<BUILD_INTERFACE:${vulkan_headers_SOURCE_DIR}/include>;$<INSTALL_INTERFACE:${HYLE_INSTALL_INCLUDEDIR}/hyle/vendor/volk>")

  # glslang/Python generate test and benchmark shaders; no product source links
  # glslang.  Keep this heavy build-time tool out of a binary SDK configuration.
  if(HYLE_BUILD_TESTS OR HYLE_BUILD_BENCH)
    block()
      set(ENABLE_OPT OFF)
      set(ENABLE_HLSL OFF)
      set(GLSLANG_TESTS OFF)
      set(GLSLANG_ENABLE_INSTALL OFF)
      set(BUILD_EXTERNAL OFF)
      set(ENABLE_SPIRV ON)
      set(ENABLE_GLSLANG_BINARIES ON)
      FetchContent_Declare(glslang
        GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
        GIT_TAG        ${GLSLANG_PIN}
        GIT_SHALLOW    TRUE)
      FetchContent_MakeAvailable(glslang)
    endblock()
    FetchContent_GetProperties(glslang)
    set_property(DIRECTORY "${glslang_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(HYLE_COMPUTE_SHADER_TOOLS_ENABLED ON)
  endif()

  set(HYLE_COMPUTE_VULKAN_ENABLED ON)
  message(STATUS
    "HYLE: Vulkan compute enabled -- private PIC volk + Vulkan-Headers 1.4.341 "
    "(shader tools=${HYLE_COMPUTE_SHADER_TOOLS_ENABLED}, cache: ${HYLE_DEPS_DIR}; "
    "loader dlopen'd at runtime)")
endif()


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

# ---- modules (component closure + dependency order) -------------------------
# sched fiber switching needs platform fcontext asm. Vendored asm covers x86_64 (MS/PE MASM,
# SysV/ELF GAS, SysV/macho GAS) AND arm64 (AAPCS ELF/macho/PE). Wire each (platform, arch)
# to its flavor; the sched CMakeLists picks the matching .S/.asm. Gate the module on having a
# usable asm flavor for this target.
set(HYLE_HAS_FCONTEXT FALSE)
if(HYLE_BUILD_SCHED)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    if(MSVC OR CMAKE_SYSTEM_NAME STREQUAL "Linux" OR APPLE)
      set(HYLE_HAS_FCONTEXT TRUE)
    endif()
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    if(APPLE OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
      set(HYLE_HAS_FCONTEXT TRUE)
    endif()
  endif()
  if(NOT HYLE_HAS_FCONTEXT)
    message(FATAL_ERROR
      "The selected component set requires sched, but HYLE has no fcontext assembly for "
      "${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}/${CMAKE_CXX_COMPILER_ID}")
  endif()
  if(MSVC)
    enable_language(ASM_MASM)
  else()
    enable_language(ASM)
  endif()
endif()

foreach(_component core error obs math platform time mem simd container util sched io compute graph)
  string(TOUPPER "${_component}" _component_upper)
  if(HYLE_BUILD_${_component_upper})
    add_subdirectory("src/${_component}")
  endif()
endforeach()
unset(_component)
unset(_component_upper)

include(HylePackage)
hyle_install_package()
include(HyleSourcePackage)
hyle_add_source_package()

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

if(HYLE_STATIC)
  set(_hyle_linkage static)
else()
  set(_hyle_linkage shared)
endif()
message(STATUS "HYLE ${PROJECT_VERSION} | ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} | C++${CMAKE_CXX_STANDARD} | sanitizer=${HYLE_SANITIZER} | linkage=${_hyle_linkage} | components=${HYLE_RESOLVED_COMPONENTS}")
unset(_hyle_linkage)
