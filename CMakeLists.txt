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

# Keep IDE projects navigable without imposing folder policy when HYLE is
# embedded in a larger parent project. Individual HYLE targets still declare
# their FOLDER below; this top-level gate is what makes Visual Studio render it.
if(PROJECT_IS_TOP_LEVEL)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")
endif()

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

# Resolve platform capability before the component graph.  io/compute/graph
# have useful sched-free product shapes on targets for which HYLE does not yet
# ship fcontext assembly; making sched an unconditional graph edge made those
# implementations unreachable even though their module CMakeLists supported
# them.  This is a capability fact, not a "selected target exists" fact.
set(HYLE_FCONTEXT_SUPPORTED FALSE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  if(MSVC OR CMAKE_SYSTEM_NAME STREQUAL "Linux" OR APPLE)
    set(HYLE_FCONTEXT_SUPPORTED TRUE)
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  if(APPLE OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(HYLE_FCONTEXT_SUPPORTED TRUE)
  endif()
endif()

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

include(HyleBuildContract)
include(HyleWarnings)
include(HyleSanitizers)
include(HyleModule)
include(HylePackageRegistry)
include(HyleIsa)
include(HyleShader)
include(HyleTrace)    # obs-02: HYLE_TRACE gate + optional tracy backend (default OFF)
include(HyleTesting)  # HYLE_TESTING gate: optional test-only instrumentation

# ---- module-owned optional dependencies -------------------------------------
# Keep cache controls declarable for presets/embedding parents, but perform SDK
# discovery and source fetching only for components in the resolved closure.
set(HYLE_DEPS_DIR "${HYLE_BINARY_DIR}/_deps" CACHE PATH
    "Build-tree directory for dependencies fetched by HYLE")
include(HyleIoDependencies)
include(HyleComputeDependencies)

if(HYLE_BUILD_IO)
  hyle_configure_io_dependencies()
endif()
if(HYLE_BUILD_COMPUTE)
  hyle_configure_compute_dependencies()
endif()

# ---- modules (component closure + dependency order) -------------------------
# sched fiber switching needs platform fcontext asm. Vendored build integration
# currently covers x86_64 Windows/Linux/macOS and arm64 Linux/macOS.  The
# component graph only selects sched transitively where that capability exists;
# an explicit unsupported sched request still fails here with a useful error.
set(HYLE_HAS_FCONTEXT FALSE)
if(HYLE_BUILD_SCHED)
  if(NOT HYLE_FCONTEXT_SUPPORTED)
    message(FATAL_ERROR
      "The selected component set requires sched, but HYLE has no fcontext assembly for "
      "${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}/${CMAKE_CXX_COMPILER_ID}")
  endif()
  set(HYLE_HAS_FCONTEXT TRUE)
  if(MSVC)
    enable_language(ASM_MASM)
  else()
    enable_language(ASM)
  endif()
endif()

foreach(_component IN LISTS HYLE_KNOWN_COMPONENTS)
  string(TOUPPER "${_component}" _component_upper)
  if(HYLE_BUILD_${_component_upper})
    add_subdirectory("src/${_component}")
  endif()
endforeach()
unset(_component)
unset(_component_upper)

# Module CMakeLists may add external/private implementation libraries after
# hyle_add_module().  Audit the completed HYLE-to-HYLE link graph here so a
# later internal edge cannot bypass the declaration-time component check.
hyle_validate_final_module_graph()

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
