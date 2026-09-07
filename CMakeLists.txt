cmake_minimum_required(VERSION 3.25)

include("${CMAKE_CURRENT_LIST_DIR}/cmake/HylePlatformRequirements.cmake")

# The checked-in C++ product header is the single version source. Read it before
# project() so package metadata and public compile-time identity cannot drift.
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/include/hyle/version.h"
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
      "Cannot read HYLE_VERSION_${_hyle_version_part} from hyle/version.h")
  endif()
endforeach()
set(_hyle_project_version
  "${_hyle_version_MAJOR}.${_hyle_version_MINOR}.${_hyle_version_PATCH}")

# platform's native macOS address-wait backend is the public os_sync API added
# in macOS 14.4. Make that loader floor part of the generated binary instead of
# silently inheriting the build host's current OS version. An embedding parent
# that already chose a deployment target remains authoritative and is validated
# after project()/component resolution below.
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND
   CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND
   (NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET OR
    CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL ""))
  set(CMAKE_OSX_DEPLOYMENT_TARGET
      "${HYLE_PLATFORM_MACOS_DEPLOYMENT_TARGET_MIN}" CACHE STRING
      "Minimum macOS version for HYLE's native platform wait backend")
endif()

project(hyle
  VERSION "${_hyle_project_version}"
  DESCRIPTION "HYbrid LEvels - general-purpose high-performance C++ infra"
  LANGUAGES CXX)

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

# This is the shared-object loader/SOVERSION identity for the C++ SDK. HYLE has
# no Native FFI. Before 1.0 every release may change C++ layout or mangling, so
# the full version is the honest loader identity; after 1.0 the conventional
# major-only identity applies.
if(PROJECT_VERSION_MAJOR EQUAL 0)
  set(HYLE_ABI_VERSION "${PROJECT_VERSION}")
else()
  set(HYLE_ABI_VERSION "${PROJECT_VERSION_MAJOR}")
endif()

set(HYLE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(HYLE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
list(APPEND CMAKE_MODULE_PATH "${HYLE_SOURCE_DIR}/cmake")
include(HyleTargetPlatform)
include(HyleDependencyVersions)
include(HyleDependencyCache)

# Resolve platform capability before the component graph. The current public
# io/compute/graph surfaces expose sched types, so selecting any of them also
# selects sched. A future review may split genuinely sched-free product roots;
# the present graph must not advertise a partial shape its headers cannot ship.
set(HYLE_FCONTEXT_SUPPORTED FALSE)
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  if(MSVC OR CMAKE_SYSTEM_NAME STREQUAL "Linux" OR APPLE)
    set(HYLE_FCONTEXT_SUPPORTED TRUE)
  endif()
elseif(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  if(APPLE OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(HYLE_FCONTEXT_SUPPORTED TRUE)
  endif()
endif()

include(HyleComponents)
if(HYLE_BUILD_SCHED AND NOT HYLE_FCONTEXT_SUPPORTED)
  message(FATAL_ERROR
    "The selected component closure requires sched, but HYLE has no fcontext "
    "assembly for ${CMAKE_SYSTEM_NAME}/${HYLE_TARGET_PROCESSOR}/"
    "${CMAKE_CXX_COMPILER_ID}. Select a sched-free component root; splitting "
    "io/compute/graph into smaller product roots is a future module decision.")
endif()
include(HyleFeatureOption)

# C is an internal implementation language, not a product surface. A selected
# mem closure needs it for mimalloc (and every current C-backed protocol/kernel
# closure already reaches mem); C++-only foundation builds should not require a
# C compiler merely because unrelated components can vendor C sources.
if(HYLE_BUILD_MEM)
  enable_language(C)
endif()

if(APPLE AND HYLE_BUILD_PLATFORM_THREAD)
  if(NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET OR
     CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
    message(FATAL_ERROR
      "HYLE's platform_thread factor requires an explicit macOS deployment target "
      "of ${HYLE_PLATFORM_MACOS_DEPLOYMENT_TARGET_MIN} or newer. An embedding "
      "parent must set "
      "CMAKE_OSX_DEPLOYMENT_TARGET before adding HYLE.")
  elseif(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS
         "${HYLE_PLATFORM_MACOS_DEPLOYMENT_TARGET_MIN}")
    message(FATAL_ERROR
      "HYLE's platform_thread factor requires macOS "
      "${HYLE_PLATFORM_MACOS_DEPLOYMENT_TARGET_MIN}+ for the public "
      "os_sync_wait_on_address backend; got "
      "CMAKE_OSX_DEPLOYMENT_TARGET='${CMAKE_OSX_DEPLOYMENT_TARGET}'.")
  endif()
endif()

# ---- C++23 baseline + toolchain gate ----------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Internal C baseline. HYLE has no public C source contract; this governs only
# selected vendored protocol/allocator translation units.
if(CMAKE_C_COMPILER_LOADED)
  set(CMAKE_C_STANDARD 17)
  set(CMAKE_C_STANDARD_REQUIRED ON)
  set(CMAKE_C_EXTENSIONS OFF)
endif()

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
if(HYLE_BUILD_TESTS)
  option(HYLE_TEST_INSTRUMENTATION
    "compile test-only fault injection and structural diagnostics" OFF)
else()
  set(HYLE_TEST_INSTRUMENTATION OFF)
endif()
option(HYLE_BUILD_BENCH "build microbenchmarks (nanobench)" OFF)
option(HYLE_INSTALL "generate install rules and a relocatable CMake package" OFF)
if(HYLE_INSTALL AND DEFINED HYLE_DEV_FACTORS)
  message(FATAL_ERROR
    "HYLE_INSTALL=ON cannot be combined with HYLE_DEV_FACTORS. Internal factor "
    "selection is a source-maintenance journey; install a complete product "
    "selected through HYLE_COMPONENTS instead.")
endif()
if(HYLE_INSTALL AND HYLE_BUILD_TESTS)
  message(FATAL_ERROR
    "HYLE_INSTALL=ON requires HYLE_BUILD_TESTS=OFF. Test configurations are validation "
    "artifacts; the instrumented form may change layout, and neither form is an SDK.")
endif()

# Lock projection is global, but vendored source requirements follow the
# selected component/feature closure.  This keeps a core-only partial source
# checkout independent from mem/container/io/sched/benchmark vendors while
# still auditing every snapshot before its first consumer is configured.
set(_hyle_active_in_tree_dependencies "")
if(HYLE_BUILD_MEM)
  list(APPEND _hyle_active_in_tree_dependencies mimalloc)
endif()
if(HYLE_BUILD_CONTAINER)
  list(APPEND _hyle_active_in_tree_dependencies xxhash wyhash)
endif()
if(HYLE_BUILD_TEXT)
  list(APPEND _hyle_active_in_tree_dependencies simdutf)
endif()
if(HYLE_BUILD_IO_CORE AND HYLE_FCONTEXT_SUPPORTED)
  list(APPEND _hyle_active_in_tree_dependencies picohttpparser)
endif()
if(HYLE_BUILD_SCHED AND HYLE_FCONTEXT_SUPPORTED)
  list(APPEND _hyle_active_in_tree_dependencies fcontext)
endif()
if(HYLE_BUILD_BENCH)
  list(APPEND _hyle_active_in_tree_dependencies nanobench)
endif()
hyle_audit_in_tree_dependencies(${_hyle_active_in_tree_dependencies})
unset(_hyle_active_in_tree_dependencies)
if(HYLE_BUILD_SIMD)
  hyle_audit_integrated_source_dependencies(highway_port)
endif()
if(HYLE_BUILD_PLATFORM_THREAD)
  hyle_audit_integrated_source_dependencies(rust_parker)
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
if(HYLE_BUILD_SIMD)
  set(HYLE_SIMD_EXACT_TARGET "${HYLE_SIMD_EXACT_TARGET}" CACHE STRING
      "Build hyle_simd for one required ISA; empty enables runtime dispatch")
  string(TOUPPER "${HYLE_SIMD_EXACT_TARGET}" _hyle_simd_exact_target)
  if(NOT _hyle_simd_exact_target MATCHES
     "^(|SCALAR|SSE4|AVX2|AVX512|NEON|SVE|RVV)$")
    message(FATAL_ERROR
      "HYLE_SIMD_EXACT_TARGET='${HYLE_SIMD_EXACT_TARGET}' is invalid. Use an "
      "empty value or SCALAR, SSE4, AVX2, AVX512, NEON, SVE, or RVV.")
  endif()
  set(HYLE_SIMD_EXACT_TARGET "${_hyle_simd_exact_target}" CACHE STRING
      "Build hyle_simd for one required ISA; empty enables runtime dispatch" FORCE)
  # As with other normalized HYLE controls, override an inherited parent normal
  # variable in this directory so all module/feature logic sees the cache value.
  set(HYLE_SIMD_EXACT_TARGET "${_hyle_simd_exact_target}")
  set_property(CACHE HYLE_SIMD_EXACT_TARGET PROPERTY STRINGS
               "" SCALAR SSE4 AVX2 AVX512 NEON SVE RVV)
  unset(_hyle_simd_exact_target)
else()
  set(HYLE_SIMD_EXACT_TARGET "")
endif()
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
# CMP0126=NEW preserves an embedding parent's normal variable when the cache is
# updated. Mirror the normalized spelling into this directory as well so later
# HYLE logic cannot still observe THREAD while the cache/GUI says tsan.
set(HYLE_SANITIZER "${_san_lc}")
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
# discovery and source acquisition only for components in the resolved closure.
# Source resources are shared across builds; binary/subbuild state is not.
if(DEFINED HYLE_DEPS_DIR)
  message(FATAL_ERROR
    "HYLE_DEPS_DIR was removed because it mixed source resources with "
    "build-local state. Use HYLE_DEP_SOURCE_CACHE for shared locked sources "
    "and HYLE_DEP_BUILD_DIR for this build's binary/subbuild directory.")
endif()
hyle_initialize_dependency_source_cache()
if(DEFINED HYLE_DEP_BUILD_DIR AND NOT HYLE_DEP_BUILD_DIR STREQUAL "")
  set(_hyle_dep_build_dir_input "${HYLE_DEP_BUILD_DIR}")
else()
  set(_hyle_dep_build_dir_input "${HYLE_BINARY_DIR}/_deps")
endif()
cmake_path(ABSOLUTE_PATH _hyle_dep_build_dir_input
  BASE_DIRECTORY "${HYLE_BINARY_DIR}" NORMALIZE
  OUTPUT_VARIABLE _hyle_dep_build_dir_normalized)
set(HYLE_DEP_BUILD_DIR "${_hyle_dep_build_dir_normalized}" CACHE PATH
    "Build-local binary/subbuild directory for HYLE dependencies" FORCE)
# Keep the effective directory value and its cache description identical for
# add_subdirectory() consumers that supplied a normal variable.
set(HYLE_DEP_BUILD_DIR "${_hyle_dep_build_dir_normalized}")
unset(_hyle_dep_build_dir_input)
unset(_hyle_dep_build_dir_normalized)
mark_as_advanced(HYLE_DEP_BUILD_DIR)
# FetchContent is directory-scoped, so keep this as a normal variable owned by
# the HYLE subproject.  Every HYLE dependency (including test-only GoogleTest)
# observes the build-local directory without rewriting an embedding parent's
# cache. Locked SOURCE_DIR overrides are resolved separately from the shared
# HYLE_DEP_SOURCE_CACHE.
set(FETCHCONTENT_BASE_DIR "${HYLE_DEP_BUILD_DIR}")
if(HYLE_BUILD_IO_CORE)
  include(HyleIoDependencies)
  hyle_configure_io_dependencies()
endif()
if(HYLE_BUILD_SIMD OR HYLE_BUILD_CONTAINER)
  include(HyleSimdDependencies)
  if(HYLE_BUILD_SIMD)
    hyle_configure_simd_provider_dependencies()
  else()
    hyle_configure_highway_dependency()
  endif()
  set(_hyle_highway_development_assets "")
  if(HYLE_BUILD_SIMD)
    # The expert SDK needs Highway's complete public header tree, not its
    # upstream tests, benchmarks or compiled .cc implementations. Project each
    # locked header through the package registry so the installed tree remains
    # self-contained without turning an SDK into a copy of the vendor checkout.
    file(GLOB_RECURSE _hyle_highway_development_headers
      CONFIGURE_DEPENDS LIST_DIRECTORIES false
      "${HYLE_HIGHWAY_SOURCE_DIR}/hwy/*.h")
    list(APPEND _hyle_highway_development_assets DEVELOPMENT_FILE_SPECS)
    foreach(_hyle_highway_header IN LISTS _hyle_highway_development_headers)
      file(RELATIVE_PATH _hyle_highway_relative
        "${HYLE_HIGHWAY_SOURCE_DIR}" "${_hyle_highway_header}")
      get_filename_component(_hyle_highway_relative_dir
        "${_hyle_highway_relative}" DIRECTORY)
      list(APPEND _hyle_highway_development_assets
        "${_hyle_highway_header}|${CMAKE_INSTALL_DATADIR}/Hyle/dependencies/highway/${_hyle_highway_relative_dir}")
    endforeach()
    unset(_hyle_highway_development_headers)
    unset(_hyle_highway_header)
    unset(_hyle_highway_relative)
    unset(_hyle_highway_relative_dir)
  endif()
  hyle_register_package_dependency(
    NAME foundation_highway
    OWNER foundation
    LOCKED_DEPENDENCIES highway
    ${_hyle_highway_development_assets}
    RUNTIME_FILE_SPECS
      "${HYLE_HIGHWAY_SOURCE_DIR}/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|highway-upstream-LICENSE")
  unset(_hyle_highway_development_assets)
  if(HYLE_BUILD_SIMD)
    hyle_register_package_dependency(
      NAME simd_xsimd
      OWNER simd
      EXPORT_TARGETS hyle_simd_expert
      LOCKED_DEPENDENCIES xsimd
      DEVELOPMENT_FILE_SPECS
        "${HYLE_SOURCE_DIR}/cmake/HyleIsa.cmake|${CMAKE_INSTALL_LIBDIR}/cmake/Hyle"
        "${HYLE_SOURCE_DIR}/cmake/HyleDispatch.cmake|${CMAKE_INSTALL_LIBDIR}/cmake/Hyle"
      DEVELOPMENT_DIRECTORY_SPECS
        "${HYLE_XSIMD_SOURCE_DIR}/include/xsimd|${CMAKE_INSTALL_DATADIR}/Hyle/dependencies/xsimd/include"
      RUNTIME_FILE_SPECS
        "${HYLE_XSIMD_SOURCE_DIR}/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|xsimd-upstream-LICENSE")
  endif()
endif()
if(HYLE_BUILD_COMPUTE)
  include(HyleComputeDependencies)
  hyle_configure_compute_dependencies()
endif()

# ---- modules (component closure + dependency order) -------------------------
# sched fiber switching needs platform fcontext asm. Vendored build integration
# currently covers x86_64 Windows/Linux/macOS and arm64 Linux/macOS.  The
# component graph only selects sched transitively where that capability exists;
# an explicit unsupported sched request still fails here with a useful error.
set(HYLE_HAS_FCONTEXT FALSE)
if(HYLE_BUILD_SCHED)
  set(HYLE_HAS_FCONTEXT TRUE)
  if(MSVC)
    enable_language(ASM_MASM)
  else()
    enable_language(ASM)
  endif()
endif()

# The append-only catalog is an identity ledger, not a construction schedule.
# Build the selected closure in its resolved topological order so natural new
# lower-level edges do not require forbidden component-history reordering.
foreach(_component IN LISTS HYLE_RESOLVED_COMPONENT_LIST)
  string(TOUPPER "${_component}" _component_upper)
  if(HYLE_BUILD_${_component_upper})
    add_subdirectory("src/${_component}")
  endif()
endforeach()
unset(_component)
unset(_component_upper)

# Product facades are consumer roots outside the factor DAG. Create them only
# after all selected factor targets exist; no module CMakeLists may link one.
hyle_add_product_facades()

# Trace backend construction is deliberately deferred until the selected
# module graph has declared whether any target consumes the gate. This keeps a
# Core-only package from manufacturing/registering an unrelated Tracy runtime.
hyle_finalize_trace()

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
message(STATUS "HYLE ${PROJECT_VERSION} | ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} | C++${CMAKE_CXX_STANDARD} | sanitizer=${HYLE_SANITIZER} | linkage=${_hyle_linkage} | products=${HYLE_RESOLVED_PRODUCTS} | factors=${HYLE_RESOLVED_COMPONENTS}")
unset(_hyle_linkage)
