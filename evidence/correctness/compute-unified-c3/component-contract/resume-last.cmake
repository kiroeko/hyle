# Configure-only checks for the component-selection and dependency-graph
# contracts.  These deliberately use fresh child projects so cache shadowing
# and source-embedding behavior are exercised exactly as consumers see them.

cmake_minimum_required(VERSION 3.25)

foreach(_required IN ITEMS HYLE_SOURCE_DIR HYLE_SMOKE_ROOT)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "Pass -D${_required}=<path>")
  endif()
  get_filename_component(${_required} "${${_required}}" ABSOLUTE)
  file(TO_CMAKE_PATH "${${_required}}" ${_required})
endforeach()

# Every generated child is a real configure journey and must inherit the same
# host-required platform floor as the delivery harness. Without this, the Mac
# contract fails while configuring a Platform-dependent child before it can
# evaluate the target assertions it was created to test.
include("${HYLE_SOURCE_DIR}/cmake/HylePlatformRequirements.cmake")
set(_hyle_generator_args "")
if(DEFINED HYLE_SMOKE_GENERATOR AND NOT HYLE_SMOKE_GENERATOR STREQUAL "")
  list(APPEND _hyle_generator_args -G "${HYLE_SMOKE_GENERATOR}")
endif()
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
  list(APPEND _hyle_generator_args
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${HYLE_PLATFORM_MACOS_DEPLOYMENT_TARGET_MIN})
endif()

set(_hyle_contract_root "${HYLE_SMOKE_ROOT}/component-contract")



function(_hyle_contract_run label expect_success)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  set(_output "${_stdout}\n${_stderr}")
  if(expect_success AND NOT _result EQUAL 0)
    message(FATAL_ERROR
      "${label} failed with exit code ${_result}:\n${_output}")
  elseif(NOT expect_success AND _result EQUAL 0)
    message(FATAL_ERROR "${label} unexpectedly succeeded")
  endif()
  set(HYLE_CONTRACT_LAST_OUTPUT "${_output}" PARENT_SCOPE)
endfunction()
set(_hyle_direct_public_edge_helper "${_hyle_contract_root}/HyleRequireDirectPublicEdge.cmake")
if(NOT EXISTS "${_hyle_direct_public_edge_helper}")
 message(FATAL_ERROR "Missing preserved contract prefix")
endif()

_hyle_contract_run("64-bit target-platform contract" TRUE
  "${CMAKE_COMMAND}"
  -DHYLE_SOURCE_DIR=${HYLE_SOURCE_DIR}
  -DHYLE_SMOKE_ROOT=${_hyle_contract_root}
  -DHYLE_SMOKE_GENERATOR=${HYLE_SMOKE_GENERATOR}
  -P "${HYLE_SOURCE_DIR}/cmake/HyleCiTargetPlatformContract.cmake")

# Exercise the strict offline preload in a self-contained local Git fixture:
# allow-listing, HEAD/tree/clean checks, license hashes, no-network cache gates,
# and the explicit non-Git escape hatch all have positive/negative coverage.
_hyle_contract_run("verified offline dependency contract" TRUE
  "${CMAKE_COMMAND}"
  -DHYLE_SOURCE_DIR=${HYLE_SOURCE_DIR}
  -DHYLE_SMOKE_ROOT=${_hyle_contract_root}
  -DHYLE_SMOKE_GENERATOR=${HYLE_SMOKE_GENERATOR}
  -P "${HYLE_SOURCE_DIR}/cmake/HyleCiOfflineDependencyContract.cmake")

# HYLE_DEP_BUILD_DIR is the one build-local binary/subbuild location for every
# HYLE-owned FetchContent dependency. Exercise the test-only path through a self-locked
# GoogleTest-shaped fixture; arbitrary fake direct overrides are intentionally
# rejected by the production provenance wrapper.
set(_fake_gtest_source "${_hyle_contract_root}/fake-googletest")
set(_fake_gtest_lock "${_hyle_contract_root}/fake-googletest-lock.json")
set(_error_test_build "${_hyle_contract_root}/error-test-build")
set(_error_dependency_cache "${_hyle_contract_root}/error-dependencies")
file(MAKE_DIRECTORY "${_fake_gtest_source}")
file(WRITE "${_fake_gtest_source}/LICENSE" "fixture license\n")
file(WRITE "${_fake_gtest_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(fake_googletest LANGUAGES CXX)\n"
"option(BUILD_GMOCK \"fixture gmock switch\" ON)\n"
"option(INSTALL_GTEST \"fixture install switch\" ON)\n"
"option(GTEST_HAS_ABSL \"fixture Abseil switch\" ON)\n"
"option(gtest_build_samples \"fixture sample switch\" ON)\n"
"option(gtest_build_tests \"fixture self-test switch\" ON)\n"
"if(BUILD_GMOCK OR INSTALL_GTEST OR GTEST_HAS_ABSL OR gtest_build_samples OR gtest_build_tests)\n"
"  message(FATAL_ERROR \"HYLE did not suppress GoogleTest optional products\")\n"
"endif()\n"
"add_library(gtest INTERFACE)\n"
"add_library(GTest::gtest ALIAS gtest)\n"
"add_library(gtest_main INTERFACE)\n"
"add_library(GTest::gtest_main ALIAS gtest_main)\n")
include("${HYLE_SOURCE_DIR}/cmake/HyleDependencyVersions.cmake")
hyle_compute_expanded_source_tree_sha256(
  "${_fake_gtest_source}" _fake_gtest_expanded_tree)
string(SHA256 _fake_gtest_license_sha256 "fixture license\n")
file(WRITE "${_fake_gtest_lock}"
"{
  \"schema_version\": 3,
  \"expanded_tree_algorithm\": \"hyle-path-hex-v1-raw-sha256\",
  \"dependencies\": [{
    \"name\": \"googletest\", \"class\": \"fetchcontent\",
    \"repository\": \"${HYLE_DEP_GOOGLETEST_REPOSITORY}\",
    \"version\": \"fixture-${HYLE_DEP_GOOGLETEST_VERSION}\",
    \"revision\": \"${HYLE_DEP_GOOGLETEST_COMMIT}\",
    \"git_tree\": \"${HYLE_DEP_GOOGLETEST_GIT_TREE}\",
    \"archive\": {
      \"url\": \"${HYLE_DEP_GOOGLETEST_ARCHIVE_URL}\",
      \"sha256\": \"${HYLE_DEP_GOOGLETEST_ARCHIVE_SHA256}\"
    },
    \"expanded_tree_sha256\": \"${_fake_gtest_expanded_tree}\",
    \"license\": { \"expression\": \"BSD-3-Clause\", \"files\": [{
      \"root\": \"dependency\", \"path\": \"LICENSE\",
      \"sha256\": \"${_fake_gtest_license_sha256}\"
    }] }
  }]
}
")
set(_fake_gtest_lock_attestation
  "${_hyle_contract_root}/fake-gtest-lock-attestation.cmake")
file(WRITE "${_fake_gtest_lock_attestation}"
"set_property(GLOBAL PROPERTY HYLE_INTERNAL_ATTESTED_DEPENDENCY_LOCK_FILE \"${_fake_gtest_lock}\")\n")
_hyle_contract_run("error-only dependency cache ownership" TRUE
  "${CMAKE_COMMAND}" -S "${HYLE_SOURCE_DIR}"
  -B "${_error_test_build}"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release
  -DHYLE_COMPONENTS=error
  -DHYLE_BUILD_TESTS=ON
  -DHYLE_BUILD_BENCH=OFF
  -DHYLE_DEP_BUILD_DIR=${_error_dependency_cache}
  -DHYLE_DEPENDENCY_LOCK_FILE=${_fake_gtest_lock}
  -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=${_fake_gtest_lock_attestation}
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${_fake_gtest_source})
if(NOT IS_DIRECTORY "${_error_dependency_cache}/googletest-build")
  message(FATAL_ERROR
    "error-only tests ignored HYLE_DEP_BUILD_DIR='${_error_dependency_cache}'")
endif()
if(IS_DIRECTORY "${_error_test_build}/_deps")
  message(FATAL_ERROR
    "error-only tests leaked FetchContent state into '${_error_test_build}/_deps'")
endif()
file(STRINGS "${_error_test_build}/CMakeCache.txt" _gtest_cache_leaks
  REGEX "^(BUILD_GMOCK|INSTALL_GTEST|GTEST_HAS_ABSL|gtest_build_samples|gtest_build_tests):")
if(_gtest_cache_leaks)
  message(FATAL_ERROR
    "GoogleTest implementation options leaked into HYLE cache: ${_gtest_cache_leaks}")
endif()
file(STRINGS "${_error_test_build}/CMakeCache.txt" _instrumentation_default
  REGEX "^HYLE_TEST_INSTRUMENTATION:BOOL=")
if(NOT _instrumentation_default STREQUAL
   "HYLE_TEST_INSTRUMENTATION:BOOL=OFF")
  message(FATAL_ERROR
    "HYLE_TEST_INSTRUMENTATION default is not OFF: ${_instrumentation_default}")
endif()

# Shared factor targets must fail closed: Unix/Mach-O definitions are hidden
# unless a public declaration carries HYLE_<FACTOR>_API. Test the target
# property directly so a header-only selected product is not forced to invent a
# binary solely for this contract.
set(_shared_visibility_source
  "${_hyle_contract_root}/shared-visibility-source")
file(MAKE_DIRECTORY "${_shared_visibility_source}")
file(WRITE "${_shared_visibility_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_shared_visibility_contract LANGUAGES CXX)\n"
"set(HYLE_COMPONENTS error)\n"
"set(HYLE_BUILD_TESTS OFF)\n"
"set(HYLE_BUILD_BENCH OFF)\n"
"set(HYLE_INSTALL OFF)\n"
"set(HYLE_STATIC OFF)\n"
"add_subdirectory(\"${HYLE_SOURCE_DIR}\" hyle)\n"
"foreach(_property IN ITEMS C_VISIBILITY_PRESET CXX_VISIBILITY_PRESET)\n"
"  get_target_property(_value hyle_error \${_property})\n"
"  if(NOT _value STREQUAL \"hidden\")\n"
"    message(FATAL_ERROR \"hyle_error \${_property}=\'\${_value}\'\")\n"
"  endif()\n"
"endforeach()\n"
"get_target_property(_hidden_inlines hyle_error VISIBILITY_INLINES_HIDDEN)\n"
"if(NOT _hidden_inlines)\n"
"  message(FATAL_ERROR \"hyle_error inline visibility is not hidden\")\n"
"endif()\n")
_hyle_contract_run("shared factor default-hidden visibility" TRUE
  "${CMAKE_COMMAND}" -S "${_shared_visibility_source}"
  -B "${_hyle_contract_root}/shared-visibility-build"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release)

message(STATUS "HYLE component contracts passed")
