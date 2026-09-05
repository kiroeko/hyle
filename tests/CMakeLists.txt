include(FetchContent)

# GoogleTest is a validation implementation detail. An embedding parent may
# already own the canonical target; reuse it instead of populating a second
# copy with colliding gtest/gtest_main target names. Otherwise fetch HYLE's
# pinned archive through HYLE_DEP_SOURCE_CACHE, keep its binary/subbuild under
# HYLE_DEP_BUILD_DIR, and keep generic GoogleTest options out of
# the parent cache/install graph. HYLE does not use GMock.
set(_hyle_owns_googletest FALSE)
if(TARGET GTest::gtest_main)
  message(STATUS "HYLE tests: reusing parent-provided GTest::gtest_main")
elseif(TARGET gtest OR TARGET gtest_main OR TARGET GTest::gtest)
  message(FATAL_ERROR
    "HYLE_BUILD_TESTS found a partial GoogleTest target set without "
    "GTest::gtest_main. Provide the canonical imported target or configure "
    "HYLE_BUILD_TESTS=OFF.")
else()
  FetchContent_Declare(
    googletest
    URL      ${HYLE_DEP_GOOGLETEST_ARCHIVE_URL}
    URL_HASH SHA256=${HYLE_DEP_GOOGLETEST_ARCHIVE_SHA256}
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
  block()
    set(gtest_force_shared_crt ON)
    set(BUILD_GMOCK OFF)
    set(INSTALL_GTEST OFF)
    set(GTEST_HAS_ABSL OFF)
    set(gtest_build_samples OFF)
    set(gtest_build_tests OFF)
    set(gtest_disable_pthreads OFF)
    set(gtest_hide_internal_symbols OFF)
    hyle_fetchcontent_make_available_locked(googletest)
  endblock()
  set(_hyle_owns_googletest TRUE)
endif()

# MSVC's ASan-enabled STL records container-annotation mismatch metadata in
# every object. A source-built GoogleTest must therefore use the same compiler
# instrumentation as HYLE's test executables; otherwise the linker correctly
# rejects the mixed annotated/unannotated ABI. Do not mutate a parent-owned
# target: an embedding parent remains responsible for supplying an
# ASan-compatible GoogleTest when it asks HYLE to build tests in that mode.
if(_hyle_owns_googletest AND MSVC AND HYLE_SANITIZER STREQUAL "asan")
  hyle_apply_sanitizer(gtest)
  hyle_apply_sanitizer(gtest_main)
endif()
unset(_hyle_owns_googletest)

add_custom_target(hyle_tests)
set_target_properties(hyle_tests PROPERTIES FOLDER "hyle/validation")

function(hyle_ensure_module_test_target module)
  if(NOT TARGET hyle_${module}_tests)
    add_custom_target(hyle_${module}_tests)
    set_target_properties(hyle_${module}_tests PROPERTIES FOLDER "hyle/validation")
    add_dependencies(hyle_tests hyle_${module}_tests)
  endif()
endfunction()

function(hyle_add_header_self_contained_test module)
  cmake_parse_arguments(ARG "" "HEADER_DIR" "HEADERS" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "hyle_add_header_self_contained_test(${module}): unrecognized arguments: "
      "${ARG_UNPARSED_ARGUMENTS}")
  endif()
  if(ARG_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR
      "hyle_add_header_self_contained_test(${module}): keywords missing values: "
      "${ARG_KEYWORDS_MISSING_VALUES}")
  endif()
  if(NOT TARGET hyle_${module})
    return()
  endif()
  set(_hyle_vs_args "")
  if(DEFINED CMAKE_GENERATOR_INSTANCE)
    set(_hyle_vs_value "${CMAKE_GENERATOR_INSTANCE}")
    if(NOT _hyle_vs_value STREQUAL "")
      list(APPEND _hyle_vs_args --vs-instance "${_hyle_vs_value}")
    endif()
  endif()
  if(DEFINED CMAKE_VS_PLATFORM_NAME)
    set(_hyle_vs_value "${CMAKE_VS_PLATFORM_NAME}")
    if(NOT _hyle_vs_value STREQUAL "")
      list(APPEND _hyle_vs_args --vs-platform "${_hyle_vs_value}")
    endif()
  endif()
  if(DEFINED CMAKE_VS_PLATFORM_TOOLSET_VERSION)
    set(_hyle_vs_value "${CMAKE_VS_PLATFORM_TOOLSET_VERSION}")
    if(NOT _hyle_vs_value STREQUAL "")
      list(APPEND _hyle_vs_args --vs-toolset-version "${_hyle_vs_value}")
    endif()
  endif()
  set(_hyle_header_args "")
  if(ARG_HEADER_DIR)
    list(APPEND _hyle_header_args --header-dir "${ARG_HEADER_DIR}")
  endif()
  foreach(_hyle_header IN LISTS ARG_HEADERS)
    list(APPEND _hyle_header_args --header "${_hyle_header}")
  endforeach()
  add_test(NAME hyle_${module}_header_self_contained_test
           COMMAND "${Python3_EXECUTABLE}"
                   "${HYLE_SOURCE_DIR}/scripts/header-self-contained.py"
                   --module "${module}"
                   --include-dir "${HYLE_SOURCE_DIR}/include"
                   --cxx "${CMAKE_CXX_COMPILER}"
                   --cxx-compiler-id "${CMAKE_CXX_COMPILER_ID}"
                   --cxx-frontend "${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}"
                   ${_hyle_header_args}
                   ${_hyle_vs_args})
  set_tests_properties(hyle_${module}_header_self_contained_test PROPERTIES
    LABELS "${module};headers")
  hyle_ensure_module_test_target(${module})
  unset(_hyle_vs_args)
  unset(_hyle_vs_value)
  unset(_hyle_header_args)
  unset(_hyle_header)
endfunction()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME hyle_bench_stat_parser_test
         COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/tooling/bench_stat_parser_test.py")
set_tests_properties(hyle_bench_stat_parser_test PROPERTIES
  LABELS "tooling;benchmark")
add_test(NAME hyle_perf_machine_test
         COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/tooling/perf_machine_test.py")
set_tests_properties(hyle_perf_machine_test PROPERTIES
  LABELS "tooling;benchmark")
add_test(NAME hyle_perf_core_test
         COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/tooling/hyle_perf_core_test.py")
set_tests_properties(hyle_perf_core_test PROPERTIES
  LABELS "tooling;benchmark")
add_test(NAME hyle_perf_nanobench_test
         COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/tooling/hyle_perf_nanobench_test.py")
set_tests_properties(hyle_perf_nanobench_test PROPERTIES
  LABELS "tooling;benchmark")
add_test(NAME hyle_perf_runner_test
         COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/tooling/hyle_perf_runner_test.py")
set_tests_properties(hyle_perf_runner_test PROPERTIES
  LABELS "tooling;benchmark")
add_test(NAME hyle_perf_controls_test
         COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/tooling/hyle_perf_controls_test.py")
set_tests_properties(hyle_perf_controls_test PROPERTIES
  LABELS "tooling;benchmark")
add_test(NAME hyle_perf_evidence_runner_test
         COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/tooling/hyle_perf_evidence_runner_test.py")
set_tests_properties(hyle_perf_evidence_runner_test PROPERTIES
  LABELS "tooling;benchmark")
add_test(NAME hyle_perf_raw_replay_test
         COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/tooling/hyle_perf_raw_replay_test.py")
set_tests_properties(hyle_perf_raw_replay_test PROPERTIES
  LABELS "tooling;benchmark")
add_test(NAME hyle_perf_build_replay_test
         COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/tooling/hyle_perf_build_replay_test.py")
set_tests_properties(hyle_perf_build_replay_test PROPERTIES
  LABELS "tooling;benchmark")
add_test(NAME hyle_module_closure_contract_test
         COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/tooling/check_module_closure_test.py")
set_tests_properties(hyle_module_closure_contract_test PROPERTIES
  LABELS "tooling;documentation;closure")
add_test(NAME hyle_module_closure_registry_test
         COMMAND "${Python3_EXECUTABLE}"
                 "${HYLE_SOURCE_DIR}/scripts/check-module-closure.py")
set_tests_properties(hyle_module_closure_registry_test PROPERTIES
  LABELS "tooling;documentation;closure")
# LOC ownership/revision checks are maintainer contracts over a Git checkout,
# not product tests. Complete source archives intentionally omit .git, so do
# not register a test there that cannot possibly resolve the audited revision.
find_package(Git QUIET)
if(Git_FOUND AND EXISTS "${HYLE_SOURCE_DIR}/.git")
  add_test(NAME hyle_loc_report_test
           COMMAND "${Python3_EXECUTABLE}"
                   "${CMAKE_CURRENT_SOURCE_DIR}/tooling/loc_report_test.py")
  set_tests_properties(hyle_loc_report_test PROPERTIES
    LABELS "tooling;closure")
else()
  message(STATUS
    "HYLE tests: skipping Git-only LOC reporter outside a Git checkout")
endif()
# Every module owns its test registration beside its test sources. A real
# subdirectory gives each module independent source/binary/variable scope while
# target names and the common output directory remain stable.
foreach(_component IN LISTS HYLE_KNOWN_COMPONENTS)
  set(_hyle_test_fragment "${CMAKE_CURRENT_SOURCE_DIR}/${_component}/CMakeLists.txt")
  if(TARGET hyle_${_component} AND NOT EXISTS "${_hyle_test_fragment}")
    message(FATAL_ERROR
      "Selected HYLE component '${_component}' has no test ownership fragment: "
      "${_hyle_test_fragment}")
  endif()
  if(TARGET hyle_${_component})
    add_subdirectory("${_component}")
    if(NOT TARGET hyle_${_component}_tests)
      message(FATAL_ERROR
        "Test ownership fragment for '${_component}' registered no module aggregate")
    endif()
    get_property(_hyle_module_targets
      DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${_component}"
      PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_hyle_module_target IN LISTS _hyle_module_targets)
      get_target_property(_hyle_module_target_type
        ${_hyle_module_target} TYPE)
      if(_hyle_module_target_type STREQUAL "EXECUTABLE")
        get_target_property(_hyle_test_owner
          ${_hyle_module_target} HYLE_TEST_MODULE)
        if(NOT _hyle_test_owner STREQUAL "${_component}")
          message(FATAL_ERROR
            "Test executable '${_hyle_module_target}' in tests/${_component} is not "
            "registered through the ${_component} test aggregate")
        endif()
      endif()
    endforeach()
  endif()
endforeach()
unset(_hyle_test_owner)
unset(_hyle_module_target_type)
unset(_hyle_module_target)
unset(_hyle_module_targets)
unset(_hyle_test_fragment)
unset(_component)
