set(_hyle_correctness_targets "")

function(hyle_add_correctness_journey name)
  cmake_parse_arguments(ARG "" "" "SOURCES;PRODUCTS;DEPS;DEFINES" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "hyle_add_correctness_journey(${name}): unrecognized arguments: "
      "${ARG_UNPARSED_ARGUMENTS}")
  endif()
  if(ARG_KEYWORDS_MISSING_VALUES OR NOT ARG_SOURCES OR NOT ARG_PRODUCTS)
    message(FATAL_ERROR
      "hyle_add_correctness_journey(${name}) requires SOURCES and PRODUCTS")
  endif()

  set(_product_targets "")
  foreach(_product IN LISTS ARG_PRODUCTS)
    if(NOT TARGET hyle::${_product})
      return()
    endif()
    list(APPEND _product_targets hyle::${_product})
  endforeach()

  set(_target "hyle_correctness_${name}_test")
  add_executable(${_target} ${ARG_SOURCES})
  set_target_properties(${_target} PROPERTIES
    FOLDER "hyle/tests/correctness"
    HYLE_TEST_MODULE "project")
  target_link_libraries(${_target} PRIVATE
    ${_product_targets} GTest::gtest_main ${ARG_DEPS})
  if(ARG_DEFINES)
    target_compile_definitions(${_target} PRIVATE ${ARG_DEFINES})
  endif()
  hyle_apply_warnings(${_target})
  hyle_apply_sanitizer(${_target})
  hyle_stage_runtime(${_target})

  add_test(NAME ${_target} COMMAND ${_target})
  set_tests_properties(${_target} PROPERTIES
    LABELS "correctness;integration"
    TIMEOUT 60)
  set_property(GLOBAL APPEND PROPERTY HYLE_CORRECTNESS_JOURNEY_TARGETS
    ${_target})
endfunction()

hyle_add_correctness_journey(data_preparation
  SOURCES data_preparation_journey_test.cc
  PRODUCTS container simd math mem)

hyle_add_correctness_journey(async_runtime
  SOURCES async_runtime_journey_test.cc
  PRODUCTS container error obs sched)

hyle_add_correctness_journey(model_file_lifecycle
  SOURCES model_file_lifecycle_journey_test.cc
  PRODUCTS io mem)

hyle_add_correctness_journey(service_connection
  SOURCES service_connection_journey_test.cc
  PRODUCTS io)

hyle_add_correctness_journey(working_process
  SOURCES working_process_journey_test.cc
  PRODUCTS process)
if(TARGET hyle_correctness_working_process_test)
  if(NOT TARGET hyle_process_child_helper)
    message(FATAL_ERROR
      "working-process journey requires the Process-owned child helper")
  endif()
  target_compile_definitions(hyle_correctness_working_process_test PRIVATE
    HYLE_PROCESS_TEST_HELPER_PATH="$<TARGET_FILE:hyle_process_child_helper>")
  add_dependencies(hyle_correctness_working_process_test
    hyle_process_child_helper)
endif()

# The remaining journeys are added beside their public-API test sources.  Keep
# this aggregate independent of hyle_<factor>_tests: a journey deliberately
# crosses several factor and product ownership boundaries.
get_property(_hyle_correctness_targets GLOBAL PROPERTY
  HYLE_CORRECTNESS_JOURNEY_TARGETS)
if(_hyle_correctness_targets)
  add_custom_target(hyle_correctness_tests)
  set_target_properties(hyle_correctness_tests PROPERTIES
    FOLDER "hyle/validation")
  add_dependencies(hyle_correctness_tests ${_hyle_correctness_targets})
  add_dependencies(hyle_tests hyle_correctness_tests)
endif()
unset(_hyle_correctness_targets)
