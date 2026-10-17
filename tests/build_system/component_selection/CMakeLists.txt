cmake_minimum_required(VERSION 3.25)
project(hyle_component_selection_test LANGUAGES NONE)

# Configure-only regression: exercise the real resolver without configuring
# dependency SDKs or building product/test binaries.
get_filename_component(HYLE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(HYLE_TARGET_PROCESSOR x86_64)
set(HYLE_COMPONENTS graph)
set(HYLE_COMPUTE_ROUTES vulkan)
set(HYLE_BUILD_TESTS OFF)
include("${HYLE_SOURCE_DIR}/cmake/HyleComponents.cmake")
set(production_without_tests "${HYLE_PRODUCTION_COMPONENT_LIST}")
set(products_without_tests "${HYLE_PRODUCTION_PRODUCT_LIST}")
if(NOT HYLE_PRODUCTION_COMPONENT_LIST STREQUAL HYLE_RESOLVED_COMPONENT_LIST OR
   HYLE_TEST_FIXTURE_COMPONENT_LIST)
  message(FATAL_ERROR "A test-disabled build must contain only production factors")
endif()
if("compute" IN_LIST HYLE_PRODUCTION_PRODUCT_LIST OR
   "compute_backend_cpu" IN_LIST HYLE_PRODUCTION_COMPONENT_LIST)
  message(FATAL_ERROR "A Vulkan Graph must not acquire CPU fixtures as production")
endif()

set(HYLE_BUILD_TESTS ON)
include("${HYLE_SOURCE_DIR}/cmake/HyleComponents.cmake")
if(NOT HYLE_PRODUCTION_COMPONENT_LIST STREQUAL production_without_tests OR
   NOT HYLE_PRODUCTION_PRODUCT_LIST STREQUAL products_without_tests OR
   NOT HYLE_RESOLVED_PRODUCT_LIST STREQUAL HYLE_PRODUCTION_PRODUCT_LIST)
  message(FATAL_ERROR "Test fixtures changed the production factors or public products")
endif()
if(NOT HYLE_BUILD_COMPUTE_BACKEND_CPU OR
   NOT "compute_backend_cpu" IN_LIST HYLE_TEST_FIXTURE_COMPONENT_LIST OR
   NOT HYLE_BUILD_COMPUTE_STATE OR
   NOT "compute_state" IN_LIST HYLE_TEST_FIXTURE_COMPONENT_LIST)
  message(FATAL_ERROR "The build must still contain the actual Graph test fixtures")
endif()
foreach(component IN LISTS HYLE_TEST_FIXTURE_COMPONENT_LIST)
  if(component IN_LIST HYLE_PRODUCTION_COMPONENT_LIST)
    message(FATAL_ERROR "A fixture was also classified as production: ${component}")
  endif()
endforeach()

# A route/root change in the same configure must not inherit membership from
# the previous traversal. Core also provides a dependency-free small case.
set(HYLE_COMPONENTS core)
set(HYLE_BUILD_TESTS OFF)
include("${HYLE_SOURCE_DIR}/cmake/HyleComponents.cmake")
if(NOT HYLE_PRODUCTION_COMPONENT_LIST STREQUAL "core" OR
   NOT HYLE_RESOLVED_COMPONENT_LIST STREQUAL "core" OR
   NOT HYLE_PRODUCTION_PRODUCT_LIST STREQUAL "core" OR
   HYLE_BUILD_COMPUTE_BACKEND_CPU OR HYLE_BUILD_GRAPH OR
   HYLE_TEST_FIXTURE_COMPONENT_LIST)
  message(FATAL_ERROR "Changing roots retained stale production/test membership")
endif()
message(STATUS "Production, fixture, product and repeated-selection checks passed")
