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
if(HYLE_BUILD_COMPUTE_STREAM OR HYLE_BUILD_IO_CORE)
  message(FATAL_ERROR "Resident Graph must not acquire file streaming or IO")
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

# Host progress is an explicit service. Selecting a Scheduler for another
# purpose must not change the Graph facade's dependency closure.
if("graph_progress" IN_LIST HYLE_PRODUCTION_COMPONENT_LIST)
  message(FATAL_ERROR "Ordinary Graph acquired the optional host progress runtime")
endif()
set(HYLE_COMPONENTS graph sched)
set(HYLE_BUILD_TESTS OFF)
include("${HYLE_SOURCE_DIR}/cmake/HyleComponents.cmake")
if(HYLE_BUILD_GRAPH_PROGRESS OR
   "graph_progress" IN_LIST _HYLE_PRODUCT_FACTORS_GRAPH OR
   "sched" IN_LIST _HYLE_DEPS_GRAPH)
  message(FATAL_ERROR "Graph + Sched implicitly acquired host progress")
endif()
set(HYLE_COMPONENTS graph graph_progress)
include("${HYLE_SOURCE_DIR}/cmake/HyleComponents.cmake")
if(NOT HYLE_BUILD_GRAPH_PROGRESS OR NOT HYLE_BUILD_SCHED OR
   NOT HYLE_BUILD_PRODUCT_GRAPH_PROGRESS OR
   "graph_progress" IN_LIST _HYLE_PRODUCT_FACTORS_GRAPH)
  message(FATAL_ERROR "Explicit host progress was not isolated from the Graph facade")
endif()
set(HYLE_COMPONENTS graph_progress)
include("${HYLE_SOURCE_DIR}/cmake/HyleComponents.cmake")
if(NOT HYLE_BUILD_GRAPH OR NOT HYLE_BUILD_SCHED OR
   NOT HYLE_BUILD_GRAPH_PROGRESS OR HYLE_BUILD_GRAPH_BATCH OR
   HYLE_BUILD_COMPUTE_ROUTE_CPU OR HYLE_BUILD_COMPUTE_ROUTE_VULKAN OR
   HYLE_BUILD_COMPUTE_PROVIDER_NATIVE OR HYLE_BUILD_COMPUTE_STREAM OR
   HYLE_BUILD_IO_CORE OR HYLE_COMPUTE_ROUTE_LIST)
  message(FATAL_ERROR "Standalone host progress acquired a route or unrelated service")
endif()

# The explicit streaming service selects only the requested route's adapter.
set(HYLE_COMPONENTS graph compute_stream)
include("${HYLE_SOURCE_DIR}/cmake/HyleComponents.cmake")
if(NOT HYLE_BUILD_COMPUTE_STREAM OR NOT HYLE_BUILD_COMPUTE_STREAM_VULKAN OR
   NOT HYLE_BUILD_IO_CORE OR HYLE_BUILD_COMPUTE_STREAM_CPU OR
   HYLE_BUILD_COMPUTE_STREAM_CUDA OR HYLE_BUILD_COMPUTE_STREAM_APPLE OR
   HYLE_BUILD_COMPUTE_BACKEND_CPU OR HYLE_BUILD_IO_TLS OR HYLE_BUILD_IO_HTTP OR
   HYLE_BUILD_GRAPH_PROGRESS)
  message(FATAL_ERROR "Graph + file streaming acquired an unselected route or service")
endif()

# Standalone CPU streaming uses the real copy backend, without a math roster.
set(HYLE_COMPONENTS compute_stream)
set(HYLE_COMPUTE_ROUTES cpu)
include("${HYLE_SOURCE_DIR}/cmake/HyleComponents.cmake")
if(NOT HYLE_BUILD_COMPUTE_STREAM_CPU OR NOT HYLE_BUILD_COMPUTE_BACKEND_CPU OR
   HYLE_BUILD_COMPUTE_ROUTE_CPU OR HYLE_BUILD_COMPUTE_PROVIDER_NATIVE OR
   HYLE_BUILD_COMPUTE_PROVIDER_ONEDNN OR HYLE_BUILD_COMPUTE_PROVIDER_GGML OR
   HYLE_BUILD_GRAPH OR HYLE_BUILD_COMPUTE_STREAM_VULKAN)
  message(FATAL_ERROR "CPU file streaming acquired a math roster or stale route")
endif()
if("compute" IN_LIST HYLE_PRODUCTION_PRODUCT_LIST OR
   "graph" IN_LIST HYLE_PRODUCTION_PRODUCT_LIST OR
   "io" IN_LIST HYLE_PRODUCTION_PRODUCT_LIST)
  message(FATAL_ERROR "The streaming service manufactured an unrequested complete product")
endif()

# A route/root change in the same configure must not inherit membership from
# the previous traversal. Core also provides a dependency-free small case.
set(HYLE_COMPONENTS core)
set(HYLE_BUILD_TESTS OFF)
include("${HYLE_SOURCE_DIR}/cmake/HyleComponents.cmake")
if(NOT HYLE_PRODUCTION_COMPONENT_LIST STREQUAL "core" OR
   NOT HYLE_RESOLVED_COMPONENT_LIST STREQUAL "core" OR
   NOT HYLE_PRODUCTION_PRODUCT_LIST STREQUAL "core" OR
   HYLE_BUILD_COMPUTE_BACKEND_CPU OR HYLE_BUILD_GRAPH OR
   HYLE_BUILD_COMPUTE_STREAM OR HYLE_BUILD_IO_CORE OR
   HYLE_TEST_FIXTURE_COMPONENT_LIST)
  message(FATAL_ERROR "Changing roots retained stale production/test membership")
endif()
message(STATUS "Production, fixture, product and repeated-selection checks passed")
