cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED HYLE_DEPENDENCY OR HYLE_DEPENDENCY STREQUAL "")
  message(FATAL_ERROR "Pass -DHYLE_DEPENDENCY=<integrated-source lock name>")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/HyleDependencyVersions.cmake")
if(NOT HYLE_DEPENDENCY IN_LIST HYLE_INTEGRATED_SOURCE_DEPENDENCIES)
  message(FATAL_ERROR
    "'${HYLE_DEPENDENCY}' is not an integrated-source dependency")
endif()
string(TOUPPER "${HYLE_DEPENDENCY}" _upper)
set(_prefix "HYLE_DEP_${_upper}")
hyle_compute_integrated_source_content_sha256(
  "${HYLE_DEPENDENCY_SOURCE_ROOT}"
  "${${_prefix}_INTEGRATED_PATHS}" _content_sha256)
message("${HYLE_DEPENDENCY}=${_content_sha256}")
