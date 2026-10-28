cmake_minimum_required(VERSION 3.25)
project(hyle_glslang_host LANGUAGES C CXX)

if(NOT HYLE_SOURCE_DIR OR NOT HYLE_DEP_SOURCE_CACHE)
  message(FATAL_ERROR "Pass HYLE_SOURCE_DIR and HYLE_DEP_SOURCE_CACHE")
endif()
set(HYLE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
include("${HYLE_SOURCE_DIR}/cmake/HyleDependencyVersions.cmake")
include("${HYLE_SOURCE_DIR}/cmake/HyleDependencyCache.cmake")
hyle_initialize_dependency_source_cache()
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps")

include(FetchContent)
FetchContent_Declare(glslang
  URL "${HYLE_DEP_GLSLANG_ARCHIVE_URL}"
  URL_HASH "SHA256=${HYLE_DEP_GLSLANG_ARCHIVE_SHA256}"
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
block()
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0126 NEW)
  set(BUILD_SHARED_LIBS OFF)
  set(BUILD_EXTERNAL OFF)
  set(BUILD_TESTING OFF)
  set(ENABLE_CTEST OFF)
  set(ENABLE_GLSLANG_BINARIES ON)
  set(ENABLE_SPIRV ON)
  set(ENABLE_OPT OFF)
  set(ENABLE_HLSL OFF)
  set(ENABLE_PCH OFF)
  hyle_fetchcontent_make_available_locked(glslang)
endblock()

if(TARGET glslang-standalone)
  set(_standalone glslang-standalone)
elseif(TARGET glslangValidator)
  set(_standalone glslangValidator)
else()
  message(FATAL_ERROR "Locked glslang did not define its standalone host compiler")
endif()
add_custom_target(hyle_glslang_host DEPENDS ${_standalone})
file(GENERATE
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/glslang-path-$<CONFIG>.txt"
  CONTENT "$<TARGET_FILE:${_standalone}>\n")
