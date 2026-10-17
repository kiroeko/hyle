include_guard(GLOBAL)

# Private availability is derived from the selected DAG by HyleComputeRoutes.

function(hyle_configure_vulkan_dependencies)
  if(_HYLE_COMPUTE_REQUEST_VULKAN STREQUAL "OFF")
    message(STATUS "HYLE: Vulkan route OFF -- SDK and shader probes/downloads skipped")
    set(_HYLE_COMPUTE_HAVE_VULKAN OFF PARENT_SCOPE)
    return()
  endif()
  # Shader compilation is a host build tool, independent of the target loader
  # and physical device. Installed binary SDKs contain embedded SPIR-V.
  find_program(HYLE_VULKAN_GLSLANG NAMES glslangValidator glslang
    HINTS "$ENV{VULKAN_SDK}" PATH_SUFFIXES Bin bin)
  if(NOT HYLE_VULKAN_GLSLANG)
    message(FATAL_ERROR "Vulkan source builds require host glslangValidator (HYLE_VULKAN_GLSLANG)")
  endif()
  execute_process(COMMAND "${HYLE_VULKAN_GLSLANG}" --version
    RESULT_VARIABLE _result OUTPUT_VARIABLE _version ERROR_VARIABLE _error
    TIMEOUT 10 OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "Vulkan host shader compiler failed: ${_error}")
  endif()
  string(REGEX MATCH "^[^\r\n]*" _compiler "${_version}")
  string(REPLACE "\\" "\\\\" _compiler "${_compiler}")
  string(REPLACE "\"" "\\\"" _compiler "${_compiler}")
  set(HYLE_VULKAN_SHADER_COMPILER "${_compiler}" PARENT_SCOPE)
  hyle_audit_in_tree_dependencies(ggml_vulkan_shaders)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  include(FetchContent)
  FetchContent_Declare(vulkan_headers
    URL "${HYLE_DEP_VULKAN_HEADERS_ARCHIVE_URL}"
    URL_HASH "SHA256=${HYLE_DEP_VULKAN_HEADERS_ARCHIVE_SHA256}"
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    EXCLUDE_FROM_ALL)
  block()
    # Configure this private vendor without rewriting an embedding parent's cache.
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0126 NEW)
    set(VULKAN_HEADERS_ENABLE_TESTS OFF)
    set(VULKAN_HEADERS_ENABLE_INSTALL OFF)
    set(VULKAN_HEADERS_ENABLE_MODULE OFF)
    hyle_fetchcontent_make_available_locked(vulkan_headers)
  endblock()
  FetchContent_GetProperties(vulkan_headers)
  set_target_properties(Vulkan-Headers PROPERTIES
    SYSTEM TRUE EXPORT_NAME "_vulkan_headers_native")
  target_include_directories(Vulkan-Headers INTERFACE
    "$<INSTALL_INTERFACE:${HYLE_INSTALL_INCLUDEDIR}>")
  set(HYLE_VULKAN_HEADERS_SOURCE_DIR "${vulkan_headers_SOURCE_DIR}" PARENT_SCOPE)
  set(HYLE_VULKAN_PYTHON "${Python3_EXECUTABLE}" PARENT_SCOPE)
  set(_HYLE_COMPUTE_HAVE_VULKAN ON PARENT_SCOPE)
endfunction()
