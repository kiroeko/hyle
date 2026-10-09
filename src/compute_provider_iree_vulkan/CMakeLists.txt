if(NOT HYLE_COMPUTE_VULKAN_ENABLED)
  hyle_add_module(compute_provider_iree_vulkan OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_provider_iree_vulkan PROPERTY
    HYLE_INTERNAL_ONLY_ABSENT TRUE)
  return()
endif()

include(FetchContent)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(Threads REQUIRED)
hyle_audit_in_tree_dependencies(iree_vulkan_runtime_patches)
foreach(_dependency IN ITEMS iree_runtime iree_flatcc iree_vulkan_headers)
  string(TOUPPER "${_dependency}" _upper)
  FetchContent_Declare(${_dependency}
    URL "${HYLE_DEP_${_upper}_ARCHIVE_URL}"
    URL_HASH "SHA256=${HYLE_DEP_${_upper}_ARCHIVE_SHA256}"
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE SOURCE_SUBDIR hyle-verified-source-only)
  hyle_fetchcontent_make_available_locked(${_dependency})
  FetchContent_GetProperties(${_dependency})
endforeach()
set(_vendor "${HYLE_SOURCE_DIR}/third_party/iree_vulkan_runtime")
# Module-owned short paths also bound MSVC object/debug-file path lengths.
set(_source "${HYLE_DEP_BUILD_DIR}/iv/src")
set(_generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(GLOB_RECURSE _inputs CONFIGURE_DEPENDS "${_vendor}/*")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${_inputs} "${CMAKE_CURRENT_SOURCE_DIR}/prepare_source.py")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_source.py"
  --source "${iree_runtime_SOURCE_DIR}" --flatcc "${iree_flatcc_SOURCE_DIR}"
  --headers "${iree_vulkan_headers_SOURCE_DIR}" --vendor "${_vendor}"
  --output "${_source}" --generated "${_generated}"
  RESULT_VARIABLE _result OUTPUT_VARIABLE _output ERROR_VARIABLE _error)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "IREE source preparation failed: ${_error}")
endif()
message(STATUS "${_output}")
block()
  # The original top-level build probes CUDA unconditionally, even with only
  # the Vulkan runtime enabled. Keep that SDK probe out of this private scope.
  set(CMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit TRUE)
  set(CUDAToolkit_FOUND FALSE)
  set(Python_EXECUTABLE "${Python3_EXECUTABLE}")
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  # Upstream ARM processor waits use GNU C inline assembly. Preserve the
  # original C dialect inside this vendor scope; HYLE's own dialect stays strict.
  if(NOT MSVC)
    set(CMAKE_C_EXTENSIONS ON)
  endif()
  set(CMAKE_C_VISIBILITY_PRESET hidden)
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
  set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
  set(BUILD_SHARED_LIBS OFF)
  # An instrumented adapter alone cannot detect accesses made by the C runtime.
  # Use the original runtime's sanitizer switch inside this private build scope.
  if(HYLE_SANITIZER STREQUAL "asan")
    set(IREE_ENABLE_ASAN ON)
    if(MSVC)
      string(APPEND CMAKE_C_FLAGS " /Zi")
      string(APPEND CMAKE_CXX_FLAGS " /Zi")
    endif()
  endif()
  if(MSVC)
    add_compile_options(/utf-8)
  endif()
  foreach(_option IN ITEMS BUILD_COMPILER BUILD_TESTS BUILD_BENCHMARKS BUILD_SAMPLES
      BUILD_PYTHON_BINDINGS BUILD_BINDINGS_TFLITE BUILD_BINDINGS_TFLITE_JAVA
      HAL_DRIVER_DEFAULTS ENABLE_CPUINFO)
    set(IREE_${_option} OFF CACHE BOOL "Private Vulkan runtime configuration" FORCE)
  endforeach()
  set(IREE_HAL_DRIVER_VULKAN ON CACHE BOOL "Private Vulkan runtime" FORCE)
  add_subdirectory("${_source}" "${HYLE_DEP_BUILD_DIR}/iv/build" EXCLUDE_FROM_ALL)
endblock()

# Upstream's unified library contains every required runtime object. Avoid
# forwarding its duplicate individual archives or build headers into the SDK.
foreach(_property IN ITEMS INTERFACE_INCLUDE_DIRECTORIES INTERFACE_COMPILE_DEFINITIONS)
  get_target_property(_usage iree_runtime_unified ${_property})
  if(_usage)
    set_property(TARGET iree_runtime_unified PROPERTY ${_property} "$<BUILD_INTERFACE:${_usage}>")
  endif()
endforeach()
set_property(TARGET iree_runtime_unified PROPERTY INTERFACE_LINK_LIBRARIES
  "iree_defs;flatcc_parsing;$<BUILD_INTERFACE:HyleIREEVulkan::Headers>")
foreach(_target IN ITEMS iree_defs flatcc_parsing)
  get_target_property(_includes ${_target} INTERFACE_INCLUDE_DIRECTORIES)
  if(_includes)
    set_property(TARGET ${_target} PROPERTY INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${_includes}>")
  endif()
endforeach()
foreach(_target IN ITEMS iree_runtime_unified iree_defs flatcc_parsing)
  set_target_properties(${_target} PROPERTIES EXPORT_NAME "_iree_${_target}")
endforeach()

hyle_add_module(compute_provider_iree_vulkan
  SOURCES native_device.cc native_program.cc "${_generated}/programs.cc")
target_include_directories(hyle_compute_provider_iree_vulkan PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
if(NOT MSVC)
  # Original C headers use C99/GNU extensions when consumed from C++. Keep
  # their diagnostics separate from the strict warnings on this adapter.
  target_include_directories(hyle_compute_provider_iree_vulkan SYSTEM PRIVATE
    "${_source}" "${_source}/runtime/src"
    "${HYLE_DEP_BUILD_DIR}/iv/build" "${HYLE_DEP_BUILD_DIR}/iv/build/runtime/src")
endif()
target_link_libraries(hyle_compute_provider_iree_vulkan PRIVATE iree_runtime_unified)
if(MSVC)
  target_compile_options(hyle_compute_provider_iree_vulkan PRIVATE /utf-8 /Zc:preprocessor)
endif()
hyle_register_package_dependency(
  NAME iree_vulkan_runtime OWNER compute_provider_iree_vulkan
  STATIC_EXPORT_TARGETS iree_runtime_unified iree_defs flatcc_parsing
  STATIC_FIND_DEPENDENCIES Threads
  LOCKED_DEPENDENCIES iree_runtime iree_flatcc iree_vulkan_headers iree_vulkan_runtime_patches
  RUNTIME_FILE_SPECS
    "${iree_runtime_SOURCE_DIR}/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|iree-LICENSE"
    "${iree_flatcc_SOURCE_DIR}/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|iree-flatcc-LICENSE"
    "${iree_flatcc_SOURCE_DIR}/NOTICE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|iree-flatcc-NOTICE"
    "${iree_flatcc_SOURCE_DIR}/include/flatcc/portable/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|iree-flatcc-portable-LICENSE"
    "${iree_vulkan_headers_SOURCE_DIR}/LICENSE.md|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|iree-vulkan-headers-LICENSE.md"
    "${iree_vulkan_headers_SOURCE_DIR}/LICENSES/Apache-2.0.txt|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|iree-vulkan-headers-Apache-2.0.txt"
    "${iree_vulkan_headers_SOURCE_DIR}/LICENSES/MIT.txt|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|iree-vulkan-headers-MIT.txt"
    "${_vendor}/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|iree-vulkan-patches-LICENSE"
    "${_vendor}/NOTICE.md|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|iree-vulkan-patches-NOTICE.md"
    "${_vendor}/programs/licenses/bitsandbytes.txt|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|iree-bitsandbytes-LICENSE"
    "${_vendor}/programs/licenses/awq.txt|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|iree-awq-LICENSE"
    "${_vendor}/programs/licenses/compressed-tensors.txt|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|iree-compressed-tensors-LICENSE"
    "${_vendor}/programs/licenses/mlx.txt|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|iree-mlx-LICENSE"
    "${_vendor}/programs/licenses/ds4.txt|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|iree-ds4-LICENSE"
    "${_vendor}/programs/licenses/deepseek.txt|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|iree-deepseek-LICENSE"
    "${_vendor}/programs/licenses/sglang.txt|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|iree-sglang-LICENSE")
