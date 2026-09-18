if(NOT HYLE_COMPUTE_VULKAN_ENABLED)
  hyle_add_module(compute_provider_ggml_vulkan OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_provider_ggml_vulkan PROPERTY
    HYLE_INTERNAL_ONLY_ABSENT TRUE)
  return()
endif()

# One owner for the locked source, exact integration patches, vendor options
# and package projection. There is no second downloader or vendor source list.
include(FetchContent)
hyle_audit_in_tree_dependencies(ggml_vulkan_runtime_patches)
FetchContent_Declare(ggml_vulkan_runtime
  URL "${HYLE_DEP_GGML_VULKAN_RUNTIME_ARCHIVE_URL}"
  URL_HASH "SHA256=${HYLE_DEP_GGML_VULKAN_RUNTIME_ARCHIVE_SHA256}"
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
  SOURCE_SUBDIR hyle-verified-source-only)
hyle_fetchcontent_make_available_locked(ggml_vulkan_runtime)
FetchContent_GetProperties(ggml_vulkan_runtime)
set(_patch_root "${HYLE_SOURCE_DIR}/third_party/ggml_vulkan_runtime")
set(_patched_source
  "${CMAKE_CURRENT_BINARY_DIR}/source-${HYLE_DEP_GGML_VULKAN_RUNTIME_PATCHES_VENDORED_TREE_SHA256}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${_patch_root}/patches.json" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_source.py")
execute_process(
  COMMAND "${HYLE_VULKAN_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_source.py"
    --source "${ggml_vulkan_runtime_SOURCE_DIR}"
    --output "${_patched_source}" --manifest "${_patch_root}/patches.json"
  RESULT_VARIABLE _patch_result OUTPUT_VARIABLE _patch_output ERROR_VARIABLE _patch_error)
if(NOT _patch_result EQUAL 0)
  message(FATAL_ERROR "ggml Vulkan source preparation failed: ${_patch_error}")
endif()
message(STATUS "${_patch_output}")

# Upstream's default ExternalProject prefix nests the shader-tool compiler
# probes past MSVC's path limit when HYLE is embedded in another project.
# This directory property is inherited only by this owner's vendor subtree.
include(ExternalProject)
set_property(DIRECTORY PROPERTY EP_PREFIX "${HYLE_DEP_BUILD_DIR}/ggml-vulkan-host")

block()
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(CMAKE_C_VISIBILITY_PRESET hidden)
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
  set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
  set(HYLE_GGML_PARENT_PROJECT_INCLUDE_BEFORE "${CMAKE_PROJECT_ggml_INCLUDE_BEFORE}")
  set(CMAKE_PROJECT_ggml_INCLUDE_BEFORE "${HYLE_SOURCE_DIR}/cmake/HyleGgmlProjectPolicy.cmake")
  set(BUILD_SHARED_LIBS OFF)
  # Avoid leaking upstream backend options into the embedding parent's cache.
  # ggml's project uses CMP0077 NEW and respects these scoped normal variables.
  foreach(_option IN ITEMS CPU CUDA HIP MUSA METAL BLAS SYCL VULKAN_CHECK_RESULTS
      VULKAN_RUN_TESTS VULKAN_VALIDATE VULKAN_DEBUG VULKAN_MEMORY_DEBUG
      OPENMP OPENMP_FETCH RPC OPENCL WEBGPU HEXAGON ZDNN VIRTGPU VIRTGPU_BACKEND
      OPENVINO ET ET_SYSEMU CANN KOMPUTE ZENDNN NATIVE CCACHE BACKEND_DL LTO
      STATIC CPU_ALL_VARIANTS
      BUILD_TESTS BUILD_EXAMPLES)
    set(GGML_${_option} OFF)
  endforeach()
  set(GGML_VULKAN ON)
  set(GGML_BUILD_COMMIT "${HYLE_DEP_GGML_VULKAN_RUNTIME_COMMIT}-hyle")
  add_subdirectory("${_patched_source}" vendor EXCLUDE_FROM_ALL)
endblock()

foreach(_target IN ITEMS ggml ggml-base ggml-vulkan)
  if(NOT TARGET ${_target})
    message(FATAL_ERROR "Locked ggml Vulkan did not define '${_target}'")
  endif()
  set_target_properties(${_target} PROPERTIES
    EXPORT_NAME "_ggml_vulkan_${_target}"
    POSITION_INDEPENDENT_CODE ON)
endforeach()
# Static SDKs retain the implementation targets, without exposing ggml's C
# headers as a second HYLE user API. HYLE owns all installed public headers.
set_property(TARGET ggml PROPERTY PUBLIC_HEADER "")
# Keep the portable HYLE header version at the same locked SDK revision used
# by the native route; the vendor's SDK also provides SPIR-V compiler headers.
target_link_libraries(ggml-vulkan PRIVATE "$<BUILD_INTERFACE:Vulkan-Headers>")

hyle_add_module(compute_provider_ggml_vulkan SOURCES metadata_arena.cc)
target_link_libraries(hyle_compute_provider_ggml_vulkan PRIVATE ggml)
set(HYLE_GGML_VULKAN_SOURCE_DIR "${_patched_source}" PARENT_SCOPE)
hyle_register_package_dependency(
  NAME ggml_vulkan_runtime OWNER compute_provider_ggml_vulkan FEATURES compute_vulkan
  STATIC_EXPORT_TARGETS ggml ggml-base ggml-vulkan
  STATIC_FIND_DEPENDENCIES Threads Vulkan
  LOCKED_DEPENDENCIES ggml_vulkan_runtime ggml_vulkan_runtime_patches
  RUNTIME_FILE_SPECS
    "${ggml_vulkan_runtime_SOURCE_DIR}/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|ggml-vulkan-runtime-LICENSE"
    "${_patch_root}/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|ggml-vulkan-patches-LICENSE"
    "${_patch_root}/NOTICE.md|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|ggml-vulkan-patches-NOTICE.md")
