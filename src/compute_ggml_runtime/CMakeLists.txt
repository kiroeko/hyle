# Shared source ownership does not imply a shared backend executor. Only the
# selected CPU/CUDA/Metal/Vulkan closures enable their native implementation targets.
set(_cpu OFF)
set(_explicit_factors "${HYLE_DEV_FACTORS}")
string(REPLACE "," ";" _explicit_factors "${_explicit_factors}")
list(TRANSFORM _explicit_factors STRIP)
list(TRANSFORM _explicit_factors TOLOWER)
if(HYLE_BUILD_COMPUTE_BACKEND_CPU OR HYLE_BUILD_COMPUTE_PROVIDER_GGML OR
   "compute_ggml_runtime" IN_LIST _explicit_factors)
  set(_cpu ON)
endif()
if(DEFINED HYLE_COMPUTE_CPU AND NOT HYLE_COMPUTE_CPU)
  set(_cpu OFF)
endif()
set(HYLE_GGML_CPU_ENABLED "${_cpu}" PARENT_SCOPE)
if(NOT _cpu AND NOT HYLE_COMPUTE_VULKAN_ENABLED AND
   NOT HYLE_COMPUTE_CUDA_ENABLED AND NOT HYLE_COMPUTE_APPLE_ENABLED)
  hyle_add_module(compute_ggml_runtime OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_ggml_runtime PROPERTY HYLE_INTERNAL_ONLY_ABSENT TRUE)
  return()
endif()

include(FetchContent)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
# One source revision per SDK. Existing GPU programs keep their qualified core;
# CPU-only builds receive the newer core and the original narrow IQ1 extension.
set(_cpu_upgrade OFF)
set(_dependency ggml)
set(_dependency_upper GGML)
set(_patch_dependency ggml_runtime_patches)
set(_patch_dependency_upper GGML_RUNTIME_PATCHES)
set(_patch_root "${HYLE_SOURCE_DIR}/third_party/ggml_runtime")
if(_cpu AND NOT HYLE_COMPUTE_CUDA_ENABLED AND
   NOT HYLE_COMPUTE_APPLE_ENABLED AND NOT HYLE_COMPUTE_VULKAN_ENABLED)
  set(_cpu_upgrade ON)
  set(_dependency ggml_cpu)
  set(_dependency_upper GGML_CPU)
  set(_patch_dependency ggml_cpu_runtime_patches)
  set(_patch_dependency_upper GGML_CPU_RUNTIME_PATCHES)
  set(_patch_root "${HYLE_SOURCE_DIR}/third_party/ggml_cpu_runtime")
endif()
set(_selected_commit "${HYLE_DEP_${_dependency_upper}_COMMIT}")
set(HYLE_GGML_CPU_PEER_REVISION "${_selected_commit}" PARENT_SCOPE)
hyle_audit_in_tree_dependencies(${_patch_dependency})
FetchContent_Declare(${_dependency}
  URL "${HYLE_DEP_${_dependency_upper}_ARCHIVE_URL}"
  URL_HASH "SHA256=${HYLE_DEP_${_dependency_upper}_ARCHIVE_SHA256}"
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
  SOURCE_SUBDIR hyle-verified-source-only)
hyle_fetchcontent_make_available_locked(${_dependency})
FetchContent_GetProperties(${_dependency} SOURCE_DIR ggml_SOURCE_DIR)
set(_manifests "${_patch_root}/patches.json")
set(_patch_args --manifest "${_patch_root}/patches.json")
set(_source_identity "${HYLE_DEP_${_patch_dependency_upper}_VENDORED_TREE_SHA256}")
set(_locked ${_dependency} ${_patch_dependency})
set(_licenses
  "${ggml_SOURCE_DIR}/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|ggml-runtime-LICENSE"
  "${_patch_root}/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|ggml-runtime-patches-LICENSE"
  "${_patch_root}/NOTICE.md|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|ggml-runtime-patches-NOTICE.md")
if(HYLE_COMPUTE_APPLE_ENABLED)
  list(APPEND _manifests "${_patch_root}/metal-patches.json")
  list(APPEND _patch_args --manifest "${_patch_root}/metal-patches.json")
endif()
if(HYLE_COMPUTE_VULKAN_ENABLED)
  hyle_audit_in_tree_dependencies(ggml_vulkan_runtime_patches)
  set(_vk_patches "${HYLE_SOURCE_DIR}/third_party/ggml_vulkan_runtime")
  list(APPEND _manifests "${_vk_patches}/patches.json")
  list(APPEND _patch_args --manifest "${_vk_patches}/patches.json")
  string(APPEND _source_identity "${HYLE_DEP_GGML_VULKAN_RUNTIME_PATCHES_VENDORED_TREE_SHA256}")
  list(APPEND _locked ggml_vulkan_runtime_patches)
  list(APPEND _licenses
    "${_vk_patches}/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|ggml-vulkan-patches-LICENSE"
    "${_vk_patches}/NOTICE.md|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|ggml-vulkan-patches-NOTICE.md")
endif()
if(HYLE_COMPUTE_CUDA_ENABLED)
  hyle_audit_in_tree_dependencies(ggml_cuda_runtime_patches)
  set(_cuda_patches "${HYLE_SOURCE_DIR}/third_party/ggml_cuda_runtime")
  list(APPEND _manifests "${_cuda_patches}/patches.json")
  list(APPEND _patch_args --manifest "${_cuda_patches}/patches.json")
  string(APPEND _source_identity "${HYLE_DEP_GGML_CUDA_RUNTIME_PATCHES_VENDORED_TREE_SHA256}")
  list(APPEND _locked ggml_cuda_runtime_patches)
  list(APPEND _licenses
    "${_cuda_patches}/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|ggml-cuda-patches-LICENSE"
    "${_cuda_patches}/NOTICE.md|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|ggml-cuda-patches-NOTICE.md")
endif()
string(SHA256 _source_identity "${_source_identity}")
string(SUBSTRING "${_source_identity}" 0 16 _source_id)
set(_cpu_provider_version "456172ec+hyle-resource-v1")
if(_cpu_upgrade)
  set(_cpu_provider_version "${_selected_commit}+hyle-resource-iq1-${_source_id}")
endif()
set(HYLE_GGML_CPU_PROVIDER_VERSION "${_cpu_provider_version}" PARENT_SCOPE)
set(_patched_source "${CMAKE_CURRENT_BINARY_DIR}/source-${_source_id}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${_manifests} "${CMAKE_CURRENT_SOURCE_DIR}/prepare_source.py")
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare_source.py"
    --source "${ggml_SOURCE_DIR}" --output "${_patched_source}" ${_patch_args}
  RESULT_VARIABLE _result OUTPUT_VARIABLE _output ERROR_VARIABLE _error)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "ggml source preparation failed: ${_error}")
endif()
message(STATUS "${_output}")
include(ExternalProject)
set_property(DIRECTORY PROPERTY EP_PREFIX "${HYLE_DEP_BUILD_DIR}/gg-${_source_id}")

block()
  if(MSVC)
    set(CMAKE_ASM_SOURCE_FILE_EXTENSIONS s S)
  endif()
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(CMAKE_C_VISIBILITY_PRESET hidden)
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
  set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
  set(HYLE_GGML_PARENT_PROJECT_INCLUDE_BEFORE "${CMAKE_PROJECT_ggml_INCLUDE_BEFORE}")
  set(CMAKE_PROJECT_ggml_INCLUDE_BEFORE "${HYLE_SOURCE_DIR}/cmake/HyleGgmlProjectPolicy.cmake")
  set(BUILD_SHARED_LIBS OFF)
  foreach(_option IN ITEMS CUDA HIP MUSA METAL BLAS SYCL VULKAN_CHECK_RESULTS
      VULKAN_RUN_TESTS VULKAN_VALIDATE VULKAN_DEBUG VULKAN_MEMORY_DEBUG
      OPENMP OPENMP_FETCH RPC OPENCL WEBGPU HEXAGON ZDNN VIRTGPU VIRTGPU_BACKEND
      OPENVINO ET ET_SYSEMU CANN KOMPUTE ZENDNN NATIVE CCACHE BACKEND_DL LTO
      STATIC CPU_ALL_VARIANTS BUILD_TESTS BUILD_EXAMPLES
      ACCELERATE LLAMAFILE CPU_HBM CPU_KLEIDIAI SSE42 AVX AVX_VNNI AVX2 BMI2
      AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16 FMA F16C AMX_TILE AMX_INT8 AMX_BF16)
    set(GGML_${_option} OFF)
  endforeach()
  set(GGML_CPU "${_cpu}")
  set(GGML_CUDA "${HYLE_COMPUTE_CUDA_ENABLED}")
  set(GGML_CUDA_GRAPHS "${HYLE_COMPUTE_CUDA_ENABLED}")
  set(GGML_CUDA_NCCL OFF)
  set(GGML_VULKAN "${HYLE_COMPUTE_VULKAN_ENABLED}")
  set(GGML_METAL "${HYLE_COMPUTE_APPLE_ENABLED}")
  # Original embedded Metal sources survive prefix relocation in every delivery.
  # Device libraries and shape pipelines are prepared cold by the native owner.
  set(GGML_METAL_EMBED_LIBRARY "${HYLE_COMPUTE_APPLE_ENABLED}")
  set(GGML_BUILD_COMMIT "${_selected_commit}-hyle")
  add_subdirectory("${_patched_source}" vendor EXCLUDE_FROM_ALL)
endblock()

if(_cpu_upgrade)
  target_compile_definitions(ggml INTERFACE HYLE_GGML_CPU_IQ1_EXTENSIONS=1)
endif()

set(_vendor_targets ggml ggml-base)
set(_sources metadata_arena.cc)
if(HYLE_COMPUTE_APPLE_ENABLED)
  list(APPEND _vendor_targets ggml-metal)
endif()
if(_cpu)
  list(APPEND _vendor_targets ggml-cpu)
  list(APPEND _sources cpu_execution.cc cpu_program.cc)
endif()
if(HYLE_COMPUTE_CUDA_ENABLED)
  list(APPEND _vendor_targets ggml-cuda)
  target_include_directories(ggml-cuda PRIVATE
    "${HYLE_SOURCE_DIR}/src/compute_provider_ggml_cuda")
  target_compile_options(ggml-cuda PRIVATE
    "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<BOOL:${MSVC}>>>:-Xcompiler=-fvisibility=hidden,-fvisibility-inlines-hidden>")
  if(NOT MSVC AND NOT HYLE_SANITIZER STREQUAL "off")
    target_compile_options(ggml-cuda PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:--forward-unknown-to-host-compiler>")
    target_link_options(ggml-cuda PRIVATE
      "$<$<LINK_LANGUAGE:CUDA>:--forward-unknown-to-host-compiler>")
  endif()
  # Explicit-frequency RoPE needs full-range sinf/cosf. Compile the unchanged
  # upstream unary unit without its global intrinsic substitution; preserve
  # every other ggml CUDA unit's options and existing object files.
  # ggml enables CUDA in its child directory; this sibling object target also
  # needs the language's compile rules in the current directory scope.
  enable_language(CUDA)
  get_target_property(_cuda_sources ggml-cuda SOURCES)
  set(_precise_unary "${_cuda_sources}")
  list(FILTER _precise_unary INCLUDE REGEX "/unary\\.cu$")
  list(LENGTH _precise_unary _precise_count)
  if(NOT _precise_count EQUAL 1)
    message(FATAL_ERROR "Locked ggml CUDA must provide exactly one unary.cu")
  endif()
  list(REMOVE_ITEM _cuda_sources ${_precise_unary})
  set_property(TARGET ggml-cuda PROPERTY SOURCES "${_cuda_sources}")
  get_target_property(_cuda_options ggml-cuda COMPILE_OPTIONS)
  string(REPLACE "-use_fast_math;" "" _precise_options "${_cuda_options}")
  if(_precise_options STREQUAL _cuda_options)
    message(FATAL_ERROR "Locked ggml CUDA fast-math options changed")
  endif()
  add_library(ggml-cuda-precise-unary OBJECT ${_precise_unary})
  target_compile_options(ggml-cuda-precise-unary PRIVATE ${_precise_options})
  target_compile_definitions(ggml-cuda-precise-unary PRIVATE
    "$<TARGET_PROPERTY:ggml-cuda,COMPILE_DEFINITIONS>")
  target_include_directories(ggml-cuda-precise-unary PRIVATE
    "$<TARGET_PROPERTY:ggml-cuda,INCLUDE_DIRECTORIES>")
  get_target_property(_cuda_dependencies ggml-cuda LINK_LIBRARIES)
  target_link_libraries(ggml-cuda-precise-unary PRIVATE ${_cuda_dependencies})
  foreach(_property CXX_STANDARD CXX_STANDARD_REQUIRED CUDA_STANDARD
                    CUDA_ARCHITECTURES CUDA_RUNTIME_LIBRARY
                    CUDA_SEPARABLE_COMPILATION)
    get_target_property(_value ggml-cuda ${_property})
    if(NOT _value MATCHES "-NOTFOUND$")
      set_property(TARGET ggml-cuda-precise-unary PROPERTY ${_property} "${_value}")
    endif()
  endforeach()
  set_target_properties(ggml-cuda-precise-unary PROPERTIES POSITION_INDEPENDENT_CODE ON)
  hyle_apply_sanitizer(ggml-cuda-precise-unary)
  target_sources(ggml-cuda PRIVATE $<TARGET_OBJECTS:ggml-cuda-precise-unary>)
endif()
if(HYLE_COMPUTE_VULKAN_ENABLED)
  list(APPEND _vendor_targets ggml-vulkan)
  target_link_libraries(ggml-vulkan PRIVATE "$<BUILD_INTERFACE:Vulkan-Headers>")
  target_include_directories(ggml-vulkan BEFORE PRIVATE
    "$<TARGET_PROPERTY:Vulkan-Headers,INTERFACE_INCLUDE_DIRECTORIES>")
  target_include_directories(ggml-vulkan PRIVATE
    "${HYLE_SOURCE_DIR}/src/compute_provider_ggml_vulkan")
  target_compile_definitions(ggml-base PRIVATE GGML_HYLE_EMBEDDED=1)
endif()
foreach(_target IN LISTS _vendor_targets)
  if(NOT TARGET ${_target})
    message(FATAL_ERROR "Locked ggml did not define '${_target}'")
  endif()
  set_target_properties(${_target} PROPERTIES
    EXPORT_NAME "_ggml_${_target}" POSITION_INDEPENDENT_CODE ON)
  hyle_apply_sanitizer(${_target})
  if(MSVC AND NOT _target STREQUAL "ggml-cuda")
    target_compile_options(${_target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/EHsc->")
  endif()
endforeach()
set_property(TARGET ggml PROPERTY PUBLIC_HEADER "")

hyle_add_module(compute_ggml_runtime SOURCES ${_sources})
target_link_libraries(hyle_compute_ggml_runtime PRIVATE ggml)
if(_cpu AND HYLE_TARGET_PROCESSOR STREQUAL "x86_64" AND
   CMAKE_SYSTEM_NAME MATCHES "^(Windows|Linux)$")
  set(_cpu_program_basename hyle_ggml_cpu_avx2_456172ec)
  if(_cpu_upgrade)
    string(SUBSTRING "${_selected_commit}" 0 8 _cpu_commit_short)
    set(_cpu_program_basename "hyle_ggml_cpu_avx2_${_cpu_commit_short}_${_source_id}")
  endif()
  include("${CMAKE_CURRENT_SOURCE_DIR}/cpu_program.cmake")
  target_link_libraries(hyle_compute_ggml_runtime PRIVATE ${CMAKE_DL_LIBS})
endif()
set(HYLE_GGML_SOURCE_DIR "${_patched_source}" PARENT_SCOPE)
set(HYLE_GGML_VULKAN_SOURCE_DIR "${_patched_source}" PARENT_SCOPE)
hyle_register_package_dependency(
  NAME ggml_runtime OWNER compute_ggml_runtime
  STATIC_EXPORT_TARGETS ${_vendor_targets}
  STATIC_FIND_DEPENDENCIES Threads
  LOCKED_DEPENDENCIES ${_locked}
  RUNTIME_FILE_SPECS ${_licenses})
if(HYLE_COMPUTE_CUDA_ENABLED)
  hyle_register_package_dependency(
    NAME ggml_cuda_sdk OWNER compute_ggml_runtime
    FEATURES compute_cuda FIND_DEPENDENCIES CUDAToolkit|13.3)
endif()
