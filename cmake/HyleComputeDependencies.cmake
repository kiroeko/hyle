include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/HyleComputeRoutes.cmake")

macro(_hyle_onednn_option name value)
  # Upstream still reads legacy aliases internally. Both spellings must shadow
  # a parent's normal variables, without changing that parent's cache policy.
  set(ONEDNN_${name} "${value}")
  set(DNNL_${name} "${value}")
endmacro()

function(hyle_configure_compute_dependencies)
  if(DEFINED _HYLE_COMPUTE_REQUEST_CPU AND NOT _HYLE_COMPUTE_REQUEST_CPU)
    message(STATUS
      "HYLE: CPU route OFF -- oneDNN dependency probe/download skipped")
    set(_HYLE_COMPUTE_HAVE_ONEDNN OFF PARENT_SCOPE)
    return()
  endif()

  _hyle_compute_cpu_provider_supported(onednn _supported)
  if(NOT _supported)
    message(STATUS "HYLE: oneDNN unavailable on ${CMAKE_SYSTEM_NAME}/${HYLE_TARGET_PROCESSOR}")
    set(_HYLE_COMPUTE_HAVE_ONEDNN OFF PARENT_SCOPE)
    return()
  endif()

  include(FetchContent)
  FetchContent_Declare(onednn
    URL      ${HYLE_DEP_ONEDNN_ARCHIVE_URL}
    URL_HASH SHA256=${HYLE_DEP_ONEDNN_ARCHIVE_SHA256}
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
  block()
    # Configure this private vendor without rewriting an embedding parent's cache.
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0126 NEW)
    # oneDNN options.cmake FORCE-writes DNNL_* cache aliases even when its
    # ONEDNN_* options are normal variables. Preserve existing parent entries,
    # including their GUI/type metadata; new vendor probes remain cached.
    get_cmake_property(_cache_variables CACHE_VARIABLES)
    set(_parent_dnnl_cache "")
    set(_cache_properties VALUE TYPE HELPSTRING ADVANCED STRINGS)
    foreach(_name IN LISTS _cache_variables)
      if(NOT _name MATCHES "^DNNL_")
        continue()
      endif()
      list(APPEND _parent_dnnl_cache "${_name}")
      foreach(_property IN LISTS _cache_properties)
        get_property(_saved_${_name}_${_property}_set CACHE "${_name}"
          PROPERTY "${_property}" SET)
        get_property(_saved_${_name}_${_property} CACHE "${_name}"
          PROPERTY "${_property}")
      endforeach()
    endforeach()
    # The HYLE product is C++23, but language mode is a target/source contract,
    # not a license to compile vendored implementation files under a newer
    # dialect than they advertise. oneDNN 3.13 still deliberately uses
    # facilities deprecated by C++23; keep its private build at C++17 while
    # HYLE targets created after this block retain C++23.
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    # BRGEMM partitions columns through HYLE; normalization primitives borrow
    # the same caller executor through oneDNN's threadpool interop. Neither
    # path creates a private pool. The locked ukernel API stays private.
    _hyle_onednn_option(LIBRARY_TYPE STATIC)
    _hyle_onednn_option(BUILD_GRAPH OFF)
    _hyle_onednn_option(BUILD_TESTS OFF)
    _hyle_onednn_option(BUILD_EXAMPLES OFF)
    _hyle_onednn_option(BUILD_DOC OFF)
    _hyle_onednn_option(CPU_RUNTIME THREADPOOL)
    _hyle_onednn_option(GPU_RUNTIME NONE)
    _hyle_onednn_option(ENABLE_WORKLOAD INFERENCE)
    _hyle_onednn_option(ENABLE_PRIMITIVE
      "MATMUL;LAYER_NORMALIZATION;BINARY;ELTWISE;REORDER")
    _hyle_onednn_option(ENABLE_PRIMITIVE_CACHE OFF)
    _hyle_onednn_option(ENABLE_CONCURRENT_EXEC ON)
    _hyle_onednn_option(EXPERIMENTAL_UKERNEL ON)
    _hyle_onednn_option(ENABLE_JIT_PROFILING OFF)
    _hyle_onednn_option(ENABLE_ITT_TASKS OFF)
    _hyle_onednn_option(VERBOSE OFF)
    _hyle_onednn_option(ENABLE_MAX_CPU_ISA OFF)
    _hyle_onednn_option(ENABLE_CPU_ISA_HINTS OFF)
    _hyle_onednn_option(WERROR OFF)
    hyle_fetchcontent_make_available_locked(onednn)
    foreach(_name IN LISTS _parent_dnnl_cache)
      # Recreate only saved entries: set_property cannot remove an upstream
      # STRINGS/ADVANCED property that the parent originally left unset.
      unset("${_name}" CACHE)
      set("${_name}" "" CACHE INTERNAL "")
      foreach(_property IN LISTS _cache_properties)
        if(_saved_${_name}_${_property}_set)
          set_property(CACHE "${_name}" PROPERTY "${_property}"
            "${_saved_${_name}_${_property}}")
        endif()
      endforeach()
    endforeach()
  endblock()
  FetchContent_GetProperties(onednn)
  include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/compute_provider_onednn/runtime_overlay.cmake")
  hyle_onednn_runtime_overlay("${onednn_SOURCE_DIR}" "${onednn_BINARY_DIR}")
  if(NOT TARGET dnnl)
    message(FATAL_ERROR "Locked oneDNN source did not define target 'dnnl'")
  endif()
  set_property(DIRECTORY "${onednn_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)
  set_target_properties(dnnl PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME "_onednn_static"
    HYLE_DEP_SOURCE_DIR "${onednn_SOURCE_DIR}"
    HYLE_DEP_BINARY_DIR "${onednn_BINARY_DIR}")
  set(HYLE_ONEDNN_SOURCE_DIR "${onednn_SOURCE_DIR}" PARENT_SCOPE)
  set(HYLE_ONEDNN_BINARY_DIR "${onednn_BINARY_DIR}" PARENT_SCOPE)
  set(_HYLE_COMPUTE_HAVE_ONEDNN ON PARENT_SCOPE)
  message(STATUS
    "HYLE: oneDNN ${HYLE_DEP_ONEDNN_VERSION} -- x64 BRGEMM, normalization and tensor primitives enabled")
endfunction()
