include_guard(GLOBAL)

if(HYLE_BUILD_COMPUTE_PROVIDER_ONEDNN OR
   HYLE_BUILD_PRODUCT_COMPUTE OR HYLE_BUILD_PRODUCT_GRAPH)
  hyle_auto_feature_option(HYLE_COMPUTE_ONEDNN
    "Build the x64 oneDNN BRGEMM MatMul provider" DEFAULT AUTO)
else()
  set(HYLE_COMPUTE_ONEDNN OFF)
endif()

function(hyle_configure_compute_dependencies)
  if(DEFINED HYLE_COMPUTE_CPU AND NOT HYLE_COMPUTE_CPU)
    if(HYLE_COMPUTE_ONEDNN STREQUAL "ON")
      message(FATAL_ERROR
        "HYLE_COMPUTE_ONEDNN=ON requires HYLE_COMPUTE_CPU=ON")
    endif()
    message(STATUS
      "HYLE: CPU route OFF -- oneDNN dependency probe/download skipped")
    set(HYLE_COMPUTE_ONEDNN_ENABLED OFF PARENT_SCOPE)
    return()
  endif()

  set(_supported FALSE)
  if(CMAKE_SYSTEM_NAME MATCHES "^(Windows|Linux|Darwin)$" AND
     HYLE_TARGET_PROCESSOR STREQUAL "x86_64")
    set(_supported TRUE)
  endif()

  if(HYLE_COMPUTE_ONEDNN STREQUAL "OFF")
    message(STATUS "HYLE: oneDNN provider OFF -- dependency download skipped")
    set(HYLE_COMPUTE_ONEDNN_ENABLED OFF PARENT_SCOPE)
    return()
  endif()
  if(NOT _supported)
    if(HYLE_COMPUTE_ONEDNN STREQUAL "ON")
      message(FATAL_ERROR
        "HYLE_COMPUTE_ONEDNN=ON is unsupported for "
        "${CMAKE_SYSTEM_NAME}/${HYLE_TARGET_PROCESSOR}")
    endif()
    message(STATUS
      "HYLE: oneDNN provider AUTO -- unsupported target, capability disabled")
    set(HYLE_COMPUTE_ONEDNN_ENABLED OFF PARENT_SCOPE)
    return()
  endif()

  include(FetchContent)
  FetchContent_Declare(onednn
    URL      ${HYLE_DEP_ONEDNN_ARCHIVE_URL}
    URL_HASH SHA256=${HYLE_DEP_ONEDNN_ARCHIVE_SHA256}
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
  block()
    # The HYLE product is C++23, but language mode is a target/source contract,
    # not a license to compile vendored implementation files under a newer
    # dialect than they advertise. oneDNN 3.13 still deliberately uses
    # facilities deprecated by C++23; keep its private build at C++17 while
    # HYLE targets created after this block retain C++23.
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    # HYLE owns execution policy and packaging. oneDNN contributes only the
    # sequential x64 BRGEMM ukernel implementation; HYLE partitions independent
    # output-column tiles through the caller's explicit executor. The locked
    # experimental API is private and must be re-audited on every upgrade.
    set(ONEDNN_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
    set(ONEDNN_BUILD_GRAPH OFF CACHE BOOL "" FORCE)
    set(ONEDNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(ONEDNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(ONEDNN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(ONEDNN_CPU_RUNTIME SEQ CACHE STRING "" FORCE)
    set(ONEDNN_GPU_RUNTIME NONE CACHE STRING "" FORCE)
    set(ONEDNN_ENABLE_WORKLOAD INFERENCE CACHE STRING "" FORCE)
    set(ONEDNN_ENABLE_PRIMITIVE MATMUL CACHE STRING "" FORCE)
    set(ONEDNN_ENABLE_PRIMITIVE_CACHE OFF CACHE BOOL "" FORCE)
    set(ONEDNN_ENABLE_CONCURRENT_EXEC ON CACHE BOOL "" FORCE)
    set(ONEDNN_EXPERIMENTAL_UKERNEL ON CACHE BOOL "" FORCE)
    set(ONEDNN_ENABLE_JIT_PROFILING OFF CACHE BOOL "" FORCE)
    set(ONEDNN_ENABLE_ITT_TASKS OFF CACHE BOOL "" FORCE)
    set(ONEDNN_VERBOSE OFF CACHE BOOL "" FORCE)
    set(ONEDNN_ENABLE_MAX_CPU_ISA OFF CACHE BOOL "" FORCE)
    set(ONEDNN_ENABLE_CPU_ISA_HINTS OFF CACHE BOOL "" FORCE)
    set(ONEDNN_WERROR OFF CACHE BOOL "" FORCE)
    hyle_fetchcontent_make_available_locked(onednn)
  endblock()
  FetchContent_GetProperties(onednn)
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
  set(HYLE_COMPUTE_ONEDNN_ENABLED ON PARENT_SCOPE)
  message(STATUS
    "HYLE: oneDNN ${HYLE_DEP_ONEDNN_VERSION} -- x64 BRGEMM ukernel provider enabled")
endfunction()
