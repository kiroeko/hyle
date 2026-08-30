include_guard(GLOBAL)

# Highway and xsimd are locked header-only implementation candidates. Each
# owning module compiles a statically selected provider body into its own
# artifact, so source/static/shared consumers do not inherit a vendor runtime
# ABI. Container currently needs only Highway; SIMD configures both providers.
function(hyle_configure_highway_dependency)
  include(FetchContent)
  FetchContent_Declare(highway
    URL      ${HYLE_DEP_HIGHWAY_ARCHIVE_URL}
    URL_HASH SHA256=${HYLE_DEP_HIGHWAY_ARCHIVE_SHA256}
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    # Populate and authenticate the source without executing upstream's project.
    # HYLE owns target selection, package shape and warning policy.
    SOURCE_SUBDIR hyle-header-only-product)
  hyle_fetchcontent_make_available_locked(highway)
  FetchContent_GetProperties(highway)
  if(NOT EXISTS "${highway_SOURCE_DIR}/hwy/highway.h" OR
     NOT EXISTS "${highway_SOURCE_DIR}/hwy/contrib/math/math-inl.h")
    message(FATAL_ERROR
      "Locked Highway source lacks the SIMD/math headers required by HYLE")
  endif()
  set(HYLE_HIGHWAY_SOURCE_DIR "${highway_SOURCE_DIR}" PARENT_SCOPE)

  if(NOT TARGET hyle_highway_headers)
    add_library(hyle_highway_headers INTERFACE)
    target_compile_features(hyle_highway_headers INTERFACE cxx_std_23)
    target_compile_definitions(hyle_highway_headers INTERFACE HWY_HEADER_ONLY)
    target_include_directories(hyle_highway_headers SYSTEM INTERFACE
      "$<BUILD_INTERFACE:${highway_SOURCE_DIR}>")
    set_target_properties(hyle_highway_headers PROPERTIES
      FOLDER "hyle/foundation/internal")
  endif()
endfunction()

function(hyle_configure_xsimd_dependency)
  include(FetchContent)
  FetchContent_Declare(xsimd
    URL      ${HYLE_DEP_XSIMD_ARCHIVE_URL}
    URL_HASH SHA256=${HYLE_DEP_XSIMD_ARCHIVE_SHA256}
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    # Populate and authenticate headers without executing upstream's project.
    SOURCE_SUBDIR hyle-header-only-product)
  hyle_fetchcontent_make_available_locked(xsimd)
  FetchContent_GetProperties(xsimd)
  if(NOT EXISTS "${xsimd_SOURCE_DIR}/include/xsimd/xsimd.hpp")
    message(FATAL_ERROR
      "Locked xsimd source lacks include/xsimd/xsimd.hpp")
  endif()
  set(HYLE_XSIMD_SOURCE_DIR "${xsimd_SOURCE_DIR}" PARENT_SCOPE)

  if(NOT TARGET hyle_xsimd_headers)
    add_library(hyle_xsimd_headers INTERFACE)
    target_compile_features(hyle_xsimd_headers INTERFACE cxx_std_23)
    target_include_directories(hyle_xsimd_headers SYSTEM INTERFACE
      "$<BUILD_INTERFACE:${xsimd_SOURCE_DIR}/include>")
    set_target_properties(hyle_xsimd_headers PROPERTIES
      FOLDER "hyle/foundation/internal")
  endif()
endfunction()

function(hyle_configure_simd_provider_dependencies)
  hyle_configure_highway_dependency()
  hyle_configure_xsimd_dependency()
  if(NOT TARGET hyle_simd_provider_headers)
    add_library(hyle_simd_provider_headers INTERFACE)
    target_link_libraries(hyle_simd_provider_headers INTERFACE
      hyle_highway_headers hyle_xsimd_headers)
    set_target_properties(hyle_simd_provider_headers PROPERTIES
      FOLDER "hyle/foundation/internal")
  endif()
  # The individual helpers publish into this wrapper's function scope. Forward
  # their authenticated source roots to the caller that owns package/license
  # registration; otherwise a wrapper call would accidentally hide them.
  set(HYLE_HIGHWAY_SOURCE_DIR "${HYLE_HIGHWAY_SOURCE_DIR}" PARENT_SCOPE)
  set(HYLE_XSIMD_SOURCE_DIR "${HYLE_XSIMD_SOURCE_DIR}" PARENT_SCOPE)
endfunction()
