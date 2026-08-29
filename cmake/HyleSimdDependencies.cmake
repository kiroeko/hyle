include_guard(GLOBAL)

# Highway owns the portable register/target mechanism used by the SIMD product.
# Keep acquisition selected-component-local and header-only at this boundary:
# HYLE's target-specific translation units compile the required implementation,
# so ordinary source/static/shared consumers do not inherit a second library ABI.
function(hyle_configure_simd_dependencies)
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
  set(HYLE_SIMD_HIGHWAY_SOURCE_DIR "${highway_SOURCE_DIR}" PARENT_SCOPE)
endfunction()
