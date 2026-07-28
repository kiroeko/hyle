include_guard(GLOBAL)

# Source delivery is independent from HYLE_STATIC/HYLE_INSTALL. Those options
# select a binary SDK shape; this target archives the complete version-controlled
# source tree for a specifically authorized recipient.
function(hyle_add_source_package)
  if(NOT PROJECT_IS_TOP_LEVEL)
    return()
  endif()

  find_package(Git QUIET)
  if(NOT Git_FOUND OR NOT EXISTS "${HYLE_SOURCE_DIR}/.git")
    message(STATUS
      "HYLE source package target unavailable (requires Git and a Git checkout)")
    return()
  endif()

  set(_hyle_source_dist_dir "${HYLE_BINARY_DIR}/dist")
  set(_hyle_source_archive
      "${_hyle_source_dist_dir}/hyle-${PROJECT_VERSION}-source.zip")
  add_custom_target(hyle_source_package
    COMMAND "${CMAKE_COMMAND}"
      "-DHYLE_SOURCE_DIR=${HYLE_SOURCE_DIR}"
      "-DHYLE_SOURCE_ARCHIVE=${_hyle_source_archive}"
      "-DHYLE_SOURCE_PREFIX=hyle-${PROJECT_VERSION}-source/"
      "-DHYLE_GIT_EXECUTABLE=${GIT_EXECUTABLE}"
      -P "${HYLE_SOURCE_DIR}/cmake/HyleCreateSourcePackage.cmake"
    BYPRODUCTS
      "${_hyle_source_archive}"
      "${_hyle_source_archive}.sha256"
    COMMENT
      "Packaging the clean Git-tracked HYLE source tree (license is unchanged)"
    VERBATIM)
  set_property(TARGET hyle_source_package PROPERTY FOLDER "hyle/distribution")
endfunction()
