include_guard(GLOBAL)

# Install/export is deliberately centralized here. Module CMakeLists describe
# factor and product targets; this file turns the already-resolved target graph into one
# relocatable package without reconstructing dependencies from filenames.
function(hyle_install_package)
  if(NOT HYLE_INSTALL)
    return()
  endif()

  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  set(_hyle_export_targets hyle_build_contract hyle_trace hyle_testing)
  set(_hyle_package_factors "")
  foreach(_component IN LISTS HYLE_KNOWN_COMPONENTS)
    if(TARGET hyle_${_component})
      list(APPEND _hyle_export_targets hyle_${_component})
      list(APPEND _hyle_package_factors "${_component}")
    endif()
  endforeach()
  set(_hyle_package_products "")
  foreach(_product IN LISTS HYLE_KNOWN_PRODUCTS)
    if(TARGET hyle_product_${_product})
      list(APPEND _hyle_export_targets hyle_product_${_product})
      list(APPEND _hyle_package_products "${_product}")
    endif()
  endforeach()
  hyle_package_collect_registry(
    _hyle_dependency_export_targets
    HYLE_PACKAGE_FEATURES
    HYLE_PACKAGE_FIND_DEPENDENCIES)
  list(APPEND _hyle_export_targets ${_hyle_dependency_export_targets})

  # An installed shared HYLE module must be able to resolve sibling HYLE
  # modules after the whole SDK prefix is moved.  A consumer executable's
  # RUNPATH is not inherited while loading transitive ELF dependencies, so
  # relying on the consumer's link line works for direct modules but fails for
  # e.g. hyle_simd -> hyle_platform -> hyle_error.  Give each installed HYLE
  # shared object its own loader-relative lookup path.  Keep this target-local:
  # source-embedding parents retain ownership of their build/install RPATH.
  if(NOT HYLE_STATIC AND UNIX)
    if(APPLE)
      set(_hyle_sibling_install_rpath "@loader_path")
    else()
      set(_hyle_sibling_install_rpath "$ORIGIN")
    endif()
    foreach(_hyle_export_target IN LISTS _hyle_export_targets)
      get_target_property(_hyle_export_type
        ${_hyle_export_target} TYPE)
      if(_hyle_export_type STREQUAL "SHARED_LIBRARY" OR
         _hyle_export_type STREQUAL "MODULE_LIBRARY")
        set_target_properties(${_hyle_export_target} PROPERTIES
          INSTALL_RPATH "${_hyle_sibling_install_rpath}"
          INSTALL_RPATH_USE_LINK_PATH OFF)
      endif()
    endforeach()
    unset(_hyle_export_target)
    unset(_hyle_export_type)
    unset(_hyle_sibling_install_rpath)
  endif()

  install(TARGETS ${_hyle_export_targets}
    EXPORT HyleTargets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT Hyle_Runtime
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Hyle_Runtime
            NAMELINK_COMPONENT Hyle_Development
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Hyle_Development)

  # Install exactly the selected factor closure. Copying the whole public
  # tree made a platform-only SDK appear usable even when its target graph had
  # omitted a direct header dependency. HyleComponents owns the header boundary
  # for ordinary and split components; its configure-time include audit ensures
  # every copied cross-component header is backed by a PUBLIC imported edge.
  set(_hyle_installed_headers "")
  foreach(_hyle_package_factor IN LISTS _hyle_package_factors)
    _hyle_component_public_headers(
      "${_hyle_package_factor}" _hyle_component_headers)
    list(APPEND _hyle_installed_headers ${_hyle_component_headers})
  endforeach()
  # obs/trace.h belongs to the hyle_trace foundation target and is consumed by
  # TRACE-declaring modules such as mem even when obs is absent. Do not leak it
  # into a Core-only SDK when no selected module consumes that foundation seam.
  get_property(_hyle_trace_consumed GLOBAL PROPERTY HYLE_TRACE_CONSUMED)
  if(_hyle_trace_consumed)
    list(APPEND _hyle_installed_headers
      "${HYLE_SOURCE_DIR}/include/hyle/obs/trace.h")
  endif()
  list(REMOVE_DUPLICATES _hyle_installed_headers)
  foreach(_hyle_public_header IN LISTS _hyle_installed_headers)
    file(RELATIVE_PATH _hyle_public_header_relative
      "${HYLE_SOURCE_DIR}/include" "${_hyle_public_header}")
    get_filename_component(_hyle_public_header_directory
      "${_hyle_public_header_relative}" DIRECTORY)
    install(FILES "${_hyle_public_header}"
      DESTINATION
        "${CMAKE_INSTALL_INCLUDEDIR}/${_hyle_public_header_directory}"
      COMPONENT Hyle_Development)
  endforeach()
  unset(_hyle_component_headers)
  unset(_hyle_installed_headers)
  unset(_hyle_package_factor)
  unset(_hyle_public_header)
  unset(_hyle_public_header_directory)
  unset(_hyle_public_header_relative)
  unset(_hyle_trace_consumed)
  install(FILES "${HYLE_SOURCE_DIR}/LICENSE"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Hyle"
    RENAME HYLE-LICENSE
    COMPONENT Hyle_Runtime)
  install(FILES "${HYLE_SOURCE_DIR}/THIRD_PARTY_NOTICES.md"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Hyle"
    RENAME THIRD_PARTY-NOTICES.md
    COMPONENT Hyle_Runtime)
  # Third-party payload follows the same module-owned registry as export and
  # feature metadata; the package layer never switches on dependency names.
  hyle_package_install_registered_assets()

  set(_hyle_package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/Hyle")
  set(HYLE_PACKAGE_COMPONENTS "${_hyle_package_products}")
  set(HYLE_PACKAGE_STATIC "${HYLE_STATIC}")

  configure_package_config_file(
    "${HYLE_SOURCE_DIR}/cmake/HyleConfig.cmake.in"
    "${HYLE_BINARY_DIR}/HyleConfig.cmake"
    INSTALL_DESTINATION "${_hyle_package_dir}")
  if(PROJECT_VERSION_MAJOR EQUAL 0)
    set(_hyle_package_compatibility ExactVersion)
  else()
    set(_hyle_package_compatibility SameMajorVersion)
  endif()
  write_basic_package_version_file(
    "${HYLE_BINARY_DIR}/HyleConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY ${_hyle_package_compatibility})
  unset(_hyle_package_compatibility)

  install(EXPORT HyleTargets
    FILE HyleTargets.cmake
    NAMESPACE hyle::
    DESTINATION "${_hyle_package_dir}"
    COMPONENT Hyle_Development)
  install(FILES
    "${HYLE_BINARY_DIR}/HyleConfig.cmake"
    "${HYLE_BINARY_DIR}/HyleConfigVersion.cmake"
    DESTINATION "${_hyle_package_dir}"
    COMPONENT Hyle_Development)
endfunction()
