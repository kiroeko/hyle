include_guard(GLOBAL)

function(_hyle_package_define_export_group group)
  foreach(_target IN LISTS ARGN)
    get_target_property(_existing "${_target}" HYLE_PACKAGE_EXPORT_GROUP)
    if(_existing AND NOT _existing STREQUAL group)
      message(FATAL_ERROR
        "Package target '${_target}' belongs to both '${_existing}' and '${group}'")
    endif()
    set_property(TARGET "${_target}" PROPERTY HYLE_PACKAGE_EXPORT_GROUP "${group}")
  endforeach()
  set(_hyle_package_groups "${_hyle_package_groups};${group}" PARENT_SCOPE)
  set(_hyle_group_${group}_targets "${ARGN}" PARENT_SCOPE)
endfunction()

function(_hyle_package_group_closure roots out_var)
  set(_pending "${roots}")
  set(_seen "")
  while(_pending)
    list(POP_FRONT _pending _group)
    if(_group IN_LIST _seen)
      continue()
    endif()
    list(APPEND _seen "${_group}")
    list(APPEND _pending ${_hyle_group_${_group}_dependencies})
  endwhile()
  set(_ordered "")
  foreach(_group IN LISTS _hyle_package_group_order)
    if(_group IN_LIST _seen)
      list(APPEND _ordered "${_group}")
    endif()
  endforeach()
  set(${out_var} "${_ordered}" PARENT_SCOPE)
endfunction()

# Install/export is deliberately centralized here. Module CMakeLists describe
# factor and product targets; this file turns the already-resolved target graph into one
# relocatable package without reconstructing dependencies from filenames.
function(hyle_install_package)
  hyle_package_collect_registry(_source_exports HYLE_PACKAGE_FEATURES _source_dependencies)
  set(HYLE_PACKAGE_COMPUTE_ROUTES "${HYLE_COMPUTE_ROUTE_LIST}")
  # Diagnostic build identity, not a cross-ABI guarantee or WIN certificate.
  # Dependency identities come from the existing registry/lock, not a parallel
  # handwritten list of Providers. Include only this package's locked inputs.
  set(_recipe "routes=${HYLE_COMPUTE_ROUTE_LIST}|features=${HYLE_PACKAGE_FEATURES}")
  string(APPEND _recipe
    "|${CMAKE_SYSTEM_NAME}/${HYLE_TARGET_PROCESSOR}"
    "|${CMAKE_CXX_COMPILER_ID}/${CMAKE_CXX_COMPILER_VERSION}"
    "|static=${HYLE_STATIC}|config=${CMAKE_BUILD_TYPE}"
    "|flags=${CMAKE_CXX_FLAGS}/${CMAKE_CXX_FLAGS_RELEASE}"
    "|crt=${CMAKE_MSVC_RUNTIME_LIBRARY}|cuda=${CMAKE_CUDA_ARCHITECTURES}"
    "|isa=${HYLE_SIMD_EXACT_TARGET}|ipo=${CMAKE_INTERPROCEDURAL_OPTIMIZATION}"
    "|ipo-release=${CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE}"
    "|configs=${CMAKE_CONFIGURATION_TYPES}|debug=${CMAKE_CXX_FLAGS_DEBUG}"
    "|relwithdebinfo=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}"
    "|cuda-compiler=${CMAKE_CUDA_COMPILER_VERSION}"
    "|cuda-flags=${CMAKE_CUDA_FLAGS}/${CMAKE_CUDA_FLAGS_RELEASE}"
    "|metal=${HYLE_APPLE_METAL_COMPILER}|vulkan=${HYLE_VULKAN_SHADER_COMPILER}"
    "|trace=${HYLE_TRACE}|sanitizer=${HYLE_SANITIZER}")
  get_directory_property(_directory_options COMPILE_OPTIONS)
  get_directory_property(_directory_definitions COMPILE_DEFINITIONS)
  string(APPEND _recipe "|options=${_directory_options}|definitions=${_directory_definitions}")
  _hyle_package_selected_dependencies(_dependencies)
  foreach(_dependency IN LISTS _dependencies)
    _hyle_package_dependency_property("${_dependency}" LOCKED_DEPENDENCIES _locked)
    foreach(_identity IN LISTS _locked)
      string(TOUPPER "${_identity}" _upper)
      string(APPEND _recipe "|${_identity}=${HYLE_DEP_${_upper}_COMMIT}:"
        "${HYLE_DEP_${_upper}_VERSION}:${HYLE_DEP_${_upper}_VENDORED_TREE_SHA256}:")
    endforeach()
  endforeach()
  string(APPEND _recipe "|ggml-cpu=${HYLE_GGML_CPU_PROVIDER_VERSION}"
    "|ggml-cuda=${HYLE_GGML_CUDA_PROVIDER_VERSION}")
  string(SHA256 HYLE_PACKAGE_COMPUTE_RECIPE_ID "${_recipe}")
  set_property(GLOBAL PROPERTY HYLE_COMPUTE_BUILD_RECIPE_ID
    "${HYLE_PACKAGE_COMPUTE_RECIPE_ID}")

  foreach(_product IN ITEMS compute graph)
    if(TARGET hyle_product_${_product})
      set_target_properties(hyle_product_${_product} PROPERTIES
        HYLE_COMPUTE_ROUTES "${HYLE_PACKAGE_COMPUTE_ROUTES}"
        HYLE_COMPUTE_RECIPE_ID "${HYLE_PACKAGE_COMPUTE_RECIPE_ID}")
      set_property(TARGET hyle_product_${_product} APPEND PROPERTY EXPORT_PROPERTIES
        HYLE_COMPUTE_ROUTES HYLE_COMPUTE_RECIPE_ID)
    endif()
  endforeach()
  if(NOT HYLE_INSTALL)
    return()
  endif()

  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  set(_hyle_export_targets "")
  set(_hyle_package_factors "")
  if(DEFINED HYLE_PRODUCTION_COMPONENT_LIST)
    set(_hyle_candidate_factors "${HYLE_PRODUCTION_COMPONENT_LIST}")
    set(_hyle_candidate_products "${HYLE_PRODUCTION_PRODUCT_LIST}")
  else()
    set(_hyle_candidate_factors "${HYLE_KNOWN_COMPONENTS}")
    set(_hyle_candidate_products "${HYLE_KNOWN_PRODUCTS}")
  endif()
  foreach(_component IN LISTS _hyle_candidate_factors)
    if(TARGET hyle_${_component})
      get_target_property(_internal_only hyle_${_component}
        HYLE_INTERNAL_ONLY_ABSENT)
      if(_internal_only)
        continue()
      endif()
      list(APPEND _hyle_export_targets hyle_${_component})
      list(APPEND _hyle_package_factors "${_component}")
    endif()
  endforeach()
  set(_hyle_package_products "")
  foreach(_product IN LISTS _hyle_candidate_products)
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

  # Each installed target has exactly one export group. A factor owns its
  # registered vendor targets, so private vendor cycles stay in the same file.
  # Separate files let a mem-only consumer avoid importing Compute targets or
  # probing their machine SDKs. CMake validates references between export files,
  # hence their actual target graph determines the import order below.
  set(_hyle_package_groups "")
  set(_factor_roots "")
  foreach(_factor IN LISTS _hyle_package_factors)
    list(APPEND _factor_roots hyle_${_factor})
    set(_hyle_group_factor_${_factor}_targets hyle_${_factor})
  endforeach()
  _hyle_package_target_closure("${_factor_roots}" _reachable_targets)
  foreach(_foundation IN ITEMS hyle_build_contract hyle_trace hyle_testing)
    if(_foundation IN_LIST _reachable_targets)
      _hyle_package_define_export_group("foundation_${_foundation}" "${_foundation}")
      list(APPEND _hyle_export_targets "${_foundation}")
    endif()
  endforeach()
  foreach(_dependency IN LISTS _dependencies)
    _hyle_package_dependency_property("${_dependency}" OWNER _owner)
    hyle_package_collect_registry(_targets _features _find_dependencies
      DEPENDENCIES "${_dependency}")
    if(_owner STREQUAL "foundation")
      set(_group "dependency_${_dependency}")
      # A metadata-only registry still needs a group for find_dependency calls.
      _hyle_package_define_export_group("${_group}" ${_targets})
      _hyle_package_dependency_property("${_dependency}" CONSUMER_COMPONENTS
        _hyle_group_${_group}_consumers)
    else()
      set(_group "factor_${_owner}")
      list(APPEND _hyle_group_${_group}_targets ${_targets})
    endif()
    list(APPEND _hyle_group_${_group}_find_dependencies ${_find_dependencies})
    list(APPEND _hyle_group_${_group}_features ${_features})
  endforeach()
  foreach(_factor IN LISTS _hyle_package_factors)
    list(REMOVE_DUPLICATES _hyle_group_factor_${_factor}_targets)
    _hyle_package_define_export_group("factor_${_factor}"
      ${_hyle_group_factor_${_factor}_targets})
  endforeach()
  foreach(_product IN LISTS _hyle_package_products)
    _hyle_package_define_export_group("product_${_product}" "hyle_product_${_product}")
  endforeach()
  list(FILTER _hyle_package_groups EXCLUDE REGEX "^$")
  list(REMOVE_DUPLICATES _hyle_package_groups)
  list(REMOVE_DUPLICATES _hyle_export_targets)
  foreach(_group IN LISTS _hyle_package_groups)
    set(_group_dependencies "")
    foreach(_target IN LISTS _hyle_group_${_group}_targets)
      _hyle_package_target_edges("${_target}" _edges)
      foreach(_edge IN LISTS _edges)
        get_target_property(_dependency_group "${_edge}" HYLE_PACKAGE_EXPORT_GROUP)
        if(_dependency_group AND NOT _dependency_group STREQUAL _group)
          list(APPEND _group_dependencies "${_dependency_group}")
        endif()
      endforeach()
    endforeach()
    string(REGEX REPLACE "^factor_" "" _factor_component "${_group}")
    foreach(_foundation_group IN LISTS _hyle_package_groups)
      if(_foundation_group MATCHES "^dependency_")
        # Header-only shared dependencies have no target link to walk. Their
        # declared component users provide the same ownership edge explicitly.
        if(_group MATCHES "^factor_" AND
           _factor_component IN_LIST _hyle_group_${_foundation_group}_consumers)
          list(APPEND _group_dependencies "${_foundation_group}")
        elseif(_group MATCHES "^product_" AND
               NOT _hyle_group_${_foundation_group}_targets AND
               NOT _hyle_group_${_foundation_group}_consumers)
          # A target-less foundation without narrower consumers is global
          # package metadata, e.g. a package-wide notice.
          list(APPEND _group_dependencies "${_foundation_group}")
        endif()
      endif()
    endforeach()
    list(REMOVE_DUPLICATES _group_dependencies)
    set(_hyle_group_${_group}_dependencies "${_group_dependencies}")
  endforeach()
  set(_hyle_package_group_order "")
  set(_pending_groups "${_hyle_package_groups}")
  while(_pending_groups)
    set(_progress FALSE)
    foreach(_group IN LISTS _pending_groups)
      set(_ready TRUE)
      foreach(_dependency_group IN LISTS _hyle_group_${_group}_dependencies)
        if(NOT _dependency_group IN_LIST _hyle_package_group_order)
          set(_ready FALSE)
          break()
        endif()
      endforeach()
      if(_ready)
        list(APPEND _hyle_package_group_order "${_group}")
        list(REMOVE_ITEM _pending_groups "${_group}")
        set(_progress TRUE)
      endif()
    endforeach()
    if(NOT _progress)
      message(FATAL_ERROR "Hyle installed target groups contain a dependency cycle: ${_pending_groups}")
    endif()
  endwhile()

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

  set(_hyle_package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/Hyle")
  set(_index "# Generated installed-product dependency graph.\n")
  string(APPEND _index "set(_Hyle_package_group_order [==[${_hyle_package_group_order}]==])\n")
  foreach(_group IN LISTS _hyle_package_group_order)
    if(_hyle_group_${_group}_targets)
      install(TARGETS ${_hyle_group_${_group}_targets}
        EXPORT HyleTargets_${_group}
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT Hyle_Runtime
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Hyle_Runtime
                NAMELINK_COMPONENT Hyle_Development
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT Hyle_Development)
      install(EXPORT HyleTargets_${_group}
        FILE "HyleTargets-${_group}.cmake"
        NAMESPACE hyle:: DESTINATION "${_hyle_package_dir}"
        COMPONENT Hyle_Development)
      string(APPEND _index "set(_Hyle_group_${_group}_file HyleTargets-${_group}.cmake)\n")
    endif()
    list(REMOVE_DUPLICATES _hyle_group_${_group}_find_dependencies)
    string(APPEND _index
      "set(_Hyle_group_${_group}_find_dependencies [==[${_hyle_group_${_group}_find_dependencies}]==])\n")
  endforeach()
  foreach(_product IN LISTS _hyle_package_products)
    _hyle_package_group_closure("product_${_product}" _product_groups)
    set(_required_groups "${_product_groups}")
    list(REMOVE_ITEM _required_groups "product_${_product}")
    string(APPEND _index
      "set(_Hyle_product_${_product}_groups [==[${_product_groups}]==])\n"
      "set(_Hyle_product_${_product}_required_groups [==[${_required_groups}]==])\n")
  endforeach()
  file(WRITE "${HYLE_BINARY_DIR}/HylePackageIndex.cmake" "${_index}")

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
  if(hyle_trace IN_LIST _reachable_targets)
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

  install(FILES
    "${HYLE_BINARY_DIR}/HyleConfig.cmake"
    "${HYLE_BINARY_DIR}/HyleConfigVersion.cmake"
    "${HYLE_BINARY_DIR}/HylePackageIndex.cmake"
    "${HYLE_SOURCE_DIR}/cmake/HyleRuntimeStaging.cmake"
    DESTINATION "${_hyle_package_dir}"
    COMPONENT Hyle_Development)
endfunction()
