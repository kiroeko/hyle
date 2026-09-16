include_guard(GLOBAL)

# External/package assets are declared once by the module that owns them.  The
# package layer projects this registry into exported targets, feature metadata,
# downstream find_dependency() calls, licenses, headers, and tools.  A spec is
# `absolute-source|relative-install-destination[|installed-name]`.
function(_hyle_package_normalized_sha256 path out_var)
  if(NOT EXISTS "${path}" OR IS_DIRECTORY "${path}" OR IS_SYMLINK "${path}")
    message(FATAL_ERROR
      "Package registry license source must be a regular file: '${path}'")
  endif()
  file(READ "${path}" _content)
  string(REPLACE "\r\n" "\n" _content "${_content}")
  string(REPLACE "\r" "\n" _content "${_content}")
  string(SHA256 _sha256 "${_content}")
  set(${out_var} "${_sha256}" PARENT_SCOPE)
endfunction()

function(_hyle_package_validate_locked_dependencies registry locked runtime_files)
  set(_runtime_hashes "")
  set(_runtime_assets "")
  foreach(_spec IN LISTS runtime_files)
    string(REPLACE "|" ";" _asset_fields "${_spec}")
    list(LENGTH _asset_fields _asset_field_count)
    if(_asset_field_count LESS 2 OR _asset_field_count GREATER 3)
      message(FATAL_ERROR
        "Package dependency '${registry}' has malformed runtime asset '${_spec}'")
    endif()
    list(GET _asset_fields 0 _source)
    if(NOT IS_ABSOLUTE "${_source}" OR NOT EXISTS "${_source}")
      message(FATAL_ERROR
        "Package dependency '${registry}' lacks runtime source '${_source}'")
    endif()
    _hyle_package_normalized_sha256("${_source}" _sha256)
    list(APPEND _runtime_hashes "${_sha256}")
    list(APPEND _runtime_assets "${_spec}")
  endforeach()

  foreach(_dependency IN LISTS locked)
    if(NOT _dependency MATCHES "^[a-z][a-z0-9_]*$")
      message(FATAL_ERROR
        "Package dependency '${registry}' has invalid locked dependency "
        "'${_dependency}'")
    endif()
    if(NOT _dependency IN_LIST HYLE_DEPENDENCY_NAMES)
      message(FATAL_ERROR
        "Package dependency '${registry}' names unlocked dependency "
        "'${_dependency}'")
    endif()
    string(TOUPPER "${_dependency}" _dependency_key)
    set(_prefix "HYLE_DEP_${_dependency_key}")
    if(NOT DEFINED ${_prefix}_CLASS OR
       NOT DEFINED ${_prefix}_LICENSE_FILE_SPECS)
      message(FATAL_ERROR
        "Package dependency '${registry}' lacks central lock projection for "
        "'${_dependency}'")
    endif()
    if(${_prefix}_CLASS STREQUAL "benchmark")
      message(FATAL_ERROR
        "Package dependency '${registry}' maps benchmark-only lock "
        "'${_dependency}' into the production package")
    endif()
    foreach(_license_spec IN LISTS ${_prefix}_LICENSE_FILE_SPECS)
      string(REPLACE "|" ";" _license_fields "${_license_spec}")
      list(LENGTH _license_fields _license_field_count)
      if(NOT _license_field_count EQUAL 3)
        message(FATAL_ERROR
          "Locked dependency '${_dependency}' has malformed license projection "
          "'${_license_spec}'")
      endif()
      list(GET _license_fields 1 _license_path)
      list(GET _license_fields 2 _license_sha256)
      list(FIND _runtime_hashes "${_license_sha256}" _runtime_index)
      if(_runtime_index EQUAL -1)
        message(FATAL_ERROR
          "Package dependency '${registry}' does not install locked license "
          "hash '${_license_sha256}' as a distinct runtime asset for "
          "'${_dependency}' path '${_license_path}'")
      endif()
      # Consume one runtime asset for each locked license path.  Keeping this a
      # multiset prevents one identical license text from silently satisfying
      # multiple dependencies or multiple required files.
      list(REMOVE_AT _runtime_hashes ${_runtime_index})
      list(REMOVE_AT _runtime_assets ${_runtime_index})
    endforeach()
  endforeach()
  if(_runtime_hashes)
    message(FATAL_ERROR
      "Package dependency '${registry}' contains unlocked runtime asset(s) "
      "after consuming every locked license: ${_runtime_assets}")
  endif()
endfunction()

function(hyle_register_package_dependency)
  cmake_parse_arguments(ARG "" "NAME;OWNER"
    "FEATURES;EXPORT_TARGETS;STATIC_EXPORT_TARGETS;FIND_DEPENDENCIES;STATIC_FIND_DEPENDENCIES;LOCKED_DEPENDENCIES;RUNTIME_FILE_SPECS;GENERATED_RUNTIME_FILE_SPECS;DEVELOPMENT_FILE_SPECS;STATIC_DEVELOPMENT_FILE_SPECS;DEVELOPMENT_DIRECTORY_SPECS;STATIC_DEVELOPMENT_DIRECTORY_SPECS;DEVELOPMENT_PROGRAM_SPECS"
    ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "hyle_register_package_dependency: unknown arguments: ${ARG_UNPARSED_ARGUMENTS}")
  endif()
  if(ARG_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR
      "hyle_register_package_dependency: keywords missing values: "
      "${ARG_KEYWORDS_MISSING_VALUES}")
  endif()
  if(NOT ARG_NAME OR NOT ARG_NAME MATCHES "^[a-z][a-z0-9_]*$")
    message(FATAL_ERROR
      "hyle_register_package_dependency requires NAME matching [a-z][a-z0-9_]*")
  endif()
  if(NOT ARG_OWNER)
    message(FATAL_ERROR
      "HYLE package dependency '${ARG_NAME}' has no OWNER")
  endif()
  if(NOT ARG_OWNER STREQUAL "foundation" AND
     NOT ARG_OWNER IN_LIST HYLE_KNOWN_COMPONENTS)
    message(FATAL_ERROR
      "HYLE package dependency '${ARG_NAME}' has unknown owner '${ARG_OWNER}'")
  endif()
  if(ARG_RUNTIME_FILE_SPECS AND NOT ARG_LOCKED_DEPENDENCIES)
    message(FATAL_ERROR
      "Package dependency '${ARG_NAME}' has runtime assets without "
      "LOCKED_DEPENDENCIES")
  endif()
  if(NOT ARG_RUNTIME_FILE_SPECS AND ARG_LOCKED_DEPENDENCIES)
    message(FATAL_ERROR
      "Package dependency '${ARG_NAME}' declares a runtime asset lock without "
      "RUNTIME_FILE_SPECS")
  endif()
  if(ARG_LOCKED_DEPENDENCIES)
    list(REMOVE_DUPLICATES ARG_LOCKED_DEPENDENCIES)
    _hyle_package_validate_locked_dependencies(
      "${ARG_NAME}" "${ARG_LOCKED_DEPENDENCIES}" "${ARG_RUNTIME_FILE_SPECS}")
  endif()
  foreach(_spec IN LISTS ARG_GENERATED_RUNTIME_FILE_SPECS)
    _hyle_package_parse_generated_asset("${ARG_NAME}" "${_spec}"
      _producer _source _destination _name)
    if(NOT _producer IN_LIST ARG_EXPORT_TARGETS)
      message(FATAL_ERROR
        "Package dependency '${ARG_NAME}' generated asset producer '${_producer}' must be an exported target of this registry")
    endif()
  endforeach()

  get_property(_registered GLOBAL PROPERTY HYLE_PACKAGE_DEPENDENCIES)
  if(ARG_NAME IN_LIST _registered)
    message(FATAL_ERROR
      "HYLE package dependency '${ARG_NAME}' is registered more than once")
  endif()

  # A central lock record has exactly one package owner. Otherwise two module
  # registries can independently project the same dependency, drift their
  # destinations/targets, and both appear locally valid. Keep this ownership
  # mechanical across the active product closure.
  foreach(_locked_dependency IN LISTS ARG_LOCKED_DEPENDENCIES)
    string(TOUPPER "${_locked_dependency}" _locked_key)
    get_property(_locked_owner GLOBAL PROPERTY
      "HYLE_PACKAGE_LOCKED_DEP_${_locked_key}_OWNER")
    if(_locked_owner)
      message(FATAL_ERROR
        "Locked dependency '${_locked_dependency}' is already owned by package "
        "registry '${_locked_owner}'; '${ARG_NAME}' cannot register it again")
    endif()
  endforeach()
  foreach(_locked_dependency IN LISTS ARG_LOCKED_DEPENDENCIES)
    string(TOUPPER "${_locked_dependency}" _locked_key)
    set_property(GLOBAL PROPERTY
      "HYLE_PACKAGE_LOCKED_DEP_${_locked_key}_OWNER" "${ARG_NAME}")
  endforeach()
  list(APPEND _registered "${ARG_NAME}")
  set_property(GLOBAL PROPERTY HYLE_PACKAGE_DEPENDENCIES "${_registered}")

  string(TOUPPER "${ARG_NAME}" _key)
  foreach(_field IN ITEMS OWNER FEATURES EXPORT_TARGETS STATIC_EXPORT_TARGETS
      FIND_DEPENDENCIES STATIC_FIND_DEPENDENCIES LOCKED_DEPENDENCIES
      RUNTIME_FILE_SPECS GENERATED_RUNTIME_FILE_SPECS DEVELOPMENT_FILE_SPECS
      STATIC_DEVELOPMENT_FILE_SPECS DEVELOPMENT_DIRECTORY_SPECS
      STATIC_DEVELOPMENT_DIRECTORY_SPECS DEVELOPMENT_PROGRAM_SPECS)
    set_property(GLOBAL PROPERTY "HYLE_PACKAGE_DEP_${_key}_${_field}"
      "${ARG_${_field}}")
  endforeach()
endfunction()

function(_hyle_package_dependency_property name field out_var)
  string(TOUPPER "${name}" _key)
  get_property(_value GLOBAL PROPERTY "HYLE_PACKAGE_DEP_${_key}_${field}")
  set(${out_var} "${_value}" PARENT_SCOPE)
endfunction()

function(hyle_package_collect_registry out_targets out_features out_find_dependencies)
  get_property(_dependencies GLOBAL PROPERTY HYLE_PACKAGE_DEPENDENCIES)
  set(_targets "")
  set(_features "")
  set(_find_dependencies "")

  foreach(_dependency IN LISTS _dependencies)
    _hyle_package_dependency_property("${_dependency}" OWNER _owner)
    if(NOT _owner STREQUAL "foundation" AND NOT TARGET hyle_${_owner})
      message(FATAL_ERROR
        "Package dependency '${_dependency}' was registered for absent component "
        "'${_owner}'")
    endif()

    _hyle_package_dependency_property("${_dependency}" EXPORT_TARGETS _dependency_targets)
    _hyle_package_dependency_property(
      "${_dependency}" FIND_DEPENDENCIES _common_find_dependencies)
    foreach(_find_dependency IN LISTS _common_find_dependencies)
      if(NOT _find_dependency MATCHES
         "^[A-Za-z0-9_+.-]+(\\|[0-9][A-Za-z0-9_.-]*)?$")
        message(FATAL_ERROR
          "Package dependency '${_dependency}' has invalid find_dependency spec "
          "'${_find_dependency}'")
      endif()
    endforeach()
    list(APPEND _find_dependencies ${_common_find_dependencies})
    if(HYLE_STATIC)
      _hyle_package_dependency_property(
        "${_dependency}" STATIC_EXPORT_TARGETS _static_targets)
      list(APPEND _dependency_targets ${_static_targets})
      _hyle_package_dependency_property(
        "${_dependency}" STATIC_FIND_DEPENDENCIES _static_find_dependencies)
      foreach(_find_dependency IN LISTS _static_find_dependencies)
        if(NOT _find_dependency MATCHES
           "^[A-Za-z0-9_+.-]+(\\|[0-9][A-Za-z0-9_.-]*)?$")
          message(FATAL_ERROR
            "Package dependency '${_dependency}' has invalid find_dependency spec "
            "'${_find_dependency}'")
        endif()
      endforeach()
      list(APPEND _find_dependencies ${_static_find_dependencies})
    endif()
    foreach(_target IN LISTS _dependency_targets)
      if(NOT TARGET ${_target})
        message(FATAL_ERROR
          "Package dependency '${_dependency}' requires missing export target "
          "'${_target}'")
      endif()
    endforeach()
    list(APPEND _targets ${_dependency_targets})

    _hyle_package_dependency_property("${_dependency}" FEATURES _dependency_features)
    list(APPEND _features ${_dependency_features})
  endforeach()

  list(REMOVE_DUPLICATES _targets)
  list(REMOVE_DUPLICATES _features)
  list(REMOVE_DUPLICATES _find_dependencies)
  set(${out_targets} "${_targets}" PARENT_SCOPE)
  set(${out_features} "${_features}" PARENT_SCOPE)
  set(${out_find_dependencies} "${_find_dependencies}" PARENT_SCOPE)
endfunction()

function(_hyle_package_validate_install_destination dependency kind destination)
  # Install destinations are portable relative paths. Reject traversal and
  # host-specific roots before install() can normalize them outside the package
  # prefix; also reject non-canonical dot/empty segments that can conceal one.
  if(NOT destination OR IS_ABSOLUTE "${destination}" OR
     destination MATCHES "^[\\/]" OR destination MATCHES "^[A-Za-z]:" OR
     destination MATCHES "(^|[\\/])\\.\\.?($|[\\/])" OR
     destination MATCHES "[\\/][\\/]" OR destination MATCHES "[|;]")
    message(FATAL_ERROR
      "Package dependency '${dependency}' ${kind} destination must be a safe "
      "portable relative path: '${destination}'")
  endif()
endfunction()

function(_hyle_package_validate_install_name dependency kind name)
  if(NOT name OR name STREQUAL "." OR name STREQUAL ".." OR
     name MATCHES "[<>:\"/\\\\|?*;]")
    message(FATAL_ERROR
      "Package dependency '${dependency}' ${kind} RENAME must be one safe "
      "basename: '${name}'")
  endif()
endfunction()

function(_hyle_package_parse_asset dependency kind spec out_source out_destination out_name)
  string(REPLACE "|" ";" _fields "${spec}")
  list(LENGTH _fields _field_count)
  if(_field_count LESS 2 OR _field_count GREATER 3)
    message(FATAL_ERROR
      "Package dependency '${dependency}' has malformed ${kind} asset '${spec}'; "
      "expected source|destination[|name]")
  endif()
  list(GET _fields 0 _source)
  list(GET _fields 1 _destination)
  set(_name "")
  if(_field_count EQUAL 3)
    list(GET _fields 2 _name)
  endif()
  if(NOT IS_ABSOLUTE "${_source}")
    message(FATAL_ERROR
      "Package dependency '${dependency}' ${kind} source is not absolute: '${_source}'")
  endif()
  if(NOT EXISTS "${_source}")
    message(FATAL_ERROR
      "Package dependency '${dependency}' lacks ${kind} source '${_source}'")
  endif()
  _hyle_package_validate_install_destination(
    "${dependency}" "${kind}" "${_destination}")
  if(_name)
    _hyle_package_validate_install_name("${dependency}" "${kind}" "${_name}")
  endif()
  set(${out_source} "${_source}" PARENT_SCOPE)
  set(${out_destination} "${_destination}" PARENT_SCOPE)
  set(${out_name} "${_name}" PARENT_SCOPE)
endfunction()

function(_hyle_package_parse_generated_asset dependency spec out_producer out_source out_destination out_name)
  # Generated binaries are deliberately separate from authenticated license
  # text. They may not exist until the declared native producer has built.
  string(REPLACE "|" ";" _fields "${spec}")
  list(LENGTH _fields _count)
  if(_count LESS 3 OR _count GREATER 4)
    message(FATAL_ERROR
      "Package dependency '${dependency}' malformed generated asset '${spec}'; expected producer|source|destination[|name]")
  endif()
  list(GET _fields 0 _producer)
  list(GET _fields 1 _source)
  list(GET _fields 2 _destination)
  set(_name "")
  if(_count EQUAL 4)
    list(GET _fields 3 _name)
    _hyle_package_validate_install_name("${dependency}" generated "${_name}")
  endif()
  if(NOT TARGET ${_producer})
    message(FATAL_ERROR "Package dependency '${dependency}' generated asset has no producer target '${_producer}'")
  endif()
  get_target_property(_imported ${_producer} IMPORTED)
  get_target_property(_type ${_producer} TYPE)
  if(_imported OR NOT _type MATCHES "^(STATIC|SHARED|MODULE)_LIBRARY$")
    message(FATAL_ERROR "Package dependency '${dependency}' generated asset needs a built producer target")
  endif()
  if(NOT IS_ABSOLUTE "${_source}" OR _source MATCHES "\\$<")
    message(FATAL_ERROR "Package dependency '${dependency}' generated source must be an absolute build-tree path")
  endif()
  set(_build_root "${CMAKE_BINARY_DIR}")
  cmake_path(IS_PREFIX _build_root "${_source}" NORMALIZE _inside_build)
  if(NOT _inside_build OR IS_DIRECTORY "${_source}")
    message(FATAL_ERROR "Package dependency '${dependency}' generated source must be a file inside the build tree: '${_source}'")
  endif()
  _hyle_package_validate_install_destination("${dependency}" generated "${_destination}")
  set(${out_producer} "${_producer}" PARENT_SCOPE)
  set(${out_source} "${_source}" PARENT_SCOPE)
  set(${out_destination} "${_destination}" PARENT_SCOPE)
  set(${out_name} "${_name}" PARENT_SCOPE)
endfunction()

function(_hyle_package_install_file_specs dependency component)
  foreach(_spec IN LISTS ARGN)
    _hyle_package_parse_asset(
      "${dependency}" file "${_spec}" _source _destination _name)
    if(_name)
      install(FILES "${_source}"
        DESTINATION "${_destination}"
        RENAME "${_name}"
        COMPONENT "${component}")
    else()
      install(FILES "${_source}"
        DESTINATION "${_destination}"
        COMPONENT "${component}")
    endif()
  endforeach()
endfunction()

function(_hyle_package_install_directory_specs dependency component)
  foreach(_spec IN LISTS ARGN)
    _hyle_package_parse_asset(
      "${dependency}" directory "${_spec}" _source _destination _name)
    if(_name)
      message(FATAL_ERROR
        "Package dependency '${dependency}' directory asset cannot use RENAME: '${_spec}'")
    endif()
    install(DIRECTORY "${_source}"
      DESTINATION "${_destination}"
      COMPONENT "${component}")
  endforeach()
endfunction()

function(_hyle_package_install_program_specs dependency component)
  foreach(_spec IN LISTS ARGN)
    _hyle_package_parse_asset(
      "${dependency}" program "${_spec}" _source _destination _name)
    if(_name)
      install(PROGRAMS "${_source}"
        DESTINATION "${_destination}"
        RENAME "${_name}"
        COMPONENT "${component}")
    else()
      install(PROGRAMS "${_source}"
        DESTINATION "${_destination}"
        COMPONENT "${component}")
    endif()
  endforeach()
endfunction()

function(hyle_package_install_registered_assets)
  get_property(_dependencies GLOBAL PROPERTY HYLE_PACKAGE_DEPENDENCIES)
  foreach(_dependency IN LISTS _dependencies)
    _hyle_package_dependency_property(
      "${_dependency}" RUNTIME_FILE_SPECS _runtime_files)
    _hyle_package_install_file_specs(
      "${_dependency}" Hyle_Runtime ${_runtime_files})
    _hyle_package_dependency_property(
      "${_dependency}" GENERATED_RUNTIME_FILE_SPECS _generated_files)
    foreach(_spec IN LISTS _generated_files)
      _hyle_package_parse_generated_asset("${_dependency}" "${_spec}"
        _producer _source _destination _name)
      if(_name)
        install(FILES "${_source}" DESTINATION "${_destination}"
          RENAME "${_name}" COMPONENT Hyle_Runtime)
      else()
        install(FILES "${_source}" DESTINATION "${_destination}" COMPONENT Hyle_Runtime)
      endif()
    endforeach()

    _hyle_package_dependency_property(
      "${_dependency}" DEVELOPMENT_FILE_SPECS _development_files)
    _hyle_package_dependency_property(
      "${_dependency}" DEVELOPMENT_DIRECTORY_SPECS _development_directories)
    if(HYLE_STATIC)
      _hyle_package_dependency_property(
        "${_dependency}" STATIC_DEVELOPMENT_FILE_SPECS _static_files)
      _hyle_package_dependency_property(
        "${_dependency}" STATIC_DEVELOPMENT_DIRECTORY_SPECS _static_directories)
      list(APPEND _development_files ${_static_files})
      list(APPEND _development_directories ${_static_directories})
    endif()
    _hyle_package_install_file_specs(
      "${_dependency}" Hyle_Development ${_development_files})
    _hyle_package_install_directory_specs(
      "${_dependency}" Hyle_Development ${_development_directories})

    _hyle_package_dependency_property(
      "${_dependency}" DEVELOPMENT_PROGRAM_SPECS _development_programs)
    _hyle_package_install_program_specs(
      "${_dependency}" Hyle_Development ${_development_programs})
  endforeach()
endfunction()
