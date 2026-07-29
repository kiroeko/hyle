include_guard(GLOBAL)

# External/package assets are declared once by the module that owns them.  The
# package layer projects this registry into exported targets, feature metadata,
# downstream find_dependency() calls, licenses, headers, and tools.  A spec is
# `absolute-source|relative-install-destination[|installed-name]`.
function(hyle_register_package_dependency)
  cmake_parse_arguments(ARG "" "NAME;OWNER"
    "FEATURES;EXPORT_TARGETS;STATIC_EXPORT_TARGETS;STATIC_FIND_DEPENDENCIES;RUNTIME_FILE_SPECS;DEVELOPMENT_FILE_SPECS;STATIC_DEVELOPMENT_FILE_SPECS;DEVELOPMENT_DIRECTORY_SPECS;STATIC_DEVELOPMENT_DIRECTORY_SPECS;DEVELOPMENT_PROGRAM_SPECS"
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

  get_property(_registered GLOBAL PROPERTY HYLE_PACKAGE_DEPENDENCIES)
  if(ARG_NAME IN_LIST _registered)
    message(FATAL_ERROR
      "HYLE package dependency '${ARG_NAME}' is registered more than once")
  endif()
  list(APPEND _registered "${ARG_NAME}")
  set_property(GLOBAL PROPERTY HYLE_PACKAGE_DEPENDENCIES "${_registered}")

  string(TOUPPER "${ARG_NAME}" _key)
  foreach(_field IN ITEMS OWNER FEATURES EXPORT_TARGETS STATIC_EXPORT_TARGETS
      STATIC_FIND_DEPENDENCIES RUNTIME_FILE_SPECS DEVELOPMENT_FILE_SPECS
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
  if(NOT _destination OR IS_ABSOLUTE "${_destination}")
    message(FATAL_ERROR
      "Package dependency '${dependency}' ${kind} destination must be relative: "
      "'${_destination}'")
  endif()
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
