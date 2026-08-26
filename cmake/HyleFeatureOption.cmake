include_guard(GLOBAL)

# Declare an optional capability whose availability may be detected, required, or
# forbidden.  A BOOL option cannot distinguish "please detect" from "this build
# contract requires the feature", which made HYLE's SDK shape depend silently on
# whichever SDK happened to be installed on the configure machine.
function(hyle_auto_feature_option name description)
  cmake_parse_arguments(ARG "" "DEFAULT" "" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS OR ARG_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR
      "hyle_auto_feature_option(${name}): use optional DEFAULT AUTO|ON|OFF")
  endif()
  if(DEFINED ARG_DEFAULT)
    set(_hyle_feature_default "${ARG_DEFAULT}")
  else()
    set(_hyle_feature_default AUTO)
  endif()
  string(TOUPPER "${_hyle_feature_default}" _hyle_feature_default)
  if(NOT _hyle_feature_default MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR
      "hyle_auto_feature_option(${name}): DEFAULT must be AUTO, ON, or OFF")
  endif()
  set(${name} "${_hyle_feature_default}" CACHE STRING
      "${description} (AUTO|ON|OFF)")
  string(TOUPPER "${${name}}" _hyle_feature_value)

  # Accept CMake's conventional boolean spellings when migrating an existing
  # build tree, then store one canonical value for logs and package manifests.
  if(_hyle_feature_value MATCHES "^(1|YES|TRUE|Y)$")
    set(_hyle_feature_value ON)
  elseif(_hyle_feature_value MATCHES "^(0|NO|FALSE|N|IGNORE|NOTFOUND|.*-NOTFOUND)$")
    set(_hyle_feature_value OFF)
  endif()
  if(NOT _hyle_feature_value MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR
      "${name}='${${name}}' is invalid. Use AUTO, ON, or OFF. "
      "AUTO detects the dependency, ON requires it, and OFF does not probe it.")
  endif()

  set(${name} "${_hyle_feature_value}" CACHE STRING
      "${description} (AUTO|ON|OFF)" FORCE)
  set_property(CACHE ${name} PROPERTY STRINGS AUTO ON OFF)
  # CMP0126=NEW preserves an embedding parent's normal variable when a cache
  # entry is updated.  Publish the canonical spelling back to the caller as
  # well, otherwise the GUI can say ON while later dependency logic still sees
  # TRUE (and misses its STREQUAL "ON" hard-fail branch).
  set(${name} "${_hyle_feature_value}" PARENT_SCOPE)
endfunction()
