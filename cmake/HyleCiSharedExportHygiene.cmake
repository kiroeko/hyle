include_guard(GLOBAL)

# Inspect the actual installed artifact rather than source spelling. This is a
# narrow Native-FFI=N/A guard, not a stable C++ ABI whitelist.
function(hyle_ci_check_shared_export_hygiene prefix)
  cmake_parse_arguments(ARG "ALLOW_EMPTY" "LABEL" "" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "hyle_ci_check_shared_export_hygiene: unexpected arguments: "
      "${ARG_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT IS_DIRECTORY "${prefix}")
    message(FATAL_ERROR "shared export scan prefix does not exist: ${prefix}")
  endif()
  if(NOT DEFINED HYLE_SOURCE_DIR OR HYLE_SOURCE_DIR STREQUAL "")
    message(FATAL_ERROR "shared export scan requires HYLE_SOURCE_DIR")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(_hyle_export_args
    "${HYLE_SOURCE_DIR}/scripts/check-shared-export-hygiene.py"
    --root "${prefix}")
  if(ARG_ALLOW_EMPTY)
    list(APPEND _hyle_export_args --allow-empty)
  endif()
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" ${_hyle_export_args}
    RESULT_VARIABLE _hyle_export_result
    OUTPUT_VARIABLE _hyle_export_stdout
    ERROR_VARIABLE _hyle_export_stderr
    COMMAND_ECHO STDOUT)
  if(NOT _hyle_export_result EQUAL 0)
    if(ARG_LABEL)
      set(_hyle_export_label "${ARG_LABEL}")
    else()
      set(_hyle_export_label "shared SDK")
    endif()
    message(FATAL_ERROR
      "${_hyle_export_label} export hygiene failed (exit "
      "${_hyle_export_result}):\n${_hyle_export_stdout}${_hyle_export_stderr}")
  endif()
  string(STRIP "${_hyle_export_stdout}" _hyle_export_stdout)
  message(STATUS "${_hyle_export_stdout}")
endfunction()
