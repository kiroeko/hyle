foreach(_required IN ITEMS HYLE_SOURCE_DIR HYLE_SOURCE_ARCHIVE
                           HYLE_SOURCE_PREFIX HYLE_GIT_EXECUTABLE)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()

execute_process(
  COMMAND "${HYLE_GIT_EXECUTABLE}" -C "${HYLE_SOURCE_DIR}"
          status --porcelain=v1 --untracked-files=normal
  RESULT_VARIABLE _status_result
  OUTPUT_VARIABLE _status_output
  ERROR_VARIABLE _status_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _status_result EQUAL 0)
  message(FATAL_ERROR "Cannot inspect HYLE Git tree: ${_status_error}")
endif()
if(NOT _status_output STREQUAL "")
  message(FATAL_ERROR
    "Refusing to create a source delivery from a dirty/untracked Git tree. "
    "Commit the exact reviewed source first. Git status:\n${_status_output}")
endif()

execute_process(
  COMMAND "${HYLE_GIT_EXECUTABLE}" -C "${HYLE_SOURCE_DIR}"
          rev-parse --verify HEAD
  RESULT_VARIABLE _commit_result
  OUTPUT_VARIABLE _commit
  ERROR_VARIABLE _commit_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _commit_result EQUAL 0)
  message(FATAL_ERROR "Cannot resolve HYLE source commit: ${_commit_error}")
endif()

get_filename_component(_archive_dir "${HYLE_SOURCE_ARCHIVE}" DIRECTORY)
file(MAKE_DIRECTORY "${_archive_dir}")
execute_process(
  COMMAND "${HYLE_GIT_EXECUTABLE}" -C "${HYLE_SOURCE_DIR}" archive
          --format=zip
          "--prefix=${HYLE_SOURCE_PREFIX}"
          "--output=${HYLE_SOURCE_ARCHIVE}"
          "${_commit}"
  RESULT_VARIABLE _archive_result
  ERROR_VARIABLE _archive_error)
if(NOT _archive_result EQUAL 0)
  message(FATAL_ERROR "Cannot create HYLE source archive: ${_archive_error}")
endif()

file(SHA256 "${HYLE_SOURCE_ARCHIVE}" _archive_sha256)
get_filename_component(_archive_name "${HYLE_SOURCE_ARCHIVE}" NAME)
file(WRITE "${HYLE_SOURCE_ARCHIVE}.sha256"
     "${_archive_sha256}  ${_archive_name}\n")
message(STATUS
  "HYLE source delivery: ${HYLE_SOURCE_ARCHIVE} (commit ${_commit})")
