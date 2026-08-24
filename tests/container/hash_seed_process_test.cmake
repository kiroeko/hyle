if(NOT DEFINED PROBE OR PROBE STREQUAL "")
  message(FATAL_ERROR "PROBE executable is required")
endif()

execute_process(
  COMMAND "${PROBE}"
  RESULT_VARIABLE first_result
  OUTPUT_VARIABLE first
  ERROR_VARIABLE first_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT first_result EQUAL 0 OR first STREQUAL "")
  message(FATAL_ERROR
    "first hash-seed probe failed (${first_result}): ${first_error}")
endif()

execute_process(
  COMMAND "${PROBE}"
  RESULT_VARIABLE second_result
  OUTPUT_VARIABLE second
  ERROR_VARIABLE second_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT second_result EQUAL 0 OR second STREQUAL "")
  message(FATAL_ERROR
    "second hash-seed probe failed (${second_result}): ${second_error}")
endif()

if(first STREQUAL second)
  message(FATAL_ERROR
    "independent processes produced the same default StringHash: ${first}")
endif()
