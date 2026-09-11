# Project the products eligible for CLOSED-component qualification from HYLE's
# central factor/product graph and unique lifecycle registry.  CI and delivery
# smokes must not maintain a second hand-written product list.

include_guard(GLOBAL)

function(hyle_ci_closed_product_csv out_variable)
  if(NOT DEFINED HYLE_SOURCE_DIR OR HYLE_SOURCE_DIR STREQUAL "")
    message(FATAL_ERROR
      "hyle_ci_closed_product_csv requires HYLE_SOURCE_DIR")
  endif()

  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}"
            "${HYLE_SOURCE_DIR}/scripts/correctness-inventory.py"
            --root "${HYLE_SOURCE_DIR}"
            --revision WORKTREE
            --check
            --print-closed-products
    RESULT_VARIABLE _hyle_closed_product_result
    OUTPUT_VARIABLE _hyle_closed_product_output
    ERROR_VARIABLE _hyle_closed_product_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _hyle_closed_product_result EQUAL 0)
    message(FATAL_ERROR
      "Cannot project CLOSED HYLE products from the central graph/registry "
      "(exit ${_hyle_closed_product_result}):\n${_hyle_closed_product_error}")
  endif()
  if(NOT _hyle_closed_product_output MATCHES
     "^[a-z0-9_]+(,[a-z0-9_]+)*$")
    message(FATAL_ERROR
      "Invalid CLOSED HYLE product projection: '${_hyle_closed_product_output}'")
  endif()

  set(${out_variable} "${_hyle_closed_product_output}" PARENT_SCOPE)
endfunction()
