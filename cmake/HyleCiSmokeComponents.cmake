# Select public products for standalone build and packaging smoke checks.

include_guard(GLOBAL)

function(hyle_ci_smoke_component_csv out_variable)
  if(DEFINED HYLE_SMOKE_COMPONENTS AND NOT HYLE_SMOKE_COMPONENTS STREQUAL "")
    string(REPLACE ";" "," _hyle_smoke_components "${HYLE_SMOKE_COMPONENTS}")
  else()
    set(_hyle_smoke_components all)
  endif()
  set(${out_variable} "${_hyle_smoke_components}" PARENT_SCOPE)
endfunction()
