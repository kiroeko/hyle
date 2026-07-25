# HYLE component selection and dependency closure.
#
# HYLE_COMPONENTS names the product roots a consumer wants.  The resolved
# HYLE_BUILD_<COMPONENT> variables are the executable form of the module graph;
# root CMake uses them both for add_subdirectory() and optional dependency
# discovery.  Keep this graph synchronized with docs/architecture/roadmap.md.

set(HYLE_KNOWN_COMPONENTS
    core error obs math platform time mem simd container util sched io compute graph)

set(HYLE_COMPONENTS "all" CACHE STRING
    "HYLE product roots to build (all or a semicolon/comma-separated component list)")

set(_hyle_requested "${HYLE_COMPONENTS}")
string(REPLACE "," ";" _hyle_requested "${_hyle_requested}")
list(TRANSFORM _hyle_requested STRIP)
list(TRANSFORM _hyle_requested TOLOWER)
list(REMOVE_ITEM _hyle_requested "")

if(NOT _hyle_requested OR "all" IN_LIST _hyle_requested)
  set(_hyle_selected ${HYLE_KNOWN_COMPONENTS})
else()
  set(_hyle_selected "")
  foreach(_component IN LISTS _hyle_requested)
    if(NOT _component IN_LIST HYLE_KNOWN_COMPONENTS)
      list(JOIN HYLE_KNOWN_COMPONENTS ", " _hyle_known_text)
      message(FATAL_ERROR
        "Unknown HYLE component '${_component}'. Known components: ${_hyle_known_text}")
    endif()
    list(APPEND _hyle_selected "${_component}")
  endforeach()
endif()

# Direct target dependencies. Header-only/cross-cutting seams that do not need a
# binary target (core headers, util-core headers, hyle/platform.h compile-time
# environment macros, hyle_trace) are intentionally absent. Complete
# io/compute/graph products include sched; sched in turn closes the runtime
# chain.
set(_HYLE_DEPS_PLATFORM error)
set(_HYLE_DEPS_OBS error)
set(_HYLE_DEPS_TIME error)
set(_HYLE_DEPS_MEM error platform)
set(_HYLE_DEPS_SIMD platform)
set(_HYLE_DEPS_CONTAINER mem platform)
set(_HYLE_DEPS_UTIL container)
set(_HYLE_DEPS_SCHED error platform time mem container)
set(_HYLE_DEPS_IO error mem container sched)
set(_HYLE_DEPS_COMPUTE error platform mem simd sched)
set(_HYLE_DEPS_GRAPH error mem container sched)

# Resolve the transitive closure. The graph is intentionally small enough that a
# fixed-point loop is clearer and safer than encoding directory order as meaning.
set(_hyle_changed TRUE)
while(_hyle_changed)
  set(_hyle_changed FALSE)
  foreach(_component IN LISTS _hyle_selected)
    string(TOUPPER "${_component}" _component_upper)
    foreach(_dependency IN LISTS _HYLE_DEPS_${_component_upper})
      if(NOT _dependency IN_LIST _hyle_selected)
        list(APPEND _hyle_selected "${_dependency}")
        set(_hyle_changed TRUE)
      endif()
    endforeach()
  endforeach()
endwhile()

foreach(_component IN LISTS HYLE_KNOWN_COMPONENTS)
  string(TOUPPER "${_component}" _component_upper)
  if(_component IN_LIST _hyle_selected)
    set(HYLE_BUILD_${_component_upper} TRUE)
  else()
    set(HYLE_BUILD_${_component_upper} FALSE)
  endif()
endforeach()

# Print in architectural/topological order rather than request order so configure
# logs double as a concise assembly manifest.
set(_hyle_resolved "")
foreach(_component IN LISTS HYLE_KNOWN_COMPONENTS)
  string(TOUPPER "${_component}" _component_upper)
  if(HYLE_BUILD_${_component_upper})
    list(APPEND _hyle_resolved "${_component}")
  endif()
endforeach()
list(JOIN _hyle_resolved "," HYLE_RESOLVED_COMPONENTS)
message(STATUS "HYLE components: requested='${HYLE_COMPONENTS}' resolved='${HYLE_RESOLVED_COMPONENTS}'")

unset(_hyle_requested)
unset(_hyle_selected)
unset(_hyle_changed)
unset(_hyle_resolved)
unset(_hyle_known_text)
unset(_component)
unset(_component_upper)
unset(_dependency)
