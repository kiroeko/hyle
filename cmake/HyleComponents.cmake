# HYLE component selection and dependency closure.
#
# HYLE_COMPONENTS names the product roots a consumer wants.  The resolved
# HYLE_BUILD_<COMPONENT> variables are the executable form of the module graph;
# root CMake uses them both for add_subdirectory() and optional dependency
# discovery.  Keep this graph synchronized with docs/architecture/roadmap.md.

# Declare every component and its direct internal link visibility once.  This
# manifest is consumed twice: the union drives pre-target component closure,
# while hyle_add_module() projects PUBLIC/PRIVATE entries into real CMake link
# edges after the dependency targets exist.  External implementation libraries
# remain module-owned and deliberately do not appear here.
function(_hyle_declare_component name)
  cmake_parse_arguments(ARG "" "" "PRIVATE_DEPS;PUBLIC_DEPS" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "HYLE component '${name}' has unrecognized manifest arguments: "
      "${ARG_UNPARSED_ARGUMENTS}")
  endif()
  if(ARG_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR
      "HYLE component '${name}' has manifest keywords missing values: "
      "${ARG_KEYWORDS_MISSING_VALUES}")
  endif()
  if(name IN_LIST HYLE_KNOWN_COMPONENTS)
    message(FATAL_ERROR "HYLE component '${name}' is declared more than once")
  endif()

  set(_private ${ARG_PRIVATE_DEPS})
  set(_public ${ARG_PUBLIC_DEPS})
  list(REMOVE_DUPLICATES _private)
  list(REMOVE_DUPLICATES _public)
  foreach(_dependency IN LISTS _private)
    if(_dependency IN_LIST _public)
      message(FATAL_ERROR
        "HYLE component '${name}' declares '${_dependency}' as both PRIVATE and PUBLIC")
    endif()
  endforeach()

  set(_all ${_private} ${_public})
  list(REMOVE_DUPLICATES _all)
  string(TOUPPER "${name}" _component_upper)
  set(_HYLE_PRIVATE_DEPS_${_component_upper} "${_private}" PARENT_SCOPE)
  set(_HYLE_PUBLIC_DEPS_${_component_upper} "${_public}" PARENT_SCOPE)
  set(_HYLE_DEPS_${_component_upper} "${_all}" PARENT_SCOPE)
  set(_known_components ${HYLE_KNOWN_COMPONENTS})
  list(APPEND _known_components "${name}")
  set(HYLE_KNOWN_COMPONENTS "${_known_components}" PARENT_SCOPE)
endfunction()

set(HYLE_KNOWN_COMPONENTS "")
set(_hyle_optional_sched_dep "")
if(HYLE_FCONTEXT_SUPPORTED)
  set(_hyle_optional_sched_dep sched)
endif()

# Declaration order is construction order and must remain topological.  The
# visibility here is the visibility of the real hyle_<module> target edge.
_hyle_declare_component(core)
_hyle_declare_component(util_core)
_hyle_declare_component(error)
_hyle_declare_component(obs PUBLIC_DEPS error)
_hyle_declare_component(math)
_hyle_declare_component(platform PUBLIC_DEPS error)
_hyle_declare_component(time PUBLIC_DEPS error)
_hyle_declare_component(mem PUBLIC_DEPS error platform util_core)
_hyle_declare_component(simd PUBLIC_DEPS error platform)
# container headers expose error/mem types; platform is used only by compiled
# implementation paths such as topology/cache-aware concurrency support.
_hyle_declare_component(container
  PRIVATE_DEPS platform
  PUBLIC_DEPS error mem)
_hyle_declare_component(util PUBLIC_DEPS container util_core)
# SchedulerOptions publicly names platform types, while time is consumed only
# by scheduler implementation TUs.
_hyle_declare_component(sched
  PRIVATE_DEPS time
  PUBLIC_DEPS error platform mem container util_core)
# io's public protocol/C ABI headers do not require consumers to compile against
# the implementation modules; static archives still propagate private edges as
# LINK_ONLY through normal CMake semantics.
_hyle_declare_component(io
  PRIVATE_DEPS error mem container ${_hyle_optional_sched_dep})
# compute publicly exposes error/mem and, where available, sched Waker types;
# platform and simd stay compiled implementation details.
_hyle_declare_component(compute
  PRIVATE_DEPS platform simd
  PUBLIC_DEPS error mem ${_hyle_optional_sched_dep})
_hyle_declare_component(graph
  PRIVATE_DEPS util_core
  PUBLIC_DEPS error mem container ${_hyle_optional_sched_dep})

unset(_hyle_optional_sched_dep)

# A source-embedding parent commonly sets this as a normal variable before
# add_subdirectory().  Capture that effective value before creating the cache
# entry, then mirror it into the cache so CMake GUI/diagnostic output describes
# the graph that is actually being generated.  Without this step CMake can keep
# a shadowed `all` cache entry while the normal variable selects one component.
if(DEFINED HYLE_COMPONENTS)
  set(_hyle_components_input "${HYLE_COMPONENTS}")
else()
  set(_hyle_components_input "all")
endif()
set(HYLE_COMPONENTS "${_hyle_components_input}" CACHE STRING
    "HYLE product roots to build (all or a semicolon/comma-separated component list)"
    FORCE)
unset(_hyle_components_input)

set(_hyle_requested "${HYLE_COMPONENTS}")
string(REPLACE "," ";" _hyle_requested "${_hyle_requested}")
list(TRANSFORM _hyle_requested STRIP)
list(TRANSFORM _hyle_requested TOLOWER)
list(REMOVE_ITEM _hyle_requested "")

if("all" IN_LIST _hyle_requested)
  list(LENGTH _hyle_requested _hyle_requested_count)
  if(NOT _hyle_requested_count EQUAL 1)
    message(FATAL_ERROR
      "HYLE_COMPONENTS='${HYLE_COMPONENTS}' mixes 'all' with explicit names; "
      "use 'all' alone or list only the requested product roots")
  endif()
  set(_hyle_selected ${HYLE_KNOWN_COMPONENTS})
elseif(NOT _hyle_requested)
  message(FATAL_ERROR
    "HYLE_COMPONENTS is explicitly empty. Omit it (or use 'all') for the full "
    "product, or provide one or more component roots. Refusing to turn an empty "
    "script/CI argument into an accidental all-component build.")
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

# HYLE_KNOWN_COMPONENTS is also the construction order. Validate the graph at
# configure time so a typo, cycle, or dependency added after its consumer cannot
# degrade into a missing linker input much later in the build.
foreach(_component IN LISTS HYLE_KNOWN_COMPONENTS)
  string(TOUPPER "${_component}" _component_upper)
  list(FIND HYLE_KNOWN_COMPONENTS "${_component}" _component_index)
  foreach(_dependency IN LISTS _HYLE_DEPS_${_component_upper})
    if(NOT _dependency IN_LIST HYLE_KNOWN_COMPONENTS)
      message(FATAL_ERROR
        "HYLE component '${_component}' names unknown dependency '${_dependency}'")
    endif()
    list(FIND HYLE_KNOWN_COMPONENTS "${_dependency}" _dependency_index)
    if(_dependency_index GREATER_EQUAL _component_index)
      message(FATAL_ERROR
        "HYLE component order is not topological: '${_component}' depends on "
        "'${_dependency}', which is not listed before it")
    endif()
  endforeach()
endforeach()
unset(_component_index)
unset(_dependency_index)

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
unset(_hyle_requested_count)
unset(_hyle_selected)
unset(_hyle_changed)
unset(_hyle_resolved)
unset(_hyle_known_text)
unset(_component)
unset(_component_upper)
unset(_dependency)
