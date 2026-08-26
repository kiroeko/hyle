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
set(HYLE_KNOWN_BUILD_DEPENDENCIES simd_vec_headers)

function(_hyle_declare_component name)
  cmake_parse_arguments(ARG "TRACE" ""
    "PRIVATE_DEPS;PUBLIC_DEPS;PUBLIC_TYPE_DEPS;PRIVATE_BUILD_DEPS" ${ARGN})
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
  foreach(_dependency IN LISTS ARG_PUBLIC_TYPE_DEPS)
    if(NOT _dependency IN_LIST _public)
      message(FATAL_ERROR
        "HYLE component '${name}' declares PUBLIC_TYPE_DEPS '${_dependency}' "
        "without the corresponding PUBLIC_DEPS edge")
    endif()
  endforeach()
  foreach(_dependency IN LISTS ARG_PRIVATE_BUILD_DEPS)
    if(NOT _dependency IN_LIST HYLE_KNOWN_BUILD_DEPENDENCIES)
      message(FATAL_ERROR
        "HYLE component '${name}' names unknown PRIVATE_BUILD_DEPS "
        "'${_dependency}'")
    endif()
  endforeach()

  set(_all ${_private} ${_public})
  list(REMOVE_DUPLICATES _all)
  string(TOUPPER "${name}" _component_upper)
  set(_HYLE_PRIVATE_DEPS_${_component_upper} "${_private}" PARENT_SCOPE)
  set(_HYLE_PUBLIC_DEPS_${_component_upper} "${_public}" PARENT_SCOPE)
  set(_HYLE_PUBLIC_TYPE_DEPS_${_component_upper}
      "${ARG_PUBLIC_TYPE_DEPS}" PARENT_SCOPE)
  set(_HYLE_PRIVATE_BUILD_DEPS_${_component_upper}
      "${ARG_PRIVATE_BUILD_DEPS}" PARENT_SCOPE)
  set(_HYLE_DEPS_${_component_upper} "${_all}" PARENT_SCOPE)
  set(_HYLE_TRACE_${_component_upper} "${ARG_TRACE}" PARENT_SCOPE)
  set(_known_components ${HYLE_KNOWN_COMPONENTS})
  list(APPEND _known_components "${name}")
  set(HYLE_KNOWN_COMPONENTS "${_known_components}" PARENT_SCOPE)
endfunction()

set(HYLE_KNOWN_COMPONENTS "")

# Declaration order is construction order and must remain topological.  The
# visibility here is the visibility of the real hyle_<module> target edge.
_hyle_declare_component(core)
_hyle_declare_component(container_intrusive PUBLIC_DEPS core)
_hyle_declare_component(util_core PUBLIC_DEPS core)
_hyle_declare_component(error PUBLIC_DEPS core)
_hyle_declare_component(obs TRACE PUBLIC_DEPS core error)
_hyle_declare_component(math)
_hyle_declare_component(time PUBLIC_DEPS core container_intrusive)
# fs.h exposes time::UnixTimestamp, so time is a real public SDK edge rather
# than a header made accidentally available by installing the whole tree.
_hyle_declare_component(platform PUBLIC_DEPS core error time)
_hyle_declare_component(mem TRACE PUBLIC_DEPS core error platform util_core)
_hyle_declare_component(simd PUBLIC_DEPS core platform)
# Container headers expose mem types; platform is used only by compiled
# implementation paths such as runtime ISA/topology support.
_hyle_declare_component(container
  PRIVATE_DEPS platform
  PRIVATE_BUILD_DEPS simd_vec_headers
  PUBLIC_DEPS core mem)
_hyle_declare_component(util PUBLIC_DEPS core error mem container util_core)
# SchedulerOptions publicly names platform types, while time is consumed only
# by scheduler implementation TUs.
_hyle_declare_component(sched
  TRACE
  PRIVATE_DEPS time
  PUBLIC_DEPS core error platform mem container util_core)
# io's C++ public headers expose allocator/status/container value types and
# header-inline sched calls (for example complete()/await_suspend()). These are
# consumer usage requirements, not only implementation closure: installed
# consumers that link hyle::io alone must see
# the corresponding import libraries and runtime DLLs.
_hyle_declare_component(io
  TRACE
  PUBLIC_DEPS core error mem container time sched)
# compute publicly exposes error/mem and sched Waker types;
# platform and simd stay compiled implementation details.
_hyle_declare_component(compute
  TRACE
  PRIVATE_DEPS platform simd
  PUBLIC_DEPS core error mem sched)
_hyle_declare_component(graph
  PRIVATE_DEPS util_core
  PUBLIC_DEPS core error mem container sched
  PUBLIC_TYPE_DEPS sched)

function(_hyle_component_public_headers name out_var)
  set(_hyle_util_core_headers
    "${HYLE_SOURCE_DIR}/include/hyle/util/random.h"
    "${HYLE_SOURCE_DIR}/include/hyle/util/ref_counted.h"
    "${HYLE_SOURCE_DIR}/include/hyle/util/scope_guard.h"
    "${HYLE_SOURCE_DIR}/include/hyle/util/spin_lock.h"
    "${HYLE_SOURCE_DIR}/include/hyle/util/type_name.h")
  if(name STREQUAL "core")
    file(GLOB_RECURSE _headers CONFIGURE_DEPENDS
      "${HYLE_SOURCE_DIR}/include/hyle/core/*.h")
    list(APPEND _headers
      "${HYLE_SOURCE_DIR}/include/hyle/platform.h"
      "${HYLE_SOURCE_DIR}/include/hyle/version.h"
      "${HYLE_SOURCE_DIR}/include/hyle/visibility.h")
  elseif(name STREQUAL "container_intrusive")
    set(_headers
      "${HYLE_SOURCE_DIR}/include/hyle/container/intrusive_list.h")
  elseif(name STREQUAL "container")
    file(GLOB_RECURSE _headers CONFIGURE_DEPENDS
      "${HYLE_SOURCE_DIR}/include/hyle/container/*.h")
    list(REMOVE_ITEM _headers
      "${HYLE_SOURCE_DIR}/include/hyle/container/intrusive_list.h")
  elseif(name STREQUAL "util_core")
    set(_headers ${_hyle_util_core_headers})
  elseif(name STREQUAL "util")
    file(GLOB_RECURSE _headers CONFIGURE_DEPENDS
      "${HYLE_SOURCE_DIR}/include/hyle/util/*.h")
    list(REMOVE_ITEM _headers ${_hyle_util_core_headers})
  else()
    file(GLOB_RECURSE _headers CONFIGURE_DEPENDS
      "${HYLE_SOURCE_DIR}/include/hyle/${name}/*.h")
  endif()
  list(SORT _headers)
  set(${out_var} "${_headers}" PARENT_SCOPE)
endfunction()

# Resolve a public include to the component that owns that header. Root-level
# platform/compiler/visibility/version contracts belong to Core. The trace macro
# header is owned by the always-exported hyle_trace foundation target and is
# already validated through hyle_add_module(... TRACE), so it has no product
# component edge of its own.
function(_hyle_public_include_owner include_path out_var)
  set(_owner "")
  if(include_path STREQUAL "hyle/platform.h" OR
     include_path STREQUAL "hyle/version.h" OR
     include_path STREQUAL "hyle/visibility.h" OR
     include_path MATCHES "^hyle/core/")
    set(_owner core)
  elseif(include_path STREQUAL "hyle/container/intrusive_list.h")
    set(_owner container_intrusive)
  elseif(include_path MATCHES
         "^hyle/util/(random|ref_counted|scope_guard|spin_lock|type_name)\\.h$")
    set(_owner util_core)
  elseif(include_path STREQUAL "hyle/obs/trace.h")
    set(_owner "")
  elseif(include_path MATCHES "^hyle/([^/]+)/")
    set(_candidate "${CMAKE_MATCH_1}")
    if(_candidate IN_LIST HYLE_KNOWN_COMPONENTS)
      set(_owner "${_candidate}")
    endif()
  endif()
  set(${out_var} "${_owner}" PARENT_SCOPE)
endfunction()

# Internal header seams are real build-target dependencies but are not product
# components and do not expand an installed component closure. This is reserved
# for implementation reuse that leaves no public header or binary requirement.
function(_hyle_source_include_build_dependency include_path out_var)
  set(_dependency "")
  if(include_path STREQUAL "hyle/simd/vec.h" OR
     include_path MATCHES "^hyle/simd/backend/")
    set(_dependency simd_vec_headers)
  endif()
  set(${out_var} "${_dependency}" PARENT_SCOPE)
endfunction()

# Every direct cross-component include in a public header must have a direct
# PUBLIC target edge. This generalizes the former Core-only check and prevents a
# partial installed SDK from working merely because unrelated headers happened
# to be copied beside it.
foreach(_hyle_component IN LISTS HYLE_KNOWN_COMPONENTS)
  _hyle_component_public_headers(
    "${_hyle_component}" _hyle_public_headers)
  string(TOUPPER "${_hyle_component}" _hyle_component_upper)
  set(_hyle_public_deps
    "${_HYLE_PUBLIC_DEPS_${_hyle_component_upper}}")
  set(_hyle_observed_public_deps "")
  foreach(_hyle_header IN LISTS _hyle_public_headers)
    file(READ "${_hyle_header}" _hyle_header_text)
    string(REGEX MATCHALL
      "#[ \t]*include[ \t]*[<\"]hyle/[^>\"\r\n]+[>\"]"
      _hyle_header_includes "${_hyle_header_text}")
    foreach(_hyle_header_include IN LISTS _hyle_header_includes)
      string(REGEX REPLACE
        "^[^<\"]*[<\"]([^>\"]+)[>\"].*$" "\\1"
        _hyle_include_path "${_hyle_header_include}")
      _hyle_public_include_owner(
        "${_hyle_include_path}" _hyle_include_owner)
      if(_hyle_include_owner AND
         NOT _hyle_include_owner STREQUAL _hyle_component AND
         NOT _hyle_include_owner IN_LIST _hyle_public_deps)
        file(RELATIVE_PATH _hyle_header_relative
          "${HYLE_SOURCE_DIR}" "${_hyle_header}")
        message(FATAL_ERROR
          "component '${_hyle_component}' public header "
          "'${_hyle_header_relative}' includes '${_hyle_include_path}' owned "
          "by '${_hyle_include_owner}', but the component does not declare "
          "PUBLIC_DEPS ${_hyle_include_owner}")
      elseif(_hyle_include_owner AND
             NOT _hyle_include_owner STREQUAL _hyle_component)
        list(APPEND _hyle_observed_public_deps "${_hyle_include_owner}")
      endif()
    endforeach()
  endforeach()
  list(REMOVE_DUPLICATES _hyle_observed_public_deps)
  set(_hyle_public_type_deps
    "${_HYLE_PUBLIC_TYPE_DEPS_${_hyle_component_upper}}")
  foreach(_hyle_public_dep IN LISTS _hyle_public_deps)
    if(NOT _hyle_public_dep IN_LIST _hyle_observed_public_deps AND
       NOT _hyle_public_dep IN_LIST _hyle_public_type_deps)
      message(FATAL_ERROR
        "component '${_hyle_component}' declares PUBLIC_DEPS "
        "'${_hyle_public_dep}', but no owned public header directly includes "
        "that component and it is not an explicit PUBLIC_TYPE_DEPS edge")
    endif()
  endforeach()
endforeach()
unset(_hyle_component)
unset(_hyle_component_upper)
unset(_hyle_header)
unset(_hyle_header_include)
unset(_hyle_header_includes)
unset(_hyle_header_relative)
unset(_hyle_header_text)
unset(_hyle_include_owner)
unset(_hyle_include_path)
unset(_hyle_observed_public_deps)
unset(_hyle_public_dep)
unset(_hyle_public_deps)
unset(_hyle_public_headers)
unset(_hyle_public_type_deps)

# Source-only coupling must be just as explicit as an exported header edge.
# Every implementation include of another HYLE component needs a direct
# PUBLIC/PRIVATE manifest dependency; the shared source include root must not
# make an undeclared dependency look independent.
foreach(_hyle_component IN LISTS HYLE_KNOWN_COMPONENTS)
  string(TOUPPER "${_hyle_component}" _hyle_component_upper)
  set(_hyle_direct_deps "${_HYLE_DEPS_${_hyle_component_upper}}")
  set(_hyle_private_build_deps
    "${_HYLE_PRIVATE_BUILD_DEPS_${_hyle_component_upper}}")
  set(_hyle_observed_build_deps "")
  file(GLOB_RECURSE _hyle_source_files CONFIGURE_DEPENDS
    "${HYLE_SOURCE_DIR}/src/${_hyle_component}/*.cc"
    "${HYLE_SOURCE_DIR}/src/${_hyle_component}/*.cpp"
    "${HYLE_SOURCE_DIR}/src/${_hyle_component}/*.cxx"
    "${HYLE_SOURCE_DIR}/src/${_hyle_component}/*.cu"
    "${HYLE_SOURCE_DIR}/src/${_hyle_component}/*.mm"
    "${HYLE_SOURCE_DIR}/src/${_hyle_component}/*.h"
    "${HYLE_SOURCE_DIR}/src/${_hyle_component}/*.hpp"
    "${HYLE_SOURCE_DIR}/src/${_hyle_component}/*.inc"
    "${HYLE_SOURCE_DIR}/src/${_hyle_component}/*.inl")
  foreach(_hyle_source_file IN LISTS _hyle_source_files)
    file(READ "${_hyle_source_file}" _hyle_source_text)
    string(REGEX MATCHALL
      "#[ \t]*include[ \t]*[<\"]hyle/[^>\"\r\n]+[>\"]"
      _hyle_source_includes "${_hyle_source_text}")
    foreach(_hyle_source_include IN LISTS _hyle_source_includes)
      string(REGEX REPLACE
        "^[^<\"]*[<\"]([^>\"]+)[>\"].*$" "\\1"
        _hyle_include_path "${_hyle_source_include}")
      _hyle_public_include_owner(
        "${_hyle_include_path}" _hyle_include_owner)
      if(_hyle_include_owner AND
         NOT _hyle_include_owner STREQUAL _hyle_component AND
         NOT _hyle_include_owner IN_LIST _hyle_direct_deps)
        _hyle_source_include_build_dependency(
          "${_hyle_include_path}" _hyle_build_dependency)
        if(_hyle_build_dependency AND
           _hyle_build_dependency IN_LIST _hyle_private_build_deps)
          list(APPEND _hyle_observed_build_deps "${_hyle_build_dependency}")
        else()
          file(RELATIVE_PATH _hyle_source_relative
            "${HYLE_SOURCE_DIR}" "${_hyle_source_file}")
          message(FATAL_ERROR
            "component '${_hyle_component}' implementation "
            "'${_hyle_source_relative}' includes '${_hyle_include_path}' owned "
            "by '${_hyle_include_owner}', but the component has no direct "
            "product dependency or matching PRIVATE_BUILD_DEPS seam")
        endif()
      endif()
    endforeach()
  endforeach()
  list(REMOVE_DUPLICATES _hyle_observed_build_deps)
  foreach(_hyle_build_dependency IN LISTS _hyle_private_build_deps)
    if(NOT _hyle_build_dependency IN_LIST _hyle_observed_build_deps)
      message(FATAL_ERROR
        "component '${_hyle_component}' declares PRIVATE_BUILD_DEPS "
        "'${_hyle_build_dependency}' but no implementation include uses it")
    endif()
  endforeach()
endforeach()
unset(_hyle_build_dependency)
unset(_hyle_direct_deps)
unset(_hyle_observed_build_deps)
unset(_hyle_private_build_deps)
unset(_hyle_source_file)
unset(_hyle_source_files)
unset(_hyle_source_include)
unset(_hyle_source_includes)
unset(_hyle_source_relative)
unset(_hyle_source_text)

# A source-embedding parent commonly sets this as a normal variable before
# add_subdirectory().  Capture that effective value before creating the cache
# entry, then mirror it into the cache so CMake GUI/diagnostic output describes
# the graph that is actually being generated.  Without this step CMake can keep
# a shadowed `all` cache entry while the normal variable selects one component.
if(DEFINED HYLE_COMPONENTS)
  set(_hyle_components_input "${HYLE_COMPONENTS}")
elseif(PROJECT_IS_TOP_LEVEL)
  set(_hyle_components_input "all")
else()
  message(FATAL_ERROR
    "An embedding parent must set HYLE_COMPONENTS before add_subdirectory(). "
    "HYLE will not turn an omitted subproject option into an accidental "
    "all-component build.")
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

# Owner-scope the trace option: a fresh Core-only/cache-light configure must not
# advertise a switch which cannot affect any selected target.
set(HYLE_TRACE_AVAILABLE FALSE)
foreach(_component IN LISTS HYLE_KNOWN_COMPONENTS)
  string(TOUPPER "${_component}" _component_upper)
  if(HYLE_BUILD_${_component_upper} AND _HYLE_TRACE_${_component_upper})
    set(HYLE_TRACE_AVAILABLE TRUE)
    break()
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
