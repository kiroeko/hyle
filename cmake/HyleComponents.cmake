# HYLE factor selection, dependency closure, and product facade catalog.
#
# HYLE_COMPONENTS names cohesive products a consumer wants. Product facades
# select one or more internal factor roots; HYLE_BUILD_<FACTOR> variables are
# the executable form of the internal DAG used by source ownership, module
# tests, and optional dependency discovery. Keep both catalogs synchronized
# with docs/architecture/build-system.md.

# Declare every component and its direct internal link visibility once.  This
# manifest is consumed twice: the union drives pre-target component closure,
# while hyle_add_module() projects PUBLIC/PRIVATE entries into real CMake link
# edges after the dependency targets exist.  External implementation libraries
# remain module-owned and deliberately do not appear here.
set(HYLE_KNOWN_BUILD_DEPENDENCIES "")
set(_HYLE_SPLIT_PUBLIC_HEADERS "")
set(_HYLE_SPLIT_PUBLIC_HEADER_OWNERS "")

function(_hyle_declare_component name)
  cmake_parse_arguments(ARG "TRACE;QUEUED" ""
    "PRIVATE_DEPS;PUBLIC_DEPS;PUBLIC_TYPE_DEPS;PRIVATE_BUILD_DEPS;SPLIT_PUBLIC_HEADERS"
    ${ARGN})
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
  if(ARG_QUEUED)
    if(ARG_TRACE OR ARG_PRIVATE_DEPS OR ARG_PUBLIC_DEPS OR
       ARG_PUBLIC_TYPE_DEPS OR ARG_PRIVATE_BUILD_DEPS OR
       ARG_SPLIT_PUBLIC_HEADERS)
      message(FATAL_ERROR
        "HYLE QUEUED component '${name}' is an identity reservation only and "
        "cannot declare target metadata before its own closure round")
    endif()
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

  set(_split_public_headers ${ARG_SPLIT_PUBLIC_HEADERS})
  set(_all_split_public_headers ${_HYLE_SPLIT_PUBLIC_HEADERS})
  set(_all_split_public_header_owners ${_HYLE_SPLIT_PUBLIC_HEADER_OWNERS})
  foreach(_header IN LISTS _split_public_headers)
    if(IS_ABSOLUTE "${_header}" OR
       NOT _header MATCHES "^hyle/[^/]+/.+")
      message(FATAL_ERROR
        "HYLE component '${name}' has invalid split public header '${_header}'; "
        "use an include-relative hyle/<namespace>/<header> path")
    endif()
    list(FIND _all_split_public_headers "${_header}" _header_index)
    if(NOT _header_index EQUAL -1)
      list(GET _all_split_public_header_owners ${_header_index}
        _existing_owner)
      message(FATAL_ERROR
        "HYLE split public header '${_header}' is owned by both "
        "'${_existing_owner}' and '${name}'")
    endif()
    list(APPEND _all_split_public_headers "${_header}")
    list(APPEND _all_split_public_header_owners "${name}")
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
  set(_HYLE_SPLIT_PUBLIC_HEADERS_${_component_upper}
      "${_split_public_headers}" PARENT_SCOPE)
  set(_HYLE_DEPS_${_component_upper} "${_all}" PARENT_SCOPE)
  set(_HYLE_TRACE_${_component_upper} "${ARG_TRACE}" PARENT_SCOPE)
  set(_HYLE_QUEUED_${_component_upper} "${ARG_QUEUED}" PARENT_SCOPE)
  set(_HYLE_SPLIT_PUBLIC_HEADERS
      "${_all_split_public_headers}" PARENT_SCOPE)
  set(_HYLE_SPLIT_PUBLIC_HEADER_OWNERS
      "${_all_split_public_header_owners}" PARENT_SCOPE)
  set(_known_components ${HYLE_KNOWN_COMPONENTS})
  list(APPEND _known_components "${name}")
  set(HYLE_KNOWN_COMPONENTS "${_known_components}" PARENT_SCOPE)
  if(NOT ARG_QUEUED)
    set(_implemented_components ${HYLE_IMPLEMENTED_COMPONENTS})
    list(APPEND _implemented_components "${name}")
    set(HYLE_IMPLEMENTED_COMPONENTS
        "${_implemented_components}" PARENT_SCOPE)
  endif()
endfunction()

set(HYLE_KNOWN_COMPONENTS "")
set(HYLE_IMPLEMENTED_COMPONENTS "")

# Declaration order is the append-only component identity order mirrored by
# module-status.json. Target construction uses HYLE_RESOLVED_COMPONENT_LIST,
# whose dependency closure is topologically sorted, so an older factor may
# acquire a newly admitted lower-level dependency without rewriting registry
# history.
# The visibility here is the visibility of the real hyle_<module> target edge.
_hyle_declare_component(core)
_hyle_declare_component(container_intrusive
  PUBLIC_DEPS core
  SPLIT_PUBLIC_HEADERS
    hyle/container/detail/intrusive_member.h
    hyle/container/intrusive_forward_list.h
    hyle/container/intrusive_hlist.h
    hyle/container/intrusive_list.h)
_hyle_declare_component(util_core
  PUBLIC_DEPS core
  SPLIT_PUBLIC_HEADERS
    hyle/util/borrow.h
    hyle/util/function_ref.h
    hyle/util/random.h
    hyle/util/ref_counted.h
    hyle/util/scope_guard.h
    hyle/util/spin_lock.h
    hyle/util/type_name.h
    hyle/util/unique_resource.h)
_hyle_declare_component(error PUBLIC_DEPS core)
_hyle_declare_component(obs TRACE
  PRIVATE_DEPS mem_reclaim
  PUBLIC_DEPS core error time)
_hyle_declare_component(math PUBLIC_DEPS core)
_hyle_declare_component(time PUBLIC_DEPS core)
_hyle_declare_component(timer_queue
  PUBLIC_DEPS core time container_intrusive
  SPLIT_PUBLIC_HEADERS
    hyle/time/common_timeout_queue.h
    hyle/time/timer_entry.h
    hyle/time/timer_queue.h)
# Host facts and raw OS memory/loader mechanisms are the dependency-light
# Platform prime. Thread/wait and synchronous filesystem remain separate
# factors while the consumer-facing Platform product aggregates all three.
_hyle_declare_component(platform PUBLIC_DEPS core error)
_hyle_declare_component(platform_thread
  PRIVATE_DEPS platform
  PUBLIC_DEPS core error time
  SPLIT_PUBLIC_HEADERS
    hyle/platform/thread.h
    hyle/platform/thread_convenience.h
    hyle/platform/thread_visibility.h
    hyle/platform/wait.h)
_hyle_declare_component(platform_fs
  PRIVATE_DEPS platform
  PUBLIC_DEPS core error time
  SPLIT_PUBLIC_HEADERS
    hyle/platform/fs.h
    hyle/platform/fs_convenience.h
    hyle/platform/fs_visibility.h)
# Reader-safe retirement is an internal factor aggregated by the consumer-facing
# Mem product. Container names it directly because its public concurrent-node
# map exposes the explicit reclamation-domain constructor.
_hyle_declare_component(mem_reclaim
  PUBLIC_DEPS core
  SPLIT_PUBLIC_HEADERS hyle/mem/reclamation.h)
_hyle_declare_component(mem TRACE PUBLIC_DEPS core error platform)
_hyle_declare_component(mem_vm
  PUBLIC_DEPS core error mem platform platform_fs util_core
  SPLIT_PUBLIC_HEADERS
    hyle/mem/fiber_stack.h
    hyle/mem/large_mapping.h
    hyle/mem/mapping.h)
_hyle_declare_component(simd PUBLIC_DEPS core platform)
# Container headers expose mem types; platform is used only by compiled
# implementation paths such as runtime ISA/topology support.
_hyle_declare_component(container
  PRIVATE_DEPS platform
  PUBLIC_DEPS core mem mem_reclaim)
_hyle_declare_component(util PUBLIC_DEPS core error mem container)
# SchedulerOptions and the public timer/synchronization APIs name platform and
# typed time values, so both are real transitive SDK requirements.
_hyle_declare_component(sched
  TRACE
  PRIVATE_DEPS container timer_queue
  PUBLIC_DEPS core container_intrusive error time platform platform_thread
              mem mem_vm util_core)
# io's C++ public headers expose allocator/status/container value types and
# header-inline sched calls (for example complete()/await_suspend()). These are
# consumer usage requirements, not only implementation closure: installed
# consumers that link hyle::io alone must see
# the corresponding import libraries and runtime DLLs.
_hyle_declare_component(io
  TRACE
  PUBLIC_DEPS core mem container time timer_queue sched)
# These names reserve independently closable IO factors before product work
# starts. QUEUED reservations deliberately have no source directory, target,
# dependencies, public headers or product-root membership. Their own closure
# rounds must remove QUEUED only after those contracts exist.
_hyle_declare_component(io_tls QUEUED)
_hyle_declare_component(io_http QUEUED)
_hyle_declare_component(io_http2 QUEUED)
_hyle_declare_component(io_transfer QUEUED)
# compute publicly exposes error/mem and sched Waker types;
# platform and simd stay compiled implementation details.
_hyle_declare_component(compute
  TRACE
  PRIVATE_DEPS platform platform_thread platform_fs simd
  PUBLIC_DEPS core error mem sched)
_hyle_declare_component(graph
  PRIVATE_DEPS util_core
  PUBLIC_DEPS core error mem container sched
  PUBLIC_TYPE_DEPS sched)

# Product facades are deliberately outside the factor DAG. They have no source,
# module status, or decision authority and must never be used by an internal
# factor edge. A facade only selects the exact factor roots that make one
# consumer-facing problem domain complete.
function(_hyle_declare_product name)
  cmake_parse_arguments(ARG "" "" "FACTORS" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "HYLE product '${name}' has unrecognized manifest arguments: "
      "${ARG_UNPARSED_ARGUMENTS}")
  endif()
  if(ARG_KEYWORDS_MISSING_VALUES OR NOT ARG_FACTORS)
    message(FATAL_ERROR "HYLE product '${name}' must declare FACTORS")
  endif()
  if(name IN_LIST HYLE_KNOWN_PRODUCTS)
    message(FATAL_ERROR "HYLE product '${name}' is declared more than once")
  endif()

  set(_factors ${ARG_FACTORS})
  set(_unique_factors ${_factors})
  list(REMOVE_DUPLICATES _unique_factors)
  if(NOT "${_factors}" STREQUAL "${_unique_factors}")
    message(FATAL_ERROR "HYLE product '${name}' repeats a factor root")
  endif()
  foreach(_factor IN LISTS _factors)
    if(NOT _factor IN_LIST HYLE_KNOWN_COMPONENTS)
      message(FATAL_ERROR
        "HYLE product '${name}' names unknown factor '${_factor}'")
    endif()
    string(TOUPPER "${_factor}" _factor_upper)
    if(_HYLE_QUEUED_${_factor_upper})
      message(FATAL_ERROR
        "HYLE product '${name}' cannot name QUEUED factor '${_factor}'; "
        "implement and audit the factor before adding it to a product facade")
    endif()
  endforeach()

  string(TOUPPER "${name}" _product_upper)
  set(_HYLE_PRODUCT_FACTORS_${_product_upper} "${_factors}" PARENT_SCOPE)
  set(_known_products ${HYLE_KNOWN_PRODUCTS})
  list(APPEND _known_products "${name}")
  set(HYLE_KNOWN_PRODUCTS "${_known_products}" PARENT_SCOPE)
endfunction()

set(HYLE_KNOWN_PRODUCTS "")
_hyle_declare_product(core FACTORS core)
_hyle_declare_product(error FACTORS error)
_hyle_declare_product(time FACTORS time timer_queue)
_hyle_declare_product(platform FACTORS platform platform_thread platform_fs)
_hyle_declare_product(mem FACTORS mem mem_vm mem_reclaim)
_hyle_declare_product(obs FACTORS obs)
_hyle_declare_product(simd FACTORS simd)
_hyle_declare_product(math FACTORS math)
_hyle_declare_product(container FACTORS container_intrusive container)
_hyle_declare_product(util FACTORS util_core util)
_hyle_declare_product(sched FACTORS sched)
_hyle_declare_product(io FACTORS io)
_hyle_declare_product(compute FACTORS compute)
_hyle_declare_product(graph FACTORS graph)

# Resolve requested product or developer factor roots before inspecting source
# text. The two selectors are intentionally mutually exclusive. Downstream
# source consumers use only HYLE_COMPONENTS; HYLE_DEV_FACTORS is a top-level
# maintainer/CI seam for independently building one internal factor.
if(DEFINED HYLE_DEV_FACTORS)
  if(DEFINED HYLE_COMPONENTS)
    message(FATAL_ERROR
      "HYLE_DEV_FACTORS and HYLE_COMPONENTS are mutually exclusive; use a "
      "fresh build tree and choose either an internal maintenance root or a "
      "consumer product")
  endif()
  if(NOT PROJECT_IS_TOP_LEVEL)
    message(FATAL_ERROR
      "HYLE_DEV_FACTORS is a top-level maintainer option and cannot be used "
      "by an embedding parent; select a product with HYLE_COMPONENTS")
  endif()
  set(_hyle_selection_mode factor)
  set(_hyle_selection_input "${HYLE_DEV_FACTORS}")
  set(HYLE_DEV_FACTORS "${_hyle_selection_input}" CACHE STRING
      "HYLE internal factor roots for top-level maintenance only" FORCE)
else()
  set(_hyle_selection_mode product)
  if(DEFINED HYLE_COMPONENTS)
    set(_hyle_selection_input "${HYLE_COMPONENTS}")
  elseif(PROJECT_IS_TOP_LEVEL)
    set(_hyle_selection_input "all")
  else()
    message(FATAL_ERROR
      "An embedding parent must set HYLE_COMPONENTS before add_subdirectory(). "
      "HYLE will not turn an omitted subproject option into an accidental "
      "all-product build.")
  endif()
  set(HYLE_COMPONENTS "${_hyle_selection_input}" CACHE STRING
      "HYLE products to build (all or a semicolon/comma-separated product list)"
      FORCE)
endif()

set(_hyle_requested "${_hyle_selection_input}")
string(REPLACE "," ";" _hyle_requested "${_hyle_requested}")
list(TRANSFORM _hyle_requested STRIP)
list(TRANSFORM _hyle_requested TOLOWER)
list(REMOVE_ITEM _hyle_requested "")

if("all" IN_LIST _hyle_requested)
  list(LENGTH _hyle_requested _hyle_requested_count)
  if(NOT _hyle_requested_count EQUAL 1)
    message(FATAL_ERROR
      "HYLE ${_hyle_selection_mode} selector '${_hyle_selection_input}' mixes "
      "'all' with explicit names; "
      "use 'all' alone or list only the requested product roots")
  endif()
  if(_hyle_selection_mode STREQUAL "factor")
    set(_hyle_requested_roots ${HYLE_IMPLEMENTED_COMPONENTS})
    set(_hyle_requested_products "")
  else()
    set(_hyle_requested_products ${HYLE_KNOWN_PRODUCTS})
    set(_hyle_requested_roots "")
    foreach(_product IN LISTS _hyle_requested_products)
      string(TOUPPER "${_product}" _product_upper)
      list(APPEND _hyle_requested_roots
        ${_HYLE_PRODUCT_FACTORS_${_product_upper}})
    endforeach()
    list(REMOVE_DUPLICATES _hyle_requested_roots)
  endif()
elseif(NOT _hyle_requested)
  if(_hyle_selection_mode STREQUAL "factor")
    message(FATAL_ERROR
      "HYLE_DEV_FACTORS is explicitly empty. Omit it for the normal product "
      "build, or provide one or more internal factor roots.")
  else()
    message(FATAL_ERROR
      "HYLE_COMPONENTS is explicitly empty. Omit it (or use 'all') for the full "
      "product, or provide one or more product roots. Refusing to turn an empty "
      "script/CI argument into an accidental all-product build.")
  endif()
else()
  set(_hyle_requested_roots "")
  set(_hyle_requested_products "")
  if(_hyle_selection_mode STREQUAL "factor")
    foreach(_factor IN LISTS _hyle_requested)
      if(NOT _factor IN_LIST HYLE_KNOWN_COMPONENTS)
        list(JOIN HYLE_KNOWN_COMPONENTS ", " _hyle_known_text)
        message(FATAL_ERROR
          "Unknown HYLE internal factor '${_factor}'. Known factors: "
          "${_hyle_known_text}")
      endif()
      string(TOUPPER "${_factor}" _factor_upper)
      if(_HYLE_QUEUED_${_factor_upper})
        message(FATAL_ERROR
          "HYLE internal factor '${_factor}' is registered but QUEUED and has "
          "no build target yet")
      endif()
      list(APPEND _hyle_requested_roots "${_factor}")
    endforeach()
  else()
    foreach(_product IN LISTS _hyle_requested)
      if(NOT _product IN_LIST HYLE_KNOWN_PRODUCTS)
        list(JOIN HYLE_KNOWN_PRODUCTS ", " _hyle_known_text)
        message(FATAL_ERROR
          "Unknown HYLE product '${_product}'. Known products: "
          "${_hyle_known_text}")
      endif()
      list(APPEND _hyle_requested_products "${_product}")
      string(TOUPPER "${_product}" _product_upper)
      list(APPEND _hyle_requested_roots
        ${_HYLE_PRODUCT_FACTORS_${_product_upper}})
    endforeach()
    list(REMOVE_DUPLICATES _hyle_requested_roots)
  endif()
endif()

# Validate every manifest edge even for an unselected component: this is
# bounded manifest metadata work and does not read unrelated product source.
foreach(_component IN LISTS HYLE_KNOWN_COMPONENTS)
  string(TOUPPER "${_component}" _component_upper)
  if(_HYLE_QUEUED_${_component_upper} AND
     _HYLE_DEPS_${_component_upper})
    message(FATAL_ERROR
      "HYLE QUEUED component '${_component}' cannot declare dependencies")
  endif()
  foreach(_dependency IN LISTS _HYLE_DEPS_${_component_upper})
    if(NOT _dependency IN_LIST HYLE_KNOWN_COMPONENTS)
      message(FATAL_ERROR
        "HYLE component '${_component}' names unknown dependency '${_dependency}'")
    endif()
    string(TOUPPER "${_dependency}" _dependency_upper)
    if(_HYLE_QUEUED_${_dependency_upper})
      message(FATAL_ERROR
        "HYLE implemented component '${_component}' cannot depend on QUEUED "
        "component '${_dependency}'")
    endif()
  endforeach()
endforeach()

# Component identity order is append-only audit history, so it cannot also be
# the mutable dependency order. Compute a stable topological order with a DFS;
# catalog order remains the tie-breaker. The same walk rejects cycles before
# any product source is inspected.
function(_hyle_visit_component component)
  string(TOUPPER "${component}" _visit_upper)
  get_property(_visit_state GLOBAL
    PROPERTY "_HYLE_COMPONENT_VISIT_${_visit_upper}")
  if(_visit_state STREQUAL "DONE")
    return()
  endif()
  if(_visit_state STREQUAL "VISITING")
    message(FATAL_ERROR
      "HYLE component dependency cycle reaches '${component}'")
  endif()
  set_property(GLOBAL PROPERTY
    "_HYLE_COMPONENT_VISIT_${_visit_upper}" "VISITING")
  foreach(_visit_dependency IN LISTS _HYLE_DEPS_${_visit_upper})
    _hyle_visit_component("${_visit_dependency}")
  endforeach()
  set_property(GLOBAL PROPERTY
    "_HYLE_COMPONENT_VISIT_${_visit_upper}" "DONE")
  get_property(_visit_order GLOBAL PROPERTY _HYLE_COMPONENT_TOPOLOGICAL_ORDER)
  list(APPEND _visit_order "${component}")
  set_property(GLOBAL PROPERTY _HYLE_COMPONENT_TOPOLOGICAL_ORDER
    "${_visit_order}")
endfunction()

set_property(GLOBAL PROPERTY _HYLE_COMPONENT_TOPOLOGICAL_ORDER "")
foreach(_component IN LISTS HYLE_KNOWN_COMPONENTS)
  string(TOUPPER "${_component}" _component_upper)
  set_property(GLOBAL PROPERTY
    "_HYLE_COMPONENT_VISIT_${_component_upper}" "")
endforeach()
foreach(_component IN LISTS HYLE_KNOWN_COMPONENTS)
  _hyle_visit_component("${_component}")
endforeach()
get_property(_hyle_topological_components GLOBAL
  PROPERTY _HYLE_COMPONENT_TOPOLOGICAL_ORDER)

# A single reverse topological pass reaches every transitive dependency,
# avoiding a fixed-point rescan of the growing selected set.
foreach(_component IN LISTS _hyle_requested_roots)
  string(TOUPPER "${_component}" _component_upper)
  set(_HYLE_SELECTED_${_component_upper} TRUE)
endforeach()
set(_hyle_resolution_order ${_hyle_topological_components})
list(REVERSE _hyle_resolution_order)
foreach(_component IN LISTS _hyle_resolution_order)
  string(TOUPPER "${_component}" _component_upper)
  if(_HYLE_SELECTED_${_component_upper})
    foreach(_dependency IN LISTS _HYLE_DEPS_${_component_upper})
      string(TOUPPER "${_dependency}" _dependency_upper)
      set(_HYLE_SELECTED_${_dependency_upper} TRUE)
    endforeach()
  endif()
endforeach()

# Preserve the computed topological order for target construction, audit and
# configure diagnostics.
set(_hyle_resolved "")
foreach(_component IN LISTS _hyle_topological_components)
  string(TOUPPER "${_component}" _component_upper)
  if(_HYLE_SELECTED_${_component_upper})
    set(HYLE_BUILD_${_component_upper} TRUE)
    list(APPEND _hyle_resolved "${_component}")
  else()
    set(HYLE_BUILD_${_component_upper} FALSE)
  endif()
endforeach()

# Product mode exposes a facade only when all of its root factors are present.
# This also exposes a complete transitive product (for example Core) when
# another requested product already built it, while refusing to manufacture a
# partial Container facade from Time's container_intrusive dependency alone.
# Exact-factor maintenance mode creates no product facade at all.
set(HYLE_RESOLVED_PRODUCT_LIST "")
foreach(_product IN LISTS HYLE_KNOWN_PRODUCTS)
  string(TOUPPER "${_product}" _product_upper)
  set(_product_complete FALSE)
  if(_hyle_selection_mode STREQUAL "product")
    set(_product_complete TRUE)
    foreach(_factor IN LISTS _HYLE_PRODUCT_FACTORS_${_product_upper})
      string(TOUPPER "${_factor}" _factor_upper)
      if(NOT HYLE_BUILD_${_factor_upper})
        set(_product_complete FALSE)
        break()
      endif()
    endforeach()
  endif()
  if(_product_complete)
    set(HYLE_BUILD_PRODUCT_${_product_upper} TRUE)
    list(APPEND HYLE_RESOLVED_PRODUCT_LIST "${_product}")
  else()
    set(HYLE_BUILD_PRODUCT_${_product_upper} FALSE)
  endif()
endforeach()

# Owner-scope the trace option: a fresh Core-only/cache-light configure must not
# advertise a switch which cannot affect any selected target.
set(HYLE_TRACE_AVAILABLE FALSE)
foreach(_component IN LISTS _hyle_resolved)
  string(TOUPPER "${_component}" _component_upper)
  if(_HYLE_TRACE_${_component_upper})
    set(HYLE_TRACE_AVAILABLE TRUE)
    break()
  endif()
endforeach()

set(HYLE_RESOLVED_COMPONENT_LIST "${_hyle_resolved}")
list(JOIN _hyle_resolved "," HYLE_RESOLVED_COMPONENTS)
list(JOIN HYLE_RESOLVED_PRODUCT_LIST "," HYLE_RESOLVED_PRODUCTS)
if(_hyle_selection_mode STREQUAL "factor")
  message(STATUS
    "HYLE developer factors: requested='${HYLE_DEV_FACTORS}' "
    "resolved='${HYLE_RESOLVED_COMPONENTS}' products='${HYLE_RESOLVED_PRODUCTS}'")
else()
  message(STATUS
    "HYLE products: requested='${HYLE_COMPONENTS}' "
    "available='${HYLE_RESOLVED_PRODUCTS}' factors='${HYLE_RESOLVED_COMPONENTS}'")
endif()

foreach(_component IN LISTS HYLE_KNOWN_COMPONENTS)
  string(TOUPPER "${_component}" _component_upper)
  unset(_HYLE_SELECTED_${_component_upper})
  set_property(GLOBAL PROPERTY
    "_HYLE_COMPONENT_VISIT_${_component_upper}" "")
endforeach()
set_property(GLOBAL PROPERTY _HYLE_COMPONENT_TOPOLOGICAL_ORDER "")
unset(_hyle_requested)
unset(_hyle_requested_count)
unset(_hyle_requested_products)
unset(_hyle_requested_roots)
unset(_hyle_selection_input)
unset(_hyle_selection_mode)
unset(_hyle_resolution_order)
unset(_hyle_topological_components)
unset(_hyle_known_text)
unset(_component)
unset(_component_upper)
unset(_dependency)
unset(_dependency_upper)
unset(_factor)
unset(_factor_upper)
unset(_product)
unset(_product_complete)
unset(_product_upper)

function(_hyle_component_public_headers name out_var)
  string(TOUPPER "${name}" _component_upper)
  set(_split_public_headers
      "${_HYLE_SPLIT_PUBLIC_HEADERS_${_component_upper}}")
  if(_split_public_headers)
    set(_headers "")
    foreach(_split_public_header IN LISTS _split_public_headers)
      list(APPEND _headers
        "${HYLE_SOURCE_DIR}/include/${_split_public_header}")
    endforeach()
  elseif(name STREQUAL "core")
    file(GLOB_RECURSE _headers CONFIGURE_DEPENDS
      "${HYLE_SOURCE_DIR}/include/hyle/core/*.h")
    list(APPEND _headers
      "${HYLE_SOURCE_DIR}/include/hyle/version.h"
      "${HYLE_SOURCE_DIR}/include/hyle/visibility.h")
  else()
    file(GLOB_RECURSE _headers CONFIGURE_DEPENDS
      "${HYLE_SOURCE_DIR}/include/hyle/${name}/*.h")
    foreach(_split_public_header IN LISTS _HYLE_SPLIT_PUBLIC_HEADERS)
      list(REMOVE_ITEM _headers
        "${HYLE_SOURCE_DIR}/include/${_split_public_header}")
    endforeach()
    if(name STREQUAL "simd")
      # Finite provider routes remain implementation-only. target.h and
      # fat_kernel.h are public expert headers shared by source/static/shared
      # delivery and are therefore installed with the selected SIMD component.
      list(FILTER _headers EXCLUDE REGEX "/hyle/simd/provider/")
    elseif(name STREQUAL "sched")
      # Cross-component worker-loop bridges are source-composition details.
      # They are available while building HYLE but are not an installed C++
      # SDK API.
      list(FILTER _headers EXCLUDE REGEX "/hyle/sched/detail/")
    endif()
  endif()
  list(SORT _headers)
  set(${out_var} "${_headers}" PARENT_SCOPE)
endfunction()

# Resolve a public include to the component that owns that header. Root-level
# Compiler/environment/visibility/version contracts belong to Core. The trace macro
# header is owned by the always-exported hyle_trace foundation target and is
# already validated through hyle_add_module(... TRACE), so it has no product
# component edge of its own.
function(_hyle_public_include_owner include_path out_var)
  set(_owner "")
  list(FIND _HYLE_SPLIT_PUBLIC_HEADERS "${include_path}"
    _split_public_header_index)
  if(NOT _split_public_header_index EQUAL -1)
    list(GET _HYLE_SPLIT_PUBLIC_HEADER_OWNERS
      ${_split_public_header_index} _owner)
  elseif(include_path STREQUAL "hyle/version.h" OR
     include_path STREQUAL "hyle/visibility.h" OR
     include_path MATCHES "^hyle/core/")
    set(_owner core)
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
  set(${out_var} "${_dependency}" PARENT_SCOPE)
endfunction()

# Every direct cross-component include in a selected public header must have a
# direct PUBLIC target edge. An all-component build checks the whole tree; a
# selected source consumer stays independent from unrelated in-progress code.
foreach(_hyle_component IN LISTS _hyle_resolved)
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

# Source-only coupling inside the selected closure must be just as explicit as
# an exported header edge. The shared include root must not make an undeclared
# dependency look independent, while unselected modules remain isolated.
foreach(_hyle_component IN LISTS _hyle_resolved)
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
unset(_hyle_resolved)
