# HyleTrace.cmake -- obs-02 instrumentation/trace gate.
#
# HYLE_TRACE (default OFF) is the master compile-time gate for the HYLE_TRACE_*
# hooks (include/hyle/obs/trace.h). OFF is the shipped/benchmarked default: the
# hooks expand to nothing, no profiler is linked, and there is ZERO dependency and
# ZERO code at the fiber-switch / io-reap hot paths (the obs-02 zero-cost law).
#
# ON wires a profiler backend. The first (only) backend is tracy, borrowed whole
# (a header tree + one client TU), selected with HYLE_TRACE_BACKEND=tracy. tracy's
# sources live OUTSIDE this repo (a sibling clone, not vendored), so its location
# is a cache path -- absent, we hard-fail with a clear message rather than compile
# half a profiler.
#
# The gate defines propagate through the PUBLIC hyle_trace INTERFACE target:
# trace.h is included across modules and in consumers, so every linked TU sees
# one consistent on/off/backend choice without polluting unrelated targets.

option(HYLE_TRACE "enable compile-time trace/instrumentation hooks (obs-02)" OFF)
set(HYLE_TRACE_BACKEND "tracy" CACHE STRING "trace backend: tracy")
set_property(CACHE HYLE_TRACE_BACKEND PROPERTY STRINGS tracy)
# Path to a tracy checkout (the dir containing public/tracy/Tracy.hpp). Not vendored
# in-tree; point this at the sibling clone when enabling trace.
set(HYLE_TRACY_DIR "" CACHE PATH "path to a tracy checkout (contains public/tracy/Tracy.hpp)")

# hyle_trace is an INTERFACE target every module links (PUBLIC). It carries the
# trace include path + gate defines. OFF => it carries nothing but the (always
# available) hyle include dir is already public, so trace.h resolves and expands
# empty. This target exists unconditionally so module CMakeLists can link it
# without an if() at every call site.
add_library(hyle_trace INTERFACE)

if(HYLE_TRACE)
  if(HYLE_TRACE_BACKEND STREQUAL "tracy")
    if(NOT HYLE_TRACY_DIR OR NOT EXISTS "${HYLE_TRACY_DIR}/public/tracy/Tracy.hpp")
      message(FATAL_ERROR
        "HYLE_TRACE=ON with backend tracy, but HYLE_TRACY_DIR does not point at a "
        "tracy checkout (need <dir>/public/tracy/Tracy.hpp). Set -DHYLE_TRACY_DIR=... "
        "Refusing to build a half-wired profiler.")
    endif()
    # The tracy client: ONE TU (TracyClient.cpp) compiled with TRACY_ENABLE. tracy
    # is a global singleton and its docs require the client TU be built exactly
    # once. HYLE is a multi-DLL build by default, so this target MUST follow
    # HYLE_STATIC: a STATIC archive linked into every hyle_*.dll creates one
    # profiler singleton per module. Shared HYLE gets one shared Tracy client;
    # static HYLE gets one archive resolved once by the final executable.
    if(HYLE_STATIC)
      add_library(hyle_tracy_client STATIC "${HYLE_TRACY_DIR}/public/TracyClient.cpp")
    else()
      add_library(hyle_tracy_client SHARED "${HYLE_TRACY_DIR}/public/TracyClient.cpp")
    endif()
    target_include_directories(hyle_tracy_client SYSTEM PUBLIC
      "$<BUILD_INTERFACE:${HYLE_TRACY_DIR}/public>"
      "$<INSTALL_INTERFACE:${HYLE_INSTALL_INCLUDEDIR}/hyle/vendor/tracy-public>")
    set_target_properties(hyle_tracy_client PROPERTIES EXPORT_NAME "_tracy_client")
    target_compile_definitions(hyle_tracy_client
      PRIVATE TRACY_ENABLE TRACY_FIBERS
      INTERFACE TRACY_ENABLE TRACY_FIBERS)
    if(NOT HYLE_STATIC)
      target_compile_definitions(hyle_tracy_client
        PRIVATE TRACY_EXPORTS
        INTERFACE TRACY_IMPORTS)
    endif()
    # position-independent so it can link into HYLE's shared libs.
    set_target_properties(hyle_tracy_client PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_link_libraries(hyle_trace INTERFACE hyle_tracy_client)
    target_include_directories(hyle_trace SYSTEM INTERFACE "${HYLE_TRACY_DIR}/public")
    # HYLE_TRACE_TRACY implies HYLE_TRACE inside trace.h; set both + tracy's own.
    target_compile_definitions(hyle_trace INTERFACE
      HYLE_TRACE HYLE_TRACE_TRACY TRACY_ENABLE TRACY_FIBERS)
    message(STATUS "HYLE_TRACE = ON (tracy @ ${HYLE_TRACY_DIR})")
  else()
    message(FATAL_ERROR "HYLE_TRACE_BACKEND='${HYLE_TRACE_BACKEND}' unknown (only 'tracy').")
  endif()
else()
  message(STATUS "HYLE_TRACE = OFF (hooks compile to nothing -- zero cost)")
endif()
set_target_properties(hyle_trace PROPERTIES EXPORT_NAME "_trace_config")
