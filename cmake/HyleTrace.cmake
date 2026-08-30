# HyleTrace.cmake -- obs-02 instrumentation/trace gate.
#
# HYLE_TRACE (default OFF) is the master compile-time gate for the HYLE_TRACE_*
# hooks (include/hyle/obs/trace.h). OFF is the shipped/benchmarked default: the
# hooks expand to nothing, no profiler is linked, and there is ZERO dependency and
# ZERO code at the fiber-switch / io-reap hot paths (the obs-02 zero-cost law).
#
# ON wires Tracy, borrowed whole (a header tree + one client TU). A backend
# selector with only one legal value was noise; add one only when a second real
# implementation exists. Tracy's
# sources live OUTSIDE this repo. The default route resolves the locked archive
# through HYLE's shared source cache; HYLE_TRACY_DIR remains an explicit,
# strictly verified source override.
#
# The gate defines propagate through the PUBLIC hyle_trace INTERFACE target:
# trace.h is included across modules and in consumers, so every linked TU sees
# one consistent on/off/backend choice without polluting unrelated targets.

if(HYLE_TRACE_AVAILABLE)
  option(HYLE_TRACE "enable compile-time trace/instrumentation hooks (obs-02)" OFF)
elseif(DEFINED HYLE_TRACE AND HYLE_TRACE)
  message(FATAL_ERROR
    "HYLE_TRACE=ON, but the selected component closure has no trace-aware "
    "target. Select obs, mem, sched, io, compute, or a consumer of one of them.")
else()
  set(HYLE_TRACE OFF)
endif()
if(HYLE_TRACE)
  # Optional explicit Tracy source override (the directory containing
  # public/tracy/Tracy.hpp). Empty means resolve the locked shared-cache source.
  set(HYLE_TRACY_DIR "" CACHE PATH
      "optional locked Tracy source override")
else()
  set(HYLE_TRACY_DIR "")
endif()

# hyle_trace is the INTERFACE target used by trace-aware modules. It carries the
# trace include path + gate defines. OFF => it carries nothing. Backend setup is
# deferred until module declarations prove at least one selected target consumes
# the gate, so an unrelated component closure cannot manufacture a profiler.
add_library(hyle_trace INTERFACE)
set_target_properties(hyle_trace PROPERTIES FOLDER "hyle/foundation")

function(hyle_finalize_trace)
  if(NOT HYLE_TRACE)
    message(STATUS "HYLE_TRACE = OFF (hooks compile to nothing -- zero cost)")
    return()
  endif()

  get_property(_hyle_trace_consumed GLOBAL PROPERTY HYLE_TRACE_CONSUMED)
  if(NOT _hyle_trace_consumed)
    message(FATAL_ERROR
      "HYLE_TRACE=ON, but the selected component closure has no target that "
      "consumes the trace gate. Select a trace-aware component or disable "
      "HYLE_TRACE; refusing to create an unrelated profiler runtime.")
  endif()
  if(HYLE_TRACY_DIR)
    hyle_verify_locked_source_input(tracy "${HYLE_TRACY_DIR}")
  else()
    hyle_dependency_cache_resolve(tracy _hyle_tracy_cached_source)
    set(HYLE_TRACY_DIR "${_hyle_tracy_cached_source}")
    unset(_hyle_tracy_cached_source)
  endif()
  if(NOT EXISTS "${HYLE_TRACY_DIR}/public/tracy/Tracy.hpp")
    message(FATAL_ERROR
      "Locked Tracy source does not contain public/tracy/Tracy.hpp: "
      "${HYLE_TRACY_DIR}")
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
  # Preserve the platform closure from Tracy's own TracyClient target even
  # though HYLE intentionally compiles the single client TU directly. In a
  # static SDK these become LINK_ONLY consumer requirements; the package
  # registry below arranges find_dependency(Threads) before target import.
  find_package(Threads REQUIRED)
  target_link_libraries(hyle_tracy_client
    PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
  if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_libraries(hyle_tracy_client PRIVATE ws2_32 dbghelp secur32)
    target_compile_definitions(hyle_tracy_client
      PUBLIC WINVER=0x0A00 _WIN32_WINNT=0x0A00)
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
  set_target_properties(hyle_tracy_client PROPERTIES FOLDER "hyle/third_party")

  target_link_libraries(hyle_trace INTERFACE hyle_tracy_client)
  # HYLE_TRACE_TRACY implies HYLE_TRACE inside trace.h; set both + tracy's own.
  target_compile_definitions(hyle_trace INTERFACE
    HYLE_TRACE HYLE_TRACE_TRACY TRACY_ENABLE TRACY_FIBERS)
  hyle_register_package_dependency(
    NAME tracy
    OWNER foundation
    FEATURES trace
    EXPORT_TARGETS hyle_tracy_client
    STATIC_FIND_DEPENDENCIES Threads
    LOCKED_DEPENDENCIES tracy
    RUNTIME_FILE_SPECS
      "${HYLE_TRACY_DIR}/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|tracy-LICENSE"
    DEVELOPMENT_DIRECTORY_SPECS
      "${HYLE_TRACY_DIR}/public/|${HYLE_INSTALL_INCLUDEDIR}/hyle/vendor/tracy-public")
  message(STATUS
    "HYLE_TRACE = ON (tracy ${HYLE_DEP_TRACY_VERSION} "
    "${HYLE_DEP_TRACY_COMMIT} @ ${HYLE_TRACY_DIR})")
endfunction()
set_target_properties(hyle_trace PROPERTIES EXPORT_NAME "_trace_config")
