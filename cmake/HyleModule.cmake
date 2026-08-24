include_guard(GLOBAL)

# Keep the executable component manifest in HyleComponents.cmake as the single
# source of truth. Its PUBLIC/PRIVATE lists generate both the pre-target
# component closure and hyle_add_module()'s real link edges. The validator below
# audits the completed targets so a later manual internal link cannot bypass the
# manifest.
function(_hyle_validate_module_dependencies name)
  if(NOT name IN_LIST HYLE_KNOWN_COMPONENTS)
    message(FATAL_ERROR
      "hyle_add_module(${name}): module is absent from HYLE_KNOWN_COMPONENTS")
  endif()

  set(_declared_dependencies "")
  foreach(_dependency IN LISTS ARGN)
    if(_dependency MATCHES "^hyle_(.+)$")
      list(APPEND _declared_dependencies "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _declared_dependencies)
  list(SORT _declared_dependencies)

  string(TOUPPER "${name}" _component_upper)
  set(_graph_dependencies "${_HYLE_DEPS_${_component_upper}}")
  list(REMOVE_DUPLICATES _graph_dependencies)
  list(SORT _graph_dependencies)

  if(NOT "${_declared_dependencies}" STREQUAL "${_graph_dependencies}")
    message(FATAL_ERROR
      "hyle_add_module(${name}): direct internal dependencies disagree with "
      "HyleComponents.cmake; target declares '${_declared_dependencies}', "
      "component graph declares '${_graph_dependencies}'")
  endif()
endfunction()

function(_hyle_validate_module_public_dependencies name)
  set(_actual_public_dependencies "")
  foreach(_dependency IN LISTS ARGN)
    if(_dependency MATCHES "^hyle_(.+)$")
      list(APPEND _actual_public_dependencies "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _actual_public_dependencies)
  list(SORT _actual_public_dependencies)

  string(TOUPPER "${name}" _component_upper)
  set(_manifest_public_dependencies
      "${_HYLE_PUBLIC_DEPS_${_component_upper}}")
  list(REMOVE_DUPLICATES _manifest_public_dependencies)
  list(SORT _manifest_public_dependencies)

  if(NOT "${_actual_public_dependencies}" STREQUAL
         "${_manifest_public_dependencies}")
    message(FATAL_ERROR
      "hyle_${name}: public internal dependencies disagree with the component "
      "manifest; target exposes '${_actual_public_dependencies}', manifest "
      "declares '${_manifest_public_dependencies}'")
  endif()
endfunction()

function(_hyle_manifest_target_dependencies name visibility out_var)
  string(TOUPPER "${name}" _component_upper)
  set(_manifest_var "_HYLE_${visibility}_DEPS_${_component_upper}")
  set(_targets "")
  foreach(_dependency IN LISTS ${_manifest_var})
    if(NOT TARGET hyle_${_dependency})
      message(FATAL_ERROR
        "hyle_add_module(${name}): manifest ${visibility} dependency target "
        "'hyle_${_dependency}' does not exist; check component construction order")
    endif()
    list(APPEND _targets hyle_${_dependency})
  endforeach()
  set(${out_var} "${_targets}" PARENT_SCOPE)
endfunction()

# Re-read the finished targets after every module CMakeLists has returned.  The
# declaration-time check above catches mistakes in hyle_add_module(), while this
# final check also catches a later target_link_libraries(hyle_x ... hyle_y)
# call that would otherwise make the executable component graph lie.
function(hyle_validate_final_module_graph)
  foreach(_name IN LISTS HYLE_KNOWN_COMPONENTS)
    if(NOT TARGET hyle_${_name})
      continue()
    endif()

    set(_actual_dependencies "")
    set(_actual_public_dependencies "")
    foreach(_property IN ITEMS LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
      get_target_property(_link_items hyle_${_name} ${_property})
      if(NOT _link_items OR _link_items MATCHES "-NOTFOUND$")
        continue()
      endif()
      foreach(_item IN LISTS _link_items)
        foreach(_candidate IN LISTS HYLE_KNOWN_COMPONENTS)
          if(_item MATCHES
             "(^|[^A-Za-z0-9_])hyle(_|::)${_candidate}([^A-Za-z0-9_]|$)")
            list(APPEND _actual_dependencies hyle_${_candidate})
            if(_property STREQUAL "INTERFACE_LINK_LIBRARIES" AND
               NOT _item MATCHES "\\$<LINK_ONLY:")
              list(APPEND _actual_public_dependencies hyle_${_candidate})
            endif()
          endif()
        endforeach()
      endforeach()
    endforeach()
    list(REMOVE_DUPLICATES _actual_dependencies)
    _hyle_validate_module_dependencies(
      "${_name}" ${_actual_dependencies})
    list(REMOVE_DUPLICATES _actual_public_dependencies)
    _hyle_validate_module_public_dependencies(
      "${_name}" ${_actual_public_dependencies})
  endforeach()
endfunction()

# hyle_add_module(<name> [SOURCES ...])
# Declares a HYLE module as a library target `hyle_<name>` with conventional
# include dirs, manifest-derived internal links, warnings, sanitizer flags, and
# a tri-state export macro. External implementation links stay in the owning
# module CMakeLists via ordinary target_link_libraries().
function(hyle_add_module name)
  cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "hyle_add_module(${name}): unrecognized arguments: "
      "${ARG_UNPARSED_ARGUMENTS}")
  endif()
  if(ARG_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR
      "hyle_add_module(${name}): keywords missing values: "
      "${ARG_KEYWORDS_MISSING_VALUES}")
  endif()

  _hyle_manifest_target_dependencies(
    "${name}" PRIVATE _hyle_private_dependencies)
  _hyle_manifest_target_dependencies(
    "${name}" PUBLIC _hyle_public_dependencies)
  _hyle_validate_module_dependencies(
    "${name}" ${_hyle_private_dependencies} ${_hyle_public_dependencies})
  _hyle_validate_module_public_dependencies(
    "${name}" ${_hyle_public_dependencies})

  set(target "hyle_${name}")
  if(ARG_SOURCES)
    if(HYLE_STATIC)
      add_library(${target} STATIC ${ARG_SOURCES})
    else()
      add_library(${target} SHARED ${ARG_SOURCES})
    endif()
  else()
    add_library(${target} INTERFACE)
  endif()

  if(ARG_SOURCES)
    target_include_directories(${target}
      PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${HYLE_INSTALL_INCLUDEDIR}>"
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    # Export the C++23 requirement to CONSUMERS: a downstream target linking hyle::<mod>
    # includes our headers (which use `concept`, std::expected, etc.), so it must compile
    # at C++23 too. CMAKE_CXX_STANDARD only governs HYLE's OWN build; PUBLIC compile
    # features propagate the floor to consumers (verified by samples/ -- a consumer that
    # didn't set the standard failed on `concept` until this was added).
    target_compile_features(${target} PUBLIC cxx_std_23)
    target_link_libraries(${target}
      PRIVATE ${_hyle_private_dependencies}
      PUBLIC ${_hyle_public_dependencies})
    # obs-02: every module sees the trace gate (defines + tracy include when ON).
    # PUBLIC so consumers that include our headers (trace.h is inlined, e.g. the
    # RAII zone object) agree on whether the hooks are real or empty -- an ODR
    # requirement. OFF (default) => hyle_trace carries nothing, zero cost.
    target_link_libraries(${target} PUBLIC hyle_build_contract hyle_trace)
    # HYLE_TESTING gate (same PUBLIC/ODR rationale as hyle_trace): carries the
    # HYLE_TESTING define in an instrumented test build so a module's own sources,
    # its test exe, and consumers all agree on whether test-only members exist. Empty
    # (zero cost) in a production-shape test or shipped build.
    target_link_libraries(${target} PUBLIC hyle_testing)
    hyle_apply_warnings(${target})
    hyle_apply_sanitizer(${target})
    # tri-state export macro: HYLE_<NAME>_API
    string(TOUPPER "${name}" up)
    # The implementation always needs its BUILD discriminator. HYLE_STATIC only
    # changes visibility; it must not make a module compile its consumer-inline
    # definitions in the same TU as out-of-line bodies (error::WithContext was
    # the first concrete duplicate-definition failure this exposed).
    target_compile_definitions(${target} PRIVATE "HYLE_${up}_BUILD")
    if(HYLE_STATIC)
      target_compile_definitions(${target} PUBLIC "HYLE_STATIC")
    endif()
  else()
    if(_hyle_private_dependencies)
      message(FATAL_ERROR
        "hyle_add_module(${name}): header-only modules cannot have PRIVATE "
        "component dependencies: ${_hyle_private_dependencies}")
    endif()
    target_include_directories(${target} INTERFACE
      "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
      "$<INSTALL_INTERFACE:${HYLE_INSTALL_INCLUDEDIR}>")
    target_compile_features(${target} INTERFACE cxx_std_23)  # header-only modules too
    target_link_libraries(${target} INTERFACE
      ${_hyle_public_dependencies}
      hyle_build_contract hyle_trace hyle_testing)
  endif()

  set_target_properties(${target} PROPERTIES
    EXPORT_NAME "${name}"
    FOLDER "hyle/modules")
  if(ARG_SOURCES AND NOT HYLE_STATIC)
    set_target_properties(${target} PROPERTIES
      VERSION "${PROJECT_VERSION}"
      SOVERSION "${HYLE_ABI_VERSION}")
  endif()
  add_library(hyle::${name} ALIAS ${target})
endfunction()

# hyle_add_test(<module> SOURCES ... [NAME <suffix>] [DEFINES ...] [OPTIONS ...] [DEPS ...])
# -> test exe linked to hyle_<module> + gtest. NAME gives a distinct target suffix so a
# module can have several test exes (e.g. simd compiles the SAME source per ISA variant
# with different DEFINES/OPTIONS -- HYLE_SIMD_TARGET_AVX2 + -mavx2 etc.).
function(hyle_add_test module)
  cmake_parse_arguments(ARG "NO_AUTORUN" "NAME" "SOURCES;DEPS;DEFINES;OPTIONS" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "hyle_add_test(${module}): unrecognized arguments: "
      "${ARG_UNPARSED_ARGUMENTS}")
  endif()
  if(ARG_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR
      "hyle_add_test(${module}): keywords missing values: "
      "${ARG_KEYWORDS_MISSING_VALUES}")
  endif()
  if(NOT TARGET hyle_${module})
    return()
  endif()
  if(ARG_NAME)
    set(target "hyle_${module}_${ARG_NAME}_test")
  else()
    set(target "hyle_${module}_test")
  endif()
  add_executable(${target} ${ARG_SOURCES})
  set_target_properties(${target} PROPERTIES
    FOLDER "hyle/tests/${module}"
    HYLE_TEST_MODULE "${module}")
  # The canonical GTest::gtest_main target already propagates GTest::gtest.
  # Linking both emits the same archive twice (and a duplicate-library warning
  # from Apple's linker) without adding any symbols.
  target_link_libraries(${target} PRIVATE hyle_${module} GTest::gtest_main ${ARG_DEPS})
  if(ARG_DEFINES)
    target_compile_definitions(${target} PRIVATE ${ARG_DEFINES})
  endif()
  if(ARG_OPTIONS)
    target_compile_options(${target} PRIVATE ${ARG_OPTIONS})
  endif()
  hyle_apply_warnings(${target})
  hyle_apply_sanitizer(${target})
  if(TARGET hyle_tests)
    if(NOT TARGET hyle_${module}_tests)
      add_custom_target(hyle_${module}_tests)
      set_target_properties(hyle_${module}_tests PROPERTIES
        FOLDER "hyle/validation")
      add_dependencies(hyle_tests hyle_${module}_tests)
    endif()
    add_dependencies(hyle_${module}_tests ${target})
  endif()
  # NO_AUTORUN: build the target but don't register it as a ctest. For ISA
  # variants (avx512) that SIGILL on a host lacking the instruction set -- built
  # for compile validation + manual SDE runs, not bare-metal auto-run.
  if(NOT ARG_NO_AUTORUN)
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES LABELS "${module}")
  endif()
endfunction()

# Register benchmark executables under stable all-selected and per-module
# aggregate targets. Direct add_executable benches use hyle_register_benches;
# hyle_add_bench registers its conventional target automatically.
function(hyle_register_bench module target)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "Cannot register missing benchmark target: ${target}")
  endif()
  # Benchmarks are consumers of HYLE's public headers, including header-only
  # targets that deliberately do not link a product library. Give every one
  # the same source encoding, language and public ISA baseline as an installed
  # consumer instead of acquiring those semantics only by accident through a
  # linked hyle_<module> target.
  target_link_libraries(${target} PRIVATE hyle_build_contract)
  if(MSVC)
    # Benchmark controls intentionally read an immutable process environment
    # (filters, sample counts and Apple QoS selection).  Centralize the CRT
    # policy on validation executables so every registered benchmark remains
    # /WX-clean instead of scattering this definition across source files.
    target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS=1)
  endif()
  get_property(_registered_module TARGET ${target} PROPERTY HYLE_BENCH_MODULE)
  if(_registered_module AND
     NOT "${_registered_module}" STREQUAL "${module}")
    message(FATAL_ERROR
      "Benchmark target '${target}' is already registered to module "
      "'${_registered_module}', cannot also register it to '${module}'")
  endif()
  set_property(TARGET ${target} PROPERTY HYLE_BENCH_MODULE "${module}")
  set_target_properties(${target} PROPERTIES FOLDER "hyle/bench/${module}")
  if(TARGET hyle_benches)
    if(NOT TARGET hyle_${module}_benches)
      add_custom_target(hyle_${module}_benches)
      set_target_properties(hyle_${module}_benches PROPERTIES
        FOLDER "hyle/validation")
      add_dependencies(hyle_benches hyle_${module}_benches)
    endif()
    add_dependencies(hyle_${module}_benches ${target})
  endif()
endfunction()

function(hyle_register_benches module)
  foreach(target IN LISTS ARGN)
    hyle_register_bench(${module} ${target})
  endforeach()
endfunction()

# hyle_add_bench(<module> SOURCES ...) -> bench exe linked to hyle_<module> + nanobench
function(hyle_add_bench module)
  cmake_parse_arguments(ARG "" "" "SOURCES;DEPS" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "hyle_add_bench(${module}): unrecognized arguments: "
      "${ARG_UNPARSED_ARGUMENTS}")
  endif()
  if(ARG_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR
      "hyle_add_bench(${module}): keywords missing values: "
      "${ARG_KEYWORDS_MISSING_VALUES}")
  endif()
  if(NOT TARGET hyle_${module})
    return()
  endif()
  set(target "hyle_${module}_bench")
  add_executable(${target} ${ARG_SOURCES})
  target_link_libraries(${target} PRIVATE hyle_${module} nanobench ${ARG_DEPS})
  hyle_apply_warnings(${target})
  hyle_register_bench(${module} ${target})
endfunction()

# hyle_stage_runtime_dependencies(<consumer-target>)
#
# Windows does not have an rpath equivalent: a source-embedded executable can
# link HYLE DLL import libraries successfully but still fail at startup unless
# the DLLs are beside the executable (or manually placed on PATH).  Do not solve
# that by overriding the parent project's global output directories.  Instead,
# consumers that want a runnable build-tree executable opt into this standard
# CMake runtime-DLL staging step. It also captures transitive runtime DLLs.
function(hyle_stage_runtime_dependencies target)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "hyle_stage_runtime_dependencies: target '${target}' does not exist")
  endif()
  if(WIN32 AND NOT HYLE_STATIC)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy -t "$<TARGET_FILE_DIR:${target}>"
              "$<TARGET_RUNTIME_DLLS:${target}>"
      COMMAND_EXPAND_LISTS
      VERBATIM)
  endif()
endfunction()
