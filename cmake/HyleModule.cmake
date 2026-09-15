include_guard(GLOBAL)

# Build-only header seams are created only when the selected product closure
# consumes them. They are real target edges for source composition, while
# BUILD_INTERFACE prevents them from leaking into static/shared package
# metadata or masquerading as independently delivered product components.
set(_hyle_active_build_dependencies "")
foreach(_hyle_component IN LISTS HYLE_KNOWN_COMPONENTS)
  string(TOUPPER "${_hyle_component}" _hyle_component_upper)
  if(HYLE_BUILD_${_hyle_component_upper})
    list(APPEND _hyle_active_build_dependencies
      ${_HYLE_PRIVATE_BUILD_DEPS_${_hyle_component_upper}})
  endif()
endforeach()
list(REMOVE_DUPLICATES _hyle_active_build_dependencies)
unset(_hyle_active_build_dependencies)
unset(_hyle_component)
unset(_hyle_component_upper)

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
    if(_dependency MATCHES "^hyle_(.+)$" AND
       CMAKE_MATCH_1 IN_LIST HYLE_KNOWN_COMPONENTS)
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

function(_hyle_manifest_private_build_dependencies name out_var)
  string(TOUPPER "${name}" _component_upper)
  set(_targets "")
  foreach(_dependency IN LISTS
      _HYLE_PRIVATE_BUILD_DEPS_${_component_upper})
    if(NOT TARGET hyle_${_dependency})
      message(FATAL_ERROR
        "hyle_add_module(${name}): PRIVATE_BUILD_DEPS target "
        "'hyle_${_dependency}' does not exist")
    endif()
    list(APPEND _targets "$<BUILD_INTERFACE:hyle_${_dependency}>")
  endforeach()
  set(${out_var} "${_targets}" PARENT_SCOPE)
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
    get_target_property(_optional_absent hyle_${_name}
      HYLE_OPTIONAL_ABSENT)
    if(_optional_absent)
      # An explicitly absent optional implementation has no PRIVATE link
      # edges. Add only the manifest's private names to the validation view:
      # real PUBLIC edges must still exist and any unexpected real edge still
      # fails the ordinary exact-graph check below.
      string(TOUPPER "${_name}" _component_upper)
      foreach(_private_dependency IN LISTS
          _HYLE_PRIVATE_DEPS_${_component_upper})
        list(APPEND _actual_dependencies hyle_${_private_dependency})
      endforeach()
      list(REMOVE_DUPLICATES _actual_dependencies)
    endif()
    _hyle_validate_module_dependencies(
      "${_name}" ${_actual_dependencies})
    list(REMOVE_DUPLICATES _actual_public_dependencies)
    _hyle_validate_module_public_dependencies(
      "${_name}" ${_actual_public_dependencies})

    string(TOUPPER "${_name}" _component_upper)
    set(_expected_build_dependencies
      "${_HYLE_PRIVATE_BUILD_DEPS_${_component_upper}}")
    if(_optional_absent)
      set(_expected_build_dependencies "")
    endif()
    set(_actual_build_dependencies "")
    get_target_property(_direct_links hyle_${_name} LINK_LIBRARIES)
    foreach(_build_dependency IN LISTS HYLE_KNOWN_BUILD_DEPENDENCIES)
      foreach(_item IN LISTS _direct_links)
        if(_item MATCHES
           "(^|[^A-Za-z0-9_])hyle_${_build_dependency}([^A-Za-z0-9_]|$)")
          list(APPEND _actual_build_dependencies "${_build_dependency}")
        endif()
      endforeach()
    endforeach()
    list(REMOVE_DUPLICATES _actual_build_dependencies)
    list(SORT _actual_build_dependencies)
    list(SORT _expected_build_dependencies)
    if(NOT "${_actual_build_dependencies}" STREQUAL
           "${_expected_build_dependencies}")
      message(FATAL_ERROR
        "hyle_${_name}: build-only header dependencies disagree with the "
        "component manifest; target has '${_actual_build_dependencies}', "
        "manifest declares '${_expected_build_dependencies}'")
    endif()
    unset(_optional_absent)
    unset(_private_dependency)
  endforeach()
endfunction()

function(_hyle_module_uses_gate name token out_var)
  _hyle_component_public_headers("${name}" _hyle_gate_files)
  foreach(_hyle_module_source IN LISTS ARGN)
    if(_hyle_module_source MATCHES "^\\$<")
      continue()
    elseif(IS_ABSOLUTE "${_hyle_module_source}")
      set(_hyle_gate_file "${_hyle_module_source}")
    else()
      set(_hyle_gate_file "${CMAKE_CURRENT_SOURCE_DIR}/${_hyle_module_source}")
    endif()
    if(EXISTS "${_hyle_gate_file}")
      list(APPEND _hyle_gate_files "${_hyle_gate_file}")
    endif()
  endforeach()
  set(_hyle_gate_used FALSE)
  foreach(_hyle_gate_file IN LISTS _hyle_gate_files)
    file(READ "${_hyle_gate_file}" _hyle_gate_contents)
    string(FIND "${_hyle_gate_contents}" "${token}" _hyle_gate_position)
    if(NOT _hyle_gate_position EQUAL -1)
      set(_hyle_gate_used TRUE)
      break()
    endif()
  endforeach()
  set(${out_var} "${_hyle_gate_used}" PARENT_SCOPE)
endfunction()

# hyle_add_module(<name> [TRACE] [TESTING] [OPTIONAL_ABSENT] [SOURCES ...])
# Declares an internal HYLE factor as `hyle_<name>` with conventional
# include dirs, manifest-derived internal links, warnings, sanitizer flags, and
# a tri-state export macro. TRACE/TESTING are explicit opt-ins and must match the
# checked-in module sources/public headers; this prevents a pure target from
# acquiring an unrelated profiler runtime or test ABI. External links stay in the
# owning module CMakeLists via ordinary target_link_libraries(). Consumer aliases
# are created separately by hyle_add_product_facades(); a factor must never link
# a product facade. OPTIONAL_ABSENT is only for the OFF/unsupported branch of a
# normally compiled optional factor: it keeps identity/public dependencies but
# deliberately has no implementation or private implementation links.
function(hyle_add_module name)
  cmake_parse_arguments(ARG "TRACE;TESTING;OPTIONAL_ABSENT" "" "SOURCES" ${ARGN})
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
  if(ARG_OPTIONAL_ABSENT AND ARG_SOURCES)
    message(FATAL_ERROR
      "hyle_add_module(${name} OPTIONAL_ABSENT): SOURCES are contradictory")
  endif()
  if(ARG_OPTIONAL_ABSENT AND (ARG_TRACE OR ARG_TESTING))
    message(FATAL_ERROR
      "hyle_add_module(${name} OPTIONAL_ABSENT): TRACE/TESTING require an "
      "implementation")
  endif()

  string(TOUPPER "${name}" _hyle_component_upper)
  if(ARG_TRACE AND NOT _HYLE_TRACE_${_hyle_component_upper})
    message(FATAL_ERROR
      "hyle_add_module(${name} TRACE): the central component manifest does "
      "not declare this trace consumer")
  elseif(_HYLE_TRACE_${_hyle_component_upper} AND NOT ARG_TRACE)
    message(FATAL_ERROR
      "hyle_add_module(${name}): the central component manifest declares a "
      "TRACE consumer but the real target omitted TRACE")
  endif()

  _hyle_module_uses_gate(
    "${name}" "HYLE_TRACE" _hyle_uses_trace ${ARG_SOURCES})
  if(ARG_TRACE AND NOT _hyle_uses_trace)
    message(FATAL_ERROR
      "hyle_add_module(${name} TRACE): no HYLE_TRACE use exists in its "
      "selected sources or public headers")
  elseif(_hyle_uses_trace AND NOT ARG_TRACE)
    message(FATAL_ERROR
      "hyle_add_module(${name}): sources/public headers use HYLE_TRACE but "
      "the module did not declare TRACE")
  endif()
  _hyle_module_uses_gate(
    "${name}" "HYLE_TESTING" _hyle_uses_testing ${ARG_SOURCES})
  if(ARG_TESTING AND NOT _hyle_uses_testing)
    message(FATAL_ERROR
      "hyle_add_module(${name} TESTING): no HYLE_TESTING use exists in its "
      "selected sources or public headers")
  elseif(_hyle_uses_testing AND NOT ARG_TESTING)
    message(FATAL_ERROR
      "hyle_add_module(${name}): sources/public headers use HYLE_TESTING but "
      "the module did not declare TESTING")
  endif()

  _hyle_manifest_target_dependencies(
    "${name}" PRIVATE _hyle_private_dependencies)
  _hyle_manifest_target_dependencies(
    "${name}" PUBLIC _hyle_public_dependencies)
  _hyle_manifest_private_build_dependencies(
    "${name}" _hyle_private_build_dependencies)
  _hyle_validate_module_dependencies(
    "${name}" ${_hyle_private_dependencies} ${_hyle_public_dependencies})
  _hyle_validate_module_public_dependencies(
    "${name}" ${_hyle_public_dependencies})

  set(target "hyle_${name}")
  set(_hyle_module_contract_targets hyle_build_contract)
  if(ARG_TRACE)
    list(APPEND _hyle_module_contract_targets hyle_trace)
    set_property(GLOBAL PROPERTY HYLE_TRACE_CONSUMED TRUE)
  endif()
  if(ARG_TESTING)
    list(APPEND _hyle_module_contract_targets hyle_testing)
  endif()
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
      PRIVATE ${_hyle_private_build_dependencies}
      PRIVATE ${_hyle_private_dependencies}
      PUBLIC ${_hyle_public_dependencies})
    # Trace-aware modules see one gate (defines + tracy include when ON). PUBLIC
    # propagation makes headers that inline trace state agree with consumers --
    # an ODR requirement. Only modules that explicitly declare TRACE acquire
    # this edge; OFF (default) carries nothing and has zero product cost.
    target_link_libraries(${target} PUBLIC ${_hyle_module_contract_targets})
    # HYLE_TESTING uses the same PUBLIC/ODR rationale: an instrumented module's
    # sources, test executable and consumers must agree on test-only members.
    # Only modules that explicitly declare TESTING acquire it; shipped builds
    # carry no definition.
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
    if((_hyle_private_dependencies OR _hyle_private_build_dependencies) AND
       NOT ARG_OPTIONAL_ABSENT)
      message(FATAL_ERROR
        "hyle_add_module(${name}): header-only modules cannot have PRIVATE "
        "product/build dependencies: ${_hyle_private_dependencies};"
        "${_hyle_private_build_dependencies}")
    endif()
    target_include_directories(${target} INTERFACE
      "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
      "$<INSTALL_INTERFACE:${HYLE_INSTALL_INCLUDEDIR}>")
    target_compile_features(${target} INTERFACE cxx_std_23)  # header-only modules too
    target_link_libraries(${target} INTERFACE
      ${_hyle_public_dependencies}
      ${_hyle_module_contract_targets})
    if(ARG_OPTIONAL_ABSENT)
      set_target_properties(${target} PROPERTIES HYLE_OPTIONAL_ABSENT TRUE)
    endif()
  endif()

  unset(_hyle_module_contract_targets)
  unset(_hyle_uses_trace)
  unset(_hyle_uses_testing)

  set_target_properties(${target} PROPERTIES
    EXPORT_NAME "_factor_${name}"
    HYLE_FACTOR_ID "${name}"
    FOLDER "hyle/modules")
  if(ARG_SOURCES AND NOT HYLE_STATIC)
    set_target_properties(${target} PROPERTIES
      # A shared factor exports only declarations carrying its HYLE_<FACTOR>_API
      # marker. Private definitions and objects pulled from static vendor
      # archives must not become accidental Native FFI on ELF or Mach-O.
      C_VISIBILITY_PRESET hidden
      CXX_VISIBILITY_PRESET hidden
      VISIBILITY_INLINES_HIDDEN YES
      VERSION "${PROJECT_VERSION}"
      SOVERSION "${HYLE_SHARED_SOVERSION}")
  endif()
endfunction()

# Create the consumer-facing product catalog after every selected factor target
# exists. Facades are graph roots only: they own no source or module state and
# cannot appear in factor dependency declarations.
function(hyle_add_product_facades)
  foreach(_product IN LISTS HYLE_KNOWN_PRODUCTS)
    string(TOUPPER "${_product}" _product_upper)
    if(NOT HYLE_BUILD_PRODUCT_${_product_upper})
      continue()
    endif()

    set(_facade_target "hyle_product_${_product}")
    if(TARGET ${_facade_target} OR TARGET hyle::${_product})
      message(FATAL_ERROR
        "HYLE product facade '${_product}' collides with an existing target")
    endif()
    set(_factor_targets "")
    foreach(_factor IN LISTS _HYLE_PRODUCT_FACTORS_${_product_upper})
      if(NOT TARGET hyle_${_factor})
        message(FATAL_ERROR
          "HYLE product '${_product}' is complete in the manifest but factor "
          "target 'hyle_${_factor}' does not exist")
      endif()
      # Some optional route roots have no public surface when absent. Keep
      # their internal graph identity without exporting an empty capability.
      get_target_property(_internal_only hyle_${_factor}
        HYLE_INTERNAL_ONLY_ABSENT)
      if(_internal_only)
        continue()
      endif()
      list(APPEND _factor_targets hyle_${_factor})
    endforeach()

    add_library(${_facade_target} INTERFACE)
    target_link_libraries(${_facade_target} INTERFACE ${_factor_targets})
    set_target_properties(${_facade_target} PROPERTIES
      EXPORT_NAME "${_product}"
      HYLE_PRODUCT_NAME "${_product}"
      FOLDER "hyle/products")
    add_library(hyle::${_product} ALIAS ${_facade_target})
  endforeach()
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

include("${CMAKE_CURRENT_LIST_DIR}/HyleRuntimeStaging.cmake")
