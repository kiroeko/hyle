hyle_add_test(core SOURCES core_test.cc)
hyle_add_header_self_contained_test(core NO_C)

add_test(NAME hyle_core_codegen_audit_parser_test
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/codegen_audit_test.py")
set_tests_properties(hyle_core_codegen_audit_parser_test PROPERTIES
  LABELS "core;codegen;tooling")

# A header-inline failure sink must not become a preemptible implementation
# symbol in every shared module that uses HYLE_FATAL_ASSERT.
if(NOT HYLE_STATIC)
  add_library(hyle_core_symbol_visibility_probe SHARED
              symbol_visibility_probe.cc)
  set_target_properties(hyle_core_symbol_visibility_probe PROPERTIES
    FOLDER "hyle/tests/core")
  target_link_libraries(hyle_core_symbol_visibility_probe PRIVATE hyle_core)
  hyle_apply_warnings(hyle_core_symbol_visibility_probe)
  hyle_ensure_module_test_target(core)
  add_dependencies(hyle_core_tests hyle_core_symbol_visibility_probe)
  add_test(NAME hyle_core_symbol_visibility_test
    COMMAND "${Python3_EXECUTABLE}"
            "${HYLE_SOURCE_DIR}/scripts/core-symbol-visibility-audit.py"
            --library "$<TARGET_FILE:hyle_core_symbol_visibility_probe>")
  set_tests_properties(hyle_core_symbol_visibility_test PROPERTIES
    LABELS "core;symbols")
endif()

# Structural codegen probe: compiler/cache primitives are too small for a
# meaningful timer benchmark, so inspect the optimized object that consumers
# actually compile. The target never enters an installed artifact.
add_library(hyle_core_codegen_probe OBJECT codegen_probe.cc)
set_target_properties(hyle_core_codegen_probe PROPERTIES
  FOLDER "hyle/tests/core"
  INTERPROCEDURAL_OPTIMIZATION OFF)
target_link_libraries(hyle_core_codegen_probe PRIVATE hyle_core)
hyle_apply_warnings(hyle_core_codegen_probe)
hyle_ensure_module_test_target(core)
add_dependencies(hyle_core_tests hyle_core_codegen_probe)
if(HYLE_SANITIZER STREQUAL "off")
  add_test(NAME hyle_core_codegen_test
    COMMAND "${Python3_EXECUTABLE}"
            "${HYLE_SOURCE_DIR}/scripts/core-codegen-audit.py"
            --object "$<TARGET_OBJECTS:hyle_core_codegen_probe>"
            --architecture "${HYLE_TARGET_PROCESSOR}"
            --configuration "$<CONFIG>")
  set_tests_properties(hyle_core_codegen_test PROPERTIES
    LABELS "core;codegen"
    SKIP_RETURN_CODE 77)
endif()
