hyle_add_test(core SOURCES core_test.cc)
hyle_add_header_self_contained_test(core NO_C)

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
            --architecture "${CMAKE_SYSTEM_PROCESSOR}"
            --configuration "$<CONFIG>")
  set_tests_properties(hyle_core_codegen_test PROPERTIES
    LABELS "core;codegen"
    SKIP_RETURN_CODE 77)
endif()
