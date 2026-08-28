hyle_add_test(math SOURCES scalar_test.cc)
hyle_add_header_self_contained_test(math)

# P0 structural probe. It is a test-only consumer object, never installed;
# reviewers inspect the optimized object/assembly on the three closure tools.
add_library(hyle_math_codegen_probe OBJECT codegen_probe.cc)
set_target_properties(hyle_math_codegen_probe PROPERTIES
  FOLDER "hyle/tests/math"
  INTERPROCEDURAL_OPTIMIZATION OFF)
target_link_libraries(hyle_math_codegen_probe PRIVATE hyle_math)
hyle_apply_warnings(hyle_math_codegen_probe)
hyle_ensure_module_test_target(math)
add_dependencies(hyle_math_tests hyle_math_codegen_probe)
