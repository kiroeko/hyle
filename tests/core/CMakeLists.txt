hyle_add_test(core SOURCES core_test.cc intrusive_list_test.cc)
# Compile the Native FFI schema with a C compiler and call every real core
# symbol. This is an ABI conformance fixture, not a source-level C product API.
hyle_add_c_test(core c_core_test.c)
hyle_add_c_surface_test(core)
hyle_add_header_self_contained_test(core)

# Test-only DSO for the managed bootstrap rejection path.  It intentionally
# exports no product symbol and is never installed.
if(TARGET hyle_core AND NOT HYLE_STATIC)
  add_library(hyle_core_ffi_incompatible_fixture SHARED EXCLUDE_FROM_ALL
              ffi_incompatible_fixture.c)
  set_target_properties(hyle_core_ffi_incompatible_fixture PROPERTIES
    FOLDER "hyle/tests/core"
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
  hyle_apply_warnings(hyle_core_ffi_incompatible_fixture)
  hyle_ensure_module_test_target(core)
  add_dependencies(hyle_core_tests hyle_core_ffi_incompatible_fixture)
endif()
