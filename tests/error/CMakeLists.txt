hyle_add_test(error SOURCES status_test.cc ffi_status_test.cc)
# C-compiler/schema/link fixture for the pre-release Native FFI candidate.
hyle_add_c_test(error c_error_test.c)
hyle_add_c_surface_test(error)
hyle_add_header_self_contained_test(error)

# Test-only shared artifacts for the managed direct-FFI journey. The producer
# is the only non-HYLE code allowed to manufacture Status handles: it enters
# through the source-internal status_to_ffi bridge, while Java/.NET consume only
# the six exported hyle_error symbols. Neither fixture is installed.
if(TARGET hyle_error AND NOT HYLE_STATIC)
  add_library(hyle_error_ffi_status_fixture SHARED EXCLUDE_FROM_ALL
              ffi_status_fixture.cc)
  target_link_libraries(hyle_error_ffi_status_fixture PRIVATE hyle_error)
  set_target_properties(hyle_error_ffi_status_fixture PROPERTIES
    FOLDER "hyle/tests/error"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
  hyle_apply_warnings(hyle_error_ffi_status_fixture)

  add_library(hyle_error_ffi_incompatible_fixture SHARED EXCLUDE_FROM_ALL
              ffi_incompatible_fixture.c)
  set_target_properties(hyle_error_ffi_incompatible_fixture PROPERTIES
    FOLDER "hyle/tests/error"
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
  hyle_apply_warnings(hyle_error_ffi_incompatible_fixture)

  hyle_ensure_module_test_target(error)
  add_dependencies(hyle_error_tests
    hyle_error_ffi_status_fixture
    hyle_error_ffi_incompatible_fixture)
endif()
