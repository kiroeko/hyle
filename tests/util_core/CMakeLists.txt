hyle_add_header_self_contained_test(util_core
  HEADER_DIR util
  HEADERS
    borrow.h
    function_ref.h
    random.h
    ref_counted.h
    scope_guard.h
    spin_lock.h
    type_name.h
    unique_resource.h)
hyle_add_test(util_core
  SOURCES
    borrow_test.cc
    function_ref_test.cc
    random_test.cc
    ref_counted_test.cc
    scope_guard_test.cc
    spin_lock_test.cc
    type_name_test.cc
    unique_resource_test.cc)

# Structural probe compiled with the selected consumer toolchain. It is not an
# installed artifact; closure evidence inspects its optimized object to verify
# that the thread-confined counter is non-atomic and NotNull unwrap is identity.
add_library(hyle_util_core_codegen_probe OBJECT codegen_probe.cc)
set_target_properties(hyle_util_core_codegen_probe PROPERTIES
  FOLDER "hyle/tests/util_core"
  INTERPROCEDURAL_OPTIMIZATION OFF)
target_link_libraries(hyle_util_core_codegen_probe PRIVATE hyle_util_core)
hyle_apply_warnings(hyle_util_core_codegen_probe)
hyle_ensure_module_test_target(util_core)
add_dependencies(hyle_util_core_tests hyle_util_core_codegen_probe)

# Scope/resource headers must remain usable in projects that disable language
# exceptions. Conditional failure guards are simply never triggered there.
add_library(hyle_util_core_no_exceptions OBJECT no_exceptions_test.cc)
target_link_libraries(hyle_util_core_no_exceptions PRIVATE hyle_util_core)
if(MSVC)
  target_compile_options(hyle_util_core_no_exceptions PRIVATE /EHs-c-)
else()
  target_compile_options(hyle_util_core_no_exceptions PRIVATE -fno-exceptions)
endif()
hyle_apply_warnings(hyle_util_core_no_exceptions)
add_dependencies(hyle_util_core_tests hyle_util_core_no_exceptions)
