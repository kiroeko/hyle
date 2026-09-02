hyle_add_header_self_contained_test(container_intrusive
  HEADER_DIR container
  HEADERS intrusive_forward_list.h intrusive_hlist.h intrusive_list.h)
hyle_add_test(container_intrusive SOURCES
  intrusive_extensions_test.cc
  intrusive_list_test.cc)
if(MSVC)
  # These tests exercise the public comparator/predicate exception guarantee.
  # The product target deliberately inherits the embedding TU's exception
  # mode, so enable unwind semantics on this test consumer only.
  target_compile_options(hyle_container_intrusive_test PRIVATE /EHsc)
endif()

add_executable(hyle_container_intrusive_no_exceptions_test
  no_exceptions_test.cc)
set_target_properties(hyle_container_intrusive_no_exceptions_test PROPERTIES
  FOLDER "hyle/tests/container_intrusive"
  HYLE_TEST_MODULE "container_intrusive")
target_link_libraries(hyle_container_intrusive_no_exceptions_test PRIVATE
  hyle_container_intrusive)
if(MSVC)
  target_compile_options(hyle_container_intrusive_no_exceptions_test PRIVATE
    /EHs-c-)
else()
  target_compile_options(hyle_container_intrusive_no_exceptions_test PRIVATE
    -fno-exceptions)
endif()
hyle_apply_warnings(hyle_container_intrusive_no_exceptions_test)
hyle_apply_sanitizer(hyle_container_intrusive_no_exceptions_test)
hyle_ensure_module_test_target(container_intrusive)
add_dependencies(hyle_container_intrusive_tests
  hyle_container_intrusive_no_exceptions_test)
add_test(NAME hyle_container_intrusive_no_exceptions_test
  COMMAND hyle_container_intrusive_no_exceptions_test)
set_tests_properties(hyle_container_intrusive_no_exceptions_test PROPERTIES
  LABELS "container_intrusive;delivery")

# Compile the header-only list through stable noinline C probes, then inspect
# that exact consumer object.  This is a structural regression gate rather
# than a timer benchmark; it deliberately avoids absolute instruction counts.
add_library(hyle_container_intrusive_codegen_probe OBJECT codegen_probe.cc)
set_target_properties(hyle_container_intrusive_codegen_probe PROPERTIES
  FOLDER "hyle/tests/container_intrusive"
  INTERPROCEDURAL_OPTIMIZATION OFF)
target_link_libraries(hyle_container_intrusive_codegen_probe
  PRIVATE hyle_container_intrusive)
hyle_apply_warnings(hyle_container_intrusive_codegen_probe)

add_executable(hyle_container_intrusive_codegen_probe_runner
  codegen_probe_runner.cc
  $<TARGET_OBJECTS:hyle_container_intrusive_codegen_probe>)
set_target_properties(hyle_container_intrusive_codegen_probe_runner PROPERTIES
  FOLDER "hyle/tests/container_intrusive"
  HYLE_TEST_MODULE "container_intrusive")
target_link_libraries(hyle_container_intrusive_codegen_probe_runner
  PRIVATE hyle_container_intrusive)
hyle_apply_warnings(hyle_container_intrusive_codegen_probe_runner)

hyle_ensure_module_test_target(container_intrusive)
add_dependencies(hyle_container_intrusive_tests
  hyle_container_intrusive_codegen_probe
  hyle_container_intrusive_codegen_probe_runner)

add_test(NAME hyle_container_intrusive_codegen_audit_parser_test
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/codegen_audit_test.py")
set_tests_properties(hyle_container_intrusive_codegen_audit_parser_test
  PROPERTIES LABELS "container_intrusive;codegen;tooling")

add_test(NAME hyle_container_intrusive_perf_evidence_parser_test
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/perf_evidence_parser_test.py")
set_tests_properties(hyle_container_intrusive_perf_evidence_parser_test
  PROPERTIES LABELS "container_intrusive;performance;tooling")

# Sanitizer instrumentation intentionally adds helper calls and therefore has
# a different machine contract.  Ordinary Debug remains registered: the audit
# runs the layout/operation probe and validates all symbols, while reserving the
# helper-free and optimizer-assumption checks for NDEBUG configurations.
if(HYLE_SANITIZER STREQUAL "off")
  add_test(NAME hyle_container_intrusive_codegen_test
    COMMAND "${Python3_EXECUTABLE}"
            "${HYLE_SOURCE_DIR}/scripts/container-intrusive-codegen-audit.py"
            --object "$<TARGET_OBJECTS:hyle_container_intrusive_codegen_probe>"
            --runner "$<TARGET_FILE:hyle_container_intrusive_codegen_probe_runner>"
            --architecture "${HYLE_TARGET_PROCESSOR}"
            --configuration "$<CONFIG>")
  set_tests_properties(hyle_container_intrusive_codegen_test PROPERTIES
    LABELS "container_intrusive;codegen")
endif()
