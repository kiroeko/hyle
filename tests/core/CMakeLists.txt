hyle_add_test(core SOURCES core_test.cc)
hyle_add_test(core NAME release_assert SOURCES release_assert_test.cc
              DEFINES NDEBUG)
hyle_add_header_self_contained_test(core)

# Release-disabled assertions must accept and reject the same C++ expressions
# as their enabled form while evaluating none of them. These are compiler
# contracts, so test them with the configured compiler instead of trying to
# infer validity from a runtime path that intentionally does not exist.
function(hyle_core_expect_release_compile name expected source_variable)
  set(CMAKE_TRY_COMPILE_CONFIGURATION Release)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  try_compile(_hyle_core_compiled
    SOURCE_FROM_VAR "${name}.cc" "${source_variable}"
    CMAKE_FLAGS
      "-DCMAKE_BUILD_TYPE=Release"
      "-DCMAKE_CXX_STANDARD=23"
      "-DCMAKE_CXX_STANDARD_REQUIRED=ON"
      "-DCMAKE_CXX_EXTENSIONS=OFF"
      "-DINCLUDE_DIRECTORIES:STRING=${HYLE_SOURCE_DIR}/include"
    NO_CACHE
    OUTPUT_VARIABLE _hyle_core_compile_output)
  if(expected AND NOT _hyle_core_compiled)
    message(FATAL_ERROR
      "Core assertion compile contract ${name} unexpectedly failed:\n"
      "${_hyle_core_compile_output}")
  elseif(NOT expected AND _hyle_core_compiled)
    message(FATAL_ERROR
      "Core assertion compile contract ${name} unexpectedly compiled")
  endif()
endfunction()

set(_hyle_core_assert_valid [=[
#define NDEBUG
#include "hyle/core/assert.h"
#if HYLE_DEBUG_ASSERT_ENABLED
#error "compile contract requires disabled Debug assertions"
#endif
template <typename T>
void valid_dependent(T value) {
  HYLE_DEBUG_ASSERT(value > T{});
  HYLE_DEBUG_ASSERT_MSG(value > T{}, "valid");
  HYLE_DEBUG_ASSERT_CONTEXT(value > T{}, "valid", HYLE_FATAL_FIELD(value));
}
void valid_context() {
  int hyle_fatal_fields = 7;
  HYLE_FATAL_ASSERT_CONTEXT(true, "valid",
                            HYLE_FATAL_FIELD(hyle_fatal_fields));
  valid_dependent(hyle_fatal_fields);
}
]=])
hyle_core_expect_release_compile(assert_valid TRUE _hyle_core_assert_valid)

set(_hyle_core_assert_non_boolean [=[
#define NDEBUG
#include "hyle/core/assert.h"
struct NotBoolean {};
void invalid_condition() { HYLE_DEBUG_ASSERT(NotBoolean{}); }
]=])
hyle_core_expect_release_compile(
  assert_reject_non_boolean FALSE _hyle_core_assert_non_boolean)

set(_hyle_core_assert_dependent [=[
#define NDEBUG
#include "hyle/core/assert.h"
template <typename T>
void invalid_dependent(T value) {
  HYLE_DEBUG_ASSERT(value.this_member_does_not_exist());
}
template void invalid_dependent<int>(int);
]=])
hyle_core_expect_release_compile(
  assert_reject_dependent FALSE _hyle_core_assert_dependent)

set(_hyle_core_assert_message [=[
#define NDEBUG
#include "hyle/core/assert.h"
struct InvalidMessage {};
template <typename T>
void invalid_message(T value) {
  HYLE_DEBUG_ASSERT_MSG(true, value);
}
template void invalid_message<InvalidMessage>(InvalidMessage);
]=])
hyle_core_expect_release_compile(
  assert_reject_message FALSE _hyle_core_assert_message)

set(_hyle_core_assert_field [=[
#define NDEBUG
#include "hyle/core/assert.h"
template <typename T>
void invalid_field(T value) {
  HYLE_DEBUG_ASSERT_CONTEXT(true, "invalid field", value);
}
template void invalid_field<int>(int);
]=])
hyle_core_expect_release_compile(
  assert_reject_field FALSE _hyle_core_assert_field)

set(_hyle_core_assert_zero_fields [=[
#define NDEBUG
#include "hyle/core/assert.h"
void invalid_zero_fields() {
  HYLE_DEBUG_ASSERT_CONTEXT(true, "zero fields");
}
]=])
hyle_core_expect_release_compile(
  assert_reject_zero_fields FALSE _hyle_core_assert_zero_fields)

set(_hyle_core_assert_nine_fields [=[
#define NDEBUG
#include "hyle/core/assert.h"
void invalid_nine_fields() {
  HYLE_DEBUG_ASSERT_CONTEXT(
      true, "nine fields",
      HYLE_FATAL_NAMED_FIELD("0", 0), HYLE_FATAL_NAMED_FIELD("1", 1),
      HYLE_FATAL_NAMED_FIELD("2", 2), HYLE_FATAL_NAMED_FIELD("3", 3),
      HYLE_FATAL_NAMED_FIELD("4", 4), HYLE_FATAL_NAMED_FIELD("5", 5),
      HYLE_FATAL_NAMED_FIELD("6", 6), HYLE_FATAL_NAMED_FIELD("7", 7),
      HYLE_FATAL_NAMED_FIELD("8", 8));
}
]=])
hyle_core_expect_release_compile(
  assert_reject_nine_fields FALSE _hyle_core_assert_nine_fields)

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
