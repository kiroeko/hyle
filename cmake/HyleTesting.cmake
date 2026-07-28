# HyleTesting.cmake -- the test-only interface gate (HYLE_TESTING).
#
# Some APIs exist ONLY to let tests reach an internal path that no production caller
# needs (e.g. a controllable Event that reports NotReady N times so the host backend can
# drive the GpuPoller suspend->wake chain without a real GPU). Such an interface must NOT
# ship in the product: it is compiled ONLY when tests are being built.
#
# HYLE_TESTING is that gate. It requires both HYLE_BUILD_TESTS and the explicit
# HYLE_TEST_INSTRUMENTATION option.  The default instrumented test build reaches
# fault-injection/structural diagnostics; a second OFF build runs the ordinary
# suite with the exact shipping layout and hot paths.  Shipping builds have
# HYLE_BUILD_TESTS=OFF, so every block vanishes with zero product cost.
#
# ODR: a test-only MEMBER changes a class's layout, so the production .cc that compiles
# the class and the test TU that includes the same header MUST agree on whether the hook
# exists. So the define is carried by an INTERFACE target every module links PUBLIC (like
# hyle_trace) -- the module's own sources, the test exe, and any consumer all see the same
# HYLE_TESTING state. A per-target private define would be an ODR hazard.
#
# CONTRACT for code behind this gate (see usage-guide "test-only interfaces"):
#   1. wrap the declaration AND definition in `#if defined(HYLE_TESTING)`;
#   2. the function/member comment MUST start with "TEST-ONLY:" stating why it is not
#      production API -- an ungated raw/helper API with no such note is a bug, not a hook.

# hyle_testing is an INTERFACE target every module links (PUBLIC), mirroring hyle_trace.
# It exists unconditionally so module CMakeLists can link it with no if() at the call site;
# it carries the HYLE_TESTING define only for an instrumented test build.
add_library(hyle_testing INTERFACE)
set_target_properties(hyle_testing PROPERTIES FOLDER "hyle/foundation")

if(HYLE_BUILD_TESTS AND HYLE_TEST_INSTRUMENTATION)
  # Installable builds forbid HYLE_BUILD_TESTS, but keep the definition explicitly
  # build-only as a second line of defence against exporting a test ABI/layout.
  target_compile_definitions(hyle_testing INTERFACE "$<BUILD_INTERFACE:HYLE_TESTING>")
  message(STATUS "HYLE_TESTING = ON (test-only interfaces compiled; not for shipped builds)")
elseif(HYLE_BUILD_TESTS)
  message(STATUS "HYLE_TESTING = OFF (tests exercise the shipping layout/hot paths)")
else()
  message(STATUS "HYLE_TESTING = OFF (test-only interfaces stripped -- zero in shipped lib)")
endif()
set_target_properties(hyle_testing PROPERTIES EXPORT_NAME "_testing_config")
