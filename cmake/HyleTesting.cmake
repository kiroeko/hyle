# HyleTesting.cmake -- the test-only interface gate (HYLE_TESTING).
#
# Some APIs exist ONLY to let tests reach an internal path that no production caller
# needs (e.g. a controllable Event that reports NotReady N times so the host backend can
# drive the GpuPoller suspend->wake chain without a real GPU). Such an interface must NOT
# ship in the product: it is compiled ONLY when tests are being built.
#
# HYLE_TESTING is that gate. It is DERIVED from HYLE_BUILD_TESTS (not a separate option):
# a test-only interface is by definition needed exactly when we build tests. When a
# consumer integrates HYLE with -DHYLE_BUILD_TESTS=OFF (usage-guide's integration recipe),
# HYLE_TESTING is undefined and every `#if defined(HYLE_TESTING)` block vanishes -- the
# shipped library has ZERO test hooks (same "zero-cost when off" law as HYLE_TRACE).
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
# it carries the HYLE_TESTING define only when HYLE_BUILD_TESTS is ON.
add_library(hyle_testing INTERFACE)

if(HYLE_BUILD_TESTS)
  target_compile_definitions(hyle_testing INTERFACE HYLE_TESTING)
  message(STATUS "HYLE_TESTING = ON (test-only interfaces compiled; not for shipped builds)")
else()
  message(STATUS "HYLE_TESTING = OFF (test-only interfaces stripped -- zero in shipped lib)")
endif()
