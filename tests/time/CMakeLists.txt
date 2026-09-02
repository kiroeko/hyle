hyle_add_header_self_contained_test(time)
hyle_add_test(time SOURCES header_test.cc clock_test.cc interop_test.cc
  manual_clock_test.cc macro_composition_test.cc unix_timestamp_test.cc)
target_include_directories(hyle_time_test PRIVATE "${PROJECT_SOURCE_DIR}/src/time")

# Compile the frozen ordinary phase through both Instant and direct chrono.
# This maintenance-only object never enters an installed artifact; on MSVC an
# optimized-object audit prevents the extra nested clock-call shell from
# returning without imposing /Ob3 on unrelated consumers.
add_library(hyle_time_codegen_probe OBJECT codegen_probe.cc)
set_target_properties(hyle_time_codegen_probe PROPERTIES
  FOLDER "hyle/tests/time"
  INTERPROCEDURAL_OPTIMIZATION OFF)
target_link_libraries(hyle_time_codegen_probe PRIVATE hyle_time)
hyle_apply_warnings(hyle_time_codegen_probe)
hyle_ensure_module_test_target(time)
add_dependencies(hyle_time_tests hyle_time_codegen_probe)
if(MSVC AND HYLE_SANITIZER STREQUAL "off")
  add_test(NAME hyle_time_codegen_test
    COMMAND "${Python3_EXECUTABLE}"
            "${HYLE_SOURCE_DIR}/scripts/time-codegen-audit.py"
            --object "$<TARGET_OBJECTS:hyle_time_codegen_probe>"
            --configuration "$<CONFIG>")
  set_tests_properties(hyle_time_codegen_test PROPERTIES
    LABELS "time;codegen"
    SKIP_RETURN_CODE 77)
endif()

if(TARGET hyle_time_test AND HYLE_TEST_INSTRUMENTATION AND
    HYLE_TARGET_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  # A fresh process is required because profiling-clock calibration is cached.
  # The force gate is compiled only under HYLE_TESTING and cannot affect a
  # shipped library.
  add_test(NAME hyle_time_tsc_calibration_test
           COMMAND hyle_time_test
                   --gtest_filter=Clock.ProfilingClockTimingCalibrationIsLowCpu)
  set_tests_properties(hyle_time_tsc_calibration_test PROPERTIES
    ENVIRONMENT "HYLE_TIME_FORCE_TSC_TIMING_CALIBRATION=1"
    LABELS "time")
endif()
