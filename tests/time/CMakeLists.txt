hyle_add_header_self_contained_test(time)
hyle_add_test(time SOURCES header_test.cc clock_test.cc
  timer_queue_test.cc common_timeout_queue_test.cc
  macro_composition_test.cc)
target_include_directories(hyle_time_test PRIVATE "${PROJECT_SOURCE_DIR}/src/time")
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
