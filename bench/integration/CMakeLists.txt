# misc-module bench (gap-fill #37): util/random (xoshiro256++ vs std::mt19937_64)
# and time/clock (monotonic/realtime vs std::chrono). obs has a dedicated runner.
add_executable(hyle_perf_control_bench perf_control_bench.cc)
target_link_libraries(hyle_perf_control_bench PRIVATE hyle_core nanobench)
hyle_apply_warnings(hyle_perf_control_bench)
set_property(TARGET hyle_perf_control_bench
  PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
hyle_register_benches(integration hyle_perf_control_bench)

if(TARGET hyle_time AND TARGET hyle_util)
  add_executable(hyle_misc_bench "${CMAKE_CURRENT_SOURCE_DIR}/../misc/misc_bench.cc")
  target_link_libraries(hyle_misc_bench PRIVATE hyle_time hyle_util nanobench)
  hyle_apply_warnings(hyle_misc_bench)
  hyle_register_benches(integration hyle_misc_bench)
endif()
