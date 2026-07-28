# misc-module bench (gap-fill #37): util/random (xoshiro256++ vs std::mt19937_64)
# and time/clock (monotonic/realtime vs std::chrono). obs has a dedicated runner.
if(TARGET hyle_time AND TARGET hyle_util)
  add_executable(hyle_misc_bench "${CMAKE_CURRENT_SOURCE_DIR}/../misc/misc_bench.cc")
  target_link_libraries(hyle_misc_bench PRIVATE hyle_time hyle_util nanobench)
  hyle_apply_warnings(hyle_misc_bench)
  hyle_register_benches(integration hyle_misc_bench)
endif()
