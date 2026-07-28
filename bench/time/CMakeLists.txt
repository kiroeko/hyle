# time timer benches: algorithm/rival certificate plus normal public-entry
# workloads. Own executables.
add_executable(hyle_timer_queue_bench timer_queue_bench.cc)
target_link_libraries(hyle_timer_queue_bench PRIVATE hyle_time nanobench)
hyle_apply_warnings(hyle_timer_queue_bench)

add_executable(hyle_time_normal_bench normal_usage_bench.cc)
target_link_libraries(hyle_time_normal_bench PRIVATE hyle_time nanobench)
hyle_apply_warnings(hyle_time_normal_bench)

hyle_register_benches(time
  hyle_timer_queue_bench
  hyle_time_normal_bench)
