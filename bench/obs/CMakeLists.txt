# obs uses its own actual-workload certificate: gated formatting, structured
# records, fanout scaling, C Status seam and multi-producer registry reads.
add_executable(hyle_obs_bench obs_bench.cc)
target_link_libraries(hyle_obs_bench PRIVATE hyle_obs nanobench)
hyle_apply_warnings(hyle_obs_bench)

if(HYLE_BENCH_ABSL)
  if(TARGET hyle_obs_bench)
    target_compile_definitions(hyle_obs_bench PRIVATE HYLE_BENCH_ABSL)
    target_link_libraries(hyle_obs_bench PRIVATE absl::log absl::log_sink)
  endif()
endif()

hyle_register_benches(obs hyle_obs_bench)
