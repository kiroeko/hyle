# obs uses its own actual-workload certificate: gated formatting, structured
# records, fanout scaling, C Status seam and multi-producer registry reads.
add_executable(hyle_obs_bench obs_bench.cc)
target_link_libraries(hyle_obs_bench PRIVATE hyle_obs nanobench)
hyle_apply_warnings(hyle_obs_bench)

# Tracy-upgrade evidence exists only in a trace-on build.  It compares HYLE's
# public macro seam with direct Tracy 0.14.1 macros under identical event
# contracts; direct backend use remains confined to this validation target.
if(TARGET hyle_tracy_client)
  add_executable(hyle_obs_tracy_bench tracy_bench.cc)
  target_link_libraries(hyle_obs_tracy_bench PRIVATE hyle_obs nanobench)
  hyle_apply_warnings(hyle_obs_tracy_bench)
  set_property(TARGET hyle_obs_tracy_bench
    PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)

  if(HYLE_STATIC)
    add_executable(hyle_obs_tracy_bench_placement tracy_bench.cc)
    target_link_libraries(hyle_obs_tracy_bench_placement PRIVATE
      hyle_obs nanobench)
    if(MSVC)
      target_link_options(hyle_obs_tracy_bench_placement PRIVATE
        /FUNCTIONPADMIN:64)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(hyle_obs_tracy_bench_placement PRIVATE
        -falign-functions=64)
    else()
      message(FATAL_ERROR
        "obs trace placement perturbation is unsupported by this compiler")
    endif()
    hyle_apply_warnings(hyle_obs_tracy_bench_placement)
    set_property(TARGET hyle_obs_tracy_bench_placement
      PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
  endif()
endif()

if(HYLE_BENCH_ABSL)
  if(TARGET hyle_obs_bench)
    target_compile_definitions(hyle_obs_bench PRIVATE HYLE_BENCH_ABSL)
    target_link_libraries(hyle_obs_bench PRIVATE absl::log absl::log_sink)
    hyle_mark_absl_benchmark_consumer(hyle_obs_bench)
  endif()
endif()

hyle_register_benches(obs hyle_obs_bench)
if(TARGET hyle_obs_tracy_bench)
  hyle_register_benches(obs hyle_obs_tracy_bench)
endif()
if(TARGET hyle_obs_tracy_bench_placement)
  hyle_register_benches(obs hyle_obs_tracy_bench_placement)
endif()
