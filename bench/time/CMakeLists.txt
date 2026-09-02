# Host-clock diagnostics plus the qualified normal consumer replay.
add_executable(hyle_time_clock_bench clock_bench.cc)
target_link_libraries(hyle_time_clock_bench PRIVATE hyle_time nanobench)
hyle_apply_warnings(hyle_time_clock_bench)

# Seastar's precise timer route is the same standard steady clock used below.
# Binding its clean source tree keeps that peer statement reviewable without
# pulling a reactor runtime into this source-equivalent host-clock benchmark.
if(DEFINED HYLE_SEASTAR_DIR AND NOT HYLE_SEASTAR_DIR STREQUAL "" AND
   NOT EXISTS "${HYLE_SEASTAR_DIR}/include/seastar/core/timer.hh")
  message(FATAL_ERROR
    "HYLE_SEASTAR_DIR does not contain include/seastar/core/timer.hh")
endif()
add_executable(hyle_time_host_consumer_bench host_consumer_bench.cc)
target_link_libraries(hyle_time_host_consumer_bench PRIVATE
  hyle_time nanobench)
hyle_apply_warnings(hyle_time_host_consumer_bench)
set_property(TARGET hyle_time_host_consumer_bench
  PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)

if(HYLE_STATIC)
  add_executable(hyle_time_host_consumer_bench_placement
    host_consumer_bench.cc)
  target_link_libraries(hyle_time_host_consumer_bench_placement PRIVATE
    hyle_time nanobench)
  if(MSVC)
    target_link_options(hyle_time_host_consumer_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hyle_time_host_consumer_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR
      "time host consumer placement perturbation is unsupported by this compiler")
  endif()
  hyle_apply_warnings(hyle_time_host_consumer_bench_placement)
  set_property(TARGET hyle_time_host_consumer_bench_placement
    PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
  hyle_register_bench(time hyle_time_host_consumer_bench_placement)
endif()

hyle_register_benches(time
  hyle_time_clock_bench
  hyle_time_host_consumer_bench)
