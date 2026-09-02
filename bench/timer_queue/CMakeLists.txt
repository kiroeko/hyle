# Timer data-structure diagnostics plus the qualified scheduler lifecycle.
add_executable(hyle_timer_queue_diagnostic_bench timer_queue_bench.cc)
target_link_libraries(hyle_timer_queue_diagnostic_bench PRIVATE
  hyle_timer_queue nanobench)
hyle_apply_warnings(hyle_timer_queue_diagnostic_bench)

add_executable(hyle_timer_queue_normal_bench normal_usage_bench.cc)
target_link_libraries(hyle_timer_queue_normal_bench PRIVATE
  hyle_timer_queue nanobench)
hyle_apply_warnings(hyle_timer_queue_normal_bench)

# Source-primary scheduler timer lifecycle replay. The Seastar algorithm is a
# reviewed local port so both sides share one TU/toolchain; qualification still
# binds the pinned clean checkout through HYLE_SEASTAR_DIR.
if(DEFINED HYLE_SEASTAR_DIR AND NOT HYLE_SEASTAR_DIR STREQUAL "" AND
   NOT EXISTS "${HYLE_SEASTAR_DIR}/include/seastar/core/timer-set.hh")
  message(FATAL_ERROR
    "HYLE_SEASTAR_DIR does not contain include/seastar/core/timer-set.hh")
endif()
add_executable(hyle_timer_queue_consumer_bench consumer_bench.cc)
target_link_libraries(hyle_timer_queue_consumer_bench PRIVATE
  hyle_timer_queue nanobench)
hyle_apply_warnings(hyle_timer_queue_consumer_bench)
set_property(TARGET hyle_timer_queue_consumer_bench
  PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)

if(HYLE_STATIC)
  add_executable(hyle_timer_queue_consumer_bench_placement consumer_bench.cc)
  target_link_libraries(hyle_timer_queue_consumer_bench_placement PRIVATE
    hyle_timer_queue nanobench)
  if(MSVC)
    target_link_options(hyle_timer_queue_consumer_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hyle_timer_queue_consumer_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR
      "timer_queue consumer placement perturbation is unsupported by this compiler")
  endif()
  hyle_apply_warnings(hyle_timer_queue_consumer_bench_placement)
  set_property(TARGET hyle_timer_queue_consumer_bench_placement
    PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
  hyle_register_bench(timer_queue
    hyle_timer_queue_consumer_bench_placement)
endif()

hyle_register_benches(timer_queue
  hyle_timer_queue_diagnostic_bench
  hyle_timer_queue_normal_bench
  hyle_timer_queue_consumer_bench)
