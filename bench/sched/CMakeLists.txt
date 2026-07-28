# sched bench: fiber switch (~23ns) + spawn throughput vs std::async (~7-9x). Now
# compiled into the default bench build (no longer withheld). It once crashed at
# teardown with mimalloc ON, but an exhaustive hunt (docs/debug/ki-1-crash-hunt.md)
# RULED OUT a HYLE logic bug -- TSan (no source race) + MI_DEBUG=3 full heap checks
# (no corruption) are clean under churn + oversubscription + 24G memory pressure,
# and the original crasher is NOT reproducible on native ext4. Evidence (SIGILL +
# clean instrumented runs) points to a 9p/WSL paging artifact, not this code. Kept
# IN the build on purpose: if it ever recurs, we investigate head-on, not hide it.
  hyle_add_bench(sched SOURCES scheduler_bench.cc)

# sched-05 fiber-aware sync primitives bench (own exe: scheduler_bench.cc already
# owns main(), so this is a second target rather than another source in it).
add_executable(hyle_sched_sync_bench sync_bench.cc)
target_link_libraries(hyle_sched_sync_bench PRIVATE hyle_sched nanobench)
hyle_apply_warnings(hyle_sched_sync_bench)

# sched-09 concurrency-primitive bench: Semaphore/RwLock/oneshot/watch/broadcast
# (the sched-09 additions sync_bench doesn't cover). Own exe. Uses Scheduler+Fiber
# so it needs fcontext like the rest of the fiber-driven sched benches.
if(HYLE_HAS_FCONTEXT)
  add_executable(hyle_sched_sync_ext_bench sync_ext_bench.cc)
  target_link_libraries(hyle_sched_sync_ext_bench PRIVATE hyle_sched nanobench)
  hyle_apply_warnings(hyle_sched_sync_ext_bench)
endif()

# sched-11 spawn_blocking bench: the blocking-offload pool round-trip + fan-out. HYLE
# rows always build (into hyle_sched); the marl blocking_call comparison row is gated
# -DHYLE_BENCH_MARL (wired in the marl block below). Own exe; needs fcontext.
if(HYLE_HAS_FCONTEXT)
  add_executable(hyle_sched_blocking_bench blocking_bench.cc)
  target_link_libraries(hyle_sched_blocking_bench PRIVATE hyle_sched nanobench)
  hyle_apply_warnings(hyle_sched_blocking_bench)
endif()

# sched-07 cancellation-tree bench: is_cancelled() hot-path (vs bare atomic) +
# cancel() propagation O(N) (wide + deep trees) + derive() per-child cost. Own exe;
# CancelScope references Fiber so it needs fcontext like the rest of sched.
if(HYLE_HAS_FCONTEXT)
  add_executable(hyle_sched_cancel_bench cancel_bench.cc)
  target_link_libraries(hyle_sched_cancel_bench PRIVATE hyle_sched nanobench)
  hyle_apply_warnings(hyle_sched_cancel_bench)
endif()

# sched-10 JoinHandle/spawn_task/when_all/when_any bench: proves the structured-
# concurrency layer is ~zero-overhead vs the raw fiber+WaitGroup(1)+slot it wraps.
# Own exe (scheduler_bench owns main). Needs fcontext like the rest of sched.
if(HYLE_HAS_FCONTEXT)
  add_executable(hyle_sched_join_bench join_bench.cc)
  target_link_libraries(hyle_sched_join_bench PRIVATE hyle_sched nanobench)
  hyle_apply_warnings(hyle_sched_join_bench)
endif()

# scheduler-layer gap benches (work-stealing scaling, spawn_on affinity,
# coroutine glue vs raw fiber spine, fiber sleep). Complements scheduler_bench
# (raw primitives) + queue_sota_bench (the work-stealing DATA STRUCTURE) by
# measuring SCHEDULER-LEVEL behaviour. Own exe. Needs fcontext (coro + timer).
if(HYLE_HAS_FCONTEXT)
  add_executable(hyle_sched_ext_bench scheduler_ext_bench.cc)
  target_link_libraries(hyle_sched_ext_bench PRIVATE hyle_sched nanobench)
  hyle_apply_warnings(hyle_sched_ext_bench)
endif()

# -DHYLE_BENCH_MARL=ON; point HYLE_MARL_DIR at a cloned marl (sibling default).
option(HYLE_BENCH_MARL "build the fiber-sync-vs-marl parity bench" OFF)
if(HYLE_BENCH_MARL)
  set(HYLE_MARL_DIR "${PROJECT_SOURCE_DIR}/../marl" CACHE PATH "cloned marl")
  if(NOT EXISTS "${HYLE_MARL_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "HYLE_BENCH_MARL=ON but no marl at ${HYLE_MARL_DIR}")
  endif()
  add_subdirectory("${HYLE_MARL_DIR}" "${HYLE_BINARY_DIR}/_bench_deps/marl" EXCLUDE_FROM_ALL)
  add_executable(hyle_sched_marl_bench sync_marl_bench.cc)
  target_link_libraries(hyle_sched_marl_bench PRIVATE hyle_sched nanobench marl)
  hyle_apply_warnings(hyle_sched_marl_bench)
  # sched-11 blocking_bench's marl comparison row (marl::blocking_call = new-thread-
  # per-call; the HYLE-pool-vs-marl-new-thread head-to-head). Same gate; links marl.
  if(TARGET hyle_sched_blocking_bench)
    target_compile_definitions(hyle_sched_blocking_bench PRIVATE HYLE_BENCH_MARL)
    target_link_libraries(hyle_sched_blocking_bench PRIVATE marl)
  endif()
  hyle_register_benches(sched hyle_sched_marl_bench)
endif()

hyle_register_benches(sched
  hyle_sched_sync_bench
  hyle_sched_sync_ext_bench
  hyle_sched_blocking_bench
  hyle_sched_cancel_bench
  hyle_sched_join_bench
  hyle_sched_ext_bench)
