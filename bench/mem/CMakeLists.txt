hyle_add_bench(mem SOURCES allocator_bench.cc DEPS mimalloc_static)

# arena/pool/stack specialized-allocator cluster vs std::pmr (std's arena/pool, no
# external clone) + raw mimalloc. Own exe (allocator_bench.cc already owns main()).
add_executable(hyle_mem_scenarios_bench scenario_bench.cc)
target_link_libraries(hyle_mem_scenarios_bench PRIVATE hyle_mem nanobench
                                                       mimalloc_static)
if(WIN32)
  target_link_libraries(hyle_mem_scenarios_bench PRIVATE psapi)
endif()
hyle_apply_warnings(hyle_mem_scenarios_bench)

# FiberStackPool is a scheduler-owned mem specialization. Certify it only
# through the real Scheduler + Fiber create/run/destroy lifecycle, with direct
# guarded VM reserve/release as the same-lifecycle baseline.
if(HYLE_HAS_FCONTEXT)
  add_executable(hyle_mem_fiber_stack_bench fiber_stack_bench.cc)
  target_link_libraries(hyle_mem_fiber_stack_bench PRIVATE hyle_sched)
  hyle_apply_warnings(hyle_mem_fiber_stack_bench)
endif()

if(HYLE_BENCH_TBB)
  target_compile_definitions(hyle_mem_scenarios_bench PRIVATE
                             HYLE_BENCH_TBB TBB_PREVIEW_MEMORY_POOL=1)
  target_link_libraries(hyle_mem_scenarios_bench PRIVATE TBB::tbbmalloc
                                                        TBB::tbb)
endif()

hyle_register_benches(mem
  hyle_mem_scenarios_bench)
if(TARGET hyle_mem_fiber_stack_bench)
  hyle_register_benches(mem hyle_mem_fiber_stack_bench)
endif()
