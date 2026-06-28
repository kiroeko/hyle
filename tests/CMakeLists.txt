include(FetchContent)

# GoogleTest (charter: unit test framework). Pinned tag.
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2)
# Use shared CRT on MSVC to match our build.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

hyle_add_test(core SOURCES core/scaffolding_test.cc)
hyle_add_test(error SOURCES error/status_test.cc error/error_handler_test.cc)
hyle_add_test(platform SOURCES platform/wait_test.cc platform/thread_test.cc platform/dynlib_test.cc)
hyle_add_test(mem SOURCES mem/allocator_test.cc mem/memory_arena_test.cc mem/pool_stack_test.cc mem/alignment_test.cc mem/pmr_test.cc mem/growable_pool_test.cc DEPS hyle_platform)
hyle_add_test(util SOURCES util/ref_counted_test.cc util/scope_guard_test.cc util/type_name_test.cc util/registry_test.cc util/random_test.cc util/spin_lock_test.cc util/demangle_test.cc DEPS hyle_container)
hyle_add_test(time SOURCES time/clock_test.cc)
hyle_add_test(math SOURCES math/scalar_test.cc)
hyle_add_test(container SOURCES container/bounded_queue_test.cc container/work_stealing_deque_test.cc container/ws_fifo_queue_test.cc container/hash_test.cc container/carray_test.cc container/small_vector_test.cc container/probe_test.cc container/swiss_table_test.cc container/string_ref_test.cc container/string_buf_test.cc container/concurrent_vector_test.cc container/string_algo_test.cc container/path_test.cc DEPS hyle_platform)
if(HYLE_HAS_FCONTEXT)
  hyle_add_test(sched SOURCES sched/fiber_test.cc sched/run_queue_test.cc sched/scheduler_test.cc)
endif()
