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
hyle_add_test(error SOURCES error/status_test.cc)
hyle_add_test(platform SOURCES platform/wait_test.cc platform/thread_test.cc)
hyle_add_test(mem SOURCES mem/allocator_test.cc)
hyle_add_test(container SOURCES container/bounded_queue_test.cc DEPS hyle_platform)
