cmake_minimum_required(VERSION 3.25)
project(hyle_sample_sched LANGUAGES CXX ASM)

set(HYLE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(HYLE_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(HYLE_COMPONENTS sched CACHE STRING "" FORCE)
add_subdirectory(../.. hyle_build)

# sched needs fcontext fiber-switch asm, wired only on x86-64 Windows(MASM)/Linux(GAS).
# The hyle::sched target therefore only EXISTS on those platforms (top CMakeLists gates
# it on HYLE_HAS_FCONTEXT). Guard so the sample degrades gracefully elsewhere.
if(TARGET hyle::sched)
  add_executable(sched_sample main.cc)
  target_link_libraries(sched_sample PRIVATE hyle::sched hyle::mem)
  hyle_stage_runtime_dependencies(sched_sample)
else()
  message(WARNING "hyle::sched not available on this platform (needs x86-64 fcontext); "
                  "skipping sched_sample.")
endif()
