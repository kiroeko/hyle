hyle_add_header_self_contained_test(platform
  HEADER_DIR platform
  HEADERS
    cpu.h
    cpu_convenience.h
    dynlib.h
    environment.h
    environment_convenience.h
    host_memory.h
    numa.h
    process.h
    random.h
    virtual_memory.h
    visibility.h)
hyle_add_test(platform SOURCES cgroup_test.cc dynlib_test.cc virtual_memory_test.cc numa_test.cc cpu_test.cc host_test.cc process_test.cc convenience_test.cc windows_macro_composition_test.cc)
target_include_directories(hyle_platform_test PRIVATE
  "${PROJECT_SOURCE_DIR}/src/platform")
if(NOT HYLE_STATIC)
  # Shared libraries deliberately hide parser seams, so tests link the same
  # object directly. Static tests already receive it from hyle_platform and
  # must not add a second definition.
  target_sources(hyle_platform_test PRIVATE
    $<TARGET_OBJECTS:hyle_platform_cgroup_parser>)
endif()
