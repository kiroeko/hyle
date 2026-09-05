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
hyle_add_test(platform SOURCES dynlib_test.cc virtual_memory_test.cc numa_test.cc cpu_test.cc host_test.cc process_test.cc convenience_test.cc windows_macro_composition_test.cc)
target_include_directories(hyle_platform_test PRIVATE
  "${PROJECT_SOURCE_DIR}/src/platform")
