hyle_add_header_self_contained_test(platform)
hyle_add_test(platform SOURCES wait_test.cc thread_test.cc dynlib_test.cc fs_test.cc virtual_memory_test.cc numa_test.cc mapped_file_test.cc cpu_test.cc convenience_test.cc)
target_include_directories(hyle_platform_test PRIVATE
  "${PROJECT_SOURCE_DIR}/src/platform")
