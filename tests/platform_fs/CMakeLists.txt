hyle_add_header_self_contained_test(platform_fs
  HEADER_DIR platform
  HEADERS fs.h fs_convenience.h fs_visibility.h)
hyle_add_test(platform_fs SOURCES fs_test.cc convenience_test.cc
  DEPS hyle_platform)
target_include_directories(hyle_platform_fs_test PRIVATE
  "${PROJECT_SOURCE_DIR}/src/platform_fs")
