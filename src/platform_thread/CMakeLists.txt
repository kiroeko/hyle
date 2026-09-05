hyle_add_module(platform_thread SOURCES thread.cc wait.cc)

# WaitOnAddress / WakeByAddress* are exported from Synchronization.lib.
if(WIN32)
  target_link_libraries(hyle_platform_thread PRIVATE Synchronization)
endif()

hyle_register_package_dependency(
  NAME platform_thread_rust_parker
  OWNER platform_thread
  LOCKED_DEPENDENCIES rust_parker
  RUNTIME_FILE_SPECS
    "${PROJECT_SOURCE_DIR}/third_party/ported/LICENSE-RUST|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|rust-parker-LICENSE-MIT")
