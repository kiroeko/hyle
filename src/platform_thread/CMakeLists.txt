hyle_add_module(platform_thread SOURCES thread.cc wait.cc)

# WaitOnAddress / WakeByAddress* are exported from Synchronization.lib.
if(WIN32)
  target_link_libraries(hyle_platform_thread PRIVATE Synchronization)
endif()
