hyle_add_module(platform SOURCES wait.cc thread.cc PUBLIC_DEPS hyle_error)

# WaitOnAddress / WakeByAddress* are exported from Synchronization.lib (API set).
if(WIN32)
  target_link_libraries(hyle_platform PRIVATE Synchronization)
endif()
