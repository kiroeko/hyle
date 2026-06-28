hyle_add_module(platform SOURCES wait.cc thread.cc dynlib.cc fs.cc PUBLIC_DEPS hyle_error)

# WaitOnAddress / WakeByAddress* are exported from Synchronization.lib (API set).
if(WIN32)
  target_link_libraries(hyle_platform PRIVATE Synchronization)
endif()

# dlopen/dlsym live in libdl on Linux (folded into libc on macOS/BSD).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_libraries(hyle_platform PRIVATE ${CMAKE_DL_LIBS})
endif()
