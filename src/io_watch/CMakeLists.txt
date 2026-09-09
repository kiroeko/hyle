set(_io_watch_sources watch.cc)
if(WIN32)
  list(APPEND _io_watch_sources watch_windows.cc)
elseif(APPLE)
  list(APPEND _io_watch_sources watch_macos.cc)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND _io_watch_sources watch_linux.cc)
else()
  message(FATAL_ERROR
    "io_watch has no native backend for ${CMAKE_SYSTEM_NAME}")
endif()

hyle_add_module(io_watch SOURCES ${_io_watch_sources})

if(APPLE)
  target_link_libraries(hyle_io_watch PRIVATE
    "-framework CoreServices"
    "-framework CoreFoundation")
endif()

unset(_io_watch_sources)
