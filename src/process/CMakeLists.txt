set(_process_sources process.cc)
if(WIN32)
  list(APPEND _process_sources process_windows.cc)
else()
  list(APPEND _process_sources process_posix.cc)
endif()

hyle_add_module(process SOURCES ${_process_sources})
