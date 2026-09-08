set(_pico_source
  "${PROJECT_SOURCE_DIR}/third_party/picohttpparser/picohttpparser.c")
hyle_add_module(io_http SOURCES http.cc "${_pico_source}")

# picohttpparser is a private syntax state machine. HTTP owns its source,
# warnings, license and package trace; it never leaks a parser type to users.
target_include_directories(hyle_io_http SYSTEM PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/picohttpparser")
if(MSVC)
  set_source_files_properties("${_pico_source}" PROPERTIES COMPILE_OPTIONS "/w")
else()
  set_source_files_properties("${_pico_source}" PROPERTIES COMPILE_OPTIONS "-w")
endif()
hyle_register_package_dependency(
  NAME picohttpparser
  OWNER io_http
  LOCKED_DEPENDENCIES picohttpparser
  RUNTIME_FILE_SPECS
    "${PROJECT_SOURCE_DIR}/third_party/picohttpparser/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|picohttpparser-LICENSE")
unset(_pico_source)
