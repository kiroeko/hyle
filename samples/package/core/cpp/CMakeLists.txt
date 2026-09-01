cmake_minimum_required(VERSION 3.25)
project(hyle_package_core_cpp LANGUAGES CXX)

find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS core)

if(NOT TARGET hyle::_factor_core)
  message(FATAL_ERROR "installed Core facade lacks its package-detail factor")
endif()
if(TARGET hyle::container_intrusive OR TARGET hyle::util_core OR
   "container_intrusive" IN_LIST Hyle_AVAILABLE_COMPONENTS OR
   "util_core" IN_LIST Hyle_AVAILABLE_COMPONENTS)
  message(FATAL_ERROR "installed package leaked an internal factor as a product")
endif()

get_target_property(_hyle_core_links hyle::core INTERFACE_LINK_LIBRARIES)
if(_hyle_core_links MATCHES "_trace_config|_testing_config|_tracy_client")
  message(FATAL_ERROR
    "installed Core unexpectedly carries instrumentation: ${_hyle_core_links}")
endif()
unset(_hyle_core_links)

get_target_property(_hyle_core_build_options
  hyle::_build_contract INTERFACE_COMPILE_OPTIONS)
if(_hyle_core_build_options MATCHES "mssse3|/EHsc")
  message(FATAL_ERROR
    "Core package leaked product ISA/exception policy: "
    "${_hyle_core_build_options}")
endif()
unset(_hyle_core_build_options)

add_executable(hyle_package_core_cpp main.cc)
target_compile_features(hyle_package_core_cpp PRIVATE cxx_std_23)
target_link_libraries(hyle_package_core_cpp PRIVATE hyle::core)

add_executable(hyle_package_core_no_exceptions no_exceptions.cc)
target_compile_features(hyle_package_core_no_exceptions PRIVATE cxx_std_23)
target_link_libraries(hyle_package_core_no_exceptions PRIVATE hyle::core)
if(MSVC OR CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
  target_compile_options(hyle_package_core_no_exceptions PRIVATE /EHs-c-)
else()
  target_compile_options(hyle_package_core_no_exceptions PRIVATE
    -fno-exceptions)
endif()

enable_testing()
add_test(NAME hyle_package_core_cpp COMMAND hyle_package_core_cpp)
add_test(NAME hyle_package_core_no_exceptions
  COMMAND hyle_package_core_no_exceptions)
