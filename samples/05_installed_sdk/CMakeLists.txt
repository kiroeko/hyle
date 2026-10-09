cmake_minimum_required(VERSION 3.25)
project(hyle_sample_installed_sdk LANGUAGES CXX)

find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS mem)

add_executable(installed_mem_cpp_consumer system_allocator.cc)
target_link_libraries(installed_mem_cpp_consumer PRIVATE hyle::mem)
hyle_stage_runtime(installed_mem_cpp_consumer)

if(NOT Hyle_AVAILABLE_COMPONENTS)
  message(FATAL_ERROR "installed HYLE package reported no components")
endif()

foreach(_internal_factor IN ITEMS container_intrusive util_core)
  if(_internal_factor IN_LIST Hyle_AVAILABLE_COMPONENTS OR
     TARGET hyle::${_internal_factor})
    message(FATAL_ERROR
      "installed HYLE package leaked internal factor '${_internal_factor}' "
      "as a consumer product")
  endif()
  if(NOT TARGET hyle::_factor_${_internal_factor})
    message(FATAL_ERROR
      "all-product SDK lacks detail target for '${_internal_factor}'")
  endif()
endforeach()
unset(_internal_factor)

enable_testing()

foreach(_component IN LISTS Hyle_AVAILABLE_COMPONENTS)
  if(NOT TARGET hyle::${_component})
    message(FATAL_ERROR
      "installed HYLE package reports '${_component}' but does not export "
      "hyle::${_component}")
  endif()
  set(_consumer "installed_${_component}_target_consumer")
  add_executable(${_consumer} all_components.cc)
  target_link_libraries(${_consumer} PRIVATE hyle::${_component})
  hyle_stage_runtime(${_consumer})
  add_test(NAME ${_consumer} COMMAND ${_consumer})
endforeach()

add_test(NAME installed_mem_cpp_consumer COMMAND installed_mem_cpp_consumer)
