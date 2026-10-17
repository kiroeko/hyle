cmake_minimum_required(VERSION 3.25)
project(hyle_sample_installed_sdk LANGUAGES CXX)

find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS mem)

add_executable(installed_mem_cpp_consumer system_allocator.cc)
target_link_libraries(installed_mem_cpp_consumer PRIVATE hyle::mem)
hyle_stage_runtime(installed_mem_cpp_consumer)

if(NOT Hyle_LOADED_COMPONENTS)
  message(FATAL_ERROR "installed HYLE package loaded no components")
endif()

enable_testing()

foreach(_component IN LISTS Hyle_LOADED_COMPONENTS)
  if(NOT TARGET hyle::${_component})
    message(FATAL_ERROR
      "installed HYLE package loaded '${_component}' but does not export "
      "hyle::${_component}")
  endif()
  set(_consumer "installed_${_component}_target_consumer")
  add_executable(${_consumer} all_components.cc)
  target_link_libraries(${_consumer} PRIVATE hyle::${_component})
  hyle_stage_runtime(${_consumer})
  add_test(NAME ${_consumer} COMMAND ${_consumer})
endforeach()

add_test(NAME installed_mem_cpp_consumer COMMAND installed_mem_cpp_consumer)
