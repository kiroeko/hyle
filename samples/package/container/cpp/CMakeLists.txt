cmake_minimum_required(VERSION 3.25)
project(hyle_package_container_cpp LANGUAGES CXX)

find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS container)

foreach(_internal_factor IN ITEMS
    container_intrusive container_concurrent container_queue container_text
    mem_reclaim)
  if(_internal_factor IN_LIST Hyle_AVAILABLE_COMPONENTS OR
     TARGET hyle::${_internal_factor})
    message(FATAL_ERROR
      "Container package leaked internal factor '${_internal_factor}' as a product")
  endif()
  if(NOT TARGET hyle::_factor_${_internal_factor})
    message(FATAL_ERROR
      "Container facade lacks detail factor '${_internal_factor}'")
  endif()
endforeach()
unset(_internal_factor)
if(NOT TARGET hyle::_factor_container)
  message(FATAL_ERROR "Container facade lacks its sequential detail factor")
endif()

add_executable(hyle_package_container_cpp main.cc)
target_compile_features(hyle_package_container_cpp PRIVATE cxx_std_23)
target_link_libraries(hyle_package_container_cpp PRIVATE hyle::container)
hyle_stage_runtime_dependencies(hyle_package_container_cpp)

enable_testing()
add_test(NAME hyle_package_container_cpp COMMAND hyle_package_container_cpp)
