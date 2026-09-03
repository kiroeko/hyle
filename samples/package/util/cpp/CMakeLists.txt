cmake_minimum_required(VERSION 3.25)
project(hyle_util_cpp_package_probe LANGUAGES CXX)

find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS util)

foreach(_internal_factor IN ITEMS container_intrusive util_core)
  if(_internal_factor IN_LIST Hyle_AVAILABLE_COMPONENTS OR
     TARGET hyle::${_internal_factor})
    message(FATAL_ERROR
      "Util package leaked internal factor '${_internal_factor}' as a product")
  endif()
endforeach()
unset(_internal_factor)

if(NOT TARGET hyle::_factor_util_core)
  message(FATAL_ERROR "Util facade lacks detail factor 'util_core'")
endif()
if(TARGET hyle::_factor_container_intrusive)
  message(FATAL_ERROR
    "Util facade unexpectedly selected detail factor 'container_intrusive'")
endif()

add_executable(hyle_util_cpp_package_probe main.cc)
target_link_libraries(hyle_util_cpp_package_probe PRIVATE hyle::util)
target_compile_features(hyle_util_cpp_package_probe PRIVATE cxx_std_23)
hyle_stage_runtime_dependencies(hyle_util_cpp_package_probe)

enable_testing()
add_test(NAME hyle_util_cpp_package_probe COMMAND hyle_util_cpp_package_probe)
