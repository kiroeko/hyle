cmake_minimum_required(VERSION 3.25)
project(hyle_package_core_cpp LANGUAGES CXX)

find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS core)

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
