cmake_minimum_required(VERSION 3.25)
project(hyle_package_container_intrusive_cpp LANGUAGES CXX)

find_package(Hyle 0.1 REQUIRED COMPONENTS container_intrusive)

add_executable(hyle_package_container_intrusive_cpp main.cc)
target_link_libraries(hyle_package_container_intrusive_cpp
  PRIVATE hyle::container_intrusive)

enable_testing()
add_test(NAME hyle_package_container_intrusive_cpp
  COMMAND hyle_package_container_intrusive_cpp)
