cmake_minimum_required(VERSION 3.25)
project(g01_cpu_resident_p1 LANGUAGES C CXX)
set(HYLE_CPU_ORIGINAL_REFERENCE_SOURCE "" CACHE PATH "Authenticated original CPU ggml peer checkout")
include("${CMAKE_CURRENT_LIST_DIR}/../cpu-CMakeLists.txt")
