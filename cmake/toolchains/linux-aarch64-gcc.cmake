# Reference Linux AArch64 cross toolchain for the whole HYLE target graph.
#
# Prefer the standard CMake entry:
#   cmake -S . -B build-cross \
#     -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/linux-aarch64-gcc.cmake
# Override CMAKE_C_COMPILER/CMAKE_CXX_COMPILER directly when a distribution uses
# different executable names. HYLE_CROSS_TRIPLE and HYLE_CROSS_GCC_VERSION only
# control discovery; they are not product options or embedded ABI metadata.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(HYLE_CROSS_TRIPLE "aarch64-linux-gnu" CACHE STRING
    "GNU target triple used to discover the AArch64 cross compiler")
set(HYLE_CROSS_GCC_VERSION "14" CACHE STRING
    "Preferred GCC major suffix for the AArch64 cross compiler")

if(NOT CMAKE_C_COMPILER)
  find_program(_hyle_cross_c
    NAMES
      "${HYLE_CROSS_TRIPLE}-gcc-${HYLE_CROSS_GCC_VERSION}"
      "${HYLE_CROSS_TRIPLE}-gcc")
  if(NOT _hyle_cross_c)
    message(FATAL_ERROR
      "No ${HYLE_CROSS_TRIPLE} GCC C compiler was found. Install the cross "
      "toolchain or pass -DCMAKE_C_COMPILER=<path>.")
  endif()
  set(CMAKE_C_COMPILER "${_hyle_cross_c}" CACHE FILEPATH "" FORCE)
endif()

if(NOT CMAKE_CXX_COMPILER)
  find_program(_hyle_cross_cxx
    NAMES
      "${HYLE_CROSS_TRIPLE}-g++-${HYLE_CROSS_GCC_VERSION}"
      "${HYLE_CROSS_TRIPLE}-g++")
  if(NOT _hyle_cross_cxx)
    message(FATAL_ERROR
      "No ${HYLE_CROSS_TRIPLE} GCC C++ compiler was found. Install the cross "
      "toolchain or pass -DCMAKE_CXX_COMPILER=<path>.")
  endif()
  set(CMAKE_CXX_COMPILER "${_hyle_cross_cxx}" CACHE FILEPATH "" FORCE)
endif()

# Cross configuration must never attempt to execute a target binary. Compiler
# feature probes that only need to compile are represented as static libraries.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Host programs (CMake, Python, shader tools) come from the build machine;
# libraries, headers, and packages must come from the target compiler/sysroot.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
