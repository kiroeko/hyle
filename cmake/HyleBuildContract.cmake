include_guard(GLOBAL)

# The public scripts use one MSVC diagnostic language for configuration and
# compilation. Repair a retained Ninja parser in this configure scope, without
# changing compiler commands or CMake's compiler-identification metadata.
if(MSVC AND CMAKE_GENERATOR MATCHES "^Ninja"
    AND "$ENV{VSLANG}" STREQUAL "1033"
    AND NOT CMAKE_CL_SHOWINCLUDES_PREFIX STREQUAL "Note: including file: ")
  block(SCOPE_FOR VARIABLES PROPAGATE CMAKE_CL_SHOWINCLUDES_PREFIX)
    # This helper has the same (language, userflags) signature since CMake 3.25.
    include(CMakeDetermineCompilerId)
    separate_arguments(CMAKE_CXX_COMPILER_ID_ARG1 NATIVE_COMMAND "${CMAKE_CXX_COMPILER_ARG1}")
    CMAKE_DETERMINE_MSVC_SHOWINCLUDES_PREFIX(CXX "")
    if(NOT CMAKE_CXX_CL_SHOWINCLUDES_PREFIX STREQUAL "Note: including file: ")
      message(FATAL_ERROR
        "Cannot detect the English MSVC /showIncludes prefix with VSLANG=1033. "
        "Check the selected compiler's English language support.")
    endif()
    set(CMAKE_CL_SHOWINCLUDES_PREFIX "${CMAKE_CXX_CL_SHOWINCLUDES_PREFIX}")
  endblock()
endif()

if(MSVC AND CMAKE_GENERATOR MATCHES "^Ninja" AND "$ENV{VSLANG}" STREQUAL "1033")
  include("${CMAKE_CURRENT_LIST_DIR}/HyleNinjaHeaderDeps.cmake")
  cmake_language(DEFER CALL _hyle_recover_ninja_headerdeps)
endif()

# Minimum public compile-semantics contract shared by every HYLE module and
# exported to package consumers. These are not warning preferences: public
# inline headers must see the same language and preprocessing contract in HYLE
# itself and in downstream TUs. Product-specific ISA and exception policy stay
# on their owning target instead of becoming a global usage requirement.
add_library(hyle_build_contract INTERFACE)
set_target_properties(hyle_build_contract PROPERTIES
  EXPORT_NAME "_build_contract"
  FOLDER "hyle/foundation")

if(MSVC)
  # Report the real C++ language level and keep source/execution text UTF-8.
  # Exception mode belongs to the embedding target/toolchain. Normal MSVC
  # projects already choose /EHsc through CMake; forcing it here would override
  # a Core-only consumer that intentionally compiles without exceptions.
  target_compile_options(hyle_build_contract INTERFACE
    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/Zc:__cplusplus>"
    "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
    # Public variadic-macro surfaces use conforming preprocessing.
    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/Zc:preprocessor>")
endif()

add_library(hyle::_build_contract ALIAS hyle_build_contract)
