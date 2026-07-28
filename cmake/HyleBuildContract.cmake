include_guard(GLOBAL)

# Public compile-semantics contract shared by every HYLE module and exported to
# package consumers. These are not warning preferences: public inline headers
# must see the same language/ISA baseline in HYLE itself and in downstream TUs.
add_library(hyle_build_contract INTERFACE)
set_target_properties(hyle_build_contract PROPERTIES
  EXPORT_NAME "_build_contract"
  FOLDER "hyle/foundation")

if(MSVC)
  # Report the real C++ language level and keep source/execution text UTF-8.
  # Language genexes keep the switches away from MASM fcontext sources.
  target_compile_options(hyle_build_contract INTERFACE
    "$<$<COMPILE_LANGUAGE:C,CXX>:/Zc:__cplusplus>"
    "$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  # HYLE's x86-64-v2 public baseline enables the SSSE3 path in header-inline
  # container probes. Propagate it so those functions cannot acquire two ODR
  # shapes between the library and its consumer.
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mssse3" HYLE_CXX_HAS_MSSSE3)
  if(HYLE_CXX_HAS_MSSSE3)
    target_compile_options(hyle_build_contract INTERFACE
      "$<$<COMPILE_LANGUAGE:C,CXX>:-mssse3>")
  endif()
endif()

add_library(hyle::_build_contract ALIAS hyle_build_contract)
