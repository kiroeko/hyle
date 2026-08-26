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
  # HYLE does not throw errors across its API, but public noexcept convenience
  # adapters catch std::bad_alloc/length_error and translate them to Status.
  # MSVC therefore needs an explicit standard C++ unwind model; leaving it at
  # the compiler default makes current STL headers emit C4530 and gives source
  # consumers a different contract. HYLE has no public C source contract, so
  # these usage requirements stay on downstream C++ translation units and do
  # not alter unrelated C files in an embedding target.
  target_compile_options(hyle_build_contract INTERFACE
    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/Zc:__cplusplus>"
    "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
    "$<$<COMPILE_LANGUAGE:CXX>:/EHsc>"
    # Public variadic-macro surfaces use conforming preprocessing.
    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/Zc:preprocessor>")
endif()

if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  # HYLE's x86-64-v2 public baseline enables the SSSE3 path in header-inline
  # container probes. Propagate it so those functions cannot acquire two ODR
  # shapes between the library and its consumer. Compiler selection must stay
  # in generator expressions: this target is exported, and a Windows-ABI SDK
  # built by clang-cl remains consumable by cl.exe (and vice versa).
  if(WIN32)
    target_compile_options(hyle_build_contract INTERFACE
      "$<$<COMPILE_LANG_AND_ID:CXX,Clang>:/clang:-mssse3>")
  else()
    target_compile_options(hyle_build_contract INTERFACE
      "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-mssse3>")
  endif()
endif()

add_library(hyle::_build_contract ALIAS hyle_build_contract)
