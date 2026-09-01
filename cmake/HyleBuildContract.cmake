include_guard(GLOBAL)

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
