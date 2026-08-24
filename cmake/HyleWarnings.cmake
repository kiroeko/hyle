# hyle_apply_warnings(<target>): unified warning flags. HYLE_WERROR gates -Werror.
# Flags are scoped to C/CXX so they are never handed to the assembler (ml64
# rejects /W4 etc.) when a target also compiles fcontext .asm.
function(hyle_apply_warnings target)
  if(MSVC)
    # cl.exe /Zc:preprocessor: conforming preprocessor (__VA_OPT__, correct
    # __VA_ARGS__) needed by HYLE's macro layer. CMake's MSVC variable also
    # covers clang-cl (the ABI/frontend family); clang-cl diagnoses this cl-only
    # switch as unused under /WX, so select it by compiler ID per language.
    # /wd4324: "structure padded due to alignas" -- expected, not a bug: HYLE
    # uses cache-line alignment pervasively (conventions §7 防伪共享); the
    # padding is the whole point (abseil/folly suppress it the same way).
    # One quoted genex per flag (a ';' inside $<...> gets list-split early).
    target_compile_options(${target} PRIVATE
      "$<$<COMPILE_LANGUAGE:C,CXX>:/W4>"
      "$<$<COMPILE_LANGUAGE:C,CXX>:/permissive->"
      "$<$<AND:$<COMPILE_LANGUAGE:C>,$<C_COMPILER_ID:MSVC>>:/Zc:preprocessor>"
      "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/Zc:preprocessor>"
      "$<$<COMPILE_LANGUAGE:C,CXX>:/wd4324>")
    if(HYLE_WERROR)
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:/WX>")
    endif()
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    if(HYLE_WERROR)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()
endfunction()
