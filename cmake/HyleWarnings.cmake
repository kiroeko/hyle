# hyle_apply_warnings(<target>): unified warning flags. HYLE_WERROR gates -Werror.
function(hyle_apply_warnings target)
  if(MSVC)
    # /Zc:preprocessor: conforming preprocessor (__VA_OPT__, correct __VA_ARGS__)
    # needed by HYLE's X-macro / variadic macro layer. Scoped to HYLE targets
    # only -- applying it globally makes the Windows SDK headers spew C5105.
    target_compile_options(${target} PRIVATE /W4 /permissive- /Zc:preprocessor)
    if(HYLE_WERROR)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    if(HYLE_WERROR)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()
endfunction()
