# hyle_apply_warnings(<target>): unified warning flags. HYLE_WERROR gates -Werror.
function(hyle_apply_warnings target)
  if(MSVC)
    # /Zc:preprocessor: conforming preprocessor (__VA_OPT__, correct __VA_ARGS__)
    # needed by HYLE's X-macro / variadic macro layer. Scoped to HYLE targets
    # only -- applying it globally makes the Windows SDK headers spew C5105.
    # /wd4324: "structure padded due to alignas" -- expected, not a bug: HYLE
    # uses cache-line alignment pervasively (conventions §7 防伪共享); the
    # padding is the whole point (abseil/folly suppress it the same way).
    target_compile_options(${target} PRIVATE /W4 /permissive- /Zc:preprocessor /wd4324)
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
