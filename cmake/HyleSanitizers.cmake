# hyle_apply_sanitizer(<target>): apply HYLE_SANITIZER (off|asan|tsan|ubsan).
# Sanitizers are mutually exclusive (charter: separate CI builds).
function(hyle_apply_sanitizer target)
  if(HYLE_SANITIZER STREQUAL "off")
    return()
  endif()
  if(MSVC)
    if(HYLE_SANITIZER STREQUAL "asan")
      target_compile_options(${target} PRIVATE /fsanitize=address)
    else()
      message(WARNING "MSVC supports only asan; HYLE_SANITIZER=${HYLE_SANITIZER} ignored")
    endif()
    return()
  endif()
  if(HYLE_SANITIZER STREQUAL "asan")
    set(flags -fsanitize=address -fno-omit-frame-pointer)
  elseif(HYLE_SANITIZER STREQUAL "tsan")
    set(flags -fsanitize=thread)
    # GCC warns (-Wtsan) that std::atomic_thread_fence "is not supported" under
    # TSan: TSan cannot model STANDALONE fences, so it won't observe the ordering
    # our Chase-Lev deque (Le 2013) relies on. This is a known, universal
    # limitation (crossbeam/folly hit it too) -- NOT a bug. We silence the
    # repeated warning here so TSan-WERROR CI is usable; the LIMITATION itself is
    # documented in container-07's doc: fence correctness rests on the faithful
    # port + stress + (future) a model checker / ARM run, NOT on TSan.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      list(APPEND flags -Wno-tsan)
    endif()
  elseif(HYLE_SANITIZER STREQUAL "ubsan")
    set(flags -fsanitize=undefined -fno-sanitize-recover=all)
  endif()
  target_compile_options(${target} PRIVATE ${flags})
  target_link_options(${target} PRIVATE ${flags})
endfunction()
