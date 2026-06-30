# hyle_apply_sanitizer(<target>): apply HYLE_SANITIZER (off|asan|tsan|ubsan).
# Sanitizers are mutually exclusive (charter: separate CI builds). The VALUE is validated
# + normalized in the top-level CMakeLists (aliases mapped, unknown values hard-fail), so
# by here it is always one of the four canonical names -- a defensive else still FATALs
# rather than silently emit no flags (an empty flag set = a sanitizer build that proves
# nothing, the exact false-pass trap we are guarding against).
function(hyle_apply_sanitizer target)
  if(HYLE_SANITIZER STREQUAL "off")
    return()
  endif()
  if(MSVC)
    if(HYLE_SANITIZER STREQUAL "asan")
      target_compile_options(${target} PRIVATE /fsanitize=address)
    else()
      # tsan/ubsan don't exist on MSVC. FAIL LOUDLY rather than build uninstrumented:
      # asking for tsan and silently getting nothing is a false pass. Use a GCC/Clang
      # build (e.g. WSL) for tsan/ubsan.
      message(FATAL_ERROR "MSVC supports only HYLE_SANITIZER=asan; '${HYLE_SANITIZER}' "
        "needs a GCC/Clang build (refusing to build uninstrumented = false pass).")
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
  else()
    message(FATAL_ERROR "hyle_apply_sanitizer: unhandled HYLE_SANITIZER='${HYLE_SANITIZER}'")
  endif()
  target_compile_options(${target} PRIVATE ${flags})
  target_link_options(${target} PRIVATE ${flags})
endfunction()
