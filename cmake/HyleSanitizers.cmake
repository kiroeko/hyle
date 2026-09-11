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
      target_compile_definitions(${target} PRIVATE
        HYLE_DETAIL_SANITIZER_ADDRESS=1)
      # /Zi is part of a valid MSVC ASan invocation (native cl emits C5072
      # without it) and gives both compiler frontends actionable reports.
      target_compile_options(${target} PRIVATE /fsanitize=address /Zi)
      get_target_property(_hyle_sanitizer_type ${target} TYPE)
      if(_hyle_sanitizer_type STREQUAL "EXECUTABLE" OR
         _hyle_sanitizer_type STREQUAL "SHARED_LIBRARY" OR
         _hyle_sanitizer_type STREQUAL "MODULE_LIBRARY")
        # CMake's MSVC-ABI Ninja rules invoke link.exe/lld-link directly rather
        # than routing the final link through cl/clang-cl. Compile-only
        # instrumentation therefore leaves every __asan_* symbol unresolved.
        # HYLE explicitly selects the dynamic-CRT runtime/thunk pair and
        # discovers those compiler-owned inputs instead of pinning a Visual
        # Studio layout or copying runtimes into the source tree.
        if(HYLE_TARGET_PROCESSOR STREQUAL "x86_64")
          set(_hyle_asan_arch x86_64)
          set(_hyle_asan_msvc_lib_arch x64)
        elseif(HYLE_TARGET_PROCESSOR STREQUAL "aarch64")
          set(_hyle_asan_arch aarch64)
          set(_hyle_asan_msvc_lib_arch arm64)
        else()
          message(FATAL_ERROR
            "Windows ASan runtime mapping is unavailable for "
            "HYLE_TARGET_PROCESSOR='${HYLE_TARGET_PROCESSOR}'")
        endif()

        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
          execute_process(
            COMMAND "${CMAKE_CXX_COMPILER}" --print-resource-dir
            OUTPUT_VARIABLE _hyle_asan_resource_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _hyle_asan_resource_result)
          if(NOT _hyle_asan_resource_result EQUAL 0 OR
             _hyle_asan_resource_dir STREQUAL "")
            message(FATAL_ERROR
              "clang-cl ASan runtime directory could not be discovered")
          endif()
          set(_hyle_asan_lib_dir
              "${_hyle_asan_resource_dir}/lib/windows")
          set(_hyle_asan_dll_dir "${_hyle_asan_lib_dir}")
        else()
          get_filename_component(_hyle_asan_compiler_dir
            "${CMAKE_CXX_COMPILER}" DIRECTORY)
          get_filename_component(_hyle_asan_toolset_dir
            "${_hyle_asan_compiler_dir}/../../.." ABSOLUTE)
          set(_hyle_asan_lib_dir
              "${_hyle_asan_toolset_dir}/lib/${_hyle_asan_msvc_lib_arch}")
          set(_hyle_asan_dll_dir "${_hyle_asan_compiler_dir}")
        endif()

        set(_hyle_asan_dynamic
            "${_hyle_asan_lib_dir}/clang_rt.asan_dynamic-${_hyle_asan_arch}.lib")
        set(_hyle_asan_dynamic_debug
            "${_hyle_asan_lib_dir}/clang_rt.asan_dbg_dynamic-${_hyle_asan_arch}.lib")
        set(_hyle_asan_thunk
            "${_hyle_asan_lib_dir}/clang_rt.asan_dynamic_runtime_thunk-${_hyle_asan_arch}.lib")
        set(_hyle_asan_dll
            "${_hyle_asan_dll_dir}/clang_rt.asan_dynamic-${_hyle_asan_arch}.dll")
        set(_hyle_asan_dll_debug
            "${_hyle_asan_dll_dir}/clang_rt.asan_dbg_dynamic-${_hyle_asan_arch}.dll")
        foreach(_hyle_asan_required IN ITEMS
            "${_hyle_asan_dynamic}" "${_hyle_asan_thunk}" "${_hyle_asan_dll}")
          if(NOT EXISTS "${_hyle_asan_required}")
            message(FATAL_ERROR
              "MSVC-ABI ASan runtime input is missing: ${_hyle_asan_required}")
          endif()
        endforeach()

        if(EXISTS "${_hyle_asan_dynamic_debug}" AND
           EXISTS "${_hyle_asan_dll_debug}")
          target_link_libraries(${target} PRIVATE
            "$<$<CONFIG:Debug>:${_hyle_asan_dynamic_debug}>"
            "$<$<NOT:$<CONFIG:Debug>>:${_hyle_asan_dynamic}>")
          set(_hyle_asan_runtime_dll
            "$<IF:$<CONFIG:Debug>,${_hyle_asan_dll_debug},${_hyle_asan_dll}>")
        else()
          target_link_libraries(${target} PRIVATE "${_hyle_asan_dynamic}")
          set(_hyle_asan_runtime_dll "${_hyle_asan_dll}")
        endif()
        target_link_options(${target} PRIVATE
          /DEBUG /INCREMENTAL:NO /INCLUDE:__asan_seh_interceptor
          "/WHOLEARCHIVE:${_hyle_asan_thunk}" /INFERASANLIBS:NO)
        add_custom_command(TARGET ${target} POST_BUILD
          COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                  "${_hyle_asan_runtime_dll}" "$<TARGET_FILE_DIR:${target}>"
          COMMENT "Copying the compiler-owned ASan runtime beside ${target}")
      endif()
      unset(_hyle_sanitizer_type)
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
    target_compile_definitions(${target} PRIVATE
      HYLE_DETAIL_SANITIZER_ADDRESS=1)
  elseif(HYLE_SANITIZER STREQUAL "tsan")
    set(flags -fsanitize=thread)
    target_compile_definitions(${target} PRIVATE
      HYLE_DETAIL_SANITIZER_THREAD=1)
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
    target_compile_definitions(${target} PRIVATE
      HYLE_DETAIL_SANITIZER_UNDEFINED=1)
  else()
    message(FATAL_ERROR "hyle_apply_sanitizer: unhandled HYLE_SANITIZER='${HYLE_SANITIZER}'")
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC PR 109224 misidentifies the allocator-first coroutine promise
    # placement new as mismatched with the required usual sized delete. Keep
    # the diagnostic visible, but do not let this compiler-model bug make every
    # otherwise valid ASan/UBSan/TSan + WERROR build impossible. Ordinary
    # sanitizer=off shipping builds retain the strict warning contract.
    target_compile_options(${target} PRIVATE
      "$<$<COMPILE_LANGUAGE:CXX>:-Wno-error=mismatched-new-delete>")
  endif()
  target_compile_options(${target} PRIVATE ${flags})
  target_link_options(${target} PRIVATE ${flags})
endfunction()
