# hyle_add_module(<name> [SOURCES ...] [DEPS ...] [PUBLIC_DEPS ...])
# Declares a HYLE module as a library target `hyle_<name>` with conventional
# include dirs, warnings, sanitizer flags, and a tri-state export macro.
function(hyle_add_module name)
  cmake_parse_arguments(ARG "" "" "SOURCES;DEPS;PUBLIC_DEPS" ${ARGN})

  set(target "hyle_${name}")
  if(ARG_SOURCES)
    add_library(${target} ${ARG_SOURCES})
  else()
    add_library(${target} INTERFACE)
  endif()

  if(ARG_SOURCES)
    target_include_directories(${target}
      PUBLIC  "${PROJECT_SOURCE_DIR}/include"
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(${target} PRIVATE ${ARG_DEPS} PUBLIC ${ARG_PUBLIC_DEPS})
    hyle_apply_warnings(${target})
    hyle_apply_sanitizer(${target})
    # tri-state export macro: HYLE_<NAME>_API
    string(TOUPPER "${name}" up)
    if(BUILD_SHARED_LIBS)
      target_compile_definitions(${target} PRIVATE "HYLE_${up}_BUILD")
    else()
      target_compile_definitions(${target} PUBLIC "HYLE_STATIC")
    endif()
  else()
    target_include_directories(${target} INTERFACE "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(${target} INTERFACE ${ARG_PUBLIC_DEPS})
  endif()

  add_library(hyle::${name} ALIAS ${target})
endfunction()

# hyle_add_test(<module> SOURCES ...) -> test exe linked to hyle_<module> + gtest
function(hyle_add_test module)
  cmake_parse_arguments(ARG "" "" "SOURCES;DEPS" ${ARGN})
  set(target "hyle_${module}_test")
  add_executable(${target} ${ARG_SOURCES})
  target_link_libraries(${target} PRIVATE hyle_${module} GTest::gtest GTest::gtest_main ${ARG_DEPS})
  hyle_apply_warnings(${target})
  hyle_apply_sanitizer(${target})
  add_test(NAME ${target} COMMAND ${target})
endfunction()

# hyle_add_bench(<module> SOURCES ...) -> bench exe linked to hyle_<module> + nanobench
function(hyle_add_bench module)
  cmake_parse_arguments(ARG "" "" "SOURCES;DEPS" ${ARGN})
  set(target "hyle_${module}_bench")
  add_executable(${target} ${ARG_SOURCES})
  target_link_libraries(${target} PRIVATE hyle_${module} nanobench ${ARG_DEPS})
  hyle_apply_warnings(${target})
endfunction()
