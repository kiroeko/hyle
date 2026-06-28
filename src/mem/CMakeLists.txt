# mem-01: Allocator vtable (+ alignment fix). mem-02: system backing = mimalloc,
# wrapped behind the vtable so mi_* never leaks (mem-design §96/106). The OS
# aligned-malloc path stays as the HYLE_USE_MIMALLOC=OFF fallback.
hyle_add_module(mem SOURCES system.cc)

option(HYLE_USE_MIMALLOC "system_allocator backing = vendored mimalloc (else OS aligned malloc)" ON)

if(HYLE_USE_MIMALLOC)
  set(_mi "${PROJECT_SOURCE_DIR}/third_party/mimalloc")
  # Whole library in one TU (mimalloc src/static.c #includes the rest). Static so
  # it folds into hyle_mem.dll; mi_* symbols stay internal (not re-exported).
  add_library(mimalloc_static STATIC "${_mi}/src/static.c")
  target_include_directories(mimalloc_static PUBLIC "${_mi}/include")
  set_target_properties(mimalloc_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
  # MSVC's C compiler chokes on mimalloc's atomic macros (they pass a TYPE arg;
  # C2275). mimalloc is built to compile as C++ there (MI_USE_CXX: std::atomic,
  # API stays extern "C" via mimalloc.h). GCC/Clang build it fine as C.
  if(MSVC)
    set_source_files_properties("${_mi}/src/static.c" PROPERTIES LANGUAGE CXX)
  endif()
  # Third-party C: keep our -Werror/-Wall off it.
  if(MSVC)
    target_compile_options(mimalloc_static PRIVATE /w)
  else()
    target_compile_options(mimalloc_static PRIVATE -w)
  endif()
  target_link_libraries(hyle_mem PRIVATE mimalloc_static)
  target_include_directories(hyle_mem PRIVATE "${_mi}/include")
  target_compile_definitions(hyle_mem PRIVATE HYLE_USE_MIMALLOC)
endif()
