# Borrowed text, lexical paths, interning and Unicode codecs. Public types stay
# under hyle/container so users continue to see one Container product; this
# target is an internal dependency/delivery boundary.
include(HyleDispatch)

set(_container_text_highway_targets scalar sse4 avx2 neon)
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  set(HYLE_CONTAINER_TEXT_X64 TRUE)
  if(NOT (MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"))
    list(APPEND _container_text_highway_targets avx512)
    set(HYLE_CONTAINER_TEXT_HAS_AVX512 TRUE)
  endif()
elseif(HYLE_TARGET_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  set(HYLE_CONTAINER_TEXT_ARM64 TRUE)
endif()

hyle_add_module(container_text
  SOURCES path.cc string_algo.cc string_buf.cc string_view.cc)
if(HYLE_CONTAINER_TEXT_HAS_AVX512)
  target_compile_definitions(hyle_container_text PRIVATE
    HYLE_CONTAINER_TEXT_HAS_AVX512=1)
endif()

hyle_add_dispatched_kernel(HIGHWAY OWN_TARGET_SET
  TARGET hyle_container_text
  NAME container_text_string
  IMPL "src/container_text/str_search_impl.h"
  TARGETS ${_container_text_highway_targets})

set_target_properties(hyle_container_text PROPERTIES
  INTERPROCEDURAL_OPTIMIZATION OFF)
if(NOT WIN32)
  # All non-HYLE symbols in the vendored amalgamation and generated variant TUs
  # stay private. HYLE_CONTAINER_TEXT_API explicitly restores default
  # visibility for the public C++ product symbols.
  set_target_properties(hyle_container_text PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
endif()

target_compile_definitions(hyle_container_text PRIVATE
  HYLE_CONTAINER_TEXT_SIMDUTF)
target_include_directories(hyle_container_text SYSTEM PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/simdutf")
target_sources(hyle_container_text PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/simdutf/simdutf.cpp")
if(MSVC)
  set_source_files_properties("${PROJECT_SOURCE_DIR}/third_party/simdutf/simdutf.cpp"
    PROPERTIES COMPILE_OPTIONS "/w")
else()
  set_source_files_properties("${PROJECT_SOURCE_DIR}/third_party/simdutf/simdutf.cpp"
    PROPERTIES COMPILE_OPTIONS "-w")
endif()

hyle_register_package_dependency(
  NAME container_text_simdutf
  OWNER container_text
  LOCKED_DEPENDENCIES simdutf
  RUNTIME_FILE_SPECS
    "${PROJECT_SOURCE_DIR}/third_party/simdutf/LICENSE-MIT|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|simdutf-LICENSE-MIT"
    "${PROJECT_SOURCE_DIR}/third_party/simdutf/LICENSE-APACHE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|simdutf-LICENSE-APACHE")
