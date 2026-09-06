hyle_add_header_self_contained_test(text
  HEADER_DIR container
  HEADERS
    path.h
    string_algo.h
    string_interner.h
    string_view.h)
hyle_add_test(text SOURCES
  path_test.cc
  string_algo_test.cc
  string_buf_test.cc
  string_view_test.cc)

if(HYLE_STATIC)
  hyle_add_test(text NAME isa_variants SOURCES isa_variant_test.cc
    DEPS hyle_platform)
  target_include_directories(hyle_text_isa_variants_test PRIVATE
    "${PROJECT_SOURCE_DIR}")
endif()
