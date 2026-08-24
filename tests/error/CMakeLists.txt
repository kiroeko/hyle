hyle_add_test(error SOURCES status_test.cc)
# close-01 sample: pure C TU (C compiler) linking only the C ABI header.
hyle_add_c_test(error c_error_test.c)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  # A downstream consumer may enable this globally; hyle_status_code remains
  # uint16_t and therefore cannot drift from the library's calling convention.
  target_compile_options(hyle_error_c_test PRIVATE -fshort-enums)
endif()
hyle_add_c_surface_test(error)
hyle_add_header_self_contained_test(error)
