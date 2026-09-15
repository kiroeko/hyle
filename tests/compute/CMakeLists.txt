hyle_add_header_self_contained_test(compute)
if(NOT HYLE_STATIC)
  add_library(hyle_compute_result_fixture SHARED result_owner_producer.cc)
  target_compile_definitions(hyle_compute_result_fixture PRIVATE HYLE_RESULT_FIXTURE_BUILD=1)
  target_link_libraries(hyle_compute_result_fixture PRIVATE hyle_compute)
  hyle_apply_warnings(hyle_compute_result_fixture)
  hyle_apply_sanitizer(hyle_compute_result_fixture)
endif()
hyle_add_test(compute SOURCES
  completion_test.cc provider_test.cc
  shape_domain_test.cc shape_program_test.cc tensor_test.cc)
if(NOT HYLE_STATIC)
  target_link_libraries(hyle_compute_test PRIVATE hyle_compute_result_fixture)
endif()
target_include_directories(hyle_compute_test PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../src/compute")
