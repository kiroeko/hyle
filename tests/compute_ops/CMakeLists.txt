hyle_add_header_self_contained_test(compute_ops
  HEADER_DIR compute
  HEADERS ops_visibility.h tensor_desc.h operation.h)

set(_hyle_compute_ops_test_deps)
if(NOT HYLE_STATIC)
  add_library(hyle_compute_ops_custom_fixture SHARED
    custom_operation_producer.cc)
  set_target_properties(hyle_compute_ops_custom_fixture PROPERTIES
    FOLDER "hyle/tests/compute_ops")
  target_compile_definitions(hyle_compute_ops_custom_fixture PRIVATE
    HYLE_COMPUTE_OPS_FIXTURE_BUILD=1)
  target_link_libraries(hyle_compute_ops_custom_fixture PRIVATE
    hyle_compute_ops)
  hyle_apply_warnings(hyle_compute_ops_custom_fixture)
  hyle_apply_sanitizer(hyle_compute_ops_custom_fixture)
  list(APPEND _hyle_compute_ops_test_deps hyle_compute_ops_custom_fixture)
endif()

if(_hyle_compute_ops_test_deps)
  hyle_add_test(compute_ops
    SOURCES operation_test.cc tensor_desc_test.cc
    DEPS ${_hyle_compute_ops_test_deps})
else()
  hyle_add_test(compute_ops
    SOURCES operation_test.cc tensor_desc_test.cc)
endif()

unset(_hyle_compute_ops_test_deps)
