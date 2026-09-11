# The AUDITED transition establishes ownership before provider product code is
# added. The implementation commit replaces this empty aggregate with the
# provider's public-header and correctness tests.
add_custom_target(hyle_compute_provider_onednn_tests)
set_target_properties(hyle_compute_provider_onednn_tests PROPERTIES
  FOLDER "hyle/validation")
if(TARGET hyle_tests)
  add_dependencies(hyle_tests hyle_compute_provider_onednn_tests)
endif()
