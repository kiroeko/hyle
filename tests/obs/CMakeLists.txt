hyle_add_header_self_contained_test(obs)
hyle_add_test(obs SOURCES log_test.cc)
hyle_add_test(obs NAME compiled_level SOURCES compiled_level_test.cc)

# Tracy listens on every interface by default. Test processes need only a local
# viewer and must not request a Windows firewall exception merely because the
# trace-on validation binary starts the backend. This runtime setting is scoped
# to CTest; it does not change the shipped Tracy client or a user's LAN policy.
if(HYLE_TRACE)
  set_tests_properties(hyle_obs_test hyle_obs_compiled_level_test PROPERTIES
    ENVIRONMENT "TRACY_ONLY_LOCALHOST=1")
endif()
