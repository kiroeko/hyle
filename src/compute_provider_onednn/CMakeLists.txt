if(HYLE_COMPUTE_ONEDNN_ENABLED)
  hyle_add_module(compute_provider_onednn SOURCES
    brgemm_internal.cc
    onednn_provider.cc)
  target_include_directories(hyle_compute_provider_onednn PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../compute")
  target_link_libraries(hyle_compute_provider_onednn PRIVATE dnnl)
  target_compile_definitions(hyle_compute_provider_onednn
    PUBLIC HYLE_COMPUTE_ONEDNN=1)
  hyle_register_package_dependency(
    NAME onednn
    OWNER compute_provider_onednn
    FEATURES compute_onednn
    STATIC_EXPORT_TARGETS dnnl
    LOCKED_DEPENDENCIES onednn
    RUNTIME_FILE_SPECS
      "${HYLE_ONEDNN_SOURCE_DIR}/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|oneDNN-LICENSE"
    STATIC_DEVELOPMENT_DIRECTORY_SPECS
      "${HYLE_ONEDNN_SOURCE_DIR}/include/oneapi|${HYLE_INSTALL_INCLUDEDIR}"
      "${HYLE_ONEDNN_BINARY_DIR}/include/oneapi|${HYLE_INSTALL_INCLUDEDIR}")
else()
  hyle_add_module(compute_provider_onednn OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_provider_onednn PROPERTY
    HYLE_INTERNAL_ONLY_ABSENT TRUE)
endif()
